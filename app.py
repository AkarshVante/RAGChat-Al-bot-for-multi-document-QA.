import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import cassio
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")

# Init AstraDB
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Streamlit UI
st.title("📄 PDF Q&A with LangChain and AstraDB")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    raw_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)

    splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200)
    texts = splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)

    vectorstore = Cassandra(embedding=embeddings, table_name="demo_table", session=None, keyspace=None)
    vectorstore.add_texts(texts[:10])
    st.success(f"Inserted {len(texts[:10])} text chunks into AstraDB.")

    index = VectorStoreIndexWrapper(vectorstore=vectorstore)

    question = st.text_input("Ask something about the document:")
    if question:
        response = index.query(question=question, llm=llm)
        st.write("### 💬 Answer:")
        st.write(response)
