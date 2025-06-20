# app.py
import os
import time
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import Cassandra
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
import cassio
from openai.error import RateLimitError, AuthenticationError

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")

# Initialize DB connection
try:
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
except Exception as e:
    st.error("Could not initialize AstraDB connection. Please check your credentials.")
    st.stop()

# Initialize OpenAI components with error handling
try:
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
except AuthenticationError:
    st.error("OpenAI API authentication failed. Check your API key.")
    st.stop()

# UI layout
st.set_page_config(page_title="Budget Speech QA Bot", layout="centered")
st.title("📝 Budget Speech Q&A Bot")
st.write("Upload a budget speech PDF, and ask questions about its contents.")

# PDF upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Reading PDF..."):
        reader = PdfReader(uploaded_file)
        raw_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Text splitting
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    # Insert into vector store with error handling
    try:
        vectorstore = Cassandra(embedding=embeddings, table_name="demo_table", session=None, keyspace=None)
        try:
            vectorstore.add_texts(texts[:5])  # Limit chunks for demo
        except RateLimitError:
            st.warning("Rate limit exceeded. Retrying after 10 seconds...")
            time.sleep(10)
            vectorstore.add_texts(texts[:5])
        st.success("Text data embedded and stored successfully.")
    except Exception as e:
        st.error(f"Vector store initialization failed: {e}")
        st.stop()

    # Ask a question
    st.subheader("Ask a question about the speech")
    user_query = st.text_input("Your question")

    if user_query:
        with st.spinner("Generating response..."):
            try:
                index = VectorStoreIndexWrapper(vectorstore=vectorstore)
                response = index.query(question=user_query, llm=llm)
                st.success("Answer:")
                st.write(response)
            except RateLimitError:
                st.error("OpenAI rate limit hit. Try again later.")
            except Exception as e:
                st.error(f"Query failed: {e}")

        with st.expander("See most relevant passage"):
            try:
                result = vectorstore.similarity_search(query=user_query, k=1)
                st.markdown(result[0].page_content)
            except Exception:
                st.warning("Could not retrieve relevant passage.")
