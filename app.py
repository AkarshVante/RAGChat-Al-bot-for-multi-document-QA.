# app.py

import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

# Load env vars
from dotenv import load_dotenv
load_dotenv()

# ========== Try loading OpenAI, fall back to Anthropic/HuggingFace if rate limited ==========
from openai.error import OpenAIError, RateLimitError
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# Fallbacks
from langchain.chat_models import ChatAnthropic
from langchain.embeddings import HuggingFaceEmbeddings

# Astra DB setup
import cassio
from cassandra.cluster import NoHostAvailable

ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
TABLE_NAME = "demo_table"

@st.cache_resource(show_spinner=False)
def load_llm_and_embeddings():
    try:
        llm = ChatOpenAI()
        embeddings = OpenAIEmbeddings()
        st.success("✅ Using OpenAI for LLM and Embeddings")
    except (OpenAIError, RateLimitError):
        st.warning("⚠️ OpenAI failed. Falling back to Anthropic + HuggingFace.")
        llm = ChatAnthropic()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embeddings

@st.cache_resource(show_spinner=False)
def init_vector_store(embeddings, texts):
    try:
        cassio.init(token=ASTRA_DB_TOKEN, database_id=ASTRA_DB_ID)
        vectorstore = Cassandra(
            embedding=embeddings,
            table_name=TABLE_NAME,
            session=None,
            keyspace=None
        )
        if not vectorstore.get_all_documents():
            vectorstore.add_texts(texts[:10])  # limit initial load
        return vectorstore
    except NoHostAvailable:
        st.error("❌ Failed to connect to AstraDB. Check credentials or DB status.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Cassandra error: {e}")
        st.stop()

# ========== UI ==========
st.set_page_config(page_title="PDF QA Bot", layout="wide")
st.title("📄🔎 PDF Chatbot with LangChain + Astra DB")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf", help="Supports one PDF at a time")
question = st.text_input("Ask a question about the document")

# ========== Main Logic ==========
if uploaded_file:
    try:
        pdf_reader = PdfReader(uploaded_file)
        raw_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        if not raw_text:
            st.error("❌ No extractable text found in PDF.")
        else:
            st.success("✅ PDF Loaded")
    except Exception as e:
        st.error(f"❌ Failed to read PDF: {e}")
        st.stop()

    # Split and embed text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_text(raw_text)

    llm, embeddings = load_llm_and_embeddings()
    vectorstore = init_vector_store(embeddings, texts)

    # Ask question
    if question:
        try:
            index = VectorStoreIndexWrapper(vectorstore=vectorstore)
            answer = index.query(question=question, llm=llm)
            st.subheader("🧠 Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"❌ Query failed: {e}")

