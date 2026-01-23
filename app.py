import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import openai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

if not OPENROUTER_API_KEY or not MODEL_NAME or not ADMIN_PASSWORD:
    st.error("Missing environment variables.")
    st.stop()

openai.api_key = OPENROUTER_API_KEY
openai.api_base = "https://openrouter.ai/api/v1"

# --------------------------------------------------
# App config
# --------------------------------------------------
st.set_page_config(page_title="Company AI Assistant")

UPLOAD_DIR = "data/uploads"
INDEX_DIR = "data/faiss_index"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# --------------------------------------------------
# ADMIN INGEST (BACKEND-LEVEL)
# --------------------------------------------------
st.sidebar.title("")
admin_input = st.sidebar.text_input("Admin access", type="password")

if admin_input == ADMIN_PASSWORD:
    st.sidebar.success("Admin mode enabled")

    uploaded_file = st.sidebar.file_uploader(
        "Upload company PDF", type=["pdf"]
    )

    if uploaded_file:
        pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        reader = PdfReader(pdf_path)
        text = ""

        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        db = FAISS.from_texts(chunks, embeddings)
        db.save_local(INDEX_DIR)

        st.sidebar.success("Document indexed successfully")

# --------------------------------------------------
# USER CHAT UI (CHAT-ONLY)
# --------------------------------------------------
st.title("Company AI Assistant")
st.caption("Ask questions about the company")

question = st.text_input("Ask a question")

if question:
    if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        st.error("Knowledge base not ready.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(INDEX_DIR, embeddings)
    docs = db.similarity_search(question, k=4)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a company AI assistant.

Rules:
- Answer ONLY using the context below.
- If the answer is not found, say exactly:
  "I don’t have this information in the provided documents."
- If the answer contains multiple items, format them as a bullet list using hyphens (-).
- If the answer is a single statement, write it as one sentence.

Context:
{context}

Question:
{question}

Answer:
"""


    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Strict document-based assistant"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Company RAG Bot"
        }
    )

    st.subheader("Answer")
    st.markdown(response["choices"][0]["message"]["content"])


