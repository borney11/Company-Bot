import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import openai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# ------------------ ENV ------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

openai.api_key = OPENROUTER_API_KEY
openai.api_base = "https://openrouter.ai/api/v1"

# ------------------ PATHS ------------------
UPLOAD_DIR = "data/uploads"
INDEX_DIR = "data/faiss_index_final"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ------------------ STREAMLIT ------------------
st.set_page_config(page_title="Company AI Assistant")

# ------------------ SESSION STATE ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "trained" not in st.session_state:
    st.session_state.trained = False  # 🔒 HARD TRAINING FLAG

# ------------------ ADMIN INGEST ------------------
st.sidebar.title("")
admin_input = st.sidebar.text_input("Admin access", type="password")

if admin_input == ADMIN_PASSWORD:
    st.sidebar.success("Admin mode enabled")

    uploaded_file = st.sidebar.file_uploader("Upload company PDF", type=["pdf"])

    if uploaded_file:
        pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        reader = PdfReader(pdf_path)
        text = "".join(
            page.extract_text() or "" for page in reader.pages
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80
        )
        chunks = splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        db = FAISS.from_texts(chunks, embeddings)
        db.save_local(INDEX_DIR)

        st.session_state.trained = True  # ✅ SET FLAG
        st.sidebar.success("Document indexed successfully")

# ------------------ UI ------------------
st.title("Company AI Assistant")
st.caption("Ask questions about the company")

# ------------------ CHAT HISTORY ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ INPUT ------------------
question = st.chat_input("Ask a question…")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    # 🔒 ABSOLUTE GUARD
    if not st.session_state.trained:
        answer = "The assistant has not been trained on any documents yet."
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(INDEX_DIR, embeddings)

    docs = db.similarity_search(question, k=2)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer ONLY using the context.
If not found, say:
"I don’t have this information in the provided documents."

If multiple items exist, format as markdown bullets using '-'.

Context:
{context}

Question:
{question}

Answer:
"""

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown("_Thinking…_")

        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Strict document-only assistant"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Company RAG Bot"
            }
        )

        raw_answer = response["choices"][0]["message"]["content"]

        # 🔧 NORMALIZE BULLETS
        answer = raw_answer.replace("•", "-")

        thinking.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
