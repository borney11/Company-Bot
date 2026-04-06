import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import openai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# ================= ENV =================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

if not OPENROUTER_API_KEY or not MODEL_NAME or not ADMIN_TOKEN:
    st.error("Server configuration missing.")
    st.stop()

openai.api_key = OPENROUTER_API_KEY
openai.api_base = "https://openrouter.ai/api/v1"

# ================= PAGE =================
st.set_page_config(page_title="Company AI Assistant")

# ================= QUERY PARAMS =================
query_params = st.query_params
CLIENT_ID = query_params.get("client")
ADMIN_MODE = query_params.get("admin")

if not CLIENT_ID:
    st.error("Invalid access link. Missing client ID.")
    st.stop()

# ================= PATHS =================
BASE_DIR = f"data/clients/{CLIENT_ID}"
UPLOAD_DIR = f"{BASE_DIR}/uploads"
INDEX_DIR = f"{BASE_DIR}/faiss"
TRAINED_FLAG = f"{BASE_DIR}/trained.flag"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ================= SESSION =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= ADMIN AUTH =================
is_admin = False

if ADMIN_MODE == "1":
    admin_input = st.sidebar.text_input("Enter Admin Password", type="password")

    if admin_input == ADMIN_TOKEN:
        is_admin = True
        st.sidebar.success("Admin mode enabled")
    elif admin_input:
        st.sidebar.error("Wrong password")

# ================= EMBEDDINGS =================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1"
)

# ================= ADMIN PANEL =================
if is_admin:
    st.sidebar.header("📂 Admin Panel")

    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        # Save file
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Read PDF
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80
        )
        chunks = splitter.split_text(text)

        # FIX: append instead of overwrite
        if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
            db = FAISS.load_local(INDEX_DIR, embeddings)
            db.add_texts(chunks)
        else:
            db = FAISS.from_texts(chunks, embeddings)

        db.save_local(INDEX_DIR)

        # mark trained
        with open(TRAINED_FLAG, "w") as f:
            f.write("trained")

        st.sidebar.success("PDF added to knowledge base")

# ================= UI =================
st.title("Company AI Assistant")
st.caption(f"Client: {CLIENT_ID}")

# ================= CHAT HISTORY =================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= CHAT =================
question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # Check training
    if not os.path.exists(TRAINED_FLAG):
        answer = "No documents uploaded yet."
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.stop()

    # Load DB
    db = FAISS.load_local(INDEX_DIR, embeddings)

    docs = db.similarity_search(question, k=3)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer ONLY using the context below.
If answer is not found, say:
"I don’t have this information in the documents."

Context:
{context}

Question:
{question}

Answer:
"""

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown("Thinking...")

        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Strict document-based assistant"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        answer = response["choices"][0]["message"]["content"]
        thinking.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
