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
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")  # platform-level admin secret

openai.api_key = OPENROUTER_API_KEY
openai.api_base = "https://openrouter.ai/api/v1"

# ------------------ STREAMLIT ------------------
st.set_page_config(page_title="Company AI Assistant")

# ------------------ CLIENT IDENTIFICATION ------------------
query_params = st.experimental_get_query_params()
CLIENT_ID = query_params.get("client", [None])[0]

if not CLIENT_ID:
    st.error("Client not specified.")
    st.stop()

# ------------------ PATHS (ISOLATED PER CLIENT) ------------------
BASE_DIR = f"data/clients/{CLIENT_ID}"
UPLOAD_DIR = f"{BASE_DIR}/uploads"
INDEX_DIR = f"{BASE_DIR}/faiss"
TRAINED_FLAG = f"{BASE_DIR}/trained.flag"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ------------------ SESSION STATE (CHAT ONLY) ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------ ADMIN MODE (HIDDEN) ------------------
is_admin = False
admin_query = query_params.get("admin", [None])[0]

if admin_query == "1":
    admin_input = st.sidebar.text_input("Admin token", type="password")
    if admin_input == ADMIN_TOKEN:
        is_admin = True
        st.sidebar.success("Admin mode enabled")

# ------------------ ADMIN INGEST ------------------
if is_admin:
    uploaded_file = st.sidebar.file_uploader(
        "Upload company PDF", type=["pdf"]
    )

    if uploaded_file:
        pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)

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

        with open(TRAINED_FLAG, "w") as f:
            f.write("trained")

        st.sidebar.success("Client knowledge indexed")

# ------------------ UI ------------------
st.title("Company AI Assistant")
st.caption("Ask questions about the company")

# ------------------ CHAT HISTORY ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ CHAT INPUT ------------------
question = st.chat_input("Ask a question…")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    # GLOBAL GUARD
    if not os.path.exists(TRAINED_FLAG):
        answer = "The assistant has not been trained on any documents yet."
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
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
        )

        answer = response["choices"][0]["message"]["content"].replace("•", "-")
        thinking.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
