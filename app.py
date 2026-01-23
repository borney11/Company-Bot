import os
import streamlit as st
from dotenv import load_dotenv
import openai

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

openai.api_key = OPENROUTER_API_KEY
openai.api_base = "https://openrouter.ai/api/v1"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Company AI Assistant")
st.title("Company AI Assistant")
st.caption("Ask questions about the company")

INDEX_DIR = "data/faiss_index"

# -----------------------------
# Chat UI
# -----------------------------
question = st.text_input("Ask a question")

if question:
    if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        st.error("Knowledge base not found.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(INDEX_DIR, embeddings)

    docs = db.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a company AI assistant.
Answer ONLY using the context.
If not found, say:
"I don’t have this information in the provided documents."

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
    st.write(response["choices"][0]["message"]["content"])
