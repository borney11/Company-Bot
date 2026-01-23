import os
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

PDF_DIR = "admin_pdfs"
INDEX_DIR = "data/faiss_index"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

all_text = ""

for file in os.listdir(PDF_DIR):
    if file.endswith(".pdf"):
        reader = PdfReader(os.path.join(PDF_DIR, file))
        for page in reader.pages:
            if page.extract_text():
                all_text += page.extract_text()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = splitter.split_text(all_text)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_texts(chunks, embeddings)
db.save_local(INDEX_DIR)

print("✅ PDFs processed and index created")
