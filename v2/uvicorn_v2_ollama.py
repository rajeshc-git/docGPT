import os
import json
import uuid
import shutil
import requests
import numpy as np
import torch
import faiss

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===================== CONFIG =====================
UPLOAD_DIR = "uploaded_docs"
INDEX_FILE = "faiss.index"
META_FILE = "faiss_meta.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gpt-oss:latest"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===================== APP =====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== EMBEDDINGS =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
EMBED_DIM = embed_model.get_sentence_embedding_dimension()

# ===================== FAISS INIT (SAFE) =====================
if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 🔒 Safety check
    if index.ntotal != len(metadata):
        print("⚠️ FAISS index and metadata mismatch. Resetting.")
        index = faiss.IndexFlatL2(EMBED_DIM)
        metadata = []
        os.remove(INDEX_FILE)
        os.remove(META_FILE)
else:
    index = faiss.IndexFlatL2(EMBED_DIM)
    metadata = []

# ===================== HELPERS =====================
def save_index():
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_and_split(path: str):
    ext = path.split(".")[-1].lower()

    if ext == "pdf":
        loader = PyPDFLoader(path)
    elif ext == "txt":
        loader = TextLoader(path)
    elif ext == "docx":
        loader = Docx2txtLoader(path)
    elif ext == "csv":
        loader = CSVLoader(path)
    elif ext == "xlsx":
        loader = UnstructuredExcelLoader(path)
    else:
        return []

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

# ===================== UPLOAD =====================
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    global index, metadata

    for file in files:
        file_id = str(uuid.uuid4())
        save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        chunks = load_and_split(save_path)
        texts = [c.page_content.strip() for c in chunks if len(c.page_content.strip()) > 50]

        if not texts:
            continue

        embeddings = embed_model.encode(texts, device=device)
        index.add(np.array(embeddings).astype("float32"))

        for t in texts:
            metadata.append({
                "text": t,
                "source": file.filename
            })

    save_index()
    return {
        "status": "stored",
        "total_vectors": index.ntotal
    }

# ===================== CHAT =====================
@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    query = data.get("message", "").strip()

    if not query or index.ntotal == 0:
        return {"answer": "No documents are available to answer this question."}

    q_emb = embed_model.encode([query], device=device)
    D, I = index.search(np.array(q_emb).astype("float32"), k=5)

    # 🔒 SAFE CONTEXT BUILDING
    context_chunks = []
    for idx in I[0]:
        if 0 <= idx < len(metadata):
            context_chunks.append(metadata[idx]["text"])

    context = "\n---\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant with only the knowledge that has been provided. Use the context below to answer the user's question also if helpfull use table to explain briefly do not answer anything outside the provided knowledge base.

Context:
{context}

Question:
{query}

Answer:
"""

    # ===================== STREAM ADAPTER (FIXED) =====================
    def stream():
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True
            },
            stream=True,
            timeout=300
        )

        for line in r.iter_lines():
            if not line:
                continue

            try:
                data = json.loads(line.decode("utf-8"))

                if "response" in data:
                    chunk = data["response"]
                    yield f"data: {json.dumps({'content': chunk})}\n\n"

                if data.get("done"):
                    yield "data: [DONE]\n\n"
                    break

            except Exception:
                continue

    return StreamingResponse(stream(), media_type="text/event-stream")
