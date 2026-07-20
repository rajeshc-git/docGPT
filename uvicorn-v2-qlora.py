import os
import json
import uuid
import shutil
import numpy as np
import torch
import faiss

from threading import Thread
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer
)

# 🔥 QLoRA
from peft import PeftModel

# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ===================== CONFIG =====================
from dotenv import load_dotenv
load_dotenv()

UPLOAD_DIR = "uploaded_docs"
INDEX_FILE = "faiss.index"
META_FILE = "faiss_meta.json"

MODEL_PATH = os.getenv("MODEL_PATH", r"C:\Users\ABI-AI\Desktop\DocGPT\hf_models\Mistral-7B-Instruct-v0.2")
LORA_PATH  = os.getenv("LORA_PATH", r"C:\Users\ABI-AI\Desktop\DocGPT\qlora_train\output2")

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


# ===================== LOAD LLM (BASE + QLoRA) =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

llm = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)

# 🔥 APPLY QLoRA ADAPTER
llm = PeftModel.from_pretrained(
    llm,
    LORA_PATH
)
print(f"✅ Using QLoRA adapter: {LORA_PATH}")
llm.eval()


# ===================== FAISS INIT =====================
if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if index.ntotal != len(metadata):
        index = faiss.IndexFlatL2(EMBED_DIM)
        metadata = []
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
    return {"status": "stored", "total_vectors": index.ntotal}


# ===================== CHAT =====================
@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    query = data.get("message", "").strip()

    if not query or index.ntotal == 0:
        return {"answer": "No documents are available to answer this question."}

    q_emb = embed_model.encode([query], device=device)
    D, I = index.search(np.array(q_emb).astype("float32"), k=5)

    context_chunks = []
    for idx in I[0]:
        if 0 <= idx < len(metadata):
            context_chunks.append(metadata[idx]["text"])

    context = "\n---\n".join(context_chunks)

    # ===================== PROMPT =====================
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer ONLY from the provided context. "
                "If the answer is not found, politely apologise. "
                "When using tables, output complete and valid markdown tables only."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}"
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(llm.device)

    attention_mask = torch.ones_like(input_ids).to(llm.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    def stream():
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        thread = Thread(target=llm.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            yield f"data: {json.dumps({'content': token})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
