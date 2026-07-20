# DocGPT • Local private Document QA & Fine-Tuning Suite

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12.3-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.116.1-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PyTorch-2.9.0--dev-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Vector%20Store-FAISS-blueviolet?style=for-the-badge" alt="FAISS" />
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/Frontend-HTML%20%26%20Tailwind-38B2AC?style=for-the-badge&logo=tailwindcss&logoColor=white" alt="Frontend" />
</p>

---

## 🌟 Overview

**DocGPT** is a production-ready, fully private document ingestion and semantic QA assistant (RAG) platform. It allows you to upload documents (PDFs, Word documents, text files, CSVs, Excel spreadsheets), automatically generate embeddings, perform vector similarity search using **FAISS**, and chat with a local **Mistral-7B-Instruct-v0.2** large language model (quantized in 4-bit via BitsAndBytes) enhanced with a custom **QLoRA** adapter.

The suite also contains fine-tuning utilities using PEFT to train your own adapters on QA datasets.

---

## ✨ Features

- **Multi-Format Ingestion**: Supports `.pdf`, `.docx`, `.txt`, `.csv`, and `.xlsx` files using LangChain document loaders.
- **Fast Similarity Search**: Uses `all-MiniLM-L6-v2` via `SentenceTransformers` to embed texts and `FAISS` for low-latency similarity retrieval.
- **Local LLM Execution**: Runs quantized Mistral-7B-Instruct-v0.2 on local CPU or GPU.
- **QLoRA Adapter Support**: Leverages Hugging Face `PEFT` models to load fine-tuned training adapters, tailoring the model output to custom contexts.
- **SSE Streaming Answers**: Chat endpoint streams responses word-by-word back to the user via Server-Sent Events.
- **Gorgeous Client Dashboard (`frontend.html`)**:
  - Glassmorphic modern dark mode with a sleek, glowing ambient background.
  - Interactive light mode supporting a pure, premium white UI.
  - Multi-page professional **PDF Export** of chat history.
  - Active session timer tracking interaction durations.
  - File upload progress indicators.
- **Fine-Tuning Utilities**: Ready-made scripts to fine-tune Mistral-7B on custom datasets.

---

## 📁 Repository Structure

```
.
├── basic_chat/
│   └── chat_hf.py             # CLI local streaming chat with base Mistral-7B
├── check/
│   ├── check_model_files.py   # Utility to check local Hugging Face model status
│   └── test_load_qlora.py     # Test script for loading base + adapter
├── finetuned_chat/
│   └── qlora_finetuned.py     # CLI local streaming chat with base + adapter
├── qlora_train/
│   ├── train.jsonl            # Fine-tuning QA training dataset
│   ├── docgpt_training_large.jsonl # Larger training dataset
│   ├── train.py               # Main QLoRA training script
│   └── trainer.py             # Model training harness and settings
├── uploaded_docs/             # Target directory for uploaded files
├── v1/
│   └── backend-old.py         # Original backend prototype
├── v2/
│   └── uvicorn_v2_ollama.py   # Alternative backend using Ollama client
├── backend-qlora-toggle.py    # FastAPI service supporting Base vs QLoRA toggle
├── backend-v2-base.py         # FastAPI base service (no adapter)
├── uvicorn-v2-qlora.py        # FastAPI service with QLoRA adapter enabled by default
├── frontend.html              # Gorgeous dashboard UI (HTML, Tailwind CSS, JS)
├── requirements.txt           # Consolidated Python requirements
├── .gitignore                 # Exclusion configuration for git
└── README.md                  # This documentation file
```

---

## ⚙️ Requirements

- **Python**: `3.12.3` (or any compatible `3.10+` release)
- **CUDA-enabled GPU** (e.g., NVIDIA RTX series) with at least 8GB VRAM is highly recommended for running Mistral-7B local inference and QLoRA fine-tuning.

---

## 🚀 Setup & Installation

### 1. Clone the Repository
Clone the repository to your local machine:
```powershell
git clone https://github.com/rajeshc-git/docGPT.git
cd docGPT
```

### 2. Configure Environment Variables
Copy the template `.env.example` file to create a `.env` file:
```powershell
cp .env.example .env
```
Open `.env` and configure your paths:
- `MODEL_PATH`: Point to your local base model folder (e.g., `C:\Users\ABI-AI\Desktop\DocGPT\hf_models\Mistral-7B-Instruct-v0.2` or simply `hf_models/Mistral-7B-Instruct-v0.2`). Alternatively, specify a Hugging Face hub model ID (e.g., `mistralai/Mistral-7B-Instruct-v0.2`) to download it automatically.
- `LORA_PATH`: Point to your local QLoRA fine-tuned output adapter folder.
- `HOST` and `PORT`: Specify server port values (defaults to `0.0.0.0` and `8000`).

### 3. Initialize Virtual Environment
Create and activate a virtual environment to manage dependencies locally:
```powershell
# Create virtual environment
python -m venv venv310

# Activate virtual environment
# Windows (PowerShell)
.\venv310\Scripts\Activate.ps1
# macOS/Linux
source venv310/bin/activate
```

### 4. Install Dependencies
Ensure `pip` is updated, then install the required Python packages:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. GPU Acceleration (CUDA) Setup (Recommended)
To enable GPU-accelerated embedding generation, local Mistral inference, and model fine-tuning, install PyTorch with CUDA support:
```powershell
# Install CUDA-enabled PyTorch (CUDA 12.8 compatible nightly release)
python -m pip install --pre torch torchvision torchaudio `
  --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify your GPU is detected by PyTorch:
```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 🏃 Running the Application

### Step 1: Start the Backend Server

You can run the FastAPI backend server using three different configuration scripts, depending on whether you want QLoRA active, togglable, or base-only:

#### Option A: QLoRA Enabled by Default (Recommended)
This runs the API using the base model loaded with the PEFT adapter:
```powershell
uvicorn uvicorn-v2-qlora:app --host 0.0.0.0 --port 8000
```

#### Option B: Togglable Backend
Allows you to toggle the PEFT adapter via a boolean variable (`USE_QLORA`) in the script:
```powershell
uvicorn backend-qlora-toggle:app --host 0.0.0.0 --port 8000
```

#### Option C: Base Model Only
Loads only the base Mistral model without fine-tuning adapters:
```powershell
uvicorn backend-v2-base:app --host 0.0.0.0 --port 8000
```

*Once the server starts, the API docs are accessible at: `http://localhost:8000/docs`.*

---

### Step 2: Open the Frontend Client

1. Open [frontend.html](file:///c:/Users/ABI-AI/Desktop/client%20projects/DocGPT/frontend.html) directly in any web browser.
2. The client resolves backend endpoints dynamically using your browser's current address (defaulting to `http://localhost:8000` or the corresponding host IP on port 8000). You do not need to modify any code parameters.
3. Upload documents using the attachment icon and chat with the document base dynamically!

---

## 🧠 QLoRA Model Fine-Tuning

To train a custom QLoRA adapter on your own QA datasets:

1. Format your training data as JSONL (QA format), e.g., similar to `qlora_train/train.jsonl`.
2. Configure paths to your base model and training files in `qlora_train/trainer.py`.
3. Run the training script:
   ```powershell
   python qlora_train/train.py
   ```
4. The output adapters will be saved to the configured output path (e.g. `qlora_train/output2/`), which can then be fed into the backend server (`uvicorn-v2-qlora.py`) by setting `LORA_PATH`.

---

## 📜 License

This project is provided for development, educational, and research use. All local model usage is subject to the model developer agreements (e.g., Mistral AI license terms).
