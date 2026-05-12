# Ask My PDF Bot — RAG Chatbot

> **Project 5** of the *Top 5 AI Industry Internship Projects* guide — implemented at **Intermediate + Advanced** level.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red) ![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green) ![BM25](https://img.shields.io/badge/Hybrid-RRF%20Fusion-purple) ![Docker](https://img.shields.io/badge/Deploy-Docker-blue)

A Retrieval-Augmented Generation chatbot that lets you upload PDFs and ask natural-language questions, with conversational memory and source citations.

---

## ✨ Features

### Intermediate
- Semantic chunking with NLTK
- BGE embeddings (`BAAI/bge-small-en-v1.5`)
- FAISS vector store
- Conversational memory (sliding window)
- Streamlit UI with PDF upload + history

### ✅ Advanced (this build)
- **Hybrid retrieval** — FAISS (semantic) + **BM25** (keyword) fused via **Reciprocal Rank Fusion**
- **Source citations** — document name + page number for every answer
- **Multi-document support** — upload many PDFs, filter answers per document
- **Live system monitor** — CPU/RAM gauges in the sidebar
- **Docker deployment** — `Dockerfile` + `docker-compose.yml`

### 🚀 Roadmap (SOTA)
- Fully offline mode (`llama-cpp-python` + Mistral-7B GGUF)
- Voice interface (Whisper + Coqui TTS)
- Async background ingestion (Celery + Redis)

---

## 🗂️ Project Structure

```
project5_rag_chatbot/
├── app.py                       # Streamlit UI (multi-doc + monitoring)
├── requirements.txt
├── Dockerfile  docker-compose.yml  .dockerignore
├── .env.example  .gitignore  README.md
├── src/
│   ├── pdf_loader.py            # PyMuPDF
│   ├── chunker.py               # NLTK semantic chunking
│   ├── embeddings.py            # BGE wrapper
│   ├── vector_store.py          # FAISS IndexFlatIP
│   ├── hybrid_retriever.py      # FAISS + BM25 + RRF
│   ├── llm.py                   # HF InferenceClient
│   ├── memory.py                # ConversationBufferMemory
│   ├── monitoring.py            # psutil CPU/RAM
│   └── rag_pipeline.py          # Orchestration
├── scripts/
│   ├── download_dataset.py      # Kaggle / arXiv NLP papers
│   └── ingest_dataset.py        # Batch FAISS index build
├── data/.gitkeep
└── docs/GITHUB_PUBLISH.md
```

---

## 🚀 Quick Start (local)

```bash
git clone https://github.com/<your-username>/ask-my-pdf-bot.git
cd ask-my-pdf-bot
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                                 # add your HF_TOKEN
python scripts/download_dataset.py                   # optional sample arXiv PDFs
streamlit run app.py                                 # → http://localhost:8501
```

Get a free Hugging Face token: <https://huggingface.co/settings/tokens>

---

## 🐳 Quick Start (Docker)

```bash
cp .env.example .env             # add HF_TOKEN
docker compose up --build
# → http://localhost:8501
```

The Docker image:
- Pre-downloads NLTK `punkt` during build (no first-run delay)
- Persists Hugging Face model cache in a named volume (`hf_cache`)
- Mounts `./data` so you can drop PDFs in and click "Ingest from data/"
- Includes a health check on `/_stcore/health`

---

## 📦 Kaggle Dataset

**Dataset:** [arXiv NLP Papers — `Cornell-University/arxiv`](https://www.kaggle.com/datasets/Cornell-University/arxiv)

Run `python scripts/download_dataset.py` to fetch 5 demo NLP papers (Transformer, BERT, GPT-3, RoBERTa, original RAG paper) from arXiv's CDN. No Kaggle credentials needed for the fallback path.

---

## 🧪 Example Questions

After ingesting NLP papers, try:
- "What problem does the Transformer paper try to solve?"
- "Compare BERT and RoBERTa pretraining objectives."
- "What datasets were used to evaluate the RAG paper?"
- "Summarize GPT-3's few-shot learning results."

Every answer cites `(filename.pdf, page N)`.

---

## ⚙️ How it Works

```
PDF ──► PyMuPDF ──► NLTK sentences ──► chunks (~500 words, 50 overlap)
                                                  │
                                                  ▼
                                       BGE-small embeddings
                                                  │
                                                  ▼
                          ┌────────── FAISS index ──────────┐
                          │                                  │
                  user query                          BM25 keyword
                          │                                  │
                          ▼                                  ▼
                  semantic ranks                       keyword ranks
                          └──────► RRF fusion ◄──────────────┘
                                          │
                                          ▼
                                top-k chunks + memory
                                          │
                                          ▼
                         Hugging Face Inference API (Mistral-7B)
                                          │
                                          ▼
                              grounded answer + citations
```

---

## 📤 Publish to GitHub

See [`docs/GITHUB_PUBLISH.md`](docs/GITHUB_PUBLISH.md) for the full 11-step guide.

Short version:

```bash
git init -b main
git add . && git commit -m "Initial commit: Ask My PDF Bot"
gh repo create ask-my-pdf-bot --public --source=. --remote=origin --push
```

---

## 📜 License

MIT
