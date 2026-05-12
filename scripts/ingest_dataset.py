"""CLI utility: ingest every PDF in ../data/ into a FAISS index on disk.

Useful for batch ingestion before running the Streamlit app, or when you want
to persist the index across runs.

Usage:
    python scripts/ingest_dataset.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make `src` importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.embeddings import Embedder       # noqa: E402
from src.pdf_loader import load_pdf       # noqa: E402
from src.chunker import chunk_pages       # noqa: E402
from src.vector_store import VectorStore  # noqa: E402


def main() -> int:
    load_dotenv(ROOT / ".env")
    data_dir = ROOT / "data"
    index_dir = ROOT / "indexes" / "default"
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {data_dir}. Run scripts/download_dataset.py first.")
        return 1

    print(f"Found {len(pdfs)} PDFs. Loading embedding model...")
    embedder = Embedder(os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    store = VectorStore(dim=embedder.dim)

    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

    for pdf in pdfs:
        print(f"  Ingesting {pdf.name}...")
        pages = load_pdf(pdf)
        chunks = chunk_pages(pages, chunk_size, chunk_overlap)
        vectors = embedder.encode_documents([c.text for c in chunks])
        store.add(vectors, chunks)
        print(f"    -> {len(chunks)} chunks")

    store.save(index_dir)
    print(f"\nSaved index ({store.size} chunks) to {index_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
