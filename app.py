"""Streamlit UI for the Ask My PDF Bot — Advanced Level.

Features:
  - Hybrid retrieval (FAISS + BM25 + RRF fusion)
  - Per-document filter (multi-document support)
  - Live system monitoring (CPU / RAM)
  - Source citations with document name + page number
  - Conversational memory
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.embeddings import Embedder
from src.llm import HFChatLLM
from src.monitoring import get_system_stats
from src.pdf_loader import load_pdf_bytes
from src.rag_pipeline import RAGPipeline

# ---------- Load environment ----------
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if hf_token:
    os.environ["HF_TOKEN"] = hf_token
else:
    print("HF_TOKEN not found")


# ---------- Streamlit config ----------
st.set_page_config(
    page_title="Ask My PDF Bot — Advanced",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Cached resources ----------
@st.cache_resource(show_spinner="Loading embedding model (first run ~30s)...")
def get_embedder() -> Embedder:
    return Embedder(
        model_name=os.getenv(
            "EMBED_MODEL",
            "BAAI/bge-small-en-v1.5"
        )
    )


@st.cache_resource(show_spinner=False)
def get_llm() -> HFChatLLM:
    return HFChatLLM(
        model=os.getenv(
            "HF_MODEL",
            "mistralai/Mistral-7B-Instruct-v0.3"
        )
    )


def init_pipeline() -> RAGPipeline:
    return RAGPipeline(
        embedder=get_embedder(),
        llm=get_llm(),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        top_k=int(os.getenv("TOP_K", "4")),
        max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "4")),
    )


# ---------- Session state ----------
if "pipeline" not in st.session_state:

    if not os.getenv("HF_TOKEN"):
        st.error(
            "⚠️ HF_TOKEN is not set.\n\n"
            "Create a `.env` file and add:\n\n"
            "HF_TOKEN=your_token_here"
        )
        st.stop()

    st.session_state.pipeline = init_pipeline()
    st.session_state.chat_history = []


pipeline: RAGPipeline = st.session_state.pipeline


# ---------- Sidebar ----------
with st.sidebar:

    st.title("📄 Ask My PDF Bot")
    st.caption(
        "RAG · Advanced Level · Hybrid + Citations + Monitoring"
    )

    # ---------- Upload PDFs ----------
    st.subheader("1. Upload PDFs")

    uploaded = st.file_uploader(
        "Drop one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded and st.button(
        "Ingest documents",
        type="primary",
        use_container_width=True
    ):

        total = 0
        bar = st.progress(0.0)

        for i, file in enumerate(uploaded, start=1):

            with st.spinner(f"Processing {file.name}..."):

                pages = load_pdf_bytes(
                    file.read(),
                    file.name
                )

                if pages:
                    total += pipeline.ingest(pages)

            bar.progress(i / len(uploaded))

        st.success(
            f"Ingested {len(uploaded)} file(s) — {total} chunks added."
        )

    # ---------- Load PDFs from data folder ----------
    data_dir = Path(__file__).parent / "data"
    pdfs_in_data = sorted(data_dir.glob("*.pdf"))

    if pdfs_in_data and st.button(
        f"Ingest {len(pdfs_in_data)} PDFs from data/",
        use_container_width=True
    ):

        from src.pdf_loader import load_pdf

        total = 0
        bar = st.progress(0.0)

        for i, pdf in enumerate(pdfs_in_data, start=1):

            with st.spinner(f"Processing {pdf.name}..."):

                pages = load_pdf(pdf)

                if pages:
                    total += pipeline.ingest(pages)

            bar.progress(i / len(pdfs_in_data))

        st.success(
            f"Ingested {len(pdfs_in_data)} file(s) — {total} chunks added."
        )

    # ---------- Filter ----------
    st.divider()

    st.subheader("2. Filter by document")

    file_filter = st.multiselect(
        "Limit answers to these PDFs (empty = all)",
        options=pipeline.ingested_files,
        default=[]
    )

    # ---------- Index status ----------
    st.divider()

    st.subheader("3. Index status")

    c1, c2 = st.columns(2)

    c1.metric("Chunks", pipeline.store.size)
    c2.metric("Files", len(pipeline.ingested_files))

    # ---------- System monitor ----------
    st.divider()

    st.subheader("4. System monitor")

    stats = get_system_stats()

    print(stats)

    # SAFE CPU VALUE
    cpu = stats.get("cpu_percent", 0)

    # SAFE RAM VALUES
    ram_percent = stats.get("ram_percent", 0)
    ram_display = stats.get("ram_display", "0 GB")

    st.progress(
        min(cpu / 100, 1.0),
        text=f"CPU · {cpu:.0f}%"
    )

    st.progress(
        min(ram_percent / 100, 1.0),
        text=f"RAM · {ram_display}"
    )

    # ---------- Reset ----------
    st.divider()

    if st.button(
        "🗑️ Reset everything",
        use_container_width=True
    ):
        pipeline.reset()
        st.session_state.chat_history = []
        st.rerun()


# ---------- Main pane ----------
st.title("Ask your PDFs anything")

st.caption(
    "Advanced RAG · BGE embeddings + FAISS + BM25 (RRF fusion) "
    "· Hugging Face Inference API."
)

# ---------- Chat history ----------
for msg in st.session_state.chat_history:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg.get("sources"):

            with st.expander("🔎 Sources"):

                for s in msg["sources"]:

                    st.markdown(
                        f"**{s['source']}** — "
                        f"page {s['page']} · "
                        f"score `{s['score']:.4f}`"
                    )

                    st.text(s["preview"])


# ---------- User question ----------
question = st.chat_input(
    "Ask a question about the uploaded documents..."
)

if question:

    if pipeline.store.size == 0:

        st.warning(
            "Please upload and ingest at least one PDF first."
        )

    else:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                start = time.time()

                result = pipeline.ask(
                    question,
                    file_filter=file_filter or None
                )

                elapsed = time.time() - start

            st.markdown(result.answer)

            sources_payload = [
                {
                    "source": h.chunk.source,
                    "page": h.chunk.page_number,
                    "score": h.score,
                    "preview": (
                        h.chunk.text[:300]
                        + (
                            "..."
                            if len(h.chunk.text) > 300
                            else ""
                        )
                    ),
                }
                for h in result.sources
            ]

            with st.expander(
                f"🔎 Sources · answered in {elapsed:.2f}s"
            ):

                for s in sources_payload:

                    st.markdown(
                        f"**{s['source']}** — "
                        f"page {s['page']} · "
                        f"RRF `{s['score']:.4f}`"
                    )

                    st.text(s["preview"])

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": sources_payload,
            }
        )