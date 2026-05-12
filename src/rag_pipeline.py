"""End-to-end RAG pipeline: ingest -> hybrid retrieve -> generate.

Advanced level: uses HybridRetriever (FAISS + BM25 with RRF fusion) instead
of pure semantic search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.chunker import chunk_pages
from src.embeddings import Embedder
from src.hybrid_retriever import HybridRetriever
from src.llm import HFChatLLM
from src.memory import ConversationBufferMemory
from src.pdf_loader import PageText
from src.vector_store import SearchHit, VectorStore


@dataclass
class RAGAnswer:
    answer: str
    sources: List[SearchHit]


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        llm: HFChatLLM,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 4,
        max_history_turns: int = 4,
    ) -> None:
        self.embedder = embedder
        self.llm = llm
        self.store = VectorStore(dim=embedder.dim)
        self.retriever = HybridRetriever(store=self.store)
        self.memory = ConversationBufferMemory(max_turns=max_history_turns)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self._ingested_files: List[str] = []

    # ---------- ingestion ----------
    def ingest(self, pages: List[PageText]) -> int:
        chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return 0
        vectors = self.embedder.encode_documents([c.text for c in chunks])
        self.store.add(vectors, chunks)
        self.retriever.rebuild_bm25()

        for page in pages:
            if page.source not in self._ingested_files:
                self._ingested_files.append(page.source)
        return len(chunks)

    # ---------- query ----------
    def ask(self, question: str, file_filter: Optional[List[str]] = None) -> RAGAnswer:
        if self.store.size == 0:
            return RAGAnswer(
                answer="No documents have been uploaded yet. Please upload a PDF first.",
                sources=[],
            )

        q_vec = self.embedder.encode_query(question)
        hits = self.retriever.search(
            query=question,
            query_vec=q_vec,
            top_k=self.top_k * 3 if file_filter else self.top_k,
        )

        # per-document filter (advanced multi-doc support)
        if file_filter:
            hits = [h for h in hits if h.chunk.source in file_filter][: self.top_k]

        if not hits:
            return RAGAnswer(
                answer="No relevant passages found in the selected documents.",
                sources=[],
            )

        context = "\n\n".join(
            f"[{h.chunk.source}, page {h.chunk.page_number}]\n{h.chunk.text}" for h in hits
        )
        history = self.memory.to_prompt()
        answer = self.llm.chat(context=context, history=history, question=question)
        self.memory.add_turn(question, answer)
        return RAGAnswer(answer=answer, sources=hits)

    # ---------- introspection ----------
    @property
    def ingested_files(self) -> List[str]:
        return list(self._ingested_files)

    def reset(self) -> None:
        self.store.reset()
        self.retriever = HybridRetriever(store=self.store)
        self.memory.clear()
        self._ingested_files = []
