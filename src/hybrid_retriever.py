"""Hybrid retriever: FAISS (semantic) + BM25 (keyword), fused via RRF.

Reciprocal Rank Fusion (RRF) is the industry-standard way to combine rankings
from different retrievers without needing to calibrate score scales.

  RRF_score(d) = sum over retrievers r of  1 / (k + rank_r(d))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from src.chunker import Chunk
from src.vector_store import SearchHit, VectorStore


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t.isalnum() or any(c.isalnum() for c in t)]


@dataclass
class HybridRetriever:
    store: VectorStore
    bm25: BM25Okapi | None = None
    corpus_tokens: List[List[str]] | None = None
    rrf_k: int = 60

    def rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from the current VectorStore metadata."""
        self.corpus_tokens = [_tokenize(c.text) for c in self.store.metadata]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(
        self,
        query: str,
        query_vec: np.ndarray,
        top_k: int = 4,
        candidate_k: int | None = None,
    ) -> List[SearchHit]:
        """Return top_k chunks fused from semantic + keyword rankings."""
        if self.store.size == 0:
            return []

        candidate_k = candidate_k or max(top_k * 4, 20)
        candidate_k = min(candidate_k, self.store.size)

        # Semantic ranking
        sem_hits = self.store.search(query_vec, top_k=candidate_k)
        sem_ranks: dict[int, int] = {}
        for rank, hit in enumerate(sem_hits, start=1):
            sem_ranks[hit.chunk.chunk_id] = rank

        # Keyword ranking
        kw_ranks: dict[int, int] = {}
        if self.bm25 is not None:
            scores = self.bm25.get_scores(_tokenize(query))
            top_idx = np.argsort(scores)[::-1][:candidate_k]
            for rank, idx in enumerate(top_idx, start=1):
                chunk_id = self.store.metadata[int(idx)].chunk_id
                kw_ranks[chunk_id] = rank

        # RRF fusion
        all_ids = set(sem_ranks) | set(kw_ranks)
        fused: List[tuple[float, Chunk]] = []
        id_to_chunk = {c.chunk_id: c for c in self.store.metadata}
        for cid in all_ids:
            score = 0.0
            if cid in sem_ranks:
                score += 1.0 / (self.rrf_k + sem_ranks[cid])
            if cid in kw_ranks:
                score += 1.0 / (self.rrf_k + kw_ranks[cid])
            fused.append((score, id_to_chunk[cid]))

        fused.sort(key=lambda x: x[0], reverse=True)
        return [SearchHit(chunk=c, score=s) for s, c in fused[:top_k]]
