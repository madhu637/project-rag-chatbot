"""FAISS-backed vector store with parallel metadata list."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np

from src.chunker import Chunk


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Chunk] = []
        self.dim = dim

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        if len(vectors) == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim mismatch: got {vectors.shape[1]}, expected {self.dim}"
            )
        self.index.add(vectors)
        self.metadata.extend(chunks)

    def reset(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []

    def search(self, query_vec: np.ndarray, top_k: int = 4) -> List[SearchHit]:
        if self.index.ntotal == 0:
            return []
        k = min(top_k, self.index.ntotal)
        scores, idxs = self.index.search(query_vec, k)
        hits: List[SearchHit] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            hits.append(SearchHit(chunk=self.metadata[int(idx)], score=float(score)))
        return hits

    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / "index.faiss"))
        with open(dir_path / "metadata.pkl", "wb") as f:
            pickle.dump({"metadata": self.metadata, "dim": self.dim}, f)

    @classmethod
    def load(cls, dir_path: str | Path) -> "VectorStore":
        dir_path = Path(dir_path)
        with open(dir_path / "metadata.pkl", "rb") as f:
            payload = pickle.load(f)
        store = cls(dim=payload["dim"])
        store.index = faiss.read_index(str(dir_path / "index.faiss"))
        store.metadata = payload["metadata"]
        return store

    @property
    def size(self) -> int:
        return int(self.index.ntotal)
