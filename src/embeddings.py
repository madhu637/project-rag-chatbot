"""BGE embedding wrapper using sentence-transformers.

BAAI/bge-small-en-v1.5 is a top performer on the MTEB benchmark for its size
(~33 MB) and runs comfortably on CPU.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        prompt = "Represent this sentence for searching relevant passages: " + query
        vector = self.model.encode(
            [prompt],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.astype("float32")
