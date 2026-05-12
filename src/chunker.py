"""Semantic chunking using NLTK sentence boundaries.

Strategy:
  1. Split each page into sentences with NLTK's PunktSentenceTokenizer.
  2. Greedily pack sentences into chunks of ~CHUNK_SIZE words.
  3. Keep CHUNK_OVERLAP words between consecutive chunks for context continuity.

This respects grammatical boundaries (no chopping mid-sentence) and beats
naive fixed-window splitting for RAG retrieval quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import nltk

from src.pdf_loader import PageText


def _ensure_nltk() -> None:
    """Download the sentence tokenizer if it's missing (idempotent)."""
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                # punkt_tab is only available in newer NLTK versions
                pass


@dataclass
class Chunk:
    text: str
    source: str
    page_number: int
    chunk_id: int


def chunk_pages(
    pages: List[PageText],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Chunk]:
    """Produce semantic chunks from a list of page texts.

    Args:
        pages: Output of pdf_loader.load_pdf().
        chunk_size: target words per chunk.
        chunk_overlap: words of overlap with previous chunk.
    """
    _ensure_nltk()

    chunks: List[Chunk] = []
    chunk_id = 0

    for page in pages:
        try:
            sentences = nltk.sent_tokenize(page.text)
        except LookupError:
            # Fallback if punkt is unavailable
            sentences = [s.strip() for s in page.text.split(". ") if s.strip()]

        current_words: List[str] = []
        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue

            # If adding this sentence would exceed chunk_size, flush.
            if current_words and len(current_words) + len(words) > chunk_size:
                chunks.append(
                    Chunk(
                        text=" ".join(current_words),
                        source=page.source,
                        page_number=page.page_number,
                        chunk_id=chunk_id,
                    )
                )
                chunk_id += 1
                # keep tail as overlap
                current_words = current_words[-chunk_overlap:] if chunk_overlap > 0 else []

            current_words.extend(words)

        if current_words:
            chunks.append(
                Chunk(
                    text=" ".join(current_words),
                    source=page.source,
                    page_number=page.page_number,
                    chunk_id=chunk_id,
                )
            )
            chunk_id += 1

    return chunks
