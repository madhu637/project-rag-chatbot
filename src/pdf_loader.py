"""PDF loader using PyMuPDF (fitz).

Extracts text page-by-page so we can keep the page number for citations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


@dataclass
class PageText:
    source: str          # filename (basename)
    page_number: int     # 1-indexed
    text: str


def load_pdf(file_path: str | Path) -> List[PageText]:
    """Read a PDF and return one PageText per page (1-indexed)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    doc = fitz.open(str(path))
    try:
        pages: List[PageText] = []
        for i, page in enumerate(doc, start=1):
            raw = page.get_text("text") or ""
            cleaned = " ".join(raw.split())  # collapse whitespace
            if cleaned.strip():
                pages.append(PageText(source=path.name, page_number=i, text=cleaned))
        return pages
    finally:
        doc.close()


def load_pdf_bytes(data: bytes, filename: str) -> List[PageText]:
    """Same as load_pdf but accepts raw bytes (used by Streamlit uploader)."""
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        pages: List[PageText] = []
        for i, page in enumerate(doc, start=1):
            raw = page.get_text("text") or ""
            cleaned = " ".join(raw.split())
            if cleaned.strip():
                pages.append(PageText(source=filename, page_number=i, text=cleaned))
        return pages
    finally:
        doc.close()
