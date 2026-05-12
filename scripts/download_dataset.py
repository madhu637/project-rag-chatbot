"""Download a small sample of arXiv NLP PDFs for demoing the RAG bot.

Strategy:
  1. Try Kaggle's official `Cornell-University/arxiv` metadata dump (JSON).
  2. Pick 5 NLP papers (cs.CL category) and download their PDFs directly from
     arXiv's public CDN — no auth required.
  3. Save to ../data/.

Why this approach?
  The full arXiv Kaggle dataset is ~1.7 GB of metadata only — papers themselves
  are not in the Kaggle blob. So we use Kaggle for metadata + arXiv for PDFs.
  If you don't have Kaggle credentials configured, the script falls back to a
  hard-coded list of well-known NLP papers.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Reliable fallback list — classic NLP / ML papers
FALLBACK_PAPERS: List[Tuple[str, str]] = [
    ("1706.03762", "attention_is_all_you_need.pdf"),       # Transformer
    ("1810.04805", "bert.pdf"),                            # BERT
    ("2005.14165", "gpt3.pdf"),                            # GPT-3
    ("1907.11692", "roberta.pdf"),                         # RoBERTa
    ("2005.11401", "rag_retrieval_augmented_generation.pdf"),  # RAG paper
]


def download_arxiv_pdf(arxiv_id: str, out_path: Path) -> bool:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"  -> {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            out_path.write_bytes(resp.read())
        return True
    except Exception as exc:
        print(f"     FAILED: {exc}")
        return False


def try_kaggle_metadata() -> List[Tuple[str, str]] | None:
    """Attempt to pull arxiv metadata via the Kaggle API; pick 5 cs.CL papers."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except ImportError:
        print("Kaggle SDK not installed. Skipping Kaggle metadata step.")
        return None

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as exc:
        print(f"Kaggle auth failed ({exc}). Falling back to curated list.")
        return None

    print("Downloading arxiv metadata from Kaggle (this is ~1 GB; may take time)...")
    try:
        api.dataset_download_files(
            "Cornell-University/arxiv",
            path=str(DATA_DIR / "raw"),
            unzip=True,
            quiet=False,
        )
    except Exception as exc:
        print(f"Kaggle download failed ({exc}). Falling back to curated list.")
        return None

    meta_file = DATA_DIR / "raw" / "arxiv-metadata-oai-snapshot.json"
    if not meta_file.exists():
        return None

    chosen: List[Tuple[str, str]] = []
    with meta_file.open() as fh:
        for line in fh:
            rec = json.loads(line)
            cats = rec.get("categories", "")
            if "cs.CL" in cats:
                arxiv_id = rec["id"]
                safe_title = rec["title"].strip().replace("\n", " ").lower()
                safe_title = "".join(c if c.isalnum() else "_" for c in safe_title)[:60]
                chosen.append((arxiv_id, f"{safe_title}.pdf"))
                if len(chosen) >= 5:
                    break
    return chosen


def main() -> int:
    print("=" * 60)
    print("Downloading arXiv NLP PDFs for the RAG chatbot demo")
    print("=" * 60)

    papers = try_kaggle_metadata() or FALLBACK_PAPERS
    if papers is FALLBACK_PAPERS:
        print("Using curated fallback list (no Kaggle auth needed).")

    ok = 0
    for arxiv_id, filename in papers:
        out = DATA_DIR / filename
        if out.exists() and out.stat().st_size > 1024:
            print(f"✓ {filename} (already present)")
            ok += 1
            continue
        print(f"Downloading {arxiv_id} -> {filename}")
        if download_arxiv_pdf(arxiv_id, out):
            ok += 1
        time.sleep(1)  # be nice to arxiv.org

    print(f"\nDone. {ok}/{len(papers)} PDFs in {DATA_DIR}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
