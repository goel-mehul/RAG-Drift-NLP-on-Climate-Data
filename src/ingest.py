"""
ingest.py
---------
Reads PDFs from data/raw/, extracts text, chunks it, and saves
chunks as JSON to data/processed/chunks.json

Each chunk has metadata so we always know WHERE it came from.
"""

import fitz  # PyMuPDF
import json
import os
import re
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/chunks.json")

CHUNK_SIZE = 600      # target tokens per chunk (we approximate: 1 token ≈ 4 chars)
CHUNK_OVERLAP = 100   # overlap in tokens between consecutive chunks

# Map filename keywords → policy level
# Edit this if your filenames differ
LEVEL_MAP = {
    "ll97": "city",
    "planyc": "city",
    "PlaNYC": "city",
    "oneNYC": "city",
    "OneNYC": "city",
    "waterfront": "city",
    "greener": "city",
    "NYS": "state",
    "nys": "state",
    "scoping": "state",
    "paris": "federal",
    "BILLS": "federal",
    "bills": "federal",
    "Chapter1": "federal",
    "PPCC": "federal",
}

def get_level(filename: str) -> str:
    """Assign city / state / federal based on filename."""
    for keyword, level in LEVEL_MAP.items():
        if keyword in filename:
            return level
    return "unknown"

def clean_text(text: str) -> str:
    """Remove junk characters that PDFs often produce."""
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # remove non-ASCII
    text = text.strip()
    return text

def chunk_text(text: str, chunk_size: int, overlap: int):
    """
    Split text into overlapping chunks by approximate token count.
    We approximate: 1 token ≈ 4 characters.
    """
    char_size = chunk_size * 4
    char_overlap = overlap * 4

    chunks = []
    start = 0
    while start < len(text):
        end = start + char_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += char_size - char_overlap  # slide forward with overlap

    return chunks

def extract_chunks_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Open one PDF, extract text page by page, chunk it,
    and return a list of chunk dicts with metadata.
    """
    filename = pdf_path.name
    level = get_level(filename)
    doc_name = pdf_path.stem  # filename without .pdf

    chunks = []
    chunk_index = 0

    try:
        pdf = fitz.open(pdf_path)
    except Exception as e:
        print(f"  ⚠ Could not open {filename}: {e}")
        return []

    # Collect all text, tracking page numbers
    page_texts = []
    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text()
        text = clean_text(text)
        if len(text) > 50:  # skip near-empty pages
            page_texts.append((page_num, text))

    pdf.close()

    # Concatenate all text (we chunk across pages for better context)
    full_text = " ".join(t for _, t in page_texts)

    # Chunk
    raw_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

    for raw_chunk in raw_chunks:
        chunk_id = f"{doc_name}_c{chunk_index:04d}"
        chunks.append({
            "chunk_id": chunk_id,
            "doc_name": doc_name,
            "source_file": filename,
            "level": level,
            "text": raw_chunk,
        })
        chunk_index += 1

    return chunks


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in data/raw/")
        return

    all_chunks = []

    for pdf_path in tqdm(pdf_files, desc="Ingesting PDFs"):
        chunks = extract_chunks_from_pdf(pdf_path)
        all_chunks.extend(chunks)
        print(f"  ✓ {pdf_path.name}: {len(chunks)} chunks, level={get_level(pdf_path.name)}")

    with open(OUT_FILE, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\n Done. {len(all_chunks)} total chunks saved to {OUT_FILE}")

    # Quick stats
    from collections import Counter
    levels = Counter(c["level"] for c in all_chunks)
    print(f" Level breakdown: {dict(levels)}")


if __name__ == "__main__":
    main()