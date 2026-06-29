"""
ingest.py
---------
Step 1 of the pipeline: turn a PDF into a searchable vector index.

What this does:
1. Extracts text from every page of the PDF.
2. Splits the text into overlapping chunks (so context isn't cut mid-sentence).
3. Converts each chunk into a vector embedding (a list of numbers that
   captures its meaning) using a free, local embedding model.
4. Stores all the vectors in a FAISS index (a fast similarity-search
   structure) and saves the chunk text + page numbers alongside it.

Run this ONCE per PDF (or again whenever you change the PDF).
    python ingest.py --pdf "DC_dutta_8th.pdf" --out "dutta_index"

To use a different PDF later, just point --pdf at a new file and pick a
new --out folder name. Nothing else in this script needs to change.
"""

import argparse
import json
import os
import re

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Tunable settings
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800       # characters per chunk (roughly 150-200 words)
CHUNK_OVERLAP = 150    # overlap between consecutive chunks, so an idea that
                       # spans a chunk boundary doesn't get split awkwardly
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, free, runs on CPU


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) for every page with text."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
        if text:
            pages.append((i + 1, text))  # 1-indexed page numbers for humans
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks of roughly `chunk_size` chars."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_index(pdf_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading {pdf_path} ...")
    pages = extract_pages(pdf_path)
    print(f"Extracted text from {len(pages)} pages.")

    # Build chunks, remembering which page each chunk came from
    records = []  # each item: {"text": ..., "page": ...}
    for page_num, page_text in pages:
        for chunk in chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP):
            if len(chunk.strip()) > 30:  # skip near-empty scraps
                records.append({"text": chunk, "page": page_num})

    print(f"Created {len(records)} chunks. Loading embedding model ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding chunks (this is the slow step, runs once)...")
    texts = [r["text"] for r in records]
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")

    # Build a FAISS index for fast similarity search (cosine via normalized
    # vectors + inner product)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save everything needed to query later
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"embedding_model": EMBEDDING_MODEL, "source_pdf": pdf_path}, f)

    print(f"Done. Index saved to '{out_dir}/'. You can now run chatbot.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the PDF to ingest")
    parser.add_argument(
        "--out", required=True, help="Folder to save the index into, e.g. dutta_index"
    )
    args = parser.parse_args()
    build_index(args.pdf, args.out)
