"""
chatbot.py
----------
Multi-book RAG chatbot for pregnancy, obstetrics, and gynecology.

Knowledge sources:
  1. DC Dutta's Textbook of Gynecology (dutta_index)
  2. Oxford Handbook of Obstetrics and Gynaecology (oxford_index)
  3. Williams Obstetrics (williams_index)
  4. preloaded_knowledge.ts (structured keyword-searchable chunks)

For every question:
1. Your question is embedded with SentenceTransformer.
2. FAISS searches all loaded indexes for the most relevant chunks.
3. Preloaded knowledge is keyword-searched for additional context.
4. All retrieved chunks are merged and sent to Gemini 2.5 Flash.
5. Gemini answers as an expert pregnancy/obstetrics assistant.

Run:
    $env:GEMINI_API_KEY="your-key-here"
    python chatbot.py --indexes dutta_index oxford_index williams_index
"""

import argparse
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import preloaded_knowledge

TOP_K_PER_INDEX = 5   # chunks retrieved per FAISS index
PRELOADED_TOP_K = 4   # chunks from preloaded_knowledge.ts

SYSTEM_PROMPT = """You are an expert pregnancy, obstetrics, and gynecology assistant \
trained on multiple authoritative medical textbooks including DC Dutta's Gynecology, \
Oxford Handbook of Obstetrics and Gynaecology, and Williams Obstetrics.

You specialize in:
- Pregnancy (signs, symptoms, trimesters, antenatal care)
- High-risk pregnancy (pre-eclampsia, gestational diabetes, placenta previa, IUGR, etc.)
- Labor, delivery, and postpartum care
- Obstetric emergencies
- Gynecological conditions and female reproductive health
- Infertility, menstruation, and menopause

You have two sources of knowledge:
1. CONTEXT — passages retrieved from multiple authoritative medical textbooks.
2. Your own expert medical knowledge about pregnancy and obstetrics.

Follow these rules:
1. Always check the CONTEXT first. Use it as your primary basis for answers.
2. If the CONTEXT is insufficient, use your own expert medical knowledge to give \
a complete, accurate answer. Do NOT say "I don't have this information."
3. Never mention page numbers, page references, or citations in your response.
4. Give thorough, complete answers covering causes, symptoms, diagnosis, \
management, and complications where relevant.
5. Explain medical jargon clearly in simple terms for non-medical readers.
6. Structure longer answers with bullet points or short paragraphs for readability.
7. Always end with a brief reminder that this is educational information and the \
user should consult a qualified doctor or midwife for personal medical advice.
"""


def load_index(index_dir: str):
    """Load a single FAISS index, chunks, and config from a folder."""
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(index_dir, "config.json")) as f:
        config = json.load(f)
    return index, chunks, config


def load_all_indexes(index_dirs: list[str]):
    """Load multiple FAISS indexes. Skip missing ones with a warning."""
    loaded = []
    for d in index_dirs:
        if not os.path.isdir(d):
            print(f"  [WARNING] Index folder '{d}' not found — skipping.")
            continue
        try:
            index, chunks, config = load_index(d)
            loaded.append((index, chunks, config))
            print(f"  Loaded: {d}  ({config.get('source_pdf', d)})")
        except Exception as e:
            print(f"  [WARNING] Could not load '{d}': {e}")
    return loaded


def retrieve_from_all(
    question: str,
    embed_model,
    indexes: list,
    top_k: int = TOP_K_PER_INDEX,
) -> list[dict]:
    """Run FAISS search across all loaded indexes and merge results."""
    q_emb = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    all_results = []
    for index, chunks, config in indexes:
        scores, ids = index.search(q_emb, top_k)
        source = config.get("source_pdf", "unknown")
        for score, idx in zip(scores[0], ids[0]):
            if idx != -1:
                chunk = chunks[idx].copy()
                chunk["_source"] = source
                chunk["_score"] = float(score)
                all_results.append(chunk)

    # Sort by relevance score descending and deduplicate by text
    all_results.sort(key=lambda x: x["_score"], reverse=True)
    seen = set()
    deduped = []
    for r in all_results:
        key = r["text"][:100]
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped[:top_k * 2]  # return top chunks across all books


def build_user_prompt(
    question: str,
    faiss_chunks: list[dict],
    preloaded_chunks: list[dict],
) -> str:
    """Merge FAISS and preloaded knowledge chunks into a single context prompt."""
    all_chunks = faiss_chunks + preloaded_chunks
    if all_chunks:
        context_blocks = "\n\n".join(c["text"] for c in all_chunks)
    else:
        context_blocks = "No specific context retrieved."

    return f"""CONTEXT (from authoritative obstetrics and gynecology textbooks):
{context_blocks}

QUESTION: {question}

Answer the question using the CONTEXT above as your primary source. Follow the system rules."""


def main(index_dirs: list[str]):
    print("\nLoading embedding model ...")
    # Get embedding model name from first available index config
    embed_model_name = "all-MiniLM-L6-v2"
    for d in index_dirs:
        cfg_path = os.path.join(d, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                embed_model_name = json.load(f).get("embedding_model", embed_model_name)
            break
    embed_model = SentenceTransformer(embed_model_name)

    print("\nLoading indexes ...")
    indexes = load_all_indexes(index_dirs)
    if not indexes:
        print("ERROR: No valid indexes found. Run ingest.py for each PDF first.")
        return

    print(f"\nLoaded {len(indexes)} book index(es) + preloaded knowledge.")

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )

    print("\n" + "="*60)
    print("Pregnancy & Obstetrics Chatbot — Ready!")
    print(f"Books loaded: {len(indexes)}")
    print("Type your question, or 'quit' to exit.")
    print("="*60 + "\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not question:
            continue

        # Retrieve from all FAISS indexes
        faiss_chunks = retrieve_from_all(question, embed_model, indexes)
        # Retrieve from preloaded knowledge (.ts file)
        preloaded_chunks = preloaded_knowledge.search(question, top_k=PRELOADED_TOP_K)
        # Build combined prompt
        user_prompt = build_user_prompt(question, faiss_chunks, preloaded_chunks)

        response = gemini_model.generate_content(user_prompt)
        print(f"\nBot: {response.text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-book pregnancy & obstetrics chatbot"
    )
    parser.add_argument(
        "--indexes",
        nargs="+",
        default=["dutta_index", "oxford_index", "williams_index"],
        help="One or more index folders created by ingest.py",
    )
    args = parser.parse_args()
    main(args.indexes)
