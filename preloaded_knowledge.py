"""
preloaded_knowledge.py
-----------------------
Parses the preloaded_knowledge.ts file and provides keyword-based
search over its structured knowledge chunks.
These chunks are merged with FAISS retrieval results in chatbot.py.
"""

import re
import os

# Path to the TypeScript knowledge file (same directory as this script)
_TS_FILE = os.path.join(os.path.dirname(__file__), "preloaded_knowledge.ts")

# Common English stop-words to ignore during keyword matching
_STOP_WORDS = {
    "a", "an", "the", "is", "in", "on", "at", "to", "of", "and", "or",
    "for", "with", "as", "by", "that", "this", "it", "are", "was", "be",
    "have", "has", "had", "do", "does", "did", "not", "but", "from",
    "what", "how", "when", "where", "which", "who", "can", "will", "may",
    "about", "more", "also", "its", "their", "they", "you", "me", "my",
    "we", "he", "she", "i", "if", "so", "up", "out", "into", "than",
    "then", "there", "any", "all", "both", "each", "during", "after",
    "before", "between", "through", "against", "because", "while",
}


def _parse_ts_file(path: str) -> list[dict]:
    """
    Parse the TypeScript file and extract all PreloadedChunk objects.
    Returns a list of dicts with keys: title, category, content.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    chunks = []
    # Match each object block between { and the next },
    pattern = re.compile(
        r'\{\s*title:\s*"([^"]+)",\s*category:\s*"([^"]+)",\s*content:\s*"([\s\S]*?)"\s*\}',
        re.MULTILINE,
    )
    for m in pattern.finditer(raw):
        title = m.group(1).strip()
        category = m.group(2).strip()
        content = m.group(3).strip()
        # Unescape common escape sequences from TypeScript strings
        content = content.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
        chunks.append({"title": title, "category": category, "content": content})
    return chunks


def _keywords(text: str) -> set[str]:
    """Extract meaningful keywords from a string."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


# Load chunks once at import time
_CHUNKS: list[dict] = _parse_ts_file(_TS_FILE)


def search(question: str, top_k: int = 4) -> list[dict]:
    """
    Keyword-overlap search over the preloaded knowledge chunks.
    Returns up to top_k chunks most relevant to the question,
    formatted to match FAISS chunk format: {page, text}.
    """
    q_kw = _keywords(question)
    scored = []
    for chunk in _CHUNKS:
        chunk_text = f"{chunk['title']} {chunk['category']} {chunk['content']}"
        chunk_kw = _keywords(chunk_text)
        overlap = len(q_kw & chunk_kw)
        if overlap > 0:
            scored.append((overlap, chunk))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for _, chunk in scored[:top_k]:
        # Format to match FAISS chunk structure expected by chatbot.py
        results.append({
            "page": f"[{chunk['category']}]",
            "text": f"**{chunk['title']}**\n{chunk['content']}",
        })
    return results


if __name__ == "__main__":
    # Quick test
    q = "What are the symptoms of pregnancy?"
    hits = search(q)
    print(f"Query: {q}\nResults: {len(hits)}\n")
    for h in hits:
        print(f"  {h['page']}: {h['text'][:120]}...")
