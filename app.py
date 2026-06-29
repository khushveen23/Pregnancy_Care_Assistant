"""
app.py
------
Flask web server for the Pregnancy & Obstetrics Voice Chatbot.

Endpoints:
  GET  /                        → Serves the web UI (index.html)
  POST /api/chat                → Text question → Gemini answer (JSON)
  POST /api/speech-to-text      → WAV audio → transcribed text (JSON)
  GET  /api/text-to-speech      → ?text=...&lang=en → MP3 audio stream

Run:
    $env:GEMINI_API_KEY="your-key"
    python app.py
Then open: http://localhost:5000
"""

import os
import uuid
import json
import tempfile

import faiss
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from gtts import gTTS
import whisper
import preloaded_knowledge

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INDEX_DIRS = ["dutta_index", "oxford_index", "williams_index"]
TOP_K_PER_INDEX = 5
PRELOADED_TOP_K = 4
WHISPER_MODEL_SIZE = "base"   # base | small | medium

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
2. If the CONTEXT is insufficient, use your own expert medical knowledge. Do NOT say "I don't have this information."
3. Never mention page numbers, page references, or citations in your response.
4. Give thorough answers covering causes, symptoms, diagnosis, management, and complications.
5. Explain medical jargon clearly in simple terms.
6. Structure longer answers with bullet points or short paragraphs.
7. Always end with a brief reminder to consult a qualified doctor or midwife.
"""

# ---------------------------------------------------------------------------
# Startup: Load models and indexes
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Pregnancy & Obstetrics Voice Chatbot — Starting up")
print("=" * 60)

print("\n[1/4] Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("[2/4] Loading FAISS indexes...")
loaded_indexes = []
for d in INDEX_DIRS:
    if not os.path.isdir(d):
        print(f"  [SKIP] '{d}' not found.")
        continue
    try:
        index = faiss.read_index(os.path.join(d, "index.faiss"))
        with open(os.path.join(d, "chunks.json"), encoding="utf-8") as f:
            chunks = json.load(f)
        with open(os.path.join(d, "config.json")) as f:
            config = json.load(f)
        loaded_indexes.append((index, chunks, config))
        print(f"  OK  {d}  ({config.get('source_pdf', d)})")
    except Exception as e:
        print(f"  FAIL {d}: {e}")

print(f"[3/4] Loading Whisper STT model ('{WHISPER_MODEL_SIZE}')...")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)

print("[4/4] Connecting to Gemini...")
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT,
)

print("\n[OK] All systems ready! Starting web server...\n")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def retrieve_from_all(question: str) -> list[dict]:
    """FAISS search across all loaded indexes, merge + deduplicate by score."""
    q_emb = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    all_results = []
    for index, chunks, config in loaded_indexes:
        scores, ids = index.search(q_emb, TOP_K_PER_INDEX)
        for score, idx in zip(scores[0], ids[0]):
            if idx != -1:
                chunk = chunks[idx].copy()
                chunk["_score"] = float(score)
                all_results.append(chunk)

    all_results.sort(key=lambda x: x["_score"], reverse=True)
    seen, deduped = set(), []
    for r in all_results:
        key = r["text"][:100]
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped[:TOP_K_PER_INDEX * 2]


def build_prompt(question: str, faiss_chunks: list, preloaded_chunks: list) -> str:
    all_chunks = faiss_chunks + preloaded_chunks
    context = "\n\n".join(c["text"] for c in all_chunks) if all_chunks else "No context retrieved."
    return f"""CONTEXT (from authoritative obstetrics and gynecology textbooks):
{context}

QUESTION: {question}

Answer the question using the CONTEXT above as your primary source. Follow the system rules."""


# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        faiss_chunks = retrieve_from_all(question)
        preloaded_chunks = preloaded_knowledge.search(question, top_k=PRELOADED_TOP_K)
        prompt = build_prompt(question, faiss_chunks, preloaded_chunks)
        response = gemini_model.generate_content(prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/speech-to-text", methods=["POST"])
def speech_to_text():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    tmp_path = os.path.join(tempfile.gettempdir(), f"stt_{uuid.uuid4()}.wav")

    try:
        audio_file.save(tmp_path)
        result = whisper_model.transcribe(tmp_path)
        return jsonify({
            "success": True,
            "text": result["text"].strip(),
            "language": result.get("language", "en"),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/api/text-to-speech")
def text_to_speech():
    text = request.args.get("text", "").strip()
    lang = request.args.get("lang", "en")
    if not text:
        return jsonify({"error": "Text is required"}), 400

    tmp_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4()}.mp3")
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(tmp_path)
        return send_file(tmp_path, mimetype="audio/mpeg", as_attachment=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
