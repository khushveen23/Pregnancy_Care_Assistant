# 🤰 Pregnancy Care Assistant — AI Voice Chatbot

An AI-powered pregnancy & obstetrics voice chatbot trained on three authoritative medical textbooks. Ask questions by **text or voice** and get detailed, accurate answers with **text-to-speech** audio playback.

---

## ✨ Features

- 🎤 **Voice Input** — Speak your question (Whisper STT)
- 🔊 **Voice Output** — Answers read aloud automatically (gTTS)
- 💬 **Text Chat** — Type questions and get instant answers
- 📚 **Multi-book RAG** — Retrieves from 3 medical textbooks simultaneously
- 🌸 **Beautiful UI** — Light pink & white responsive web interface
- 🖼️ **Embeddable** — Drop into any website via `<iframe>`

---

## 📚 Knowledge Sources

| Book | Coverage |
|---|---|
| DC Dutta's Gynecology | Gynecology, obstetric complications, GTD |
| Oxford Handbook of Obstetrics & Gynaecology | Quick-reference obstetrics (856 pages) |
| Williams Obstetrics | Comprehensive obstetrics (1,376 pages) |

---

## 🗂️ Project Structure

```
model_online_api/
├── app.py                   # Flask backend (Chat + STT + TTS APIs)
├── ingest.py                # PDF → FAISS index builder
├── chatbot.py               # Terminal chatbot (CLI version)
├── preloaded_knowledge.py   # Keyword search on preloaded_knowledge.ts
├── preloaded_knowledge.ts   # Structured pregnancy Q&A knowledge base
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html           # Web UI (pink theme)
├── dynamic_stt.ipynb        # Speech-to-Text experiments
└── dynamic_tts.ipynb        # Text-to-Speech experiments
```

> **Note:** PDF files, FAISS indexes (`.faiss`), and the `.venv` folder are excluded from this repo (see `.gitignore`). You must build them locally.

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd model_online_api
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your PDF textbooks
Place your PDF books in the project root:
- `DC_dutta_8th.pdf`
- `Oxford Handbook of Obstetrics and Gynaecology - Sally Collins.pdf`
- `Williams Obstetrics-1376hlm.pdf`

### 5. Build FAISS indexes
```bash
python ingest.py --pdf "DC_dutta_8th.pdf" --out "dutta_index"
python ingest.py --pdf "Oxford Handbook of Obstetrics and Gynaecology - Sally Collins.pdf" --out "oxford_index"
python ingest.py --pdf "Williams Obstetrics-1376hlm.pdf" --out "williams_index"
```
> ⚠️ Williams Obstetrics is 1,376 pages — indexing takes ~15 minutes.

### 6. Get a Gemini API Key
Get your free key at [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## 🚀 Running the Web App

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"
$env:PYTHONIOENCODING="utf-8"
python app.py
```

Then open your browser at: **http://localhost:5000**

---

## 🖼️ Embed in Any Website

```html
<iframe 
  src="http://localhost:5000" 
  width="860" 
  height="600" 
  frameborder="0"
  style="border-radius:16px; box-shadow:0 8px 32px rgba(233,30,140,0.2);"
></iframe>
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | `{ "question": "..." }` → `{ "answer": "..." }` |
| `POST` | `/api/speech-to-text` | Upload WAV audio → `{ "text": "..." }` |
| `GET`  | `/api/text-to-speech` | `?text=...&lang=en` → MP3 audio stream |

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, FAISS, SentenceTransformers
- **AI**: Google Gemini 2.5 Flash (LLM), OpenAI Whisper (STT), gTTS (TTS)
- **Frontend**: Vanilla HTML/CSS/JS (no frameworks)
- **Embeddings**: `all-MiniLM-L6-v2` via SentenceTransformers

---

## ⚠️ Disclaimer

This chatbot is for **educational purposes only**. Always consult a qualified doctor or midwife for medical advice.
