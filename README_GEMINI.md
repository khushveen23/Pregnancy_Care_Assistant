# PDF Chatbot (RAG) — Gemini API version

This is a retrieval-augmented chatbot: it doesn't "train" a model on your
PDF in the fine-tuning sense. Instead it indexes the PDF so it can find the
exact relevant passages, then asks Gemini to answer using ONLY those
passages — with page citations, and a clear "not in the book" response
when something isn't covered.

## How it works
```
PDF → extract text → split into chunks → embed chunks → FAISS index   (ingest.py)
question → embed question → find closest chunks → Gemini answers from them   (chatbot_gemini.py)
```

## Setup

```bash
pip install -r requirements.txt
pip install google-generativeai
```

Get a **free** Gemini API key (no credit card needed):
1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API key"
3. Copy the key

Set it as an environment variable:
```bash
export GEMINI_API_KEY="your-key-here"     # Mac/Linux
setx GEMINI_API_KEY "your-key-here"        # Windows (restart terminal after)
```

## Step 1 — Build the index (run once per PDF)

```bash
python ingest.py --pdf "DC_dutta_8th.pdf" --out "dutta_index"
```

Reads all pages, chunks them, embeds them locally (free, no API calls
needed for this step), and saves the index into `dutta_index/`. Takes a
few minutes the first time — it's a one-time cost per PDF.

## Step 2 — Chat

```bash
python chatbot_gemini.py --index dutta_index
```

Ask questions like:
```
You: What are the risk factors for pre-eclampsia?
Bot: ... answer based only on the book, with page citations (p. 123) ...
```

If you ask something the book doesn't cover, it'll say so instead of
guessing — that's the grounding rule in `SYSTEM_PROMPT` inside
`chatbot_gemini.py`.

Type `quit` or `exit` to leave the chat.

## Using a different PDF later

No code changes needed — just re-run ingest with a new file and folder
name:
```bash
python ingest.py --pdf "some_other_book.pdf" --out "other_index"
python chatbot_gemini.py --index "other_index"
```
You can keep multiple indexes side by side and switch with `--index`.

## Gemini free tier — what to expect
- No credit card required to get a key.
- Free tier has rate limits (requests per minute and per day) — fine for
  development, demos, and personal use; you'll hit limits only under
  heavy/rapid use.
- `gemini-1.5-flash` (the default model in the script) is fast and sits
  comfortably within free-tier limits. `gemini-1.5-pro` is higher quality
  but has a smaller free quota — change `MODEL_NAME` in
  `chatbot_gemini.py` if you want to try it.
- If you ever hit a quota error, it'll show up as a `ResourceExhausted`
  exception — just wait a bit or switch back to `flash`.

## Notes / things you can tune
- `CHUNK_SIZE` / `CHUNK_OVERLAP` in `ingest.py` — smaller chunks = more
  precise retrieval but less context per chunk. 800/150 is a solid
  default for textbook prose.
- `TOP_K` in `chatbot_gemini.py` — how many chunks get pulled in per
  question. Raise it (e.g. to 8) if answers feel incomplete.
- Embeddings (`all-MiniLM-L6-v2`) run fully on your machine — no API cost
  for ingestion. Only the answer-generation step calls Gemini.
- For a scanned (image-only) PDF, `pypdf` extracts no text — you'd need
  OCR first. Your DC Dutta PDF extracts cleanly, so this isn't an issue.
- One document per index, by design — keeps answers grounded to a single
  trusted source instead of blending in outside knowledge, which matters
  for a medical chatbot.
