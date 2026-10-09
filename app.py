"""
Flask server. Flow:

    1. POST /api/documents          PDF -> topics -> chunks -> embeddings, saved under data/<doc_id>
    2. GET  /api/documents/<id>     topic list again (e.g. after a page refresh)
    3. Buttons on a topic:
       POST .../notes        notes for one topic in a chosen mode   (topic text, cached)
       POST .../quiz         MCQs for one topic or the whole PDF    (topic text + RAG for the page source)
       POST .../explain      "I didn't understand this"             (RAG)
       POST .../suggestions  ready-made doubts to click             (topic text, cached)
       POST .../ask          answer a doubt from the book           (RAG)
    4. POST /api/tts                notes -> MP3
"""
import io
import logging
import os

import groq
import markdown
import nh3
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from gtts import gTTS
from werkzeug.exceptions import HTTPException

load_dotenv()

from utils.chunker import make_chunks
from utils.document_store import (
    DocumentNotFound, get_cached, load_document, save_document, set_cached,
)
from utils import llm
from utils.pdf_reader import extract_sections
from utils.retriever import embed_chunks, embed_query, top_k

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

MODES = {"beginner": "Beginner", "exam": "Exam Ready", "deep": "Deep Dive"}
RAG_TOP_K = 4
MAX_TTS_CHARS = 6000

if not os.getenv("GROQ_API_KEY"):
    log.warning("GROQ_API_KEY is not set: copy .env.example to .env and add your key, or AI features will fail.")


@app.route("/")
def home():
    return render_template("index.html")


# ── Upload once ───────────────────────────────────────────────────────────────

@app.post("/api/documents")
def upload_document():
    file = request.files.get("pdf_file")
    if not file or not file.filename:
        return _error("Please choose a PDF file.", 400)
    if not file.filename.lower().endswith(".pdf"):
        return _error("Only PDF files are allowed.", 400)

    try:
        # Read straight from memory: the PDF never touches the disk
        sections = extract_sections(io.BytesIO(file.read()))
    except Exception:
        log.exception("Could not parse PDF")
        return _error("Could not read this PDF. Please check that it is a valid PDF file.", 400)

    if not sections:
        return _error("No text found in this PDF. It may be a scanned/image-only PDF.", 400)

    chunks = make_chunks(sections)
    vectors = embed_chunks([c["text"] for c in chunks])
    filename = os.path.basename(file.filename)[:120]
    doc_id = save_document(filename, sections, chunks, vectors)
    log.info("Stored %s: %d topics, %d chunks", doc_id, len(sections), len(chunks))

    return jsonify(_summary(load_document(doc_id)))


@app.get("/api/documents/<doc_id>")
def get_document(doc_id):
    return jsonify(_summary(load_document(doc_id)))


# ── Topic actions ─────────────────────────────────────────────────────────────

@app.post("/api/documents/<doc_id>/notes")
def notes(doc_id):
    doc = load_document(doc_id)
    data = request.get_json(silent=True) or {}
    topic = _topic(doc, data.get("topic_id"))
    mode = data.get("mode", "beginner")
    if mode not in MODES:
        return _error("Invalid mode selected.", 400)

    # Same topic + same mode -> same notes, so don't pay for the AI call twice
    cache_key = f"notes:{topic['id']}:{mode}"
    notes_md = get_cached(doc_id, cache_key)
    if notes_md is None:
        notes_md = llm.generate_notes(topic["title"], topic["text"], mode)
        set_cached(doc_id, cache_key, notes_md)

    return jsonify({"html": _to_html(notes_md), "mode": MODES[mode]})


@app.post("/api/documents/<doc_id>/quiz")
def quiz(doc_id):
    doc = load_document(doc_id)
    data = request.get_json(silent=True) or {}

    if data.get("topic_id") is None:
        texts = [s["text"] for s in doc["sections"]]  # whole PDF
    else:
        texts = [_topic(doc, data["topic_id"])["text"]]

    questions = llm.generate_quiz(texts, count=5)
    if not questions:
        return _error("Could not create a quiz from this part. Please try again.", 500)

    # For each question, find the passage in the book that supports the answer
    for q in questions:
        query = f"{q['question']} {q['options'][q['answer_idx']]}"
        chunk = doc["chunks"][top_k(embed_query(query), doc["vectors"], k=1)[0][0]]
        q["source"] = {"page": chunk["page"], "text": _snippet(chunk["text"])}

    return jsonify({"questions": questions})


@app.post("/api/documents/<doc_id>/explain")
def explain(doc_id):
    doc = load_document(doc_id)
    data = request.get_json(silent=True) or {}
    topic = _topic(doc, data.get("topic_id"))

    # Search with the topic title + its opening lines, so related passages
    # from other parts of the book are found too
    query = topic["title"] + "\n" + " ".join(topic["text"].split()[:60])
    chunks = _retrieve(doc, query)
    answer = llm.explain_simply(topic["title"], llm.format_excerpts(chunks, doc["sections"]))
    return jsonify({"html": _to_html(answer), "sources": _sources(doc, chunks)})


@app.post("/api/documents/<doc_id>/suggestions")
def suggestions(doc_id):
    doc = load_document(doc_id)
    data = request.get_json(silent=True) or {}
    topic = _topic(doc, data.get("topic_id"))

    cache_key = f"suggestions:{topic['id']}"
    questions = get_cached(doc_id, cache_key)
    if questions is None:
        questions = llm.suggest_questions(topic["title"], topic["text"])
        set_cached(doc_id, cache_key, questions)
    return jsonify({"questions": questions})


@app.post("/api/documents/<doc_id>/ask")
def ask(doc_id):
    doc = load_document(doc_id)
    question = ((request.get_json(silent=True) or {}).get("question") or "").strip()[:500]
    if not question:
        return _error("Please type or pick a question.", 400)

    chunks = _retrieve(doc, question)
    answer = llm.answer_question(question, llm.format_excerpts(chunks, doc["sections"]))
    return jsonify({"html": _to_html(answer), "sources": _sources(doc, chunks)})


# ── Audio ─────────────────────────────────────────────────────────────────────

@app.post("/api/tts")
def text_to_speech():
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()[:MAX_TTS_CHARS]
    if not text:
        return _error("No text provided.", 400)

    fp = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(fp)
    fp.seek(0)
    return send_file(fp, mimetype="audio/mpeg")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _retrieve(doc, query):
    """RAG retrieval: the chunks most similar in meaning to the query, in book order."""
    hits = top_k(embed_query(query), doc["vectors"], k=RAG_TOP_K)
    return sorted((doc["chunks"][i] for i, _ in hits), key=lambda c: c["id"])


def _topic(doc, topic_id):
    try:
        return doc["sections"][int(topic_id)]
    except (TypeError, ValueError, IndexError):
        raise DocumentNotFound(f"topic {topic_id}")


def _summary(doc):
    return {
        "doc_id": doc["doc_id"],
        "filename": doc["filename"],
        "topics": [
            {"id": s["id"], "title": s["title"], "page": s["page"], "word_count": s["word_count"]}
            for s in doc["sections"]
        ],
    }


def _sources(doc, chunks):
    titles = {s["id"]: s["title"] for s in doc["sections"]}
    return [
        {"page": c["page"], "topic": titles.get(c["section_id"], ""), "text": _snippet(c["text"])}
        for c in chunks
    ]


def _snippet(text, limit=280):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _to_html(md_text):
    # The LLM's output is untrusted: strip any script/style/event handlers before sending
    return nh3.clean(markdown.markdown(md_text, extensions=["extra", "fenced_code"]))


def _error(message, status):
    return jsonify({"error": message}), status


@app.errorhandler(DocumentNotFound)
def document_not_found(e):
    return _error("This PDF session was not found. Please upload the PDF again.", 404)


@app.errorhandler(413)
def file_too_large(e):
    return _error("File too large. Maximum allowed size is 16MB.", 413)


@app.errorhandler(groq.RateLimitError)
def rate_limited(e):
    return _error("The AI service is busy right now. Please wait a minute and try again.", 429)


@app.errorhandler(llm.MissingApiKey)
def missing_api_key(e):
    return _error("The server has no GROQ_API_KEY. Copy .env.example to .env and add your key.", 500)


@app.errorhandler(Exception)
def unexpected_error(e):
    if isinstance(e, HTTPException):
        return _error(e.description, e.code)
    log.exception("Unexpected error")
    return _error("Something went wrong. Please try again.", 500)


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
