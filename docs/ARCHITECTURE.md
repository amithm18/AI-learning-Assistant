# How it works

This file explains the design in plain words, so it can be walked through in an interview.

## The idea in one line

A student who has never used an AI tool uploads their textbook chapter, taps a topic, and gets notes,
a quiz, a simpler explanation, or an answer to a doubt, all taken **only from their own book, with page numbers**.
The student never writes a prompt; every button builds its prompt on the server.

## The pipeline

```
                 UPLOAD (once per PDF)                                    USING IT (many times)
 ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │   PDF    │──▶│  pdf_reader  │──▶│   chunker    │──▶│  retriever   │──▶  data/<doc_id>/
 └──────────┘   │ PDF → topics │   │ topics →     │   │ chunks →     │      document.json (topics, chunks)
                │ (headings,   │   │ ~200-word    │   │ embeddings   │      vectors.npy   (embeddings)
                │  page nos.)  │   │ chunks       │   │ (384 numbers)│      cache.json    (AI results)
                └──────────────┘   └──────────────┘   └──────────────┘
                                                                              │
     Tap a topic ──▶ Make notes / Quiz me ──▶ topic text ──────────────▶ llm ─┤
     "I didn't understand" / Ask a doubt ──▶ search chunks (RAG) ──────▶ llm ─┘──▶ answer + page sources
```

| File | Job |
|---|---|
| `utils/pdf_reader.py` | Reads words with font size/boldness, removes page numbers, headers/footers and equation debris, detects headings, and splits the PDF into **topics** |
| `utils/chunker.py` | Cuts each topic into ~200-word **chunks** with ~40 words of overlap, keeping the page number of each chunk |
| `utils/retriever.py` | Turns text into **embeddings** with a small local model and finds the most similar chunks (cosine similarity) |
| `utils/document_store.py` | Saves topics, chunks, vectors and cached AI output under `data/<doc_id>/` |
| `utils/llm.py` | Every prompt sent to Groq (Llama 3.3 70B): notes, quiz, simple explanation, suggested doubts, answers |
| `app.py` | Flask API that connects the pieces |
| `templates/index.html` | Single page: upload → topic list → topic page |

## Two kinds of "context" (the key design decision)

1. **I know exactly which text is needed → use the topic directly.**
   When the student taps "Make notes" on *Stacks*, the text is the Stacks section. No search needed.
2. **The input is free-form → search (RAG).**
   A doubt like "why is push O(1)?" or "I didn't understand this" could be answered by any part of the book,
   so the question is embedded and the 4 closest chunks are sent to the LLM along with their page numbers.

RAG is also used to cite quiz answers: each question plus its correct option is searched,
and the best matching passage is shown as "From your PDF, page N" after the student answers.

## Why these choices

| Decision | Why |
|---|---|
| Topics from headings (font size / bold) | Matches how the book is organised, and gives students a table of contents to tap instead of a prompt box |
| Topics capped at ~1200 words | One topic = one LLM call, so notes come back in seconds instead of processing the whole book |
| Chunks of ~200 words for search | A question should match the paragraph about it, not a whole chapter. Small chunks give precise matches; overlap stops a sentence being cut in half |
| Split on whole lines | Code blocks keep their line breaks and indentation |
| Local embedding model (`bge-small-en-v1.5` via fastembed) | Free, no extra API key, fast enough on CPU, no PyTorch needed |
| numpy instead of a vector database | One PDF has a few hundred chunks; a matrix multiply finds the best ones in milliseconds. A vector DB (Chroma, FAISS) would only be needed for thousands of documents |
| Upload once, then reuse a `doc_id` | The old version re-uploaded and re-parsed the PDF for every action |
| Cache notes per (topic, mode) | Same input gives the same notes, so we don't pay for the AI call twice (Groq free tier has rate limits) |
| Prompts say "use ONLY the excerpts" and "say you couldn't find it" | Reduces hallucination; page numbers let the student verify |
| Validate + shuffle quiz questions | LLMs sometimes return 3 options, a wrong index, or always put the answer first |

## Problems in the first version and how they were fixed

| Problem | Fix |
|---|---|
| Every button re-uploaded and re-parsed the whole PDF | Upload once → `doc_id` → everything reuses stored topics/chunks |
| Notes for a long PDF = dozens of sequential LLM calls in one HTTP request (slow, hits rate limits) | Student picks a topic; one call per topic, cached |
| Uploaded file saved using the browser's filename (path traversal, two users with the same filename clash) | PDF is read in memory and never saved; `doc_id` is a server-generated UUID checked with a regex |
| LLM HTML inserted into the page unsanitised | Markdown → HTML → `nh3.clean()` on the server |
| `debug=True` always on | Only when `FLASK_DEBUG=1` |
| Page numbers were lost | Every line, chunk and answer keeps its page number |
| Repeating headers/footers and equation fragments polluted notes | Removed in `pdf_reader` |
| PDF opened twice; lines grouped by exact `top` value (slightly misaligned words split into separate lines) | One pass; words within 3pt are the same line |
| Quiz trusted the model's JSON blindly | `validate_question()` checks 4 unique options and a valid answer index, then shuffles |
| Raw exception text sent to the browser | Errors are logged on the server; the student sees a friendly message (rate limit, missing key, bad PDF) |

## Known limitations (good "what would you improve?" answers)

- **Two-column pages / margin notes**: lines are rebuilt left-to-right across the page, so a side caption can get mixed into a body line.
  Fix: detect columns (big horizontal gaps) and read each column separately.
- **Scanned PDFs** have no text layer. Fix: OCR (e.g. Tesseract) as a fallback.
- **Search is meaning-only.** Exact keywords like `malloc` can be missed. Fix: hybrid search (BM25 keyword score + embedding score).
- **Embedding runs on CPU at upload** (~15s for a 25-page chapter). Fix: background job with a progress bar.
- **`data/` grows forever.** Fix: delete documents older than N days.
- **No user accounts.** The `doc_id` in the browser's localStorage is the only "session".

## Interview cheat sheet

- **What is RAG?** Retrieval-Augmented Generation: before asking the LLM, find the relevant parts of *your* data
  and put them in the prompt, so the answer is grounded in that data instead of the model's memory.
- **What is an embedding?** A list of numbers representing the meaning of a text. Similar meanings → vectors pointing the same way.
- **What is cosine similarity?** The cosine of the angle between two vectors (1 = same direction). With unit-length vectors it's just a dot product.
- **Why chunk overlap?** So information on a chunk boundary is complete in at least one chunk.
- **Why not send the whole PDF to the LLM?** Cost, speed, rate limits, and models answer worse when the relevant part is buried in a long context.
- **How do you reduce hallucination?** Restrict the prompt to the given excerpts, allow "I couldn't find this", show sources with page numbers, low temperature.
