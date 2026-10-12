# Architecture

## Goal

Many students have never used an AI tool and don't know how to write a prompt. This app removes the prompt entirely:
a student uploads a chapter PDF, taps a topic, and taps a button. Every result (notes, a quiz, a simpler explanation,
an answer to a doubt) is generated **only from their own PDF, with page numbers** so it can be checked against the book.
Each button builds its full prompt on the server.

## Pipeline

```
                 UPLOAD (once per PDF)                                    USING IT (many times)
 ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │   PDF    │──▶│  pdf_reader  │──▶│   chunker    │──▶│  retriever   │──▶  data/<doc_id>/
 └──────────┘   │ PDF → topics │   │ topics →     │   │ chunks →     │      document.json (topics, chunks)
                │ (headings,   │   │ ~200-word    │   │ embeddings   │      vectors.npy   (embeddings)
                │  page nos.)  │   │ chunks       │   │ (384 numbers)│      cache.json    (AI results)
                └──────────────┘   └──────────────┘   └──────────────┘
                                                                              │
     Tap a topic ──▶ Notes / Quiz ──────────▶ topic text ──────────────▶ llm ─┤
     "I didn't understand" / Ask a doubt ──▶ search chunks (RAG) ──────▶ llm ─┘──▶ answer + page sources
```

| File | Responsibility |
|---|---|
| `utils/pdf_reader.py` | Reads words with font size and boldness, removes page numbers, repeating headers/footers and equation debris, detects headings, and splits the PDF into **topics** |
| `utils/chunker.py` | Cuts each topic into ~200-word **chunks** with ~40 words of overlap, keeping each chunk's page number |
| `utils/retriever.py` | Turns text into **embeddings** with a small local model and finds the most similar chunks (cosine similarity) |
| `utils/document_store.py` | Saves topics, chunks, vectors and cached AI output under `data/<doc_id>/` |
| `utils/llm.py` | Every prompt sent to Groq: notes, quiz, simpler explanation, suggested doubts, answers |
| `app.py` | Flask API that connects the pieces, renders Markdown to safe HTML, and maps errors to friendly messages |
| `templates/index.html` | Single-page frontend: upload → topic list → topic page (Notes / Quiz / Ask tabs) |

## How context is chosen

1. **The needed text is known → use the topic directly.**
   "Make notes" and "Quiz" on *Stacks* use the Stacks section. No search is needed.
2. **The input is free-form → search (RAG).**
   A doubt like "why is push O(1)?" or "I didn't understand this" could be answered by any part of the book,
   so the query is embedded and the 4 closest chunks are sent to the LLM along with their page numbers.

Retrieval is also used to cite quiz answers: each question plus its correct option is searched,
and the best matching passage is shown as "From your PDF, page N" after the student answers.

## Design decisions

| Decision | Reason |
|---|---|
| Topics from headings (font size / bold) | Matches how the book is organised, and gives students a table of contents to tap instead of a prompt box |
| Topics capped at ~1200 words, tiny ones merged | One topic fits in one LLM call, so notes come back in seconds instead of processing the whole book |
| ~200-word chunks with ~40-word overlap for search | A question should match the paragraph about it, not a whole chapter; overlap keeps sentences on a boundary intact |
| Split on whole lines | Code blocks keep their line breaks and indentation |
| Local embedding model (`BAAI/bge-small-en-v1.5` via fastembed) | Free, no extra API key, fast enough on CPU, no PyTorch needed |
| numpy instead of a vector database | One PDF has a few hundred chunks; a matrix multiply ranks them in milliseconds. A vector database would only pay off at thousands of documents |
| Upload once, reuse a `doc_id` | The PDF is parsed and embedded once; every later action reuses the stored data |
| Cache notes per (topic, mode) and suggested questions per topic | Same input gives the same output, so repeat views cost no AI calls |
| Prompts restrict the model to the given text and allow "I couldn't find this" | Reduces hallucination; page numbers let the student verify |
| Validate and shuffle quiz questions | Models sometimes return 3 options, a wrong answer index, or always put the answer first |
| Configurable model with fallbacks | `GROQ_MODEL` picks the model (default `openai/gpt-oss-120b`); if it isn't available, the next model in the fallback list is tried |

## Frontend

- **Three screens**: upload, topic list, and a topic page with **Notes / Quiz / Ask** tabs, so results never pile up in one long page.
- **Header**: large on the upload screen; once a PDF is open the sun sets behind the mountains and it shrinks to a sticky bar.
  Dark mode swaps the sun for a crescent moon and stars. Animations are disabled when the device asks for reduced motion.
- **Quiz**: one question at a time, then a score summary. "Retry the same questions" and "Practise the ones I got wrong"
  reuse questions already in the browser (no AI call); only "Give me new questions" calls the server.
- **Progress**: per-topic "done" ticks and best quiz scores are stored in the browser's `localStorage`, per PDF.
- **Sessions**: a new visit always starts on the upload screen, with the last PDF offered as "Continue where you left off".
  `sessionStorage` remembers the current screen, so refreshing a tab returns to the same topic.
- **Accessibility**: real buttons and radio inputs, arrow-key navigation between tabs, visible focus rings,
  right/wrong answers shown with icons and text (not color alone), and status messages announced to screen readers.

## Security and robustness

- The uploaded PDF is read in memory and never written to disk.
- `doc_id` is a server-generated UUID and is validated before it is used in a file path.
- LLM output is treated as untrusted: Markdown → HTML → `nh3.clean()` before it reaches the browser.
- Markdown lists are normalised before rendering (blank line before a list, 4-space nesting) so notes display correctly.
- Errors are logged on the server; the student sees a friendly message (rate limit, invalid key, missing model, bad PDF).
- Flask debug mode is off unless `FLASK_DEBUG=1`.

## Known limitations

- **Two-column pages and margin notes**: lines are rebuilt left-to-right across the page, so a side caption can mix into a body line.
  A fix would detect columns from large horizontal gaps and read each column separately.
- **Scanned PDFs** have no text layer. An OCR fallback (e.g. Tesseract) would cover them.
- **Search is meaning-only**, so exact keywords like `malloc` can be missed. Hybrid search (BM25 + embeddings) would help.
- **Embedding runs on the CPU during upload** (~15 s for a 25-page chapter). A background job with real progress would hide this.
- **`data/` is never cleaned up.** Old documents should expire after N days.
- **No user accounts**: progress lives in the browser, so it doesn't follow the student to another device.
