"""
Saves each processed PDF to disk so it is parsed and embedded only once.

data/<doc_id>/
    document.json  -> filename, topics (sections) and chunks
    vectors.npy    -> one embedding vector per chunk (same order as chunks)
    cache.json     -> AI results already generated (notes per topic + mode, suggestions)

The original PDF is never stored.
"""
import json
import re
import uuid
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class DocumentNotFound(Exception):
    pass


def save_document(filename, sections, chunks, vectors):
    doc_id = uuid.uuid4().hex
    folder = _folder(doc_id)
    folder.mkdir(parents=True)

    document = {
        "doc_id": doc_id,
        "filename": filename,
        # "lines" was only needed to build chunks with page numbers
        "sections": [{k: v for k, v in s.items() if k != "lines"} for s in sections],
        "chunks": chunks,
    }
    (folder / "document.json").write_text(json.dumps(document), encoding="utf-8")
    np.save(folder / "vectors.npy", vectors)
    return doc_id


def load_document(doc_id):
    folder = _folder(doc_id)
    if not (folder / "document.json").exists():
        raise DocumentNotFound(doc_id)

    document = json.loads((folder / "document.json").read_text(encoding="utf-8"))
    document["vectors"] = np.load(folder / "vectors.npy")
    return document


def get_cached(doc_id, key):
    return _read_cache(doc_id).get(key)


def set_cached(doc_id, key, value):
    cache = _read_cache(doc_id)
    cache[key] = value
    (_folder(doc_id) / "cache.json").write_text(json.dumps(cache), encoding="utf-8")


def _read_cache(doc_id):
    path = _folder(doc_id) / "cache.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _folder(doc_id):
    # doc_id comes from the browser: only accept our own uuid format, so it can't point outside data/
    if not re.fullmatch(r"[0-9a-f]{32}", doc_id or ""):
        raise DocumentNotFound(doc_id)
    return DATA_DIR / doc_id
