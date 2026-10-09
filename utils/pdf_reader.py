"""
Step 1 of the pipeline: PDF -> topics (sections).

1. Read every word with its font size, font name and position (pdfplumber).
2. Rebuild lines, dropping page numbers and headers/footers that repeat on every page.
3. A line is a heading if it is bigger (or bold) compared to normal body text.
4. Everything between two headings becomes one section, which the UI shows as a "topic".
"""
import re
import logging
from collections import Counter

import pdfplumber

from utils.chunker import split_lines

logging.getLogger("pdfminer").setLevel(logging.ERROR)

LINE_TOLERANCE = 3        # words whose tops are within 3pt belong to the same line
MIN_SECTION_WORDS = 40    # smaller sections are merged into the next one
MAX_SECTION_WORDS = 1200  # larger sections are split into "part 1", "part 2", ...
MAX_TITLE_CHARS = 80


def extract_sections(pdf_file):
    """
    pdf_file: a path or a file-like object.
    Returns a list of sections: {id, title, page, text, word_count, lines}
    where lines is a list of {"text", "page"} (used later for chunking with page numbers).
    """
    pages = _read_pages(pdf_file)
    pages = _remove_headers_and_footers(pages)
    lines = [line for page in pages for line in page]
    if not lines:
        return []

    sections = _split_at_headings(lines)
    sections = _merge_tiny_sections(sections)
    sections = _split_huge_sections(sections)

    result = []
    for i, section in enumerate(sections):
        text = "\n".join(line["text"] for line in section["lines"])
        result.append({
            "id": i,
            "title": section["title"][:MAX_TITLE_CHARS],
            "page": section["lines"][0]["page"],
            "text": text,
            "word_count": len(text.split()),
            "lines": section["lines"],
        })
    return result


# ── Reading lines ─────────────────────────────────────────────────────────────

def _read_pages(pdf_file):
    """Returns one list of lines per page."""
    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size", "fontname"])
            words.sort(key=lambda w: (w["top"], w["x0"]))

            lines, current = [], []
            for word in words:
                if current and abs(word["top"] - current[0]["top"]) > LINE_TOLERANCE:
                    lines.append(_make_line(current, page_no))
                    current = []
                current.append(word)
            if current:
                lines.append(_make_line(current, page_no))

            pages.append([line for line in lines if line])
    return pages


def _make_line(words, page_no):
    words.sort(key=lambda w: w["x0"])
    text = re.sub(r"\s+", " ", " ".join(w["text"] for w in words)).strip()

    # Skip empty lines, and debris from figures/equations like "A 2 B 5 S S" (no real word).
    # Lines with code symbols are kept, so code such as "}" or "x = 5" survives.
    if not text or not (re.search(r"[A-Za-z]{3,}", text) or re.search(r"[{}()\[\];=<>]", text)):
        return None

    size = Counter(round(w["size"], 1) for w in words).most_common(1)[0][0]
    bold_words = sum(1 for w in words if any(b in w["fontname"] for b in ("Bold", "Heavy", "Black")))

    return {
        "text": text,
        "size": size,
        "bold": bold_words / len(words) >= 0.6,
        "word_count": len(text.split()),
        "page": page_no,
    }


def _remove_headers_and_footers(pages):
    """Drop lines at the top/bottom of a page that repeat on at least half the pages."""
    if len(pages) < 3:
        return pages

    def key(text):
        return re.sub(r"\d+", "#", text.lower())  # "Page 3" and "Page 4" count as the same

    def is_edge(i, page):
        return i < 2 or i >= len(page) - 2

    counts = Counter()
    for page in pages:
        counts.update({key(line["text"]) for i, line in enumerate(page) if is_edge(i, page)})
    repeated = {k for k, c in counts.items() if c >= len(pages) / 2}

    return [
        [line for i, line in enumerate(page) if not (is_edge(i, page) and key(line["text"]) in repeated)]
        for page in pages
    ]


# ── Building sections ─────────────────────────────────────────────────────────

def _split_at_headings(lines):
    body_size = _body_font_size(lines)
    has_bigger_text = any(line["size"] > body_size + 1 for line in lines)

    sections = [{"title": "Introduction", "lines": [], "found_heading": False}]
    for line in lines:
        if _is_heading(line, body_size, has_bigger_text):
            current = sections[-1]
            if not current["lines"] and current["found_heading"]:
                # Two heading lines in a row ("Chapter 3" + "Data Structures") -> one title
                current["title"] += " " + line["text"]
            else:
                sections.append({"title": line["text"], "lines": [], "found_heading": True})
        else:
            sections[-1]["lines"].append({"text": line["text"], "page": line["page"]})

    return [s for s in sections if s["lines"]]


def _body_font_size(lines):
    """The font size used by the most words is the normal body text size."""
    counts = Counter()
    for line in lines:
        counts[line["size"]] += line["word_count"]
    return counts.most_common(1)[0][0]


def _is_heading(line, body_size, has_bigger_text):
    text = line["text"]
    # Headings are short, don't end like a sentence, and contain a real word (not just math symbols)
    if line["word_count"] > 10 or text.endswith((".", ",", ";")) or not re.search(r"[A-Za-z]{3,}", text):
        return False
    if has_bigger_text:
        # If the PDF uses bigger fonts for headings, only trust size (bold body words are not headings)
        return line["size"] > body_size + 1
    return line["bold"]


def _merge_tiny_sections(sections):
    """A section with almost no text is merged into the following section."""
    merged = []
    carry = []
    for section in sections:
        section_words = sum(len(l["text"].split()) for l in section["lines"])
        if section_words < MIN_SECTION_WORDS:
            if section["found_heading"]:
                carry.append({"text": section["title"], "page": section["lines"][0]["page"]})
            carry += section["lines"]
            continue
        section["lines"] = carry + section["lines"]
        carry = []
        merged.append(section)

    if carry:
        if merged:
            merged[-1]["lines"] += carry
        else:
            merged.append({"title": sections[0]["title"], "lines": carry, "found_heading": False})
    return merged


def _split_huge_sections(sections):
    """Very long sections become parts, so one topic is never too big for one AI call."""
    no_headings = len(sections) == 1 and not sections[0]["found_heading"]
    result = []
    for section in sections:
        parts = split_lines(section["lines"], MAX_SECTION_WORDS, overlap_words=0)
        for n, part in enumerate(parts, start=1):
            if no_headings:
                first, last = part[0]["page"], part[-1]["page"]
                title = f"Page {first}" if first == last else f"Pages {first}–{last}"
            elif len(parts) > 1:
                title = f"{section['title']} (part {n})"
            else:
                title = section["title"]
            result.append({"title": title, "lines": part, "found_heading": section["found_heading"]})
    return result
