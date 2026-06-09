import re
import pdfplumber
from collections import Counter
import warnings
import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")



def extract_text_from_pdf(pdf_path):
    """
    General purpose PDF text extractor and chunker.
    Works on any subject PDF regardless of topic, language or formatting.

    Strategy:
    1. Try font-size hierarchy first (works for most well-formatted PDFs)
    2. If font sizes are uniform, fall back to bold-only detection
    3. If even bold is uniform, fall back to paragraph-boundary chunking
    Each fallback ensures we always produce usable chunks.
    """
    structured_lines, body_size, has_size_variation = _extract_structured_lines(pdf_path)

    if has_size_variation:
        # Best case: PDF uses different font sizes for headings
        chunks = _chunk_by_score(structured_lines, min_score=2, max_words=900)
    else:
        # Fallback: try bold-only signal
        chunks = _chunk_by_score(structured_lines, min_score=1, max_words=900)

    # If chunking produced too few or too many chunks, use paragraph fallback
    full_text = " ".join(l["text"] for l in structured_lines)
    total_words = len(full_text.split())
    expected_min = max(2, total_words // 900)
    expected_max = max(15, total_words // 150)

    if len(chunks) < expected_min or len(chunks) > expected_max:
        chunks = _chunk_by_paragraphs(full_text, max_words=700)

    chunks = _filter_code_heavy_chunks(chunks)
    return [c.strip() for c in chunks if len(c.split()) >= 30]


# ── Step 1: Extract lines with heading scores ─────────────────────────────────

def _extract_structured_lines(pdf_path):
    """
    Extract lines with a heading score based on two signals:
      - Font size larger than body = +2 points (strongest signal)
      - Bold font = +1 point (weaker signal, used as fallback)

    Score 3 = bold + larger  → definitely a major heading
    Score 2 = larger only    → likely a major heading
    Score 1 = bold only      → maybe a heading (used as fallback)
    Score 0 = body text

    Also returns whether the PDF has meaningful font size variation,
    so the caller can decide which signal to trust.
    """
    all_lines = []
    all_sizes = []

    with pdfplumber.open(pdf_path) as pdf:
        # First pass: collect all font sizes document-wide
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["size", "fontname"])
            all_sizes.extend(round(w.get("size", 12), 1) for w in words)

    if not all_sizes:
        return [], 12.0, False

    body_size = Counter(all_sizes).most_common(1)[0][0]
    unique_sizes = set(round(s, 1) for s in all_sizes)
    # Meaningful size variation = at least one size more than 1pt larger than body
    has_size_variation = any(s > body_size + 1.0 for s in unique_sizes)

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["size", "fontname"])
            if not words:
                continue

            lines_by_top = {}
            for w in words:
                lines_by_top.setdefault(round(w["top"]), []).append(w)

            for top in sorted(lines_by_top):
                lw = lines_by_top[top]
                text = " ".join(w["text"] for w in lw).strip()
                text = re.sub(r'[ \t]+', ' ', text)

                if not text or re.match(r'^\d+$', text):
                    continue

                sizes = [round(w.get("size", body_size), 1) for w in lw]
                fonts = [w.get("fontname", "") for w in lw]
                dominant_size = Counter(sizes).most_common(1)[0][0]
                bold_ratio = sum(
                    1 for f in fonts
                    if any(b in f for b in ["Bold", "Heavy", "Black"])
                ) / len(fonts)

                is_larger = dominant_size > body_size + 1.0
                is_bold = bold_ratio >= 0.6
                word_count = len(text.split())
                ends_clean = not text.rstrip().endswith(('.', ',', ';'))

                # Score heading signals
                score = (2 if is_larger else 0) + (1 if is_bold else 0)

                # Extra filter: long lines or lines ending mid-sentence
                # are almost never real headings regardless of font
                if word_count > 10 or not ends_clean:
                    score = 0

                all_lines.append({
                    "text": text,
                    "score": score,
                    "size": dominant_size,
                    "word_count": word_count
                })

    return all_lines, body_size, has_size_variation


# ── Step 2: Chunk at heading boundaries ───────────────────────────────────────

def _chunk_by_score(structured_lines, min_score, max_words):
    """
    Split text into chunks at lines that meet the minimum heading score.
    Sub-headings below the threshold stay inside their parent chunk.
    """
    sections = []
    current = []

    for line in structured_lines:
        if line["score"] >= min_score and current:
            text = "\n".join(l["text"] for l in current).strip()
            if text:
                sections.append(text)
            current = [line]
        else:
            current.append(line)

    if current:
        text = "\n".join(l["text"] for l in current).strip()
        if text:
            sections.append(text)

    # Split oversized sections at sentence boundaries
    final = []
    for section in sections:
        if len(section.split()) <= max_words:
            final.append(section)
        else:
            final.extend(_split_at_sentences(section, max_words))

    return final


# ── Step 3: Paragraph fallback chunker ───────────────────────────────────────

def _chunk_by_paragraphs(text, max_words=700):
    """
    Last resort chunker — splits at blank lines between paragraphs.
    Never splits mid-sentence. Used when font signals are unreliable.
    Targets 400-700 words per chunk for good AI note generation.
    """
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks, current, count = [], [], 0

    for para in paragraphs:
        wc = len(para.split())
        if count + wc > max_words and count >= 200 and current:
            chunks.append("\n\n".join(current))
            current, count = [para], wc
        else:
            current.append(para)
            count += wc

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _split_at_sentences(text, max_words):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, count = [], [], 0
    for s in sentences:
        wc = len(s.split())
        if count + wc > max_words and current:
            chunks.append(" ".join(current))
            current, count = [s], wc
        else:
            current.append(s)
            count += wc
    if current:
        chunks.append(" ".join(current))
    return chunks


# ── Step 4: Filter pure code chunks ──────────────────────────────────────────

def _filter_code_heavy_chunks(chunks):
    """
    Skip chunks that are overwhelmingly raw code with no explanation.
    Uses { } ; as strict code indicators. 70% threshold is conservative
    to avoid filtering normal text that contains brackets.
    """
    filtered = []
    for chunk in chunks:
        lines = [l.strip() for l in chunk.split('\n') if l.strip()]
        if not lines:
            continue
        code_lines = sum(
            1 for l in lines
            if sum(1 for c in l if c in '{};') >= 2
        )
        if code_lines / len(lines) < 0.70:
            filtered.append(chunk)
    return filtered