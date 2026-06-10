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
    1. Extract lines and calculate heading scores and vertical gaps (for paragraph detection).
    2. Group lines into blocks (paragraphs/code blocks/headings), ensuring no block exceeds max_words.
    3. Group blocks into chunks of up to max_words (e.g. 600 words), preserving layout and newlines.
    """
    structured_lines, body_size, has_size_variation = _extract_structured_lines(pdf_path)

    if not structured_lines:
        return []

    # 1. Group lines into logical blocks
    # Max words per block is 2000
    blocks = _group_lines_into_blocks(structured_lines, max_words=2000)

    # 2. Determine threshold score for headings
    min_score = 2 if has_size_variation else 1

    # 3. Group blocks into chunks
    chunks = _chunk_blocks(blocks, min_score=min_score, max_words=2000)

    # Clean up and filter out empty or extremely small chunks (less than 10 words)
    cleaned_chunks = []
    for c in chunks:
        c_strip = c.strip()
        if len(c_strip.split()) >= 10:
            cleaned_chunks.append(c_strip)

    return cleaned_chunks


def _extract_structured_lines(pdf_path):
    """
    Extract lines with a heading score based on size/boldness and identify paragraph starts.
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
    has_size_variation = any(s > body_size + 1.0 for s in unique_sizes)

    with pdfplumber.open(pdf_path) as pdf:
        prev_bottom = None
        prev_page_idx = None

        for page_idx, page in enumerate(pdf.pages):
            words = page.extract_words(extra_attrs=["size", "fontname"])
            if not words:
                continue

            lines_by_top = {}
            for w in words:
                lines_by_top.setdefault(round(w["top"]), []).append(w)

            for top in sorted(lines_by_top):
                lw = lines_by_top[top]
                lw.sort(key=lambda w: w["x0"])
                text = " ".join(w["text"] for w in lw).strip()
                text = re.sub(r'[ \t]+', ' ', text)

                # Skip empty text or line numbers / page numbers
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

                score = (2 if is_larger else 0) + (1 if is_bold else 0)

                # Extra filter: long lines or lines ending mid-sentence are not headings
                if word_count > 10 or not ends_clean:
                    score = 0

                # Determine if this line starts a new paragraph / section
                dominant_top = min(w["top"] for w in lw)
                dominant_bottom = max(w["bottom"] for w in lw)
                line_height = dominant_bottom - dominant_top

                is_paragraph_start = False
                if prev_page_idx is not None and page_idx != prev_page_idx:
                    is_paragraph_start = True
                elif prev_bottom is not None:
                    gap = dominant_top - prev_bottom
                    # If vertical gap is more than 50% of line height, or more than 6pt, it's a paragraph break
                    if gap > max(6.0, line_height * 0.5):
                        is_paragraph_start = True

                prev_bottom = dominant_bottom
                prev_page_idx = page_idx

                all_lines.append({
                    "text": text,
                    "score": score,
                    "size": dominant_size,
                    "word_count": word_count,
                    "is_paragraph_start": is_paragraph_start
                })

    return all_lines, body_size, has_size_variation


def _group_lines_into_blocks(structured_lines, max_words=2000):
    """
    Group lines into logical paragraph blocks.
    A new block is started when a heading is encountered, when paragraph start is detected,
    or if the block word count would exceed max_words (safety split).
    """
    blocks = []
    current_block = []
    current_block_words = 0

    for line in structured_lines:
        line_words = line["word_count"]
        # Start new block if heading, paragraph break, or block gets too large
        if (line["score"] > 0 or line["is_paragraph_start"] or (current_block_words + line_words > max_words)) and current_block:
            blocks.append(current_block)
            current_block = [line]
            current_block_words = line_words
        else:
            current_block.append(line)
            current_block_words += line_words

    if current_block:
        blocks.append(current_block)

    return blocks


def _chunk_blocks(blocks, min_score, max_words=2000):
    """
    Group blocks of lines into chunks up to max_words, ensuring newlines and spacing are preserved.
    """
    chunks = []
    current_chunk_blocks = []
    current_word_count = 0

    for block in blocks:
        block_text = "\n".join(l["text"] for l in block).strip()
        if not block_text:
            continue

        block_words = len(block_text.split())

        # Check if the block starts with a heading
        first_line = block[0]
        is_heading = first_line["score"] >= min_score

        # Start a new chunk if:
        # 1. Heading block is found and current chunk is already somewhat filled (avoid tiny chunks)
        # 2. Or, adding this block would exceed the target word size
        if (is_heading and current_word_count >= 500) or \
           (current_word_count + block_words > max_words and current_chunk_blocks):
            
            # Emit current chunk
            chunk_content = []
            for b in current_chunk_blocks:
                chunk_content.append("\n".join(l["text"] for l in b))
            chunks.append("\n\n".join(chunk_content))

            # Start new chunk
            current_chunk_blocks = [block]
            current_word_count = block_words
        else:
            current_chunk_blocks.append(block)
            current_word_count += block_words

    if current_chunk_blocks:
        chunk_content = []
        for b in current_chunk_blocks:
            chunk_content.append("\n".join(l["text"] for l in b))
        chunks.append("\n\n".join(chunk_content))

    return chunks