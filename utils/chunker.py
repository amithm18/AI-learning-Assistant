"""
Step 2 of the pipeline: topics -> small overlapping chunks for search (RAG).

Topics can be ~1000 words, which is too broad for search: a question about "push"
should find the exact paragraph about push, not a whole chapter. So each topic is
cut into ~200-word chunks. Neighbouring chunks share ~40 words of overlap so a
sentence that falls on a boundary is still complete in one of them.

We split on whole lines (never in the middle of a line) so code stays formatted.
"""

CHUNK_WORDS = 200
OVERLAP_WORDS = 40


def make_chunks(sections):
    """Returns a list of chunks: {id, section_id, page, text}."""
    chunks = []
    for section in sections:
        for part in split_lines(section["lines"], CHUNK_WORDS, OVERLAP_WORDS):
            chunks.append({
                "id": len(chunks),
                "section_id": section["id"],
                "page": part[0]["page"],
                "text": "\n".join(line["text"] for line in part),
            })
    return chunks


def split_lines(lines, chunk_words, overlap_words):
    """
    Groups lines ({"text", "page"}) into parts of about chunk_words words.
    The last lines of each part (about overlap_words words) are repeated at the
    start of the next part.
    """
    parts = []
    current, count, new_lines = [], 0, 0

    for line in lines:
        current.append(line)
        count += _words(line)
        new_lines += 1

        if count >= chunk_words:
            parts.append(current)
            # Keep the tail of this part as the start of the next one (never the whole part)
            tail, tail_count = [], 0
            for previous in reversed(current[1:]):
                if tail_count >= overlap_words:
                    break
                tail.insert(0, previous)
                tail_count += _words(previous)
            current, count, new_lines = tail, tail_count, 0

    if new_lines:  # don't emit a part that is only overlap
        parts.append(current)
    return parts


def _words(line):
    return len(line["text"].split())
