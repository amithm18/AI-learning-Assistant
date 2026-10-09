from utils.chunker import make_chunks, split_lines


def lines(n_lines, words_per_line=10, page=1):
    return [{"text": " ".join(f"w{i}_{j}" for j in range(words_per_line)), "page": page} for i in range(n_lines)]


def test_split_respects_size_and_keeps_every_line():
    parts = split_lines(lines(50), chunk_words=100, overlap_words=0)
    assert all(sum(len(l["text"].split()) for l in p) >= 100 for p in parts[:-1])
    assert [l for p in parts for l in p] == lines(50)  # no overlap -> nothing lost, nothing repeated


def test_overlap_repeats_the_tail_of_the_previous_part():
    parts = split_lines(lines(50), chunk_words=100, overlap_words=20)
    for previous, current in zip(parts, parts[1:]):
        assert previous[-2:] == current[:2]


def test_no_part_is_only_overlap():
    parts = split_lines(lines(10), chunk_words=100, overlap_words=20)
    assert len(parts) == 1


def test_chunks_keep_section_and_page():
    section = {"id": 3, "lines": lines(5, page=7) + lines(30, page=8)}
    chunks = make_chunks([section])
    assert chunks[0]["page"] == 7
    assert chunks[-1]["page"] == 8
    assert {c["section_id"] for c in chunks} == {3}
    assert [c["id"] for c in chunks] == list(range(len(chunks)))
