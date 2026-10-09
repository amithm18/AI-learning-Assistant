from utils.llm import validate_question


def raw(**overrides):
    q = {"question": "What does push do?", "options": ["Adds on top", "Removes", "Sorts", "Searches"],
         "answer_idx": 0, "explanation": "push adds an element."}
    q.update(overrides)
    return q


def test_valid_question_keeps_the_correct_answer_after_shuffling():
    for _ in range(20):
        q = validate_question(raw())
        assert q["options"][q["answer_idx"]] == "Adds on top"


def test_rejects_broken_questions():
    assert validate_question(raw(answer_idx=4)) is None
    assert validate_question(raw(answer_idx="x")) is None
    assert validate_question(raw(options=["a", "b", "c"])) is None
    assert validate_question(raw(options=["a", "a", "b", "c"])) is None
    assert validate_question(raw(question="")) is None
    assert validate_question({"options": []}) is None
