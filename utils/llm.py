"""
Step 4 of the pipeline: all calls to the LLM (Groq, Llama 3.3 70B).

The student never writes a prompt. Every button in the UI maps to one function
here, and each function builds the full prompt itself:

    Make notes        -> generate_notes()
    Quiz me           -> generate_quiz()
    I didn't get it   -> explain_simply()     (RAG)
    Suggested doubts  -> suggest_questions()
    Ask a doubt       -> answer_question()    (RAG)
"""
import json
import logging
import os
import random
import re

from groq import Groq

log = logging.getLogger(__name__)

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

AUDIENCE = (
    "You are an expert Computer Science teacher for PU (Pre-University) students in India. "
    "The student does NOT know how to use AI tools, so your answer must be clear, complete "
    "and self-explanatory."
)

_client = None


class MissingApiKey(RuntimeError):
    pass


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise MissingApiKey("GROQ_API_KEY is not set")
        _client = Groq(api_key=api_key)
    return _client


def _chat(prompt, temperature=0.1, max_tokens=2000, json_mode=False):
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return response.choices[0].message.content


# ── Output cleanup ────────────────────────────────────────────────────────────

FILLER_PATTERNS = [
    r"^here are .*?notes.*?:\s*",
    r"^here is .*?summary.*?:\s*",
    r"^based on the .*?text.*?:\s*",
    r"^the following .*?notes.*?:\s*",
    r"^note:.*?\n",
    r"^in summary.*?\n",
    r"^to summarize.*?\n",
]


def clean_model_output(text):
    text = text.strip()
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^[-=]{3,}\n", "", text)
    return text.strip()


def format_excerpts(chunks, sections):
    """Retrieved chunks -> numbered excerpts with page numbers, for the prompt."""
    titles = {s["id"]: s["title"] for s in sections}
    return "\n\n".join(
        f"[Excerpt {i} | page {c['page']} | topic: {titles.get(c['section_id'], '')}]\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )


# ── Notes ─────────────────────────────────────────────────────────────────────

def generate_notes(topic_title, text, mode):
    prompt = f"""{AUDIENCE}

SUBJECT: {SUBJECT_CONTEXT}
TOPIC: {topic_title}
MODE: {MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["beginner"])}

STRICT RULES — never break these:
1. ONLY use information from the TEXT below. Do not add anything from outside knowledge.
2. Skip anything unclear or missing. Do not mention that you skipped it.
3. Start directly with the first ## heading. No introduction, no conclusion, no meta commentary.
4. Technical terms and programming statements must stay exactly as written.
5. Do NOT hallucinate variables, syntax, or facts not present in the text.
6. When code, algorithms, or syntax examples are present in the text, write them inside code blocks with the appropriate markdown language identifier (e.g. ```python, ```java, ```cpp, ```html, or ```).

{FORMAT_INSTRUCTIONS.get(mode, FORMAT_INSTRUCTIONS["beginner"])}

TEXT:
{text}
"""
    return clean_model_output(_chat(prompt))


SUBJECT_CONTEXT = (
    "Computer Science — topics like programming, data structures, algorithms, "
    "computer networks, operating systems, database management, and computer architecture. "
    "When present in the text, always include: key algorithms with step-by-step logic, "
    "code syntax/snippets (properly formatted), data structure definitions and their operations, "
    "time and space complexities (Big O notation), diagram descriptions (like block diagrams or network topologies), "
    "and clear explanations of system components or design patterns."
)

MODE_INSTRUCTIONS = {
    "beginner": (
        "BEGINNER MODE — The student is learning this computer science topic for the first time. "
        "Use simple, conversational language. Explain all jargon, acronyms, and technical terms. "
        "Use everyday analogies to make abstract programming/hardware concepts intuitive. "
        "Write clear comments for any code blocks, explaining what each line does. "
        "Do not assume any prior coding or system knowledge."
    ),
    "exam": (
        "EXAM MODE — The student is preparing for their computer science board exam. "
        "Focus strictly on high-yield exam topics: precise definitions, syntax rules, exact code snippets, "
        "key differences (e.g. Stack vs Queue), algorithms step-by-step, and complexity values. "
        "Include a short 'Likely Exam Questions' section at the end of each topic (e.g., 2-mark definitions and 5-mark code/system descriptions)."
    ),
    "deep": (
        "DEEP UNDERSTANDING MODE — The student wants to master the core principles of the computer science topic. "
        "Explain the 'why' and 'how' behind concepts (e.g., how recursion affects the stack memory, memory leaks, performance trade-offs). "
        "Analyze time and space complexity in detail. "
        "Provide fully explained code examples with trace tables or step-by-step execution flow."
    ),
}

FORMAT_INSTRUCTIONS = {
    "beginner": """OUTPUT FORMAT — follow exactly:

## Topic Name
[2-3 simple sentences explaining what this topic is and why it matters in Computer Science. Use simple everyday analogies.]

**Key Concepts:**
- First key concept as a short clear sentence
- Second key concept as a short clear sentence
  - Detailed sub-point if needed
- Third key concept as a short clear sentence

**In Simple Words:**
[One analogy or real-world comparison that makes this CS concept easy to understand, e.g., memory like post boxes.]

**Code Snippet / Definition (if present in text):**
```[language]
Write the code snippet, pseudocode, or definition here.
Explain each key line or term on a new line.
```""",

    "exam": """OUTPUT FORMAT — follow exactly:

## Topic Name
[One line: what this topic is about.]

**Key Points & Definitions:**
- Precise definition of terms
- Key differences or features (e.g. key-value pairs, characteristics)
- Algorithm steps or syntax rules

**Important Code / Algorithm / Definition (if present in text):**
```[language]
Exact code, algorithm steps, or syntax template.
Explain crucial parts briefly.
```

**Likely Exam Questions:**
- [One probable 2-mark question from this topic]
- [One probable 5-mark question from this topic]""",

    "deep": """OUTPUT FORMAT — follow exactly:

## Topic Name
[2-3 sentences: what this topic is, why it is designed this way, and what problem it solves in system design or computation.]

**Detailed Analysis:**
- First concept explained in full detail
- Second concept explained in full detail
  - Complexity analysis or design trade-offs
- Third concept explained in full detail

**Code Snippet / Process / Architecture (if present in text):**
```[language]
Complete code snippet, execution trace, or architectural process steps.
```

**Why This Works & Under the Hood:**
[1-2 sentences explaining the underlying execution mechanism or hardware/memory behavior.]

**Real World Application / Use Case (if present in text):**
[One real world scenario where this concept is applied, e.g. web routing, database indexing.]""",
}


# ── Quiz ──────────────────────────────────────────────────────────────────────

def generate_quiz(texts, count=5):
    """
    Generates `count` MCQs spread across the given texts (topics).
    Every question is validated and its options are shuffled.
    """
    if not texts:
        return []

    # More texts than questions -> pick texts spread evenly through the PDF
    if len(texts) > count:
        step = (len(texts) - 1) / max(count - 1, 1)
        texts = [texts[round(i * step)] for i in range(count)]

    per_text = [0] * len(texts)
    for i in range(count):
        per_text[i % len(texts)] += 1

    questions, seen = [], set()
    for text, n in zip(texts, per_text):
        if n == 0:
            continue
        for raw in _questions_from_text(text, n):
            question = validate_question(raw)
            if question and question["question"].lower() not in seen:
                seen.add(question["question"].lower())
                questions.append(question)
    return questions[:count]


def _questions_from_text(text, count):
    prompt = f"""You are an expert Computer Science exam question generator for PU students.
Based ONLY on the text below, generate exactly {count} multiple-choice question(s) that could appear in a computer science exam.
Each question must have exactly 4 different options and exactly one correct option.

TEXT:
{text}

Return JSON in exactly this shape:
{{
  "questions": [
    {{
      "question": "The question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer_idx": 0,
      "explanation": "Why the correct option is right, based on the text."
    }}
  ]
}}
answer_idx is the 0-based position of the correct option in "options".
"""
    try:
        data = json.loads(_chat(prompt, temperature=0.3, json_mode=True))
        return data.get("questions", [])
    except json.JSONDecodeError:
        log.warning("Quiz response was not valid JSON")
        return []


def validate_question(raw):
    """Returns a clean question dict, or None if the model gave a broken one."""
    try:
        question = str(raw["question"]).strip()
        options = [str(o).strip() for o in raw["options"]]
        answer_idx = int(raw["answer_idx"])
        explanation = str(raw.get("explanation", "")).strip()
    except (KeyError, TypeError, ValueError):
        return None

    if not question or len(options) != 4 or len(set(options)) != 4 or not 0 <= answer_idx < 4:
        return None

    # Models like to put the right answer first, so shuffle and track where it went
    correct = options[answer_idx]
    random.shuffle(options)
    return {
        "question": question,
        "options": options,
        "answer_idx": options.index(correct),
        "explanation": explanation,
    }


# ── RAG: explain simply / answer a doubt ─────────────────────────────────────

def explain_simply(topic_title, excerpts):
    prompt = f"""{AUDIENCE}

The student read notes on "{topic_title}" and clicked "I didn't understand this".
Explain the topic again, much more simply, as if to a 15-year-old seeing it for the first time.

RULES:
1. Use ONLY the facts in the EXCERPTS below (taken from the student's own textbook).
2. Start with one everyday analogy, then explain in 3-5 short points, then one tiny example.
3. If there is code, show a very small piece of it in a code block with a comment on each line.
4. Mention the page number in brackets after important facts, like (p. 12).
5. Maximum 250 words. No introduction like "Sure!" — start directly.

EXCERPTS:
{excerpts}
"""
    return clean_model_output(_chat(prompt, temperature=0.3, max_tokens=900))


def answer_question(question, excerpts):
    prompt = f"""{AUDIENCE}

A student asked a doubt about their textbook. The question may be short or informal
(for example just "stack?"): understand it the way a student would mean it.

RULES:
1. Answer using ONLY the EXCERPTS below (from the student's own textbook).
2. If the excerpts do not contain the answer, reply exactly: "I couldn't find this in your PDF."
3. Put the page number in brackets after the facts you use, like (p. 12).
4. Keep it short and simple: at most 200 words. Use a code block if code helps.
5. Start directly with the answer.

STUDENT'S QUESTION: {question}

EXCERPTS:
{excerpts}
"""
    return clean_model_output(_chat(prompt, temperature=0.2, max_tokens=700))


def suggest_questions(topic_title, text, count=3):
    """Ready-made doubts the student can click instead of typing."""
    prompt = f"""A PU Computer Science student is studying the topic "{topic_title}".
Write {count} short questions (max 12 words each) that a student would typically be confused about,
and that CAN be answered from the TEXT below.

TEXT:
{text}

Return JSON: {{"questions": ["...", "...", "..."]}}
"""
    try:
        data = json.loads(_chat(prompt, temperature=0.4, max_tokens=300, json_mode=True))
        questions = [str(q).strip() for q in data.get("questions", []) if str(q).strip()]
        return questions[:count]
    except json.JSONDecodeError:
        return []
