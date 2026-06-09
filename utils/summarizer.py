import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── Filler pattern cleaner ────────────────────────────────────────────────────

FILLER_PATTERNS = [
    r"^here are .*?notes.*?:\s*",
    r"^here are .*?study notes.*?:\s*",
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
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^[-=]{3,}\n', '', text)
    return text.strip()


# ── Subject auto detection ────────────────────────────────────────────────────

def detect_subject(chunks):
    """
    Detect the subject of the PDF from the first chunk.
    Returns 'physics', 'biology', or 'unknown'.
    Uses a tiny API call — only first 400 words, 10 output tokens max.
    """
    if not chunks:
        return "unknown"

    # Use only first chunk, max 400 words to keep tokens minimal
    sample = " ".join(chunks[0].split()[:400])

    prompt = f"""Read this text and reply with ONLY one word — either 'physics' or 'biology'.
Choose 'physics' if the text contains laws of motion, forces, energy, electricity, optics, waves, thermodynamics, or similar physics topics.
Choose 'biology' if the text contains cells, organisms, genetics, evolution, photosynthesis, human body systems, or similar biology topics.
If you are not sure, reply 'unknown'.
Reply with ONE word only. No explanation.

TEXT:
{sample}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        result = response.choices[0].message.content.strip().lower()
        if result in ["physics", "biology"]:
            return result
        return "unknown"

    except Exception as e:
        print(f"Subject detection failed: {e}")
        return "unknown"


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_notes(chunks, subject="physics", mode="beginner"):
    all_notes = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1} of {len(chunks)}...")
        notes = summarize_chunk(chunk, subject, mode)
        if notes:
            all_notes.append(notes)
    return "\n\n---\n\n".join(all_notes)


# ── Prompt selector ───────────────────────────────────────────────────────────

def summarize_chunk(text, subject, mode):
    subject_context = get_subject_context(subject)
    mode_instructions = get_mode_instructions(mode)
    format_instructions = get_format_instructions(mode)

    prompt = f"""You are an expert study notes generator for PU (Pre-University) students in India.
You are helping a student who does NOT know how to use AI — your notes must be clear, complete, and self-explanatory.

SUBJECT: {subject_context}
MODE: {mode_instructions}

STRICT RULES — never break these:
1. ONLY use information from the TEXT below. Do not add anything from outside knowledge.
2. Skip anything unclear or missing. Do not mention that you skipped it.
3. Start directly with the first ## heading. No introduction, no conclusion, no meta commentary.
4. Technical terms must always stay in English exactly as written.
5. Do NOT hallucinate formulas, values, or facts not present in the text.

{format_instructions}

TEXT:
{text}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content
        return clean_model_output(raw)

    except Exception as e:
        print(f"Error summarizing chunk: {e}")
        return None


# ── Subject contexts ──────────────────────────────────────────────────────────

def get_subject_context(subject):
    contexts = {
        "physics": (
            "Physics — PU level. "
            "When present in the text, always include: laws with their exact statements, "
            "formulas with all variables explained, units of measurement, "
            "and numerical examples if available."
        ),
        "biology": (
            "Biology — PU level. "
            "When present in the text, always include: exact definitions of terms, "
            "step-by-step processes, names of scientists or discoverers, "
            "and real-life examples or applications if available."
        ),
    }
    return contexts.get(subject, "General Science — PU level.")


# ── Mode instructions ─────────────────────────────────────────────────────────

def get_mode_instructions(mode):
    modes = {
        "beginner": (
            "BEGINNER MODE — The student is reading this topic for the first time. "
            "Use the simplest possible language. Explain every term. "
            "Use real-life comparisons where the text supports it. "
            "Do not assume any prior knowledge."
        ),
        "exam": (
            "EXAM MODE — The student is preparing for their PU board exam. "
            "Focus strictly on what is most likely to be asked in an exam. "
            "Be concise and precise. Include definitions, laws, formulas, and diagrams descriptions. "
            "Add a short 'Likely Exam Questions' section at the end of each topic."
        ),
        "deep": (
            "DEEP UNDERSTANDING MODE — The student wants to fully understand the topic, not just memorize. "
            "Explain the 'why' and 'how' behind every concept. "
            "Connect ideas together where the text supports it. "
            "Include all details, examples, and reasoning present in the text."
        ),
    }
    return modes.get(mode, modes["beginner"])


# ── Format instructions per mode ─────────────────────────────────────────────

def get_format_instructions(mode):
    if mode == "beginner":
        return """OUTPUT FORMAT — follow exactly:

## Topic Name
[2-3 simple sentences explaining what this topic is and why it matters. Use simple everyday language.]

**Key Points:**
- First key point as a short clear sentence
- Second key point as a short clear sentence
  - Sub point if needed, indented with two spaces
- Third key point as a short clear sentence

**In Simple Words:**
[One analogy or real-life comparison that makes this concept easy to remember. Only include if the text supports it.]

**Formula / Definition (if present in text):**
```
Write the exact formula or definition here
Explain each variable or term on a new line
```

---"""

    elif mode == "exam":
        return """OUTPUT FORMAT — follow exactly:

## Topic Name
[One line: what this topic is about.]

**Key Points:**
- First key point as a short clear sentence
- Second key point as a short clear sentence
  - Sub point if needed
- Third key point as a short clear sentence

**Important Formula / Law / Definition (if present in text):**
```
Exact formula, law statement, or definition
Variable meanings if applicable
```

**Likely Exam Questions:**
- [One probable 2-mark question from this topic]
- [One probable 5-mark question from this topic]

---"""

    elif mode == "deep":
        return """OUTPUT FORMAT — follow exactly:

## Topic Name
[2-3 sentences: what this topic is, why it exists, and what problem it solves.]

**Detailed Explanation:**
- First concept explained in full
- Second concept explained in full
  - Sub detail if needed
- Third concept explained in full

**Formula / Process / Definition (if present in text):**
```
Exact formula or step-by-step process
Full explanation of each component
```

**Why This Works:**
[1-2 sentences explaining the reasoning or principle behind this concept. Only if supported by the text.]

**Real World Application (if present in text):**
[One real world use case or example from the text.]

---"""

    return ""