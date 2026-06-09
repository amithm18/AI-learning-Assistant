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


# ── Subject auto detection stub ────────────────────────────────────────────────

def detect_subject(chunks):
    """
    Stub detection function, since subject is now fixed to Computer Science.
    """
    return "computer_science"


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_notes(chunks, subject="computer_science", mode="beginner"):
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

    prompt = f"""You are an expert study notes generator for PU (Pre-University) students in India studying Computer Science.
You are helping a student who does NOT know how to use AI — your notes must be clear, complete, self-explanatory, and preserve code block layout.

SUBJECT: {subject_context}
MODE: {mode_instructions}

STRICT RULES — never break these:
1. ONLY use information from the TEXT below. Do not add anything from outside knowledge.
2. Skip anything unclear or missing. Do not mention that you skipped it.
3. Start directly with the first ## heading. No introduction, no conclusion, no meta commentary.
4. Technical terms and programming statements must stay exactly as written.
5. Do NOT hallucinate variables, syntax, or facts not present in the text.
6. When code, algorithms, or syntax examples are present in the text, write them inside code blocks with the appropriate markdown language identifier (e.g. ```python, ```java, ```cpp, ```html, or ```).

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
    return (
        "Computer Science — topics like programming, data structures, algorithms, "
        "computer networks, operating systems, database management, and computer architecture. "
        "When present in the text, always include: key algorithms with step-by-step logic, "
        "code syntax/snippets (properly formatted), data structure definitions and their operations, "
        "time and space complexities (Big O notation), diagram descriptions (like block diagrams or network topologies), "
        "and clear explanations of system components or design patterns."
    )


# ── Mode instructions ─────────────────────────────────────────────────────────

def get_mode_instructions(mode):
    modes = {
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
    return modes.get(mode, modes["beginner"])


# ── Format instructions per mode ─────────────────────────────────────────────

def get_format_instructions(mode):
    if mode == "beginner":
        return """OUTPUT FORMAT — follow exactly:

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
```

---"""

    elif mode == "exam":
        return """OUTPUT FORMAT — follow exactly:

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
- [One probable 5-mark question from this topic]

---"""

    elif mode == "deep":
        return """OUTPUT FORMAT — follow exactly:

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
[One real world scenario where this concept is applied, e.g. web routing, database indexing.]

---"""

    return ""