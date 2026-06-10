import re
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def before_sleep_log(retry_state):
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    print(f"[Gemini API Retry] Attempt {retry_state.attempt_number} failed. Exception: {exception}. Retrying...")

@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    before_sleep=before_sleep_log,
    reraise=True
)
def _call_gemini_api(model, contents, config):
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )


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


# ── Language constraints selector ─────────────────────────────────────────────

def get_language_instructions(language):
    if language == "kanglish":
        return """LANGUAGE & TRANSLITERATION CONSTRAINTS (CRITICAL):
- Explain all concepts using a conversational, everyday mix of Kannada and English (Kanglish) written ONLY using English letters (Latin script).
- Write in the style of a friendly, expert senior teaching a junior student in college. Use common, conversational Kannada words (e.g. 'madthare', 'barutthe', 'hege', 'yake andre', 'nodona', 'tumba', 'sikkapatte') naturally.
- Write Kannada words using ONLY English alphabets (Latin characters). For example, write "ee topic thumba important" instead of "ಈ ಟಾಪಿಕ್ ತುಂಬಾ ಇಂಪಾರ್ಟೆಂಟ್".
- DO NOT use any Kannada script (Kannada alphabets/characters like ಕನ್ನಡ).
- Keep all technical terms, code keywords, variable names, functions, operators, and core programming concepts strictly in standard English (e.g. "loop", "variables", "CPU", "memory", "recursion", "array").
- CRITICAL: You must preserve all Markdown formatting (like ## headings, **bold** labels, newlines, lists, and code blocks) exactly as specified in the OUTPUT FORMAT.
- Each bullet point must be on a new line starting with `- ` or `  - `. Do not collapse them into a single line or paragraph."""
    else:
        return """LANGUAGE CONSTRAINTS (CRITICAL):
- Explain all concepts strictly in standard, grammatically correct English.
- Write in the style of a clear, friendly, and expert computer science tutor.
- Keep all descriptions, headings, analogies, and notes strictly in standard English.
- CRITICAL: You must preserve all Markdown formatting (like ## headings, **bold** labels, newlines, lists, and code blocks) exactly as specified in the OUTPUT FORMAT.
- Each bullet point must be on a new line starting with `- ` or `  - `. Do not collapse them into a single line or paragraph."""


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_notes(chunks, subject="computer_science", mode="beginner", language="kanglish"):
    all_notes = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            import time
            time.sleep(2.0)  # Avoid hitting API Rate Limits (RPM)
        print(f"Processing chunk {i + 1} of {len(chunks)}...")
        notes = summarize_chunk(chunk, subject, mode, language)
        if notes:
            all_notes.append(notes)
    return "\n\n---\n\n".join(all_notes)


# ── Few-shot formatting examples ──────────────────────────────────────────────

def get_few_shot_example(mode, language):
    if language == "kanglish":
        if mode == "beginner":
            return """FEW-SHOT FORMATTING EXAMPLE (CRITICAL - match this style, spacing, and newlines exactly):

## What is a Queue?
Queue andre computer science nalli ondu linear data structure. Elements na order nalli place madoke idhanna use madthare. Idhu elements na queue type list nalli store madutthe.

**Key Concepts:**
- **FIFO Principle:** First In First Out structure. Line nalli yaaru first niltharo, avrige ticket first sigutthe.
- **Insert Operation (Enqueue):** Elements na standard queue end nalli insert madodu.
- **Delete Operation (Dequeue):** Queue starting position inda element remove madodu.

**In Simple Words:**
Ondu cinema hall ticket line nodi. Line nalli first bandavnge ticket first sigutthe (Dequeue), matte hosa janaru line end ge add agthare (Enqueue).

**Code Snippet / Definition (if present in text):**
```cpp
// Dequeue (Remove element from front)
void dequeue() {
    if (front == -1) {
        cout << "Queue Empty";
    } else {
        front++;
    }
}
```

---"""
        elif mode == "exam":
            return """FEW-SHOT FORMATTING EXAMPLE (CRITICAL - match this style, spacing, and newlines exactly):

## Queue Data Structure
Queue operations and basic board definitions.

**Key Points & Definitions:**
- **Definition:** Queue is a linear data structure working on FIFO (First In First Out) principle.
- **Enqueue:** Process of adding an element to the rear end of the queue. Complexity is O(1).
- **Dequeue:** Process of deleting an element from the front end of the queue. Complexity is O(1).

**Important Code / Algorithm / Definition (if present in text):**
```cpp
// Standard Enqueue Operation
void enqueue(int element) {
    if (rear == MAX - 1) {
        cout << "Queue Overflow";
    } else {
        if (front == -1) front = 0;
        rear++;
        queue[rear] = element;
    }
}
```

**Likely Exam Questions:**
- **2-Mark Question:** What is Queue? Mention the principle on which it works.
- **5-Mark Question:** Explain the Enqueue and Dequeue operations with neat algorithm steps.

---"""
        else: # deep
            return """FEW-SHOT FORMATTING EXAMPLE (CRITICAL - match this style, spacing, and newlines exactly):

## What is a Queue?
Queue andre computer science nalli ondu linear data structure. Elements na ordered list nalli sequential processing ge organize madoke idhu use agutthe. FIFO design patterns implement madakke idhanna use madthare.

**Detailed Analysis:**
- **FIFO Principle:** First entry in, first exit. First in-flight request first process agutthe.
- **Basic Operations:**
  - **Enqueue:** Back side end segment inda elements na queue ge add madoke.
  - **Dequeue:** Front side segment inda elements na queue inda remove madoke.
- **Memory Management:** Memory allocation array storage (static) or pointer linking (dynamic) irabahudu.

**Why This Works & Under the Hood:**
Queue sequential processing and flow synchronization manage madutthe. CPU task scheduling, print buffering, and asynchronous packet transmission nalli idhu core structure aagi behave madutthe.

**Real World Application / Use Case (if present in text):**
Operating system processes priority scheduling nalli buffers use madi messages store madoke queue use madthare.

---"""
    else: # English
        if mode == "beginner":
            return """FEW-SHOT FORMATTING EXAMPLE (CRITICAL - match this style, spacing, and newlines exactly):

## What is a Queue?
A Queue is a linear data structure in computer science. It is used to place elements in a specific order, storing elements in a list-like sequential structure.

**Key Concepts:**
- **FIFO Principle:** First In First Out structure. The first person in line is the first to be served.
- **Insert Operation (Enqueue):** Adding new items to the tail or rear end of the queue.
- **Delete Operation (Dequeue):** Removing items from the head or front of the queue.

**In Simple Words:**
Think of a movie ticket counter line. The person who stands first gets the ticket first (Dequeue), and new people join the line at the very end (Enqueue).

**Code Snippet / Definition (if present in text):**
```cpp
// Dequeue (Remove element from front)
void dequeue() {
    if (front == -1) {
        cout << "Queue Empty";
    } else {
        front++;
    }
}
```

---"""
        elif mode == "exam":
            return """FEW-SHOT FORMATTING EXAMPLE (CRITICAL - match this style, spacing, and newlines exactly):

## Queue Data Structure
Standard definitions and operations for exams.

**Key Points & Definitions:**
- **Definition:** Queue is a linear data structure working on FIFO (First In First Out) principle.
- **Enqueue:** Operation of inserting an element to the rear end of the queue. Complexity is O(1).
- **Dequeue:** Operation of removing an element from the front end of the queue. Complexity is O(1).

**Important Code / Algorithm / Definition (if present in text):**
```cpp
// Standard Enqueue Operation
void enqueue(int element) {
    if (rear == MAX - 1) {
        cout << "Queue Overflow";
    } else {
        if (front == -1) front = 0;
        rear++;
        queue[rear] = element;
    }
}
```

**Likely Exam Questions:**
- **2-Mark Question:** What is Queue? Mention the principle on which it works.
- **5-Mark Question:** Explain the Enqueue and Dequeue operations with neat algorithm steps.

---"""
        else: # deep
            return """FEW-SHOT FORMATTING EXAMPLE (CRITICAL - match this style, spacing, and newlines exactly):

## What is a Queue?
A Queue is a linear data structure in computer science that stores elements in an ordered list. It operates on the FIFO (First In First Out) principle, meaning that the element inserted first is the one that gets removed first.

**Detailed Analysis:**
- **FIFO Principle:** The first element added is the first to be processed, similar to a real-world ticket line.
- **Basic Operations:**
  - **Enqueue:** Adding an element to the rear of the queue.
  - **Dequeue:** Removing an element from the front of the queue.
- **Memory Management:** Queue memory allocation can be either static (using arrays) or dynamic (using pointers).

**Why This Works & Under the Hood:**
Queues follow sequential access logic. This structure is essential for CPU task scheduling, printer spooling, and network buffering where order of arrival must be preserved.

**Real World Application / Use Case (if present in text):**
Operating systems use queues to store incoming messages or manage priority processing in task buffers.

---"""


# ── Prompt selector ───────────────────────────────────────────────────────────

def summarize_chunk(text, subject, mode, language="kanglish"):
    subject_context = get_subject_context(subject)
    mode_instructions = get_mode_instructions(mode)
    format_instructions = get_format_instructions(mode)
    language_instructions = get_language_instructions(language)
    few_shot_example = get_few_shot_example(mode, language)

    prompt = f"""You are an expert study notes generator for PU (Pre-University) students in India studying Computer Science.
You are helping a student who does NOT know how to use AI — your notes must be clear, complete, self-explanatory, and preserve code block layout.

{language_instructions}

SUBJECT: {subject_context}
MODE: {mode_instructions}

{few_shot_example}

STRICT RULES — never break these:
1. ONLY use information from the TEXT below. Do not add anything from outside knowledge.
2. Skip anything unclear or missing. Do not mention that you skipped it.
3. Start directly with the first ## heading. No introduction, no conclusion, no meta commentary.
4. Technical terms and programming statements must stay exactly as written in English.
5. Do NOT hallucinate variables, syntax, or facts not present in the text.
6. When code, algorithms, or syntax examples are present in the text, write them inside code blocks with the appropriate markdown language identifier (e.g. ```python, ```java, ```cpp, ```html, or ```).
7. Never output any Kannada script/characters. Use only English letters.
8. Follow the markdown format EXACTLY. Write headings with `##`, use double asterisks for labels, and ensure every list item starts on a new line. Do not merge bullet points.
9. Identify and silently discard any page numbers, header/footer text, or corrupted OCR lines from the source TEXT. Do not let them interrupt the flow of the notes.
10. If a topic is cut off or incomplete in the text, summarize the available part clearly and self-containedly without inventing facts.
11. If comments are added inside code blocks, write them in simple English (or simple Kanglish/English mix if Kanglish is selected).
12. Merge duplicate points into a single concise note under the relevant concept. Do not repeat the same information.

{format_instructions}

TEXT:
{text}
"""

    try:
        response = _call_gemini_api(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192,
            ),
        )
        raw = response.text
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
        return """OUTPUT FORMAT — For EVERY major concept or topic found in the text, generate a separate notes section using this format:

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
        return """OUTPUT FORMAT — For EVERY major concept or exam topic found in the text, generate a separate notes section using this format:

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
        return """OUTPUT FORMAT — For EVERY major concept or architecture component found in the text, generate a separate notes section using this format:

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


# ── Interactive Quiz Generation ──────────────────────────────────────────────

import json

def generate_quiz(chunks):
    """
    Generate exactly 5 multiple choice questions spread evenly across the text chunks.
    """
    if not chunks:
        return []

    num_questions = 5
    num_chunks = len(chunks)
    questions = []

    if num_chunks >= num_questions:
        # Select 5 chunks spread evenly
        indices = [int(i * (num_chunks - 1) / (num_questions - 1)) for i in range(num_questions)]
        indices = sorted(list(set(indices)))
        
        # Pad indices to size 5 if duplicate math reduced the count
        for i in range(num_chunks):
            if len(indices) >= num_questions:
                break
            if i not in indices:
                indices.append(i)
        indices.sort()

        for step_idx, idx in enumerate(indices):
            if step_idx > 0:
                import time
                time.sleep(6.0)  # Avoid hitting API Rate Limits (RPM)
            qs = _generate_questions_from_chunk(chunks[idx], count=1)
            if qs:
                questions.extend(qs)
    else:
        # Distribute questions among fewer chunks
        questions_per_chunk = [0] * num_chunks
        for i in range(num_questions):
            questions_per_chunk[i % num_chunks] += 1

        called_count = 0
        for idx, count in enumerate(questions_per_chunk):
            if count > 0:
                if called_count > 0:
                    import time
                    time.sleep(6.0)  # Avoid hitting API Rate Limits (RPM)
                called_count += 1
                qs = _generate_questions_from_chunk(chunks[idx], count=count)
                if qs:
                    questions.extend(qs)

    # Return exactly 5 questions
    return questions[:num_questions]


def _generate_questions_from_chunk(chunk, count=1):
    """
    Calls Gemini 2.5 Flash in JSON mode to generate MCQs from a chunk.
    """
    prompt = f"""You are an expert Computer Science exam question generator.
Based ONLY on the text below, generate exactly {count} multiple-choice question(s) that could appear in a computer science exam.

TEXT:
{chunk}

You must return your response in JSON format matching this schema:
{{
  "questions": [
    {{
      "question": "The question text here",
      "options": [
        "Option A text",
        "Option B text",
        "Option C text",
        "Option D text"
      ],
      "answer_idx": 0, // 0-indexed integer (0 for A, 1 for B, 2 for C, 3 for D) indicating the correct answer
      "explanation": "A detailed explanation of why the correct option is right based on the text."
    }}
  ]
}}
"""
    try:
        response = _call_gemini_api(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )
        data = json.loads(response.text)
        return data.get("questions", [])
    except Exception as e:
        print(f"Error generating quiz question: {e}")
        return []
