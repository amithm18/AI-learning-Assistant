# 🎓 AI Computer Science Learning Assistant

An AI-powered interactive study web application designed to help Computer Science students transform dense PDF textbooks, lecture notes, and syllabus materials into structured study notes, interactive quizzes, and Text-to-Speech (TTS) audio lessons.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-f34f29?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-fastembed%20%2B%20numpy-8B5CF6?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 🌟 Overview

Many students have never used an AI tool and don't know how to write a prompt. This app removes the prompt entirely:
upload a chapter PDF, tap a topic, and tap a button. Everything the student gets (notes, quizzes, simpler explanations,
answers to doubts) comes **only from their own PDF, with page numbers** to check it against the book.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how it works and why it is designed this way.

---

## ✨ Features

- 📚 **Automatic topic list**: the PDF's headings are detected from font size and boldness, so the chapter is split into topics the student can tap, like a table of contents.
- 🎯 **3 learning modes for notes**:
  - 🌱 **Beginner**: simple language, everyday analogies, commented code.
  - 📝 **Exam Ready**: precise definitions, syntax, and likely 2-mark and 5-mark questions.
  - 🔬 **Deep Dive**: complexity analysis, trace tables, how things work underneath.
- ⚡ **Quiz per topic or for the whole PDF**: 5 validated MCQs with shuffled options. After answering, the student sees the explanation and **the passage and page in the book** it came from.
- 🤔 **"I didn't understand this"**: one tap re-explains the topic more simply, using the most relevant passages found anywhere in the book (RAG).
- 💬 **Ask a doubt without prompting**: tap one of the suggested questions, or type a doubt in any words. The answer is built only from matching passages of the PDF and cites page numbers (RAG).
- 🔊 **Listen to notes**: Text-to-Speech audio (gTTS), skipping code blocks.
- ⚡ **Fast and cheap**: the PDF is processed once; notes are cached per topic and mode.

---

## 🛠️ Tech Stack

| Component | Technology / Library |
| :--- | :--- |
| **Backend** | [Flask](https://flask.palletsprojects.com/) (Python 3.10+) |
| **LLM** | [Groq](https://groq.com/) (`llama-3.3-70b-versatile`) |
| **PDF parsing** | [pdfplumber](https://github.com/jsvine/pdfplumber) |
| **Embeddings (RAG)** | [fastembed](https://github.com/qdrant/fastembed) with `BAAI/bge-small-en-v1.5`, running locally |
| **Vector search** | numpy (cosine similarity) |
| **HTML sanitising** | [nh3](https://github.com/messense/nh3) |
| **Audio** | [gTTS](https://github.com/pndurette/gTTS) |
| **Frontend** | HTML, CSS, vanilla JavaScript |

---

## 📁 Project Structure

```
AI-learning-assistant/
├── app.py                    # Flask API: upload once, then notes / quiz / explain / ask / tts
├── utils/
│   ├── pdf_reader.py         # PDF -> topics (heading detection, page numbers, header/footer removal)
│   ├── chunker.py            # topics -> ~200-word overlapping chunks for search
│   ├── retriever.py          # embeddings + cosine-similarity search (the "R" in RAG)
│   ├── document_store.py     # saves topics, chunks, vectors and cached AI output in data/<doc_id>/
│   └── llm.py                # every prompt sent to Groq
├── templates/index.html      # single page: upload -> topic list -> topic page
├── tests/                    # pytest unit tests (chunking, quiz validation, search)
├── docs/ARCHITECTURE.md      # design explanation
├── data/                     # created at runtime, ignored by git
├── .env.example
├── requirements.txt
└── requirements-dev.txt
```

---

## ⚡ Quickstart & Installation Guide

Follow these steps to get the project running on your local machine:

### 1. Prerequisites
- **Python 3.10** or higher installed on your system.
- A free **Groq API Key** (Get one at [console.groq.com](https://console.groq.com/)).

### 2. Clone the Repository
```bash
git clone https://github.com/amithm18/AI-learning-Assistant.git
cd AI-learning-Assistant
```

### 3. Create & Activate a Virtual Environment

- **On Windows:**
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```
- **On macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables
Create a `.env` file in the root directory by copying `.env.example`:
```bash
cp .env.example .env
```
Open `.env` and insert your **Groq API Key**:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 6. Run the Application
```bash
python app.py
```
The first upload downloads the small embedding model (~130MB) once, so it takes longer than later uploads.
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📖 How to Use

1. **Upload** your Computer Science chapter PDF (max 16MB). The app finds its topics.
2. **Tap a topic** from the list.
3. Pick **Beginner**, **Exam Ready** or **Deep Dive**, then tap **Make notes** or **Quiz me**.
4. Didn't get it? Tap **I didn't understand this** for a simpler explanation.
5. Have a doubt? Tap a suggested question or type your own. Every answer shows the pages it came from.
6. Tap ▶ on the notes to listen to them.

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | The web app |
| `/api/documents` | `POST` | Upload a PDF (`pdf_file`). Returns `doc_id` and the topic list |
| `/api/documents/<doc_id>` | `GET` | Topic list for an uploaded PDF |
| `/api/documents/<doc_id>/notes` | `POST` | `{topic_id, mode}` → notes HTML (cached) |
| `/api/documents/<doc_id>/quiz` | `POST` | `{topic_id}` or `{topic_id: null}` for the whole PDF → 5 MCQs with source page |
| `/api/documents/<doc_id>/explain` | `POST` | `{topic_id}` → simpler explanation + sources (RAG) |
| `/api/documents/<doc_id>/suggestions` | `POST` | `{topic_id}` → 3 clickable questions (cached) |
| `/api/documents/<doc_id>/ask` | `POST` | `{question}` → answer + sources (RAG) |
| `/api/tts` | `POST` | `{text}` → MP3 audio |

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the PDF extraction algorithms, add new subject prompts, or enhance the UI:

1. Fork the Project repository.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📜 License

Distributed under the MIT License.

---

<p align="center">
  Made with ❤️ for Computer Science Students
</p>
