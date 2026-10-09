# 🎓 AI Computer Science Learning Assistant

An AI-powered interactive study web application designed to help Computer Science students transform dense PDF textbooks, lecture notes, and syllabus materials into structured study notes, interactive quizzes, and Text-to-Speech (TTS) audio lessons.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-f34f29?style=for-the-badge)

---

## 🌟 Overview

Studying Computer Science requires understanding complex algorithms, code syntax, architectural concepts, and exam questions. The **AI Computer Science Learning Assistant** simplifies this process by parsing uploaded PDF chapters, analyzing line heights and section headers, and generating context-preserving study notes customized to your learning goals.

Whether you are learning a programming concept for the first time, preparing for upcoming board exams, or conducting a deep-dive analysis into system architecture, this tool generates instant notes, interactive quizzes, and playable audio.

---

## ✨ Features

- 📄 **Layout-Aware PDF Extraction**: Employs `pdfplumber` to extract text while maintaining font size variation, line breaks, code blocks, and structural heading boundaries.
- 🎯 **3 Tailored AI Learning Modes**:
  - 🚀 **Beginner Mode**: Simple, jargon-free explanations, real-world analogies (e.g. memory boxes, post offices), and commented code breakdowns.
  - 🎯 **Exam Ready Mode**: Precise definitions, high-yield syntax templates, and probable 2-mark & 5-mark board exam questions.
  - 🔬 **Deep Dive Mode**: Architectural insights, time and space complexity ($O(n)$ notation) analysis, step-by-step trace tables, and real-world system applications.
- ❓ **Interactive Quiz Generator**: Automatically creates 5 multiple-choice questions (MCQs) evenly distributed across PDF content, featuring real-time option checking and detailed explanation boxes.
- 🔊 **Text-to-Speech (TTS) Audio Lessons**: Converts generated notes into smooth MP3 audio using Google Text-to-Speech (`gTTS`), enabling on-the-go audio learning.
- 🎨 **Glassmorphism UI**: Built with a dark mode aesthetic, smooth CSS glowing animations, VS Code-style syntax-highlighted code blocks, and full responsiveness.

---

## 🛠️ Tech Stack

| Component | Technology / Library |
| :--- | :--- |
| **Backend Framework** | [Flask](https://flask.palletsprojects.com/) (Python 3.8+) |
| **AI Model & Inference** | [Groq SDK](https://groq.com/) (`llama-3.3-70b-versatile`) |
| **PDF Extraction Engine** | [pdfplumber](https://github.com/jsvine/pdfplumber), `pdfminer.six` |
| **Audio Synthesis** | [gTTS](https://github.com/pndurette/gTTS) (Google Text-to-Speech) |
| **Markdown Rendering** | `Python-Markdown` with extended code formatting |
| **Frontend UI** | HTML5, Modern CSS3 (Glassmorphism, CSS Tokens), Vanilla JS (Fetch API) |

---

## 📁 Project Structure

```
AI-learning-assistant/
│
├── app.py                      # Core Flask server handling web routes & API endpoints
├── utils/
│   ├── pdf_reader.py           # Intelligent layout analysis, heading detection & PDF chunking
│   └── summarizer.py           # Groq LLM prompt pipelines, note generators & quiz builders
│
├── templates/
│   └── index.html              # Responsive single-page interface with tab navigation & audio player
│
├── uploads/                    # Temporary storage for uploaded PDF files during processing
├── .env                        # Private environment variables (API Keys - ignored by git)
├── .env.example                # Configuration template for environment variables
├── .gitignore                  # Files and directories ignored by Git
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚡ Quickstart & Installation Guide

Follow these steps to get the project running on your local machine:

### 1. Prerequisites
- **Python 3.8** or higher installed on your system.
- A free **Groq API Key** (Get one at [console.groq.com](https://console.groq.com/)).

### 2. Clone the Repository
```bash
git clone https://github.com/amithm18/AI-learning-assistant.git
cd AI-learning-assistant
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
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📖 How to Use

1. **Select Learning Mode**: Choose between **Beginner**, **Exam Ready**, or **Deep Dive** depending on your study goals.
2. **Upload PDF**: Drag & drop or browse to select your Computer Science PDF document (max size: 16MB).
3. **Generate Notes / Quiz**:
   - Click **Generate Study Notes** to extract structured markdown notes with code snippets.
   - Click **Generate Interactive Quiz** to solve 5 auto-generated multiple-choice questions.
4. **Listen to Notes**: Use the built-in custom audio player to stream Text-to-Speech audio explanations.

---

## 🔌 API Endpoints Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Renders the main Web UI dashboard |
| `/upload` | `POST` | Accepts a PDF file & mode parameter, extracts text, and returns formatted HTML study notes |
| `/quiz` | `POST` | Accepts a PDF file and returns 5 structured JSON multiple-choice questions with explanations |
| `/tts` | `POST` | Accepts JSON text and returns an MP3 audio binary stream generated via Google TTS |

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

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  Made with ❤️ for Computer Science Students
</p>
