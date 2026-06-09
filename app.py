from flask import Flask, render_template, request, jsonify
from utils.pdf_reader import extract_text_from_pdf
from utils.summarizer import generate_notes
from dotenv import load_dotenv
import markdown
import os

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "pdf_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["pdf_file"]
    mode = request.form.get("mode", "beginner").lower()

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    if mode not in ["beginner", "exam", "deep"]:
        return jsonify({"error": "Invalid mode selected"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

    try:
        file.save(file_path)

        chunks = extract_text_from_pdf(file_path)
        if not chunks:
            return jsonify({"error": "Could not extract text from this PDF. It may be scanned or image-based."}), 400

        raw_notes = generate_notes(chunks, subject="computer_science", mode=mode)
        if not raw_notes:
            return jsonify({"error": "Failed to generate notes. Please try again."}), 500

        notes_html = markdown.markdown(raw_notes, extensions=["extra", "fenced_code"])

        mode_labels = {
            "beginner": "Beginner",
            "exam": "Exam Ready",
            "deep": "Deep Dive"
        }

        return jsonify({
            "notes": notes_html,
            "chunk_count": len(chunks),
            "subject": "Computer Science",
            "mode": mode_labels.get(mode, mode)
        }), 200

    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large. Maximum allowed size is 16MB."}), 413


if __name__ == "__main__":
    app.run(debug=True)