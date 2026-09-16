from flask import Flask, request, jsonify
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)


model = SentenceTransformer("all-MiniLM-L6-v2")


# Root route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Resume Matcher API is running",
        "endpoint": "/match-resume",
        "method": "POST"
    }), 200



def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# Extract text from DOCX
def extract_docx_text(file):
    document = Document(file)
    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


@app.route("/match-resume", methods=["POST"])
def match_resume():

    try:

        # Check resume file
        if "resume" not in request.files:
            return jsonify({
                "error": "Resume file is required"
            }), 400

        file = request.files["resume"]

        # Check empty file
        if file.filename == "":
            return jsonify({
                "error": "No file selected"
            }), 400

        # Get requirements
        requirements = request.form.get("   ")

        if not requirements:
            return jsonify({
                "error": "Job requirements are required"
            }), 400

        # Check file format
        filename = file.filename.lower()

        if filename.endswith(".pdf"):

            resume_text = extract_pdf_text(file)

        elif filename.endswith(".docx"):

            resume_text = extract_docx_text(file)

        else:

            return jsonify({
                "error": "Wrong file format. Only PDF and DOCX are allowed."
            }), 400

      
        if not resume_text.strip():

            return jsonify({
                "error": "Resume does not contain readable text"
            }), 400

        
        resume_embedding = model.encode(
            resume_text,
            convert_to_tensor=True
        )

        
        requirement_embedding = model.encode(
            requirements,
            convert_to_tensor=True
        )

        # Calculate similarity
        similarity = util.cos_sim(
            resume_embedding,
            requirement_embedding
        ).item()

    
        match_percentage = round(
            max(0, min(similarity * 100, 100)),
            2
        )
        # checking code for githutb

        if match_percentage >= 80:

            status = "Excellent Match"

        elif match_percentage >= 60:

            status = "Good Match"

        elif match_percentage >= 40:

            status = "Partial Match"

        else:

            status = "Low Match"


        return jsonify({
            "filename": file.filename,
            "match_percentage": match_percentage,
            "status": status
        }), 200

    except Exception as e:

        return jsonify({
            "error": "Something went wrong while processing the resume",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)