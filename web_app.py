import os
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Import functions from the existing image_data_extraction.py file.
# The preprocessed version yields better results.
from image_data_extraction import extract_text_preprocessed, extract_receipt_data

# Initialize the Flask application
app = Flask(__name__, template_folder='.')

# Folder where uploaded files will be temporarily saved
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Serves the main page (index.html)."""
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def ocr_image():
    """Receives an image file, applies OCR, and returns the text as JSON."""
    if 'file' not in request.files:
        return jsonify({"error": "No file found in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Perform the OCR process
            extracted_text = extract_text_preprocessed(filepath)
            return jsonify({"text": extracted_text})
        except Exception as e:
            return jsonify({"error": f"An error occurred during OCR processing: {str(e)}"}), 500
        finally:
            # Delete the temporary file after processing
            if os.path.exists(filepath):
                os.remove(filepath)

@app.route('/ai', methods=['POST'])
def ai_image():
    """Receives an image file, analyzes it with Gemini AI, and returns JSON."""
    if 'file' not in request.files:
        return jsonify({"error": "No file found in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Perform AI (Gemini) receipt/invoice extraction
            receipt_data = extract_receipt_data(filepath)
            return jsonify({"text": json.dumps(receipt_data, indent=2, ensure_ascii=False)})
        except Exception as e:
            return jsonify({"error": f"An error occurred during AI processing: {str(e)}"}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)