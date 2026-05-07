# 📸 Image Data Extraction & AI Analysis Web App

A powerful, Python-based web application and command-line tool for extracting various types of data from images. It combines traditional computer vision techniques with modern Generative AI to provide comprehensive image analysis.

## ✨ Features

- **🔍 Optical Character Recognition (OCR):** Extracts raw text from images using Tesseract OCR with built-in image preprocessing (denoising, thresholding) for higher accuracy.
- **🤖 AI-Powered Receipt/Invoice Analysis:** Utilizes the **Google Gemini 2.5 Flash API** to intelligently parse receipts and invoices, returning structured JSON data (store name, items, prices, tax, totals).
- **📍 EXIF & GPS Metadata:** Extracts hidden metadata from photos, including camera make/model, timestamps, and GPS coordinates.
- **📊 Table Extraction:** Detects grid lines in images and extracts tabular data directly into Pandas DataFrames.
- **🔳 Barcode & QR Code Scanner:** Detects and decodes multiple barcodes or QR codes from a single image.
- **🌐 Drag-and-Drop Web Interface:** A sleek, modern Flask-based web UI that allows users to easily upload images and choose between standard OCR or AI extraction.

---

## 🛠️ Prerequisites

Before running this project, you need to install some system-level dependencies.

**For macOS (using Homebrew):**
```bash
# Install Tesseract OCR and Turkish language support
brew install tesseract
brew install tesseract-lang

# Install Zbar (required for barcode/QR reading)
brew install zbar
```
*(Note for Apple Silicon Mac users: You might need to export the Zbar library path if Python cannot find it: `export DYLD_LIBRARY_PATH="$(brew --prefix zbar)/lib:$DYLD_LIBRARY_PATH"`)*

---

## 🚀 Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/Aleynaklc/Image-extraction.git
cd Image-extraction
```

**2. Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up your Google Gemini API Key:**
To use the AI extraction feature, you must have a Gemini API key. Set it as an environment variable in your terminal:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

---

## 💻 Usage

### 1. Web Application (Recommended)
Start the Flask server:
```bash
python web_app.py
```
Then, open your web browser and go to: `http://127.0.0.1:5001`
Simply drag and drop an image into the designated area and choose either **"Normal OCR"** or **"Yapay Zeka (Gemini API)"**.

### 2. Command Line Tool
You can also run the full extraction suite (OCR, Metadata, Barcode, AI) directly from the terminal:
```bash
python image_data_extraction.py path/to/your/image.jpg
```

---

## 🛡️ License

This project is open-source and available under the MIT License.