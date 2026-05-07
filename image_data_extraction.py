"""
Image Data Extraction in Python
================================
Covers: OCR (text), tables, barcodes/QR codes, EXIF metadata, and AI-based extraction.
Install dependencies:
    pip install pillow pytesseract opencv-python pyzbar pandas anthropic
    # Also install Tesseract OCR engine: https://github.com/tesseract-ocr/tesseract
"""

# ─────────────────────────────────────────────
# 1. TEXT EXTRACTION (OCR) WITH PYTESSERACT
# ─────────────────────────────────────────────
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def extract_text(image_path: str) -> str:
    """Extract all text from an image using Tesseract OCR."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='tur+eng')
    return text.strip()


def extract_text_with_confidence(image_path: str) -> list[dict]:
    """Extract text with bounding boxes and confidence scores."""
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    results = []
    for i, word in enumerate(data["text"]):
        if word.strip() and int(data["conf"][i]) > 0:
            results.append({
                "text": word,
                "confidence": int(data["conf"][i]),
                "x": data["left"][i],
                "y": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            })
    return results


# ─────────────────────────────────────────────
# 2. IMAGE PREPROCESSING FOR BETTER OCR
# ─────────────────────────────────────────────
import cv2
import numpy as np

def preprocess_for_ocr(image_path: str) -> Image.Image:
    """
    Improve OCR accuracy by:
    - Converting to grayscale
    - Denoising
    - Adaptive thresholding (handles uneven lighting)
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh)


def extract_text_preprocessed(image_path: str) -> str:
    """Run OCR on a preprocessed (cleaned) version of the image."""
    clean_img = preprocess_for_ocr(image_path)
    return pytesseract.image_to_string(clean_img, lang='tur+eng').strip()


# ─────────────────────────────────────────────
# 3. TABLE EXTRACTION FROM IMAGES
# ─────────────────────────────────────────────
import pandas as pd

def extract_table_from_image(image_path: str) -> pd.DataFrame | None:
    """
    Detect and extract tabular data from an image using OpenCV.
    Works best on clear, bordered tables.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect horizontal and vertical lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Combine lines to find cell intersections
    grid = cv2.add(h_lines, v_lines)
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Extract bounding boxes for each cell, sorted by position
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 10:  # filter noise
            cells.append((x, y, w, h))

    if not cells:
        print("No table structure detected.")
        return None

    # Group cells into rows by y-position (within 10px tolerance)
    cells.sort(key=lambda c: (c[1] // 10, c[0]))
    rows: dict[int, list] = {}
    for x, y, w, h in cells:
        row_key = y // 10
        rows.setdefault(row_key, []).append((x, y, w, h))

    # OCR each cell and build the table
    color_img = cv2.imread(image_path)
    table_data = []
    for row_key in sorted(rows):
        row_cells = sorted(rows[row_key], key=lambda c: c[0])
        row_texts = []
        for x, y, w, h in row_cells:
            cell_img = color_img[y:y+h, x:x+w]
            cell_pil = Image.fromarray(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(cell_pil, config="--psm 7").strip()
            row_texts.append(text)
        if any(row_texts):
            table_data.append(row_texts)

    if not table_data:
        return None

    # Use first row as header if it looks like one
    df = pd.DataFrame(table_data[1:], columns=table_data[0]) if table_data else pd.DataFrame()
    return df


# ─────────────────────────────────────────────
# 4. BARCODE & QR CODE EXTRACTION
# ─────────────────────────────────────────────
from pyzbar.pyzbar import decode

def extract_barcodes(image_path: str) -> list[dict]:
    """Detect and decode all barcodes and QR codes in an image."""
    img = Image.open(image_path)
    decoded = decode(img)
    results = []
    for item in decoded:
        results.append({
            "type": item.type,          # e.g. "QRCODE", "EAN13", "CODE128"
            "data": item.data.decode("utf-8"),
            "rect": {
                "left": item.rect.left,
                "top": item.rect.top,
                "width": item.rect.width,
                "height": item.rect.height,
            }
        })
    return results


# ─────────────────────────────────────────────
# 5. EXIF METADATA EXTRACTION
# ─────────────────────────────────────────────
from PIL.ExifTags import TAGS, GPSTAGS

def extract_metadata(image_path: str) -> dict:
    """Extract EXIF metadata (camera info, GPS, timestamps) from a photo."""
    img = Image.open(image_path)
    raw_exif = img._getexif()
    if not raw_exif:
        return {"error": "No EXIF data found"}

    metadata = {}
    for tag_id, value in raw_exif.items():
        tag = TAGS.get(tag_id, tag_id)
        metadata[tag] = value

    return metadata


def extract_gps(image_path: str) -> dict | None:
    """Extract GPS coordinates from a photo's EXIF data."""
    metadata = extract_metadata(image_path)
    gps_raw = metadata.get("GPSInfo")
    if not gps_raw:
        return None

    gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}

    def dms_to_decimal(dms, ref):
        d, m, s = dms
        decimal = float(d) + float(m) / 60 + float(s) / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)

    try:
        lat = dms_to_decimal(gps["GPSLatitude"], gps["GPSLatitudeRef"])
        lon = dms_to_decimal(gps["GPSLongitude"], gps["GPSLongitudeRef"])
        return {"latitude": lat, "longitude": lon}
    except KeyError:
        return None


# ─────────────────────────────────────────────
# 6. AI-BASED EXTRACTION (CLAUDE API)
#    -> UPDATED TO USE GOOGLE GEMINI
# ─────────────────────────────────────────────
import json
import os
from google import genai

def extract_with_ai(image_path: str, prompt: str) -> str:
    """
    Use Google's Gemini Pro Vision API to extract structured data from any image.
    Great for receipts, forms, invoices, charts, handwriting, etc.

    Requires the GOOGLE_API_KEY environment variable to be set.

    Args:
        image_path: Path to the image file
        prompt: Natural language instruction, e.g. "Extract all line items
                and totals from this receipt as JSON."
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Lütfen GOOGLE_API_KEY ortam değişkenini ayarlayın.")
    client = genai.Client(api_key=api_key)

    img = Image.open(image_path)

    # Gemini API can sometimes return markdown ```json ... ```, let's ask it not to.
    full_prompt = f"{prompt}\n\nYanıt olarak sadece JSON metnini ver, başka bir şey yazma."

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[full_prompt, img]
        )
        # Clean up potential markdown fences
        text_response = response.text.strip()
        if text_response.startswith("```json"):
            text_response = text_response[7:]
        if text_response.endswith("```"):
            text_response = text_response[:-3]
        return text_response.strip()
    except Exception as e:
        # Handle potential API errors, content blocking, etc.
        return f'{{"error": "Gemini API call failed", "details": "{str(e)}"}}'


def extract_receipt_data(image_path: str) -> dict:
    """Extract structured receipt data using Gemini."""
    prompt = """Analyze this image. If it's a receipt or invoice, extract the information.
    If it's not a receipt/invoice, leave the fields null but briefly explain what you see in the image in English in the "description" field.
    RETURN ONLY a valid JSON object in the following structure:
    {
        "store": "store name or null",
        "date": "date or null",
        "items": [{"name": "...", "qty": 1, "price": 0.00}],
        "subtotal": 0.00,
        "tax": 0.00,
        "total": 0.00,
        "description": "brief description of what is in the image"
    }
    """

    raw = extract_with_ai(image_path, prompt)
    try:
        # Gemini might still add comments or other text, so we need to find the JSON block
        json_start = raw.find('{')
        json_end = raw.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = raw[json_start:json_end]
            return json.loads(json_str)
        else:
            raise json.JSONDecodeError("No JSON object found in the response", raw, 0)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON from AI response", "raw_response": raw}


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os

    # --- SETUP FOR TESTING ---
    # Write the full path of the image you want to test in the line below.
    path = "/Users/aleynakilic/Downloads/ornek_resim.jpg"  # <--- RESMİN YOLUNU BURAYA GİRİN

    if len(sys.argv) > 1:
        path = sys.argv[1]
    elif not os.path.exists(path):
        print("Please run the code as 'python image_data_extraction.py <image_path>' or update the 'path' variable in the code.")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"Extracting data from: {path}")
    print('='*50)

    # 1. Basic OCR
    print("\n[1] RAW TEXT (OCR)")
    print(extract_text(path))

    # 2. Metadata
    print("\n[2] METADATA")
    meta = extract_metadata(path)
    for k in ["Make", "Model", "DateTime", "ImageWidth", "ImageLength"]:
        if k in meta:
            print(f"  {k}: {meta[k]}")

    # 3. GPS
    print("\n[3] GPS COORDINATES")
    gps = extract_gps(path)
    print(gps if gps else "  No GPS data found")

    # 4. Barcodes / QR
    print("\n[4] BARCODES / QR CODES")
    codes = extract_barcodes(path)
    if codes:
        for c in codes:
            print(f"  [{c['type']}] {c['data']}")
    else:
        print("  None detected")

    # 5. AI extraction (requires GOOGLE_API_KEY env var)
    # print("\n[5] AI-BASED EXTRACTION")
    # print(extract_with_ai(path, "Describe what data is visible in this image."))
    print("\n[5] AI-BASED EXTRACTION (Receipt/Invoice)")
    try:
        receipt_data = extract_receipt_data(path)
        print(json.dumps(receipt_data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"  AI analysis failed: {e}")
        print("  Make sure the 'GOOGLE_API_KEY' environment variable is set.")
