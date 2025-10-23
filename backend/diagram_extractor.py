
# --- Updated diagram_extractor.py section (for filtering diagrams) ---
import fitz  # PyMuPDF
import pytesseract
import cv2
import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_images_filtered(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    diagram_paths = []
    for page_num in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(page_num)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 5:
                image_path = os.path.join(output_folder, f"diagram_p{page_num}_{xref}.png")
                pix.save(image_path)
                if is_diagram_image(image_path):
                    diagram_paths.append(image_path)
                else:
                    os.remove(image_path)
            pix = None
    return diagram_paths

def is_diagram_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 10:
        return True
    text = pytesseract.image_to_string(gray)
    if len(text.strip()) > 100:
        return False
    return True

def is_relevant_diagram(image_path, question, threshold=0.35):
    image_text = extract_text_from_image(image_path)
    if not image_text.strip():
        return False
    sim_score = hybrid_similarity(question, image_text)
    print(f"[INFO] Hybrid similarity between diagram and question: {sim_score}")
    return sim_score > threshold

def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh)
    return text