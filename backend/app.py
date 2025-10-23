import os
import time
import fitz
import cv2
import numpy as np
import pytesseract
from flask import Flask, render_template_string, request, send_from_directory, redirect, url_for, flash
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
import re
from graphviz import Digraph
import torch
import torchvision
from torchvision import transforms as T
import layoutparser as lp 
import matplotlib.pyplot as plt

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = r'D:\PROGRAMING\research\static\diagrams'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

llm = Llama(model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=4096, n_threads=8)

knowledge_base = {
    "Newton": "Newton's laws of motion are a cornerstone in physics, explaining how objects behave under forces.",
    "photosynthesis": "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
    "UI Path": "UI Path is a leading Robotic Process Automation (RPA) platform that enables organizations to automate repetitive tasks.",
    "control flow": "Control flow activities in UI Path determine the execution path through an automation workflow based on decisions, loops, and conditions."
}
KNOWLEDGE_WEIGHT = 0.3  

try:
    layout_model = lp.Detectron2LayoutModel(
        "HYPJUDY/layoutlmv3-base-finetuned-publaynet",
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        label_map={0:"Text", 1:"Title", 2:"Figure", 3:"Table", 4:"List"}
    )
    print("LayoutParser Detectron2LayoutModel loaded successfully.")
except AttributeError as e:
    print("LayoutParser does not have Detectron2LayoutModel. Please install layoutparser[detectron2] with version >=1.0.0.")
    layout_model = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn.eval().to(device)
transform = T.Compose([T.ToTensor()])

############################## HTML TEMPLATE ##############################
html_template = """
<!doctype html>
<html>
<head>
    <title>AI Answer with Diagrams</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; background: #f8f8f8; }
        .answer { background: #fff; padding: 15px; border-radius: 10px; box-shadow: 0 0 10px #ccc; opacity: 0; animation: fadeIn 2s forwards; }
        .diagrams { margin-top: 20px; }
        .diagram-container { margin-bottom: 20px; }
        .diagram-container img { max-width: 90%; display: block; margin: 10px auto; border-radius: 10px; }
        .explanation { font-style: italic; text-align: center; margin-top: 5px; color: #555; }
        .metrics { margin-top: 20px; }
        pre { white-space: pre-wrap; font-size: 16px; }
        .error { color: red; }
        .slider-container { margin: 15px 0; }
        .slider-container label { display: inline-block; width: 150px; }
        .marks-slider { width: 300px; vertical-align: middle; }
        .marks-value { display: inline-block; width: 30px; text-align: center; margin-left: 10px; }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
    <script>
        function updateMarksValue(val) {
            document.getElementById('marks-value').innerText = val;
        }
    </script>
</head>
<body>
    <h1>AI Answer with Relevant Diagrams</h1>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="error">{{ messages[0] }}</div>
      {% endif %}
    {% endwith %}
    <form action="/process" method="post" enctype="multipart/form-data">
        <label>Upload PDF:</label><br>
        <input type="file" name="pdf_file" required><br><br>
        <label>Ask Question:</label><br>
        <textarea name="question" rows="4" cols="50" required></textarea><br><br>
        <div class="slider-container">
            <label for="marks">Answer Length (Marks):</label>
            <input type="range" id="marks" name="marks" class="marks-slider" min="1" max="10" value="5" step="1" oninput="updateMarksValue(this.value)">
            <span id="marks-value" class="marks-value">5</span>
        </div><br>
        <input type="submit" value="Get Answer">
    </form>
    {% if answer %}
    <div class="answer">
        <h2>Answer:</h2>
        <pre>{{ answer }}</pre>
        {% if diagrams %}
        <div class="diagrams">
            <h2>Relevant Diagram(s) and Explanations:</h2>
            {% for diagram, explanation in diagrams.items() %}
                <div class="diagram-container">
                    <img src="{{ url_for('serve_diagram', filename=diagram) }}" alt="Diagram">
                    <div class="explanation">{{ explanation }}</div>
                </div>
            {% endfor %}
        </div>
        {% endif %}
        {% if metrics_graph %}
        <div class="metrics">
            <h2>Performance Metrics:</h2>
            <img src="{{ url_for('serve_diagram', filename=metrics_graph) }}" alt="Performance Metrics Graph">
        </div>
        {% endif %}
    </div>
    {% endif %}
</body>
</html>
"""

########################## UTILITY FUNCTIONS ##########################

def clean_llama_answer(text):
    """Clean LLaMA generated text by removing any lines or markdown syntax related to diagrams."""
    lines = text.splitlines()
    cleaned_lines = [line for line in lines if not line.strip().lower().startswith("diagram")]
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', cleaned)
    return cleaned.strip()

def extract_images_with_context(pdf_path, output_folder):
    """Extract images and nearby text from the PDF."""
    doc = fitz.open(pdf_path)
    images_info = []
    for page_num, page in enumerate(doc):
        text_blocks = page.get_text("blocks")
        text_blocks.sort(key=lambda b: b[1])
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 5:  # Grayscale or RGB
                image_name = f"diagram_p{page_num}_{xref}.png"
                image_path = os.path.join(output_folder, image_name)
                pix.save(image_path)
                # Get context text: the nearest text block above the image
                context_text = next((block[4].strip() for block in reversed(text_blocks) if block[1] < img[2]), "")
                images_info.append((image_name, context_text))
            pix = None
    return images_info

def is_diagram_image(image_path):
    """Check if the image is likely a diagram using a simple heuristic."""
    image = cv2.imread(image_path)
    if image is None:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 10:
        return True
    try:
        text = pytesseract.image_to_string(gray)
    except:
        text = ""
    return len(text.strip()) < 100

def extract_text_from_image(image_path):
    """Extract text from the image using OCR."""
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return pytesseract.image_to_string(thresh)
    except:
        return ""

def extract_text_from_pdf(pdf_path):
    """Extract all text from the PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def lexical_overlap(query, text):
    """Simple lexical overlap: fraction of common words."""
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    if not query_words:
        return 0
    return len(query_words.intersection(text_words)) / len(query_words)

def hybrid_similarity(query, text, alpha=0.7, beta=0.3):
    """
    Compute a hybrid similarity score:
    alpha * dense cosine similarity + beta * lexical overlap.
    """
    dense_sim = util.cos_sim(embedder.encode(query), embedder.encode(text)).item()
    lex_sim = lexical_overlap(query, text)
    return alpha * dense_sim + beta * lex_sim

def is_relevant_diagram(image_path, question, surrounding_text, threshold=0.35):
    """
    Determine if the image is relevant as a diagram.
    Uses a hybrid similarity score and checks that the combined text length is sufficient.
    """
    if not os.path.exists(image_path):
        return False
    image_text = extract_text_from_image(image_path)
    combined_text = f"{image_text} {surrounding_text}".strip()
    if len(combined_text) < 20:
        return False
    sim_score = hybrid_similarity(question, combined_text)
    print(f"[DEBUG] Hybrid similarity for {os.path.basename(image_path)}: {sim_score}")
    return sim_score > threshold

def augment_prompt_with_knowledge(question):
    """Append structured knowledge from our simple knowledge base if keywords match."""
    added_context = ""
    for key, value in knowledge_base.items():
        if key.lower() in question.lower():
            added_context += f"\nNote: {value}"
    return added_context

def determine_answer_length(marks):
    """
    Map marks value (1-10) to answer length parameters.
    Returns a tuple of (max_tokens, detail_level)
    """
    # Calculate max tokens based on marks (100-1500 token range)
    max_tokens = int(100 + (marks - 1) * 155)
    
    # Determine detail level instruction
    if marks <= 3:
        detail_level = "brief and concise, focusing only on core concepts"
    elif marks <= 6:
        detail_level = "moderately detailed, covering main points with some examples"
    else:
        detail_level = "comprehensive and thorough, with detailed explanations and examples"
        
    return max_tokens, detail_level

def generate_llama_answer(question, marks):
    """
    Generate a detailed answer using the LLaMA model.
    The answer length is adjusted based on the marks value (1-10).
    """
    knowledge_context = augment_prompt_with_knowledge(question)
    max_tokens, detail_level = determine_answer_length(marks)
    
    prompt = (f"Using only the provided textbook or notes, answer the following question in a {detail_level} manner. "
              f"The answer should be approximately {max_tokens/7} to {max_tokens/5} words long. "
              f"Provide a professional explanation that strictly addresses the question. "
              f"Do not include any textual representations of diagrams.\n\n"
              f"Question: {question}\n{knowledge_context}\n\nAnswer:")
    
    print(f"[DEBUG] Generating LLaMA answer with {max_tokens} max tokens...")
    output = llm(prompt, max_tokens=max_tokens)
    return output['choices'][0]['text'].strip()

def detect_and_crop_diagram(image_path, metadata=None):
    """
    Use LayoutParser to detect layout elements and crop regions labeled as "Figure".
    If metadata (e.g., bounding box) is available, use it. Otherwise, use the LayoutParser model.
    If LayoutParser model is not available, fall back to returning the full image.
    """
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    if layout_model is None:
        return image_path
    pil_image = lp.io.load_image(image_path)
    layout = layout_model.detect(pil_image)
    # Look for layout elements labeled as "Figure" with a high confidence
    figures = [b for b in layout if b.type == "Figure" and b.score > 0.8]
    if figures:
        best_fig = max(figures, key=lambda b: b.score)
        x1, y1, x2, y2 = map(int, best_fig.coordinates)
        cropped = img[y1:y2, x1:x2]
        crop_path = image_path.replace(".png", "_crop.png")
        cv2.imwrite(crop_path, cropped)
        return crop_path
    else:
        return image_path

def explain_diagram(image_path, question, marks):
    """
    Generate a text-based flowchart explanation for the given diagram.
    The explanation detail is adjusted based on the marks value.
    """
    cropped_path = detect_and_crop_diagram(image_path)
    ocr_text = extract_text_from_image(cropped_path)
    
    # Adjust explanation detail based on marks
    if marks <= 3:
        detail_level = "brief and simple"
        max_tokens = 80
    elif marks <= 6:
        detail_level = "moderately detailed"
        max_tokens = 120
    else:
        detail_level = "comprehensive and thorough"
        max_tokens = 200
    
    if not ocr_text.strip():
        explanation_prompt = f"Provide a {detail_level}, text-based flowchart explanation for a typical diagram related to '{question}'."
    else:
        explanation_prompt = (f"Based on the following extracted text from a diagram:\n\n{ocr_text}\n\n"
                            f"Generate a {detail_level}, text-based flowchart explanation that relates to the diagram and the question: '{question}'")
    
    print(f"[DEBUG] Generating explanation for diagram: {os.path.basename(cropped_path)} with {max_tokens} max tokens")
    output = llm(explanation_prompt, max_tokens=max_tokens)
    return output['choices'][0]['text'].strip()

def plot_performance_metrics(metrics, output_path):
    """
    Plot a bar chart of performance metrics.
    Metrics include: Diagrams Detected, Avg Diagram Similarity, Answer-Text Similarity, and Response Time.
    """
    keys = list(metrics.keys())
    values = list(metrics.values())
    plt.figure(figsize=(8, 4))
    bars = plt.bar(keys, values, color='skyblue')
    
    # Add value labels on top of each bar
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                str(round(values[i], 2) if isinstance(values[i], float) else values[i]),
                ha='center', va='bottom')
    
    plt.title("Performance Metrics")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

########################## FLASK ROUTES #############################

@app.route('/', methods=['GET'])
def index():
    return render_template_string(html_template)

@app.route('/process', methods=['POST'])
def process():
    start_time = time.time()
    pdf_file = request.files.get('pdf_file')
    question = request.form.get('question', '').strip()
    marks = int(request.form.get('marks', 5))  # Default to 5 if not provided
    
    if not pdf_file or pdf_file.filename == '':
        flash('Please upload a PDF file.')
        return redirect(url_for('index'))
    if not question:
        flash('Please enter your question.')
        return redirect(url_for('index'))

    # Save uploaded PDF
    pdf_path = os.path.join(UPLOAD_FOLDER, 'uploaded.pdf')
    pdf_file.save(pdf_path)

    # Extract full textbook text for answer accuracy metric
    textbook_text = extract_text_from_pdf(pdf_path)

    # Extract images and surrounding text from the PDF
    images_info = extract_images_with_context(pdf_path, UPLOAD_FOLDER)
    diagram_explanations = {}  # Dictionary to hold diagram filename and explanation
    diagram_similarities = []  # List to hold similarity scores for diagrams

    # Process each extracted image
    for image_name, context_text in images_info:
        image_path = os.path.join(UPLOAD_FOLDER, image_name)
        if is_diagram_image(image_path) and is_relevant_diagram(image_path, question, context_text):
            cropped_image_path = detect_and_crop_diagram(image_path, metadata=None)
            explanation = explain_diagram(cropped_image_path, question, marks)
            diagram_explanations[os.path.basename(cropped_image_path)] = explanation
            # Record hybrid similarity score for the diagram
            score = hybrid_similarity(question, extract_text_from_image(image_path))
            diagram_similarities.append(score)

    # Generate final text answer using an enhanced prompt with knowledge augmentation (text-only answer)
    answer_html = generate_llama_answer(question, marks)

    # Compute performance metrics
    processing_time = time.time() - start_time
    num_diagrams = len(diagram_explanations)
    avg_similarity = np.mean(diagram_similarities) if diagram_similarities else 0
    
    # Compute answer-text similarity: compare the generated answer with textbook content
    answer_embedding = embedder.encode(answer_html)
    textbook_embedding = embedder.encode(textbook_text)
    answer_text_similarity = util.cos_sim(answer_embedding, textbook_embedding).item()

    metrics = {
        "Diagrams Detected": num_diagrams,
        "Avg Diagram Similarity": round(avg_similarity, 2),
        "Answer-Text Similarity": round(answer_text_similarity, 2),
        "Response Time (s)": round(processing_time, 2)
    }
    metrics_path = os.path.join(UPLOAD_FOLDER, "performance_metrics.png")
    plot_performance_metrics(metrics, metrics_path)

    print(f"[DEBUG] Relevant diagrams and explanations: {diagram_explanations}")
    print(f"[DEBUG] Total processing time: {processing_time:.2f} seconds")

    return render_template_string(
        html_template, 
        answer=answer_html, 
        diagrams=diagram_explanations, 
        metrics_graph="performance_metrics.png"
    )

@app.route('/static/diagrams/<path:filename>')
def serve_diagram(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

########################## MAIN RUN #############################
if __name__ == "__main__":
    app.run(debug=True)