import os
import fitz  
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


pdf_folder = "new_pdfs"  
model_path = "ref_star_classifier"
max_length = 512


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


label_map = {0: "4*", 1: "3*", 2: "2*", 3: "1*"}


def predict_from_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
        return label_map[pred], round(confidence, 4)


def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = " ".join([page.get_text() for page in doc])
        return " ".join(text.strip().split())  # clean whitespace
    except Exception as e:
        print(f"❌ Failed to extract {pdf_path}: {e}")
        return None


print("📂 Scanning folder:", pdf_folder)
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(pdf_folder, filename)
        print(f"\n🔍 Processing: {filename}")

        text = extract_text_from_pdf(filepath)
        if text:
            label, confidence = predict_from_text(text)
            print(f"✅ Predicted: {label} (Confidence: {confidence})")
        else:
            print("❌ Skipped due to extraction failure.")
