from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import fitz  # PyMuPDF
from src.preprocessing import clean_resume

app = FastAPI(title="TalentLens Resume Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

class ResumeRequest(BaseModel):
    text: str

def extract_top_keywords(vector, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = vector.toarray()[0]
    top_indices = np.argsort(tfidf_scores)[-top_n:][::-1]
    return [feature_names[i] for i in top_indices]

@app.post("/predict")
def predict_resume(data: ResumeRequest):
    cleaned = clean_resume(data.text)
    vector = vectorizer.transform([cleaned])

    decision_scores = model.decision_function(vector)
    pred_index = np.argmax(decision_scores)
    confidence = float(
        (decision_scores[0][pred_index] / np.sum(np.abs(decision_scores))) * 100
    )

    category = label_encoder.inverse_transform([pred_index])[0]
    keywords = extract_top_keywords(vector)

    return {
        "category": category,
        "confidence": round(abs(confidence), 2),
        "keywords": keywords
    }

# 📄 PDF UPLOAD ENDPOINT
@app.post("/predict-pdf")
async def predict_pdf(file: UploadFile = File(...)):
    pdf_text = ""
    pdf = fitz.open(stream=await file.read(), filetype="pdf")

    for page in pdf:
        pdf_text += page.get_text()

    cleaned = clean_resume(pdf_text)
    vector = vectorizer.transform([cleaned])

    decision_scores = model.decision_function(vector)
    pred_index = np.argmax(decision_scores)
    confidence = float(
        (decision_scores[0][pred_index] / np.sum(np.abs(decision_scores))) * 100
    )

    category = label_encoder.inverse_transform([pred_index])[0]
    keywords = extract_top_keywords(vector)

    return {
        "category": category,
        "confidence": round(abs(confidence), 2),
        "keywords": keywords
    }
