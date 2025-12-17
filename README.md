# TalentLens – AI-Powered Resume Analyzer

TalentLens is a  **Resume Screening & Analysis system** built using **FastAPI (backend)** and **Next.js (frontend)**. It uses **Machine Learning (TF-IDF + SVM)** to classify resumes, provide confidence scores, extract top keywords, and supports **PDF resume uploads**.

This project is suitable for:

* Resume screening systems
* ATS-style applications
---

## ✨ Features

* 📄 Resume classification (Data Science, HR, Dev, etc.)
* 📊 Confidence score for predictions
* 🔑 Top keywords influencing the prediction
* 📎 PDF resume upload support
* 🧠 ML pipeline using TF‑IDF + Linear SVM
* 🌐 Full‑stack architecture (FastAPI + Next.js)

---

## 🧱 Tech Stack

### Backend

* Python 3.10+
* FastAPI
* Scikit‑learn
* NLTK
* Joblib
* PyMuPDF (PDF parsing)

### Frontend

* Next.js (App Router)
* TypeScript
* Tailwind CSS (optional styling)

---

## 📁 Folder Structure

```
TalentLens/
│
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── model.pkl               # Trained ML model
│   ├── vectorizer.pkl          # TF‑IDF vectorizer
│   ├── label_encoder.pkl       # Label encoder
│   └── src/
│       └── preprocessing.py    # Text cleaning logic
│
└── frontend/
    ├── app/
    │   └── page.tsx
    ├── components/
    │   └── ResumeForm.tsx
    └── package.json
```

---

## ⚙️ Backend Setup & Run

### 1️⃣ Create Virtual Environment

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```powershell
pip install fastapi uvicorn scikit-learn nltk joblib pymupdf
```

### 3️⃣ Download NLTK Resources

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4️⃣ Run Backend Server

```powershell
uvicorn main:app --reload
```

Backend will be live at:

```
http://localhost:8000
```

---

## 🌐 Frontend Setup & Run

### 1️⃣ Install Dependencies

```bash
cd frontend
npm install
```

### 2️⃣ Run Development Server

```bash
npm run dev
```

Frontend will be live at:

```
http://localhost:3000
```

---

## 🔌 API Endpoints

### 🔹 Predict Using Text

**POST** `/predict`

```json
{
  "text": "Your resume content here"
}
```

**Response:**

```json
{
  "category": "Data Science",
  "confidence": 87.23,
  "keywords": ["python", "machine", "learning"]
}
```

---

### 🔹 Predict Using PDF

**POST** `/predict-pdf`

* Form‑Data
* Key: `file`
* Value: Resume PDF

---

## 🧠 ML Model Details

* Text Preprocessing:

  * Lowercasing
  * Stopword removal
  * Lemmatization
* Vectorization:

  * TF‑IDF
* Classifier:

  * Linear Support Vector Machine (LinearSVC)
* Confidence:

  * Derived from decision function margins
* Keywords:

  * Top TF‑IDF weighted terms

---

## 🧪 Common Issues & Fixes

### ❌ "Failed to fetch" in frontend

✔ Ensure backend is running on port `8000`
✔ Ensure CORS is enabled in FastAPI
✔ Use `http://localhost:8000` (not 127.0.0.1 sometimes)

### ❌ sklearn version warning

✔ Re‑train model using same sklearn version
✔ Or downgrade sklearn to match saved model

---

## 🚀 Future Enhancements

* ATS score vs Job Description
* Resume strengths & weaknesses (GenAI)
* JD‑Resume similarity scoring
* Authentication (Admin / Recruiter)
* Cloud deployment (Vercel + Render)

---

## 📌 Resume / Interview Description

> Built an AI‑powered resume analyzer using FastAPI and Next.js with ML‑based classification, confidence scoring, keyword extraction, and PDF resume parsing.

---

## 👩‍💻 Author

**Neha Gade**
Final Year Computer Engineering Student
Project: TalentLens – Resume Analyzer

---

⭐ If you like this project, consider starring the repository!
