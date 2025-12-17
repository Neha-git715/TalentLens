# TalentLens – ML-Powered Resume Analyzer

TalentLens is an **end-to-end Machine Learning project** that analyzes resume text and predicts the most relevant job category using **Natural Language Processing (NLP)** and **supervised learning**. The trained ML model is served via a **FastAPI backend** and consumed by a **Next.js frontend**.

This project demonstrates the **complete ML lifecycle** — data preprocessing, feature engineering, model training, evaluation, inference, and full-stack deployment.

---

## 🚀 Key Features

* Resume text preprocessing using NLP (NLTK)
* TF-IDF based feature extraction
* Supervised ML classification of resumes into job roles
* Model evaluation with accuracy & classification report
* FastAPI backend exposing prediction API
* Next.js frontend for user interaction
* Modular, production-ready project structure

---

## 🧠 Why This Is an ML Project

TalentLens is **not just a UI or rule-based system**. It includes:

✔ Dataset-driven learning
✔ NLP text cleaning & normalization
✔ Vectorization (TF-IDF)
✔ Supervised classification model
✔ Model training & evaluation
✔ Saved ML artifacts (`.pkl` files)
✔ Real-time inference via API

---

## 🛠 Tech Stack

### Machine Learning & NLP

* Python
* Scikit-learn
* Pandas, NumPy
* NLTK (tokenization, lemmatization, stopwords)
* TF-IDF Vectorizer

### Backend

* FastAPI
* Joblib (model persistence)
* Uvicorn

### Frontend

* Next.js (TypeScript)
* React
* Tailwind CSS

---

## 📂 Project Structure

```
TalentLens/
│
├── backend/
│   ├── src/
│   │   ├── preprocessing.py   # NLP cleaning logic
│   │   ├── train.py            # Model training pipeline
│   │   ├── predict.py          # Inference logic
│   │   └── __init__.py
│   │
│   ├── data/
│   │   └── resume_dataset.csv  # Training dataset
│   │
│   ├── model.pkl               # Trained ML model
│   ├── vectorizer.pkl          # TF-IDF vectorizer
│   ├── label_encoder.pkl       # Encoded labels
│   ├── app.py                  # FastAPI server
│   ├── requirements.txt
│   └── README.md
│
├── frontend/
│   ├── components/
│   ├── app/
│   ├── package.json
│   └── ...
│
└── README.md
```

---

## ⚙️ How to Run the Project (From Scratch)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/TalentLens.git
cd TalentLens/backend
```

---

### 2️⃣ Create Virtual Environment (Recommended)

**Windows**

```powershell
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Train the ML Model (One-Time)

⚠️ Must be run from `backend/`

```bash
python -m src.train
```

This step:

* Cleans resume text
* Converts text to TF-IDF vectors
* Trains a supervised classifier
* Evaluates performance
* Saves trained artifacts (`.pkl` files)

---

### 5️⃣ Test Resume Prediction (CLI)

```bash
python -m src.predict
```

Expected output:

```
Predicted Category: Python Developer
```

---

### 6️⃣ Run Backend API (FastAPI)

```bash
python app.py
```

Open API documentation:

```
http://127.0.0.1:8000/docs
```

---

### 7️⃣ Run Frontend (Next.js)

```bash
cd ../frontend
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:3000
```

---

## 📊 Model Details

* **Input**: Raw resume text
* **Preprocessing**: Tokenization, stopword removal, lemmatization
* **Feature Engineering**: TF-IDF
* **Model Type**: Supervised multi-class classifier
* **Evaluation Metric**: Accuracy, Precision, Recall, F1-score

Sample accuracy achieved:

```
Accuracy: ~82%
```

---

## 🔌 API Endpoint

### POST `/predict`

**Request Body**

```json
{
  "text": "Python developer with experience in Django and ML"
}
```

**Response**

```json
{
  "prediction": "Python Developer"
}
```

---

## 📌 Important Notes

* Always run ML scripts using `python -m src.<file>`
* Dataset must exist in `backend/data/`
* NLTK resources auto-download on first run
* This project uses **classical ML**, not deep learning

---

## 🎯 Interview Explanation (Short)

> “TalentLens is an end-to-end ML resume classification system. I preprocess resume text using NLP, extract TF-IDF features, and train a supervised classifier. The trained model is exposed via a FastAPI backend and integrated with a Next.js frontend for real-time predictions.”

---

## 🔮 Future Improvements

* Handle class imbalance (SMOTE / class weights)
* Add confidence scores
* Resume keyword highlighting
* PDF resume upload
* Model comparison (SVM vs Logistic Regression)
* Deployment using Docker & cloud

---

## 👩‍💻 Author

**Neha Gade**

---

⭐ If you like this project, don’t forget to star the repository!
