import os
import joblib
from src.preprocessing import clean_resume

# Absolute paths (robust)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))


def predict_category(resume_text: str) -> str:
    cleaned = clean_resume(resume_text)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)
    return label_encoder.inverse_transform(pred)[0]


if __name__ == "__main__":
    sample_resume = """
    Python developer with experience in Django, REST APIs,
    machine learning and data analysis.
    """
    print("Predicted Category:", predict_category(sample_resume))
