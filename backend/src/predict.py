import joblib
from preprocessing import clean_resume

model = joblib.load("../model.pkl")
vectorizer = joblib.load("../vectorizer.pkl")
label_encoder = joblib.load("../label_encoder.pkl")

def predict_category(resume_text):
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
