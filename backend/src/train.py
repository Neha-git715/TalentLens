import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from preprocessing import clean_resume

# Load data
df = pd.read_csv("../data/resume_dataset.csv", encoding="utf-8")

# Clean text
df["cleaned_resume"] = df["Resume"].apply(clean_resume)

# Encode labels
le = LabelEncoder()
df["Category"] = le.fit_transform(df["Category"])

X = df["cleaned_resume"]
y = df["Category"]

# Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=1500,
    sublinear_tf=True
)

X_vec = vectorizer.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "svm": LinearSVC(),
    "nb": MultinomialNB()
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    print(f"{name} accuracy: {score}")

    if score > best_score:
        best_score = score
        best_model = model

# Save artifacts
joblib.dump(best_model, "../model.pkl")
joblib.dump(vectorizer, "../vectorizer.pkl")
joblib.dump(le, "../label_encoder.pkl")

print("Training complete. Best model saved.")
