import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from src.preprocessing import clean_resume

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("data/resume_dataset.csv")

df["cleaned_resume"] = df["Resume"].apply(clean_resume)

# ----------------------------
# Encode labels
# ----------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Category"])

# ----------------------------
# TF-IDF Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words="english"
)
X = vectorizer.fit_transform(df["cleaned_resume"])

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Models
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()
}

best_model = None
best_accuracy = 0

# ----------------------------
# Train & Evaluate
# ----------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# ----------------------------
# Save best model & artifacts
# ----------------------------
joblib.dump(best_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n✅ Training complete.")
print(f"🏆 Best Model: {type(best_model).__name__}")
print(f"📈 Accuracy: {best_accuracy:.4f}")
