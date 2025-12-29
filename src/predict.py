import joblib
from src.preprocess import clean_text
from src.embedder import generate_embeddings

model = joblib.load("models/phishing_model.pkl")

def predict_email(text):
    text = clean_text(text)
    embedding = generate_embeddings([text])
    prob = model.predict_proba(embedding)[0][1]

    if prob > 0.55:
        return "⚠️ Phishing"
    else:
        return "✅ Legitimate"

