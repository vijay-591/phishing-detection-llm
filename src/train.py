import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocess import clean_text
from src.embedder import generate_embeddings
from sklearn.svm import SVC

df = pd.read_csv("data/phishing_emails.csv")

df["text"] = df["text"].apply(clean_text)

X = generate_embeddings(df["text"].tolist())
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel="linear", probability=True, class_weight="balanced")

model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

joblib.dump(model, "models/phishing_model.pkl")
