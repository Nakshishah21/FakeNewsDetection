import pandas as pd
import numpy as np
import re
import joblib
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

nltk.download('stopwords')

# -----------------------------
# 1 Load Dataset
# -----------------------------

data = pd.read_csv("data/WELFake_Dataset.csv")

# remove unnecessary column
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])

# remove missing values
data = data.dropna()

# combine title + text
data["content"] = data["title"] + " " + data["text"]

# -----------------------------
# 2 Text Preprocessing
# -----------------------------

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()
    text = re.sub(r'\W',' ',text)
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

data["content"] = data["content"].apply(clean_text)

# -----------------------------
# 3 Features & Labels
# -----------------------------

X = data["content"]
y = data["label"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4 TF-IDF Vectorization
# -----------------------------

vectorizer = TfidfVectorizer(max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 5 Model Comparison
# -----------------------------

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Linear SVC": LinearSVC(),

    "Naive Bayes": MultinomialNB(),

    "Random Forest": RandomForestClassifier(n_estimators=100)

}

best_model = None
best_accuracy = 0
best_model_name = ""

print("\nModel Performance:\n")

for name, model in models.items():

    model.fit(X_train_vec, y_train)

    pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, pred)

    print(name, "Accuracy:", acc)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_accuracy)

from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
# -----------------------------
# 6 Detailed Evaluation
# -----------------------------

final_pred = best_model.predict(X_test_vec)

print("\nClassification Report:\n")

print(classification_report(y_test, final_pred))

# -----------------------------
# 7 Save Model
# -----------------------------

joblib.dump(best_model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel and Vectorizer Saved Successfully")

