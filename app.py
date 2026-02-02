# ---------------- Imports ----------------------
from flask import Flask, render_template, request, redirect, flash # type: ignore
import os
import re
import string
import numpy as np # type: ignore
import joblib # type: ignore
import nltk # type: ignore
from nltk.corpus import stopwords # type: ignore
from transformers import pipeline # type: ignore
from PIL import Image # type: ignore

# -------------------- NLTK SETUP --------------------
nltk.download("stopwords", quiet=True)

# -------------------- FLASK APP --------------------
app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------- LOAD FAKE NEWS MODEL --------------------
try:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("✅ Fake news model loaded")
except Exception as e:
    model = None
    vectorizer = None
    print("❌ Error loading fake news model:", e)

# -------------------- LOAD IMAGE DETECTION MODELS --------------------
print("🔥 Loading AI image detection models...")

IMAGE_MODELS = {
    "Ateeqq": pipeline(
        "image-classification",
        model="Ateeqq/ai-vs-human-image-detector"
    )
}

print("✅ Image detection models loaded")

# -------------------- TEXT CLEANING --------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = " ".join(
        word for word in text.split()
        if word not in stopwords.words("english")
    )
    return text

# -------------------- IMAGE LABEL NORMALIZATION --------------------
def normalize_label(label):
    label = label.lower()
    if "ai" in label or "fake" in label:
        return "AI"
    return "REAL"

# -------------------- IMAGE DETECTION LOGIC --------------------
def detect_ai_image(image_path):
    image = Image.open(image_path).convert("RGB")

    votes = []
    confidences = []

    for name, detector in IMAGE_MODELS.items():
        result = detector(image)[0]
        label = normalize_label(result["label"])
        score = result["score"]

        votes.append(label)
        confidences.append(score)

        print(f"{name}: {label} ({score:.2f})")

    final_label = max(set(votes), key=votes.count)
    avg_confidence = round(np.mean(confidences) * 100, 2)

    return final_label, avg_confidence

# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------- FAKE NEWS CHECK ----------
@app.route("/check_news", methods=["GET", "POST"])
def check_news():
    if request.method == "POST":
        if model is None or vectorizer is None:
            flash("Fake news model not loaded")
            return redirect("/check_news")

        title = request.form.get("title", "")
        article = request.form.get("article", "")
        text = clean_text(title + " " + article)

        if len(text.split()) < 5:
            flash("Please enter more meaningful text")
            return redirect("/check_news")

        vect = vectorizer.transform([text])
        prediction = model.predict(vect)[0]
        prob = model.predict_proba(vect)[0]

        label = "REAL" if prediction == 1 else "FAKE"
        confidence = round(np.max(prob) * 100, 2)

        return render_template(
            "result_news.html",
            label=label,
            confidence=confidence,
            reason=f"Model confidence: {confidence}%"
        )

    return render_template("check_news.html")

# ---------- IMAGE CHECK ----------
@app.route("/check_image", methods=["GET", "POST"])
def check_image():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No image uploaded")
            return redirect("/check_image")

        file = request.files["image"]
        if file.filename == "":
            flash("No image selected")
            return redirect("/check_image")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        label, confidence = detect_ai_image(filepath)

        os.remove(filepath)

        return render_template(
            "result_image.html",
            label=label,
            confidence=f"{confidence}%"
        )

    return render_template("check_image.html")

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    print("🚀 AI Detection Flask App Running")
    app.run(debug=True)
