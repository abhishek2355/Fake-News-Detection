# ---------------- Imports ----------------------
from flask import Flask, render_template, request, redirect, flash
import os, re, string, warnings
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
from PIL import Image
import librosa
import torch
from moviepy.editor import VideoFileClip

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)

# ---------------- Flask App --------------------
app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- Fake News Model ----------------
try:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("✅ Fake news model loaded")
except:
    model, vectorizer = None, None
    print("❌ Fake news model not loaded")

# =================================================
# IMAGE MODELS (MULTI-MODEL ENSEMBLE)
# =================================================
print("🖼️ Loading image detection models...")
IMAGE_MODELS = {}

IMAGE_MODEL_LIST = [
    ("Ateeqq", "Ateeqq/ai-vs-human-image-detector")
]

for name, model_id in IMAGE_MODEL_LIST:
    try:
        IMAGE_MODELS[name] = pipeline(
            "image-classification",
            model=model_id
        )
        print(f"✅ Image model loaded: {name}")
    except Exception as e:
        print(f"❌ Image model failed ({name}):", e)

# =================================================
# AUDIO MODELS (MULTI-MODEL ENSEMBLE)
# =================================================
print("🎵 Loading audio detection models...")
AUDIO_MODELS = []

AUDIO_MODEL_LIST = [
    "Hemgg/Deepfake-audio-detection"
]

for model_id in AUDIO_MODEL_LIST:
    try:
        AUDIO_MODELS.append(
            pipeline(
                "audio-classification",
                model=model_id,
                device=0 if torch.cuda.is_available() else -1
            )
        )
        print(f"✅ Audio model loaded: {model_id}")
    except Exception as e:
        print(f"❌ Audio model failed ({model_id}):", e)

# =================================================
# HELPERS
# =================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return " ".join(
        w for w in text.split()
        if w not in stopwords.words("english")
    )

def normalize_label(label):
    label = label.lower()
    if any(x in label for x in ["ai", "fake", "spoof", "synth"]):
        return "AI"
    return "REAL"

# =================================================
# IMAGE ENSEMBLE
# =================================================
def detect_ai_image(image_path):
    image = Image.open(image_path).convert("RGB")
    votes, confidences = [], []

    for name, model in IMAGE_MODELS.items():
        try:
            result = model(image)[0]
            label = normalize_label(result["label"])
            score = result["score"]

            votes.append(label)
            confidences.append(score)
            print(f"[IMAGE] {name}: {label} ({score:.2f})")
        except:
            pass

    if not votes:
        return "REAL", 50.0

    final_label = max(set(votes), key=votes.count)
    avg_conf = np.mean(
        [c for v, c in zip(votes, confidences) if v == final_label]
    )

    return final_label, round(avg_conf * 100, 2)

# =================================================
# AUDIO ENSEMBLE
# =================================================
def detect_ai_audio(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio, _ = librosa.effects.trim(audio)

        votes, confidences = [], []

        for model in AUDIO_MODELS:
            outputs = model(audio)

            for o in outputs:
                label = normalize_label(o["label"])
                score = o["score"]

                votes.append(label)
                confidences.append(score)

                print(f"[AUDIO] {label} ({score:.2f})")
                break

        if not votes:
            return "REAL", 50.0

        final_label = max(set(votes), key=votes.count)
        avg_conf = np.mean(
            [c for v, c in zip(votes, confidences) if v == final_label]
        )

        return final_label, round(avg_conf * 100, 2)

    except Exception as e:
        print("Audio error:", e)
        return "REAL", 50.0

# =================================================
# VIDEO (IMAGE + AUDIO ENSEMBLE)
# =================================================
def detect_ai_video(video_path):
    try:
        video = VideoFileClip(video_path)

        frame_path = os.path.join(UPLOAD_FOLDER, "frame.jpg")
        video.save_frame(frame_path, t=1)

        frame_label, frame_conf = detect_ai_image(frame_path)

        audio_label, audio_conf = "REAL", 50.0
        if video.audio:
            audio_path = os.path.join(UPLOAD_FOLDER, "audio.wav")
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            audio_label, audio_conf = detect_ai_audio(audio_path)
            os.remove(audio_path)

        video.close()
        os.remove(frame_path)

        votes = [frame_label, audio_label]
        confidences = [frame_conf, audio_conf]

        final_label = max(set(votes), key=votes.count)
        final_conf = round(np.mean(confidences), 2)

        print(f"[VIDEO] Frame={frame_label}, Audio={audio_label}")
        return final_label, final_conf

    except Exception as e:
        print("Video error:", e)
        return "REAL", 50.0

# =================================================
# ROUTES
# =================================================
@app.route("/")
def home():
    return render_template("index.html")

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

@app.route("/check_image", methods=["GET", "POST"])
def check_image():
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            flash("No image uploaded")
            return redirect("/check_image")

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        label, confidence = detect_ai_image(path)
        os.remove(path)

        return render_template("result_image.html", label=label, confidence=f"{confidence}%")

    return render_template("check_image.html")

@app.route("/check_audio", methods=["GET", "POST"])
def check_audio():
    if request.method == "POST":
        file = request.files.get("audio")
        if not file:
            flash("No audio uploaded")
            return redirect("/check_audio")

        path = os.path.join(UPLOAD_FOLDER, "temp.wav")
        file.save(path)

        label, confidence = detect_ai_audio(path)
        os.remove(path)

        return render_template("result_audio.html", label=label, confidence=f"{confidence}%")

    return render_template("check_audio.html")

@app.route("/check_video", methods=["GET", "POST"])
def check_video():
    if request.method == "POST":
        file = request.files.get("video")
        if not file:
            flash("No video uploaded")
            return redirect("/check_video")

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        label, confidence = detect_ai_video(path)
        os.remove(path)

        return render_template("result_video.html", label=label, confidence=f"{confidence}%")

    return render_template("check_video.html")

# =================================================
# RUN
# =================================================
if __name__ == "__main__":
    print("🚀 AI Detection App Running (ENSEMBLE MODE)")
    app.run(debug=True)