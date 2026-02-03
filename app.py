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
import librosa # type: ignore
import torch # type: ignore
import torchaudio # type: ignore
from moviepy.editor import VideoFileClip # type: ignore
import warnings
warnings.filterwarnings("ignore")

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

# -------------------- LOAD AUDIO/VIDEO DETECTION MODELS --------------------
print("🎵 Loading AI audio/video detection models...")

AUDIO_MODEL_1 = None

try:
    AUDIO_MODEL_1 = pipeline(
        "audio-classification",
        model="Hemgg/Deepfake-audio-detection",
        device=0 if torch.cuda.is_available() else -1
    )
    print("✅ AST audio model loaded")
except Exception as e:
    print("❌ Error loading AST model:", e)

# Video detection will use frame-based analysis + audio
print("✅ Audio/video detection setup complete")

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

# -------------------- AUDIO EXTRACTION & DETECTION --------------------
def extract_audio_features(audio_path):
    """Extract audio features for deepfake detection"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract features (MFCCs, spectral features)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Simple statistical features
        features = []
        for mfcc in mfccs:
            features.extend([np.mean(mfcc), np.std(mfcc), np.max(mfcc), np.min(mfcc)])
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        
        return np.array(features)
    except:
        return None

def detect_ai_audio(audio_path):
    try:
        # -------- Load & normalize audio --------
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # -------- Model prediction --------
        outputs = AUDIO_MODEL_1(audio)
        print("MODEL OUTPUT:", outputs)

        fake_score = 0.0

        for o in outputs:
            label = o["label"].lower()

            if label in ["aivoice", "ai_voice", "fake", "spoof"]:
                fake_score = o["score"]
                break
            elif label in ["humanvoice", "human", "real", "bonafide"]:
                fake_score = 1.0 - o["score"]

        # -------- Optional heuristic tweak --------
        smoothness = np.std(audio)
        if smoothness < 0.01:
            fake_score = min(fake_score + 0.05, 1.0)

        label = "AI" if fake_score >= 0.5 else "REAL"
        confidence = round(fake_score * 100, 2)

        return label, confidence

    except Exception as e:
        print("Audio detection error:", e)
        return "REAL", 50.0

# -------------------- VIDEO DETECTION LOGIC --------------------
def detect_ai_video(video_path):
    """Detect AI-generated video by analyzing frames + audio"""
    try:
        # Extract first frame for image analysis
        video = VideoFileClip(video_path)
        frame_path = os.path.join(app.config["UPLOAD_FOLDER"], "temp_frame.jpg")
        video.save_frame(frame_path, t=1)  # Save frame at 1 second
        
        # Analyze frame
        frame_label, frame_conf = detect_ai_image(frame_path)
        
        # Extract audio if available
        audio_label, audio_conf = "REAL", 50.0
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], "temp_audio.wav")
        if video.audio:
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            audio_label, audio_conf = detect_ai_audio(audio_path)
        
        # Clean up
        video.close()
        if os.path.exists(frame_path):
            os.remove(frame_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Combine scores
        votes = [frame_label, audio_label]
        final_label = max(set(votes), key=votes.count)
        avg_confidence = round((frame_conf + audio_conf) / 2, 2)
        
        print(f"Frame: {frame_label} ({frame_conf}%), Audio: {audio_label} ({audio_conf}%)")
        
        return final_label, avg_confidence
        
    except Exception as e:
        print(f"Video detection error: {e}")
        return "REAL", 50.0

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

# ---------- AUDIO CHECK ----------
@app.route("/check_audio", methods=["GET", "POST"])
def check_audio():
    if request.method == "POST":
        # More robust file checking
        if "audio" not in request.files:
            flash("No audio file in request")
            return render_template("check_audio.html")
        
        file = request.files["audio"]
        
        # Check if file object exists and has content
        if file is None:
            flash("No audio file selected")
            return render_template("check_audio.html")
            
        if not file.filename or file.filename == "":
            flash("No audio file selected")
            return render_template("check_audio.html")

        # Secure filename and check extension
        filename = os.path.splitext(file.filename)[0] + ".wav"  # Force .wav for consistency
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        try:
            file.save(filepath)
        except Exception as e:
            flash(f"Failed to save file: {str(e)}")
            return render_template("check_audio.html")

        # Verify file was saved and exists
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            flash("File upload failed - empty file")
            return render_template("check_audio.html")

        try:
            label, confidence = detect_ai_audio(filepath)
        except Exception as e:
            flash(f"Analysis failed: {str(e)}")
            os.remove(filepath)
            return render_template("check_audio.html")
        
        # Clean up
        os.remove(filepath)

        print(label, confidence)
        return render_template(
            "result_audio.html",
            label=label,
            confidence=f"{confidence}%"
        )

    return render_template("check_audio.html")

# ---------- VIDEO CHECK ----------
@app.route("/check_video", methods=["GET", "POST"])
def check_video():
    if request.method == "POST":
        print(request.files)
        if "video" not in request.files:
            flash("No video file uploaded")
            print("No video file uploaded")
            return redirect("/check_video")

        file = request.files["video"]
        if file.filename == "":
            flash("No video selected")
            print("No video selected")
            return redirect("/check_video")

        # Support common video formats
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        ext = os.path.splitext(file.filename.lower())[1]
        if ext not in allowed_extensions:
            flash("Please upload MP4, AVI, MOV, MKV, or WebM files")
            return redirect("/check_video")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        label, confidence = detect_ai_video(filepath)
        os.remove(filepath)

        return render_template(
            "result_video.html",
            label=label,
            confidence=f"{confidence}%"
        )

    return render_template("check_video.html")

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    print("🚀 AI Detection Flask App Running")
    app.run(debug=True)
