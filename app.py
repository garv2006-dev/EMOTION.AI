import os
from pathlib import Path
import pickle
import re
import string
import warnings
from flask import Flask, jsonify, render_template, request

# Suppress scikit-learn version warnings when unpickling
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logisctic_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

# Load ML artifacts safely
try:
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with VECTORIZER_PATH.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    MODEL_LOADED = True
except Exception as e:
    model = None
    vectorizer = None
    MODEL_LOADED = False
    print(f"[Warning] Failed to load model files: {e}")

EMOTION_META = {
    "anger": {
        "name": "Anger",
        "emoji": "😠",
        "color": "#ef4444",
        "bg_glow": "rgba(239, 68, 68, 0.25)",
        "description": "Feeling strong displeasure, hostility, or frustration."
    },
    "fear": {
        "name": "Fear",
        "emoji": "😨",
        "color": "#a855f7",
        "bg_glow": "rgba(168, 85, 247, 0.25)",
        "description": "Apprehension or anxiety caused by anticipation of danger."
    },
    "joy": {
        "name": "Joy",
        "emoji": "✨",
        "color": "#f59e0b",
        "bg_glow": "rgba(245, 158, 11, 0.25)",
        "description": "High energy, happiness, excitement, and contentment."
    },
    "love": {
        "name": "Love",
        "emoji": "💖",
        "color": "#ec4899",
        "bg_glow": "rgba(236, 72, 153, 0.25)",
        "description": "Deep affection, warmth, fondness, or tenderness."
    },
    "sadness": {
        "name": "Sadness",
        "emoji": "😢",
        "color": "#3b82f6",
        "bg_glow": "rgba(59, 130, 246, 0.25)",
        "description": "Feeling down, sorrowful, unhappy, or heartbroken."
    },
    "surprise": {
        "name": "Surprise",
        "emoji": "😲",
        "color": "#06b6d4",
        "bg_glow": "rgba(6, 182, 212, 0.25)",
        "description": "Feeling amazed, startled, or encountering the unexpected."
    }
}

EMOTIONS = ["anger", "fear", "joy", "love", "sadness", "surprise"]

SAMPLE_SENTENCES = [
    {"text": "I am so happy and excited for today! Everything is working out perfectly.", "category": "joy"},
    {"text": "I am really nervous and terrified about the upcoming exam tomorrow.", "category": "fear"},
    {"text": "I am feeling so hopeless, lonely, and sad about how things turned out.", "category": "sadness"},
    {"text": "I cannot believe you lied to me, I am absolutely furious!", "category": "anger"},
    {"text": "I cherish every single moment we spend together, you mean everything to me.", "category": "love"},
    {"text": "Wow, I was totally astonished when I received that secret gift!", "category": "surprise"}
]


def clean_text(text: str) -> str:
    """Preprocess text by lowercasing, removing punctuation and digits."""
    if not text:
        return ""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\d+", "", text).strip()


@app.route("/")
def home():
    """Render main web application interface."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "degraded",
        "model_loaded": MODEL_LOADED,
        "emotions_supported": len(EMOTIONS)
    })


@app.route("/api/examples", methods=["GET"])
def examples():
    """Return benchmark example sentences."""
    return jsonify({
        "status": "success",
        "examples": SAMPLE_SENTENCES
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict emotional probabilities for provided text."""
    if not MODEL_LOADED:
        return jsonify({"status": "error", "message": "ML model is not loaded"}), 500

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"status": "error", "message": "Please enter non-empty text to analyze."}), 400

    cleaned = clean_text(text)
    if not cleaned:
        return jsonify({"status": "error", "message": "Text contains no valid words after cleaning."}), 400

    # Vectorize and predict probabilities
    features = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(features)[0]

    prob_dict = {EMOTIONS[i]: float(probabilities[i]) for i in range(len(EMOTIONS))}
    
    # Sort emotions by highest probability
    sorted_emotions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    top_emotion_key, top_prob = sorted_emotions[0]

    ranked = []
    for key, prob in sorted_emotions:
        meta = EMOTION_META.get(key, {})
        ranked.append({
            "key": key,
            "name": meta.get("name", key.capitalize()),
            "emoji": meta.get("emoji", "🎭"),
            "color": meta.get("color", "#6366f1"),
            "probability": prob,
            "percentage": round(prob * 100, 1),
            "description": meta.get("description", "")
        })

    top_meta = EMOTION_META.get(top_emotion_key, {})

    # Extract word tokens comparison
    original_words = text.split()
    cleaned_words = cleaned.split()

    return jsonify({
        "status": "success",
        "input_text": text,
        "cleaned_text": cleaned,
        "word_count": len(original_words),
        "top_emotion": {
            "key": top_emotion_key,
            "name": top_meta.get("name", top_emotion_key.capitalize()),
            "emoji": top_meta.get("emoji", "🎭"),
            "color": top_meta.get("color", "#6366f1"),
            "bg_glow": top_meta.get("bg_glow", "rgba(99, 102, 241, 0.25)"),
            "confidence": round(top_prob * 100, 1),
            "description": top_meta.get("description", "")
        },
        "probabilities": prob_dict,
        "ranked_emotions": ranked,
        "word_tokens": {
            "original": original_words,
            "cleaned": cleaned_words
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
