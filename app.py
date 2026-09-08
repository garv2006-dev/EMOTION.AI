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

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"status": "error", "message": "Endpoint not found."}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": "error", "message": "Internal server error occurred."}), 500


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logisctic_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

EMOTIONS = ["anger", "fear", "joy", "love", "sadness", "surprise"]


def create_fallback_model():
    """Train a baseline TF-IDF + Logistic Regression model if pickle files fail to load."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    training_data = [
        ("i am so happy delighted excited joyful ecstatic thrilled glad cheerful", "joy"),
        ("this is wonderful fantastic amazing awesome great delightful news", "joy"),
        ("i cherish every single moment with you my love affection passion", "love"),
        ("i love you so much adoration warmth tenderness fondness heart caring", "love"),
        ("i am terrified scared afraid nervous frightened anxious horrified panic fear", "fear"),
        ("this scary dangerous situation gives me anxiety panic fear dread horror", "fear"),
        ("i am feeling hopeless lonely depressed extremely sad gloomy heartbroken sorrowful", "sadness"),
        ("crying unhappy heartbroken disappointed sorrowful grief miserable depressed", "sadness"),
        ("i am furious angry outraged mad annoyed enraged hostile bitter", "anger"),
        ("i hate this disgusting lie disrespect betrayal insult furious anger rage", "anger"),
        ("wow astonishing incredible unexpected surprising shock amazed stunned bewildered", "surprise"),
        ("i was totally amazed and astonished by this unexpected news surprise", "surprise")
    ]
    texts, labels = zip(*training_data)
    label_to_idx = {emo: i for i, emo in enumerate(EMOTIONS)}
    y = [label_to_idx[l] for l in labels]

    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    return clf, vec


# Load ML artifacts safely with automatic fallback
MODEL_LOAD_ERROR = None
MODEL_SOURCE = "pickle"
try:
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with VECTORIZER_PATH.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    print(f"[Warning] Failed to load pickled model files: {e}. Initializing fallback model...")
    try:
        model, vectorizer = create_fallback_model()
        MODEL_LOADED = True
        MODEL_SOURCE = "fallback"
        print("[Info] Fallback ML model initialized successfully.")
    except Exception as fallback_err:
        model = None
        vectorizer = None
        MODEL_LOADED = False
        MODEL_SOURCE = "none"
        print(f"[Error] Failed to initialize fallback model: {fallback_err}")

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
        "model_source": MODEL_SOURCE,
        "model_error": MODEL_LOAD_ERROR,
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
        err_msg = f"ML model is not loaded. Details: {MODEL_LOAD_ERROR or 'Unknown error'}"
        return jsonify({"status": "error", "message": err_msg}), 500

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
