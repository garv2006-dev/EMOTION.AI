from pathlib import Path
import pickle
import re
import string
import warnings
import gradio as gr

# Suppress scikit-learn version warnings when unpickling
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logisctic_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)
with VECTORIZER_PATH.open("rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

EMOTIONS = ["anger", "fear", "joy", "love", "sadness", "surprise"]


def clean_text(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\d", "", text)


def predict_emotion(text: str):
    if not text or not text.strip():
        return {}
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(features)[0]
    
    return {EMOTIONS[i]: float(probabilities[i]) for i in range(len(EMOTIONS))}



demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(
        lines=3, 
        placeholder="Type a sentence... (e.g., I am feeling super excited and happy today!)",
        label="Input Text"
    ),
    outputs=gr.Label(num_top_classes=6, label="Detected Emotions"),
    title="🎭 Emotion Detection Model",
    description="Enter text to classify emotional tone into anger, fear, joy, love, sadness, or surprise.",
    examples=[
        ["I am so happy and excited for today!"],
        ["I am really nervous and worried about the exam tomorrow."],
        ["I am feeling really sad and hopeless lately."],
        ["I cannot believe you did that, I am so furious!"],
        ["Wow, what an incredible and unexpected surprise!"]
    ]
)

if __name__ == "__main__":
    demo.launch()

