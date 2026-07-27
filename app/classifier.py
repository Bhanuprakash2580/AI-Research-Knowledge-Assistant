import os
from pathlib import Path
import importlib

try:
    tf = importlib.import_module("tensorflow")
    keras_layers = importlib.import_module("tensorflow.keras.layers")
    keras = importlib.import_module("tensorflow.keras")
    TextVectorization = keras_layers.TextVectorization
    Dense = keras_layers.Dense
    Sequential = keras.Sequential
except Exception:  # pragma: no cover - optional dependency
    tf = None
    keras_layers = None
    keras = None
    TextVectorization = Dense = Sequential = None


MODEL_PATH = os.getenv("TF_CLASSIFIER_PATH", "models/classifier.h5")


def load_classifier():
    if tf is None:
        return None
    p = Path(MODEL_PATH)
    if not p.exists():
        return None
    return tf.keras.models.load_model(str(p))


def predict_category(text: str):
    if tf is None:
        lowered = (text or "").lower()
        if "vision" in lowered or "image" in lowered or "camera" in lowered:
            category = "CV"
        elif "language" in lowered or "nlp" in lowered or "text" in lowered:
            category = "NLP"
        elif "robot" in lowered or "control" in lowered:
            category = "Robotics"
        elif "security" in lowered or "attack" in lowered or "malware" in lowered:
            category = "Security"
        elif "cloud" in lowered or "server" in lowered or "deploy" in lowered:
            category = "Cloud"
        elif "learning" in lowered or "model" in lowered or "train" in lowered or "dataset" in lowered or "prediction" in lowered:
            category = "ML"
        else:
            category = "AI"
        return {"category": category, "confidence": 0.6, "note": "TensorFlow not installed; used heuristic fallback"}

    model = load_classifier()
    if model is None:
        return {"category": "unknown", "confidence": 0.0, "note": "No classifier trained"}
    pred = model.predict([text])
    idx = int(pred.argmax(axis=1)[0])
    mapping = ["AI", "ML", "CV", "NLP", "Robotics", "Security", "Cloud"]
    cat = mapping[idx] if idx < len(mapping) else "other"
    return {"category": cat, "confidence": float(pred.max())}
