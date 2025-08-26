# webapp/model_loader.py
import joblib
from pathlib import Path

# Project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

VEC_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
NB_PATH = MODELS_DIR / "model_nb.joblib"
SVM_PATH = MODELS_DIR / "model_svm.joblib"
RF_PATH = MODELS_DIR / "model_rf.joblib"

def load(model_name="nb"):
    """Load the vectorizer and the selected ML model."""
    vec = joblib.load(VEC_PATH)
    
    if model_name == "nb":
        model = joblib.load(NB_PATH)
    elif model_name == "svm":
        model = joblib.load(SVM_PATH)
    elif model_name == "rf":
        model = joblib.load(RF_PATH)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return vec, model
