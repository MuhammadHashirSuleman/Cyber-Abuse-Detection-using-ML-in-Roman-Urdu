# src/config.py
import os

# Base directory = root of your project (goes up one from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directory and file
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "cyber_abuse_data_roman_urdu.csv")

# Models directory and files
MODELS_DIR = os.path.join(BASE_DIR, "models")
NB_PATH = os.path.join(MODELS_DIR, "model_nb.joblib")
VEC_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")

# Outputs directory
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
