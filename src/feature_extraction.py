# src/features.py
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from src.config import MODELS_DIR
import os


def build_vectorizer(ngram_range=(1,2), max_features=20000):
    vec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    return vec


def fit_vectorizer(vec, texts, save_path=None):
    X = vec.fit_transform(texts)
    if save_path:
        joblib.dump(vec, save_path)
    return X, vec


def load_vectorizer(path):
    return joblib.load(path)