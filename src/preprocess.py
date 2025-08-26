# src/preprocess.py
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

nltk.download('stopwords')

# Roman Urdu-specific simple normalization map (extend as needed)
NORMALIZATION_MAP = {
    'hai': 'hai',
    'hy': 'hai',
    'haii': 'hai',
    'han': 'hain',
    'nhi': 'nahi',
    'nahi': 'nahi'
}

STOPWORDS = set(stopwords.words('english'))  # keep english stopwords for now; you may add Roman Urdu stopwords


def normalize_token(tok: str) -> str:
    tok = tok.lower()
    if tok in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[tok]
    return tok


def clean_text(text: str) -> str:
    text = str(text) if text is not None else ''
    if pd.isna(text):
        return ''
    # remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # remove mentions and hashtags (or keep hashtags if desired)
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'#[A-Za-z0-9_]+', ' ', text)
    # remove punctuation and numbers (keep letters only)
    text = re.sub(r'[^A-Za-z\s]', ' ', text)
    # collapse repeated letters: e.g., coooool -> cool
    text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # token-based normalization
    tokens = [normalize_token(t) for t in text.split() if t.lower() not in STOPWORDS]
    return ' '.join(tokens)


def preprocess_dataframe(df, text_col='tweets', label_col='label'):
    # Handle missing values
    missing_before = df[text_col].isna().sum()
    print(f"Missing values in '{text_col}': {missing_before}")
    df[text_col] = df[text_col].fillna('')

    # Remove duplicates
    dup_before = df.duplicated().sum()
    print(f"Duplicate rows before: {dup_before}")
    df = df.drop_duplicates()
    dup_after = df.duplicated().sum()

    # Basic noise detection: very short rows
    short_texts = df[df[text_col].str.len() < 2]
    print(f"Short texts (<2 chars): {len(short_texts)}")
    # We will keep them, but they can be removed if desired

    # Clean text
    df['cleaned'] = df[text_col].apply(clean_text)

    # Ensure label strings
    df[label_col] = df[label_col].astype(str).str.strip()

    return df