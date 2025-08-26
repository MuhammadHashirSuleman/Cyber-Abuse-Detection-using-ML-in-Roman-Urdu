# src/train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

from src.config import DATA_PATH, MODELS_DIR
from src.preprocess import preprocess_dataframe
from src.feature_extraction import build_vectorizer, fit_vectorizer
from src.utils import load_data, safe_string_labels


def train_all():
    df = load_data(DATA_PATH)
    df = safe_string_labels(df, 'label')
    df = preprocess_dataframe(df, text_col='tweets', label_col='label')

    X_text = df['cleaned'].values
    y = df['label'].values

    # split
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.25, random_state=42, stratify=y)

    # vectorizer
    vec = build_vectorizer(ngram_range=(1,2), max_features=15000)
    X_train, vec = fit_vectorizer(vec, X_train_text, save_path=os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))

    # transform test
    X_test = vec.transform(X_test_text)

    results = {}

    # 1) MultinomialNB
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    joblib.dump(nb, os.path.join(MODELS_DIR, 'model_nb.joblib'))
    results['nb'] = (nb, y_test, y_pred_nb, acc_nb)

    # 2) LinearSVC
    svm = LinearSVC(max_iter=10000)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    joblib.dump(svm, os.path.join(MODELS_DIR, 'model_svm.joblib'))
    results['svm'] = (svm, y_test, y_pred_svm, acc_svm)

    # 3) RandomForest
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    joblib.dump(rf, os.path.join(MODELS_DIR, 'model_rf.joblib'))
    results['rf'] = (rf, y_test, y_pred_rf, acc_rf)

    # Save a small summary
    with open(os.path.join(MODELS_DIR, 'train_results.txt'), 'w') as f:
        for name, (_, _, _, acc) in results.items():
            f.write(f"{name}: {acc}\n")

    print('Training completed. Models saved to models/ folder.')
    return results

if __name__ == '__main__':
    train_all()