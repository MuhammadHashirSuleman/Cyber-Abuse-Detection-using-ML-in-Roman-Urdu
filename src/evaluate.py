# src/evaluate.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from .config import MODELS_DIR, OUTPUTS_DIR, DATA_PATH
from .utils import load_data, safe_string_labels
from .preprocess import preprocess_dataframe


MODEL_FILES = {
    'nb': 'model_nb.joblib',
    'svm': 'model_svm.joblib',
    'rf': 'model_rf.joblib'
}


def plot_confusion(y_true, y_pred, labels, outpath):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.matshow(cm, cmap='viridis')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center', color='white')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.savefig(outpath)
    plt.close()


def evaluate_all():
    # load data
    df = load_data(DATA_PATH)
    df = safe_string_labels(df, 'label')
    df = preprocess_dataframe(df, text_col='tweets', label_col='label')

    # split the same way as training: we will re-split for evaluation (note: for exact reproduction use saved test split)
    from sklearn.model_selection import train_test_split
    X_text = df['cleaned'].values
    y = df['label'].values
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.25, random_state=42, stratify=y)

    # load vectorizer
    vec = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
    X_test = vec.transform(X_test_text)

    metrics = []
    labels = np.unique(y_test)

    for key, fname in MODEL_FILES.items():
        model = joblib.load(os.path.join(MODELS_DIR, fname))
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"Model: {key} -- acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")

        # confusion
        outpath = os.path.join(OUTPUTS_DIR, f'confusion_{key}.png')
        plot_confusion(y_test, y_pred, labels=labels, outpath=outpath)

        metrics.append({'model': key, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})

    dfm = pd.DataFrame(metrics)
    dfm.to_csv(os.path.join(OUTPUTS_DIR, 'metrics_summary.csv'), index=False)
    print('Evaluation finished. Confusion matrices and metrics saved in outputs/')

if __name__ == '__main__':
    evaluate_all()