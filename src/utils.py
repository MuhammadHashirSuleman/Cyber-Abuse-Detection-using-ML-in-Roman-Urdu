# src/utils.py
import pandas as pd
import numpy as np


def load_data(path):
    # Prevent "NA" or "na" from being treated as NaN
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, na_values=[])
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", keep_default_na=False, na_values=[])
    return df



def safe_string_labels(df, label_col='label'):
    # Ensure labels like 'NA' are not interpreted as NaN
    df[label_col] = df[label_col].astype(str).str.strip()
    return df