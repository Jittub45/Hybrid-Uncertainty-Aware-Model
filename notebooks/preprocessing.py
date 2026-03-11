"""
preprocessing.py — Encode labels, split data, and scale features.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import FEATURES


def preprocess(df: pd.DataFrame, feature_cols: list = None):
    """
    Returns:
        X, y, X_train, X_test, y_train, y_test,
        X_train_sc, X_test_sc, scaler, le
    """
    print("\n[3] Preprocessing ...")

    if feature_cols is None:
        feature_cols = FEATURES

    le = LabelEncoder()
    y  = le.fit_transform(df["label"])
    X  = df[feature_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"    Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"    Classes: {list(le.classes_)}")

    return X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler, le
