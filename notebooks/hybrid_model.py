"""
hybrid_model.py — Hybrid Stacked Ensemble as described in the research paper.

Paper Section II-C (Model Development):
  "Base learners are LightGBM, XGBoost, and CatBoost"
  "TabNet is a second learner to capture the interactions between complicated features"
  "The ultimate suggestions are the result of integrating the results of every base
   model, where a meta-learner is trained based on logistic regression."

We implement: LightGBM + XGBoost + CatBoost → StackingClassifier (LR meta-learner)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from config import MODELS_DIR


def build_stacking_model() -> StackingClassifier:
    """
    Build the paper's hybrid stacked ensemble:
      Base learners  : LightGBM, XGBoost, CatBoost
      Meta-learner   : Logistic Regression
    """
    base_learners = [
        ("lgbm", LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            num_leaves=63, random_state=42, verbose=-1
        )),
        ("xgb", XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            eval_metric="mlogloss", random_state=42, verbosity=0
        )),
        ("catboost", CatBoostClassifier(
            iterations=300, learning_rate=0.05,
            depth=6, random_state=42, verbose=0
        )),
    ]
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)

    stacking = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    return stacking


def train_hybrid_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
):
    """
    Train and evaluate the hybrid stacking model.

    Returns:
        stacking_model, test_accuracy
    """
    print("\n[HYBRID] Training Hybrid Stacked Ensemble (LightGBM + XGBoost + CatBoost → LR) ...")

    stacking = build_stacking_model()
    stacking.fit(X_train, y_train)

    y_pred = stacking.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cv     = cross_val_score(stacking, X_train, y_train, cv=5, scoring="accuracy").mean()

    print(f"    Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    CV  Accuracy  : {cv:.4f}  ({cv*100:.2f}%)")
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    _plot_confusion_matrix(y_test, y_pred, le.classes_)

    return stacking, acc


def _plot_confusion_matrix(y_test, y_pred, class_names) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Confusion Matrix — Hybrid Stacked Ensemble", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix_hybrid.png"), dpi=150)
    plt.close()
    print("    Saved → models/confusion_matrix_hybrid.png")
