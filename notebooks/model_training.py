"""
model_training.py — Train multiple classifiers, evaluate, and plot results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import MODELS_DIR

# Models that require scaled input
SCALE_MODELS = {"Logistic Regression", "K-Nearest Neighbors", "SVM"}


def get_models() -> dict:
    return {
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=5),
        "Decision Tree"       : DecisionTreeClassifier(random_state=42),
        "Random Forest"       : RandomForestClassifier(n_estimators=200, random_state=42),
        "Extra Trees"         : ExtraTreesClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost"             : XGBClassifier(n_estimators=200, eval_metric="mlogloss", random_state=42),
        "SVM"                 : SVC(kernel="rbf", probability=True, random_state=42),
        "Naive Bayes"         : GaussianNB(),
    }


def train_models(
    X_train, X_test, y_train, y_test,
    X_train_sc, X_test_sc,
    le: LabelEncoder,
):
    """
    Train all models, print results, save comparison chart and confusion matrix.

    Returns:
        models (dict), results_df (DataFrame), best_name (str)
    """
    print("\n[5] Training models ...")

    models  = get_models()
    results = []

    for name, model in models.items():
        Xtr, Xte = (X_train_sc, X_test_sc) if name in SCALE_MODELS else (X_train, X_test)
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        acc    = accuracy_score(y_test, y_pred)
        cv     = cross_val_score(model, Xtr, y_train, cv=5, scoring="accuracy").mean()
        results.append({"Model": name, "Test Accuracy": acc, "CV Accuracy (5-fold)": cv})
        print(f"    {name:<25} | Test: {acc:.4f} | CV: {cv:.4f}")

    results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
    print("\n    --- Ranked Results ---")
    print(results_df.to_string(index=False))

    _plot_model_comparison(results_df)

    best_name   = results_df.iloc[0]["Model"]
    best_model  = models[best_name]
    Xte_best    = X_test_sc if best_name in SCALE_MODELS else X_test
    y_pred_best = best_model.predict(Xte_best)

    print(f"\n    Best model: {best_name}")
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))
    _plot_confusion_matrix(y_test, y_pred_best, le.classes_, best_name)

    return models, results_df, best_name


def _plot_model_comparison(results_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=results_df, x="Test Accuracy", y="Model", palette="RdYlGn", orient="h", ax=ax)
    for bar in ax.patches:
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center", fontsize=10,
        )
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_xlim(0.8, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print("    Saved → models/model_comparison.png")


def _plot_confusion_matrix(y_test, y_pred, class_names, model_name: str) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("    Saved → models/confusion_matrix.png")
