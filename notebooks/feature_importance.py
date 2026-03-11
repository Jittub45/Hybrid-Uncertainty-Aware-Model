"""
feature_importance.py — Compute and plot feature importances using Random Forest.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from config import FEATURES, MODELS_DIR


def feature_importance(X_train: np.ndarray, y_train: np.ndarray) -> None:
    print("\n[4] Feature importance ...")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    feat_df = (
        pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
    )
    print(feat_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="mako", ax=ax)
    ax.set_title("Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("    Saved → models/feature_importance.png")
