"""
explainability.py — SHAP-based feature explanations as described in the research paper.

Paper Section II-E (Decision Transparency and Explainability):
  "the final ensemble is examined with the help of SHAP (SHapley Additive exPlanations).
   SHAP values are a measure of the contribution made by each feature (e.g., nitrogen,
   rainfall, market price) towards the prediction made."
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from config import MODELS_DIR


def run_shap_analysis(model, X_train: np.ndarray, X_test: np.ndarray, feature_names: list) -> None:
    """
    Generate SHAP summary plot and bar plot for the given model.
    Uses TreeExplainer for tree-based models (LightGBM, XGBoost, CatBoost, RF).
    """
    print("\n[SHAP] Running SHAP explainability analysis ...")

    # Use a sample for speed (max 500 rows)
    sample_size = min(500, X_train.shape[0])
    X_sample    = X_train[:sample_size]

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        # Fallback to KernelExplainer for non-tree models (e.g., stacking)
        print("    TreeExplainer failed, falling back to KernelExplainer (slower) ...")
        background  = shap.kmeans(X_sample, 10)
        explainer   = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test[:100])

    # Handle multi-class: shap_values is a list — use mean absolute across classes
    if isinstance(shap_values, list):
        shap_arr = np.array(shap_values)           # (n_classes, n_samples, n_features)
        mean_abs = np.abs(shap_arr).mean(axis=(0, 1))  # (n_features,)
        # For summary plot, pick class 0 as representative
        sv_for_plot = shap_values[0]
    else:
        mean_abs    = np.abs(shap_values).mean(axis=0)
        sv_for_plot = shap_values

    # --- Bar plot: global feature importance ---
    import pandas as pd
    feat_imp = (
        pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs})
        .sort_values("Mean |SHAP|", ascending=False)
    )
    print("\n    SHAP Feature Importance (Mean |SHAP value|):")
    print(feat_imp.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    import seaborn as sns
    sns.barplot(data=feat_imp, x="Mean |SHAP|", y="Feature", palette="flare", ax=ax)
    ax.set_title("SHAP Global Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "shap_feature_importance.png"), dpi=150)
    plt.close()
    print("    Saved → models/shap_feature_importance.png")

    # --- Beeswarm summary plot ---
    try:
        plt.figure(figsize=(12, 7))
        shap.summary_plot(
            sv_for_plot, X_test,
            feature_names=feature_names,
            show=False, plot_size=(12, 7),
        )
        plt.title("SHAP Summary Plot (Beeswarm)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, "shap_summary_plot.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("    Saved → models/shap_summary_plot.png")
    except Exception as e:
        print(f"    Warning: beeswarm plot skipped ({e})")
