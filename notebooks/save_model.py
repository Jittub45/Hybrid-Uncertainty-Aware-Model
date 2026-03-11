"""
save_model.py — Save final model artifacts and run a sanity check.
"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import MODELS_DIR


def save_artifacts(
    models: dict,
    results_df,
    best_name: str,
    tuned_rf,
    tuned_acc: float,
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    le: LabelEncoder,
):
    """
    Compare tuned RF vs original best model, retrain on full data, and save.

    Returns:
        final_name (str), final_acc (float)
    """
    print("\n[7] Saving artifacts ...")

    orig_acc = results_df.iloc[0]["Test Accuracy"]

    if tuned_acc >= orig_acc:
        final_model = tuned_rf
        final_name  = "Tuned Random Forest"
    else:
        final_model = models[best_name]
        final_name  = best_name

    # Retrain on full dataset before saving
    final_model.fit(X, y)

    joblib.dump(final_model, os.path.join(MODELS_DIR, "crop_model.pkl"))
    joblib.dump(scaler,      os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(le,          os.path.join(MODELS_DIR, "label_encoder.pkl"))

    final_acc = max(tuned_acc, orig_acc)
    print(f"    Final model : {final_name}  (accuracy: {final_acc*100:.2f}%)")
    print("    Saved → models/crop_model.pkl")
    print("    Saved → models/scaler.pkl")
    print("    Saved → models/label_encoder.pkl")

    return final_name, final_acc


def sanity_check() -> None:
    print("\n[8] Sanity check ...")

    model = joblib.load(os.path.join(MODELS_DIR, "crop_model.pkl"))
    le    = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    # Known rice-like sample from dataset
    sample = np.array([[90, 42, 43, 20.88, 82.00, 6.50, 202.94]])
    crop   = le.inverse_transform(model.predict(sample))[0]
    print(f"    Test input [rice-like]  →  predicted: {crop}")
