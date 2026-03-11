"""
train.py — Crop Recommendation System: Main pipeline orchestrator.
============================================================
Run: python notebooks/train.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib

from data_loader import load_data
from eda import run_eda
from feature_engineering import engineer_features
from preprocessing import preprocess
from feature_importance import feature_importance
from model_training import train_models
from tuning import tune_random_forest
from hybrid_model import train_hybrid_model
from explainability import run_shap_analysis
from save_model import save_artifacts, sanity_check
from config import FEATURES, ALL_FEATURES, MODELS_DIR


def main():
    print("=" * 60)
    print("   CROP RECOMMENDATION SYSTEM — FULL PIPELINE")
    print("   (Paper: Hybrid Uncertainty-Aware Model)")
    print("=" * 60)

    # ── Phase 1: Load & EDA ───────────────────────────────────
    df = load_data()
    run_eda(df)

    # ── Phase 2: Baseline — original 7 features ───────────────
    print("\n--- BASELINE (7 features) ---")
    X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler, le = preprocess(df, FEATURES)
    feature_importance(X_train, y_train)
    models, results_df, best_name = train_models(
        X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, le
    )
    tuned_rf, tuned_acc = tune_random_forest(X_train, y_train, X_test, y_test)
    final_name, final_acc = save_artifacts(
        models, results_df, best_name,
        tuned_rf, tuned_acc,
        X, y, scaler, le
    )
    sanity_check()

    # ── Phase 3: Feature Engineering (paper Section II-B) ─────
    print("\n--- FEATURE ENGINEERING (12 features) ---")
    df_eng = engineer_features(df)
    (X_eng, y_eng,
     X_tr_eng, X_te_eng,
     y_tr_eng, y_te_eng,
     X_tr_sc_eng, X_te_sc_eng,
     scaler_eng, le_eng) = preprocess(df_eng, ALL_FEATURES)

    # ── Phase 4: Hybrid Stacked Ensemble (paper Section II-C) ─
    print("\n--- HYBRID STACKED ENSEMBLE (LightGBM + XGBoost + CatBoost → LR) ---")
    hybrid_model, hybrid_acc = train_hybrid_model(
        X_tr_eng, X_te_eng, y_tr_eng, y_te_eng, le_eng
    )

    # ── Phase 5: SHAP Explainability (paper Section II-E) ─────
    # Use LightGBM base learner for SHAP (tree-based, faster)
    lgbm_base = hybrid_model.named_estimators_["lgbm"]
    run_shap_analysis(lgbm_base, X_tr_eng, X_te_eng, ALL_FEATURES)

    # ── Phase 6: Save hybrid model if it beats baseline ───────
    if hybrid_acc > final_acc:
        hybrid_model.fit(X_eng, y_eng)
        joblib.dump(hybrid_model, os.path.join(MODELS_DIR, "crop_model.pkl"))
        joblib.dump(scaler_eng,   os.path.join(MODELS_DIR, "scaler.pkl"))
        joblib.dump(le_eng,       os.path.join(MODELS_DIR, "label_encoder.pkl"))
        final_name = "Hybrid Stacked Ensemble (LightGBM+XGBoost+CatBoost→LR)"
        final_acc  = hybrid_acc
        print(f"\n    Hybrid model saved (beats baseline) → models/crop_model.pkl")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETE")
    print(f"   Baseline Model   : {best_name}  ({tuned_acc*100:.2f}%)")
    print(f"   Hybrid Model     : Stacked Ensemble  ({hybrid_acc*100:.2f}%)")
    print(f"   Final Model      : {final_name}")
    print(f"   Final Accuracy   : {final_acc*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
