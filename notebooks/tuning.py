"""
tuning.py — Hyperparameter tuning for Random Forest using GridSearchCV.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Run GridSearchCV on Random Forest.

    Returns:
        best_estimator (RandomForestClassifier), tuned_acc (float)
    """
    print("\n[6] Hyperparameter tuning (Random Forest) ...")

    param_grid = {
        "n_estimators"     : [100, 200, 300],
        "max_depth"        : [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf" : [1, 2],
    }

    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1,
    )
    gs.fit(X_train, y_train)

    tuned_acc = accuracy_score(y_test, gs.best_estimator_.predict(X_test))
    print(f"    Best params   : {gs.best_params_}")
    print(f"    Best CV acc   : {gs.best_score_:.4f}")
    print(f"    Tuned test acc: {tuned_acc:.4f}")

    return gs.best_estimator_, tuned_acc
