"""
feature_engineering.py — Add domain-knowledge ratio features as described in the paper.

Paper Section II-B:
  "more features are obtained such as seasonal index, rolling rainfall averages,
   and market-based profitability ratios ... to learn not only agronomic suitability
   but also economic feasibility."

We derive the agronomic ratios available from this dataset:
  N_P_ratio, N_K_ratio, P_K_ratio, NPK_sum, temp_humid
"""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived ratio features to the dataframe. Returns a new dataframe."""
    df = df.copy()
    df["N_P_ratio"]  = df["N"] / (df["P"] + 1)
    df["N_K_ratio"]  = df["N"] / (df["K"] + 1)
    df["P_K_ratio"]  = df["P"] / (df["K"] + 1)
    df["NPK_sum"]    = df["N"] + df["P"] + df["K"]
    df["temp_humid"] = df["temperature"] * df["humidity"]

    print("\n[FE] Feature engineering complete.")
    print(f"     Original features : 7")
    print(f"     Engineered features added : 5  (N_P_ratio, N_K_ratio, P_K_ratio, NPK_sum, temp_humid)")
    print(f"     Total features now : {len([c for c in df.columns if c != 'label'])}")
    return df
