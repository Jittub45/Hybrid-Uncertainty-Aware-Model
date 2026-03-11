"""
data_loader.py — Load the crop recommendation dataset.
"""

import pandas as pd
from config import DATA_PATH


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"[1] Dataset loaded  →  shape: {df.shape}")
    return df
