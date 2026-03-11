"""
config.py — Shared constants for the Crop Recommendation pipeline.
"""

import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Features after engineering (base + derived)
ALL_FEATURES = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
    "N_P_ratio", "N_K_ratio", "P_K_ratio", "NPK_sum", "temp_humid",
]
