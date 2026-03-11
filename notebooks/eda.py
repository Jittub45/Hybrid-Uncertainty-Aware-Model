"""
eda.py — Exploratory Data Analysis for the Crop Recommendation dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import FEATURES, MODELS_DIR


def run_eda(df: pd.DataFrame) -> None:
    print("\n[2] Running EDA ...")

    print(f"    Columns      : {list(df.columns)}")
    print(f"    Missing vals : {df.isnull().sum().sum()}")
    print(f"    Duplicates   : {df.duplicated().sum()}")
    print(f"    Crops        : {df['label'].nunique()}  ({list(df['label'].unique())})")
    print("\n    Descriptive statistics:")
    print(df.describe().to_string())

    _plot_crop_distribution(df)
    _plot_feature_histograms(df)
    _plot_boxplots(df)
    _plot_correlation_heatmap(df)


def _plot_crop_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    counts = df["label"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax)
    ax.set_title("Crop Distribution", fontweight="bold")
    ax.set_xlabel("Crop")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "crop_distribution.png"), dpi=150)
    plt.close()
    print("    Saved → models/crop_distribution.png")


def _plot_feature_histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, col in enumerate(FEATURES):
        axes[i].hist(df[col], bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        axes[i].set_title(f"Distribution of {col}", fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    axes[-1].set_visible(False)
    plt.suptitle("Feature Distributions", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "feature_distributions.png"), dpi=150)
    plt.close()
    print("    Saved → models/feature_distributions.png")


def _plot_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, col in enumerate(FEATURES):
        sns.boxplot(y=df[col], ax=axes[i], color="lightcoral")
        axes[i].set_title(f"Boxplot: {col}", fontweight="bold")
    axes[-1].set_visible(False)
    plt.suptitle("Box Plots — Outlier Detection", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "boxplots.png"), dpi=150)
    plt.close()
    print("    Saved → models/boxplots.png")


def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    corr = df[FEATURES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        mask=mask, linewidths=0.5, square=True,
        cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print("    Saved → models/correlation_heatmap.png")
