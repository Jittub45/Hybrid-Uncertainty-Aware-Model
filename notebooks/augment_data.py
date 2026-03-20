"""
augment_data.py - Expand Crop_recommendation.csv to a larger synthetic dataset.

Usage:
    python notebooks/augment_data.py --target-rows 11000
"""

import argparse
import math
import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "Crop_recommendation_augmented_11000.csv")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
LABEL_COL = "label"


def _mahalanobis_sq(x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    diff = x - mean
    return np.einsum("ij,jk,ik->i", diff, inv_cov, diff)


def _synthesize_class(
    class_df: pd.DataFrame,
    n_new: int,
    rng: np.random.Generator,
    feature_min: np.ndarray,
    feature_max: np.ndarray,
    noise_scale: float,
) -> np.ndarray:
    if n_new <= 0:
        return np.empty((0, len(FEATURES)))

    real = class_df[FEATURES].to_numpy(dtype=float)
    n_real, d = real.shape

    if n_real < 3:
        idx = rng.integers(0, n_real, size=n_new)
        return real[idx]

    cov = np.cov(real, rowvar=False)
    trace = float(np.trace(cov))
    reg = max(1e-8, trace / max(1, d) * 1e-5)
    cov = cov + np.eye(d) * reg
    inv_cov = np.linalg.pinv(cov)

    mean = real.mean(axis=0)
    real_md_sq = _mahalanobis_sq(real, mean, inv_cov)
    md_sq_threshold = float(np.quantile(real_md_sq, 0.98) * 1.35)

    samples = []
    total = 0
    max_tries = 60

    for _ in range(max_tries):
        if total >= n_new:
            break

        batch = max(64, (n_new - total) * 3)
        anchor_idx = rng.integers(0, n_real, size=batch)
        anchors = real[anchor_idx]

        noise = rng.multivariate_normal(mean=np.zeros(d), cov=cov * noise_scale, size=batch)
        synth = anchors + noise
        synth = np.clip(synth, feature_min, feature_max)

        md_sq = _mahalanobis_sq(synth, mean, inv_cov)
        keep = synth[md_sq <= md_sq_threshold]

        if keep.size:
            samples.append(keep)
            total += keep.shape[0]

    if total < n_new:
        left = n_new - total
        idx = rng.integers(0, n_real, size=left)
        fallback = real[idx].copy()
        std = np.maximum(real.std(axis=0, ddof=1), 1e-6)
        fallback += rng.normal(0.0, std * 0.02, size=fallback.shape)
        fallback = np.clip(fallback, feature_min, feature_max)
        samples.append(fallback)

    out = np.vstack(samples)[:n_new]
    return out


def build_augmented_dataset(
    input_path: str,
    output_path: str,
    target_rows: int,
    random_seed: int = 42,
    noise_scale: float = 0.08,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    missing_cols = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if target_rows <= len(df):
        raise ValueError(
            f"target_rows ({target_rows}) must be greater than current rows ({len(df)})."
        )

    classes = sorted(df[LABEL_COL].unique().tolist())
    n_classes = len(classes)
    target_per_class = int(math.ceil(target_rows / n_classes))

    feature_min = df[FEATURES].min().to_numpy(dtype=float)
    feature_max = df[FEATURES].max().to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)

    per_class_frames = []
    for crop in classes:
        class_df = df[df[LABEL_COL] == crop].copy()
        n_existing = len(class_df)
        n_new = max(0, target_per_class - n_existing)

        synth = _synthesize_class(
            class_df=class_df,
            n_new=n_new,
            rng=rng,
            feature_min=feature_min,
            feature_max=feature_max,
            noise_scale=noise_scale,
        )

        synth_df = pd.DataFrame(synth, columns=FEATURES)
        synth_df[LABEL_COL] = crop

        merged = pd.concat([class_df[FEATURES + [LABEL_COL]], synth_df], ignore_index=True)
        per_class_frames.append(merged)

    out_df = pd.concat(per_class_frames, ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Keep N/P/K as integers to match the original dataset style.
    for col in ["N", "P", "K"]:
        out_df[col] = np.rint(out_df[col]).astype(int)

    # Trim to exact requested size while preserving approximate class balance.
    if len(out_df) > target_rows:
        keep_per_class = target_rows // n_classes
        remainder = target_rows % n_classes
        final_parts = []
        for i, crop in enumerate(classes):
            k = keep_per_class + (1 if i < remainder else 0)
            part = out_df[out_df[LABEL_COL] == crop].sample(n=k, random_state=random_seed + i)
            final_parts.append(part)
        out_df = pd.concat(final_parts, ignore_index=True)
        out_df = out_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment crop dataset to a larger row count.")
    parser.add_argument("--input", default=INPUT_PATH, help="Path to input CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to output CSV")
    parser.add_argument("--target-rows", type=int, default=11000, help="Target total rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--noise-scale", type=float, default=0.08, help="Noise multiplier for covariance")
    args = parser.parse_args()

    out_df = build_augmented_dataset(
        input_path=args.input,
        output_path=args.output,
        target_rows=args.target_rows,
        random_seed=args.seed,
        noise_scale=args.noise_scale,
    )

    counts = out_df[LABEL_COL].value_counts().sort_index()
    print(f"Saved augmented dataset: {args.output}")
    print(f"Shape: {out_df.shape}")
    print("Class counts (first 10):")
    print(counts.head(10).to_string())


if __name__ == "__main__":
    main()
