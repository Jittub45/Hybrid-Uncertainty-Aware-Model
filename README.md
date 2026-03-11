# 🌾 Hybrid Uncertainty-Aware Precision Crop Recommendation System

> An AI-powered agricultural decision support system that tells farmers **exactly which crop to grow** — backed by a research-grade stacked ensemble, SHAP explainability, and a real-time web interface.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?logo=flask)
![LightGBM](https://img.shields.io/badge/LightGBM-Base%20Learner-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Base%20Learner-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-Base%20Learner-yellow)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet)

---

## 🧠 What This Project Does

Farmers often make crop decisions based on intuition or tradition — leading to yield loss and resource waste. This system solves that by analyzing **7 measurable soil and climate parameters** and recommending the single best crop from 22 possibilities with predicted confidence scores.

The model is not just accurate — it is **explainable**. SHAP values reveal *why* a particular crop was recommended, making it trustworthy for real-world agricultural use.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                            │
│   N  |  P  |  K  |  Temp  |  Humidity  |  pH  |  Rainfall  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                        │
│   N_P_ratio | N_K_ratio | P_K_ratio | NPK_sum | temp_humid  │
│                  (7 → 12 features)                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ LightGBM │ │ XGBoost  │ │ CatBoost │  ← Base Learners
        │ (500 est)│ │ (300 est)│ │(300 iter)│
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │             │
             └──────────┬──┘─────────────┘
                        │  5-fold CV Stacking
                        ▼
              ┌──────────────────┐
              │ Logistic         │  ← Meta-Learner
              │ Regression       │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  SHAP Analysis   │  ← Explainability
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Flask REST API  │  ← Deployment
              │  + Web UI        │
              └──────────────────┘
```

---

## 📊 ML Pipeline (6 Phases)

| Phase | Description |
|-------|-------------|
| **1. Data Loading & EDA** | Load 2,200-row dataset; distribution plots, correlation heatmaps |
| **2. Baseline Benchmarking** | Train 9 classifiers on raw 7 features; rank by test + 5-fold CV accuracy |
| **3. Feature Engineering** | Derive 5 agronomic ratio features → 12 total |
| **4. Hybrid Stacked Ensemble** | LightGBM + XGBoost + CatBoost → Logistic Regression meta-learner |
| **5. SHAP Explainability** | Global feature importance + beeswarm summary plot |
| **6. Model Selection & Save** | Auto-select whichever model (baseline vs. hybrid) achieves higher test accuracy |

---

## 🤖 Models Benchmarked

| Model | Type |
|-------|------|
| Logistic Regression | Linear |
| K-Nearest Neighbors (k=5) | Instance-based |
| Decision Tree | Tree |
| Random Forest (200 est.) | Ensemble |
| Extra Trees (200 est.) | Ensemble |
| Gradient Boosting (100 est.) | Boosting |
| XGBoost (200 est.) | Boosting |
| SVM (RBF kernel) | Kernel-based |
| Naive Bayes | Probabilistic |
| **Stacked Ensemble** ⭐ | LightGBM + XGBoost + CatBoost → LR |

> The stacked ensemble uses **5-fold cross-validation** internally during stacking and is evaluated separately on the held-out test set.

---

## 🔬 Feature Engineering

The raw 7 sensor inputs are extended with 5 agronomic ratio features:

| Feature | Formula | Agronomic Meaning |
|---------|---------|-------------------|
| `N_P_ratio` | N / (P + 1) | Nitrogen-to-Phosphorus balance |
| `N_K_ratio` | N / (K + 1) | Nitrogen-to-Potassium balance |
| `P_K_ratio` | P / (K + 1) | Phosphorus-to-Potassium balance |
| `NPK_sum` | N + P + K | Total macronutrient load |
| `temp_humid` | temperature × humidity | Heat-moisture stress index |

---

## 🌱 Supported Crops (22 Classes)

| | | | |
|--|--|--|--|
| 🍎 Apple | 🍌 Banana | 🫘 Blackgram | 🟡 Chickpea |
| 🥥 Coconut | ☕ Coffee | 🌿 Cotton | 🍇 Grapes |
| 🌾 Jute | 🫘 Kidney Beans | 🟤 Lentil | 🌽 Maize |
| 🥭 Mango | 🫘 Moth Beans | 🟢 Mung Bean | 🍈 Muskmelon |
| 🍊 Orange | 🍑 Papaya | 🫘 Pigeon Peas | 🍎 Pomegranate |
| 🍚 Rice | 🍉 Watermelon | | |

---

## 🧬 Explainability (SHAP)

After training, the system automatically generates:
- **`models/shap_feature_importance.png`** — bar chart of mean absolute SHAP values per feature
- **`models/shap_summary_plot.png`** — beeswarm plot showing feature impact direction per sample
- **`models/confusion_matrix_hybrid.png`** — confusion matrix for the stacked ensemble
- **`models/model_comparison.png`** — side-by-side accuracy comparison of all 9 baselines

This makes recommendations **auditable** — you can see which soil nutrients drove a specific prediction.

---

## 📁 Project Structure

```
├── app/
│   ├── flask_app.py          # Web server — GET / and POST /predict
│   └── templates/
│       └── index.html        # Interactive UI with live predictions
├── data/
│   └── Crop_recommendation.csv   # 2,200 samples × 8 columns (7 features + label)
├── models/                   # Auto-created; stores .pkl artifacts + plots
├── notebooks/
│   ├── train.py              # ▶ Main pipeline entry point
│   ├── config.py             # Shared paths and feature lists
│   ├── data_loader.py        # CSV loader
│   ├── eda.py                # Exploratory analysis & plots
│   ├── feature_engineering.py
│   ├── preprocessing.py      # Label encoding, 80/20 split, StandardScaler
│   ├── model_training.py     # 9 baseline classifiers
│   ├── tuning.py             # GridSearchCV for Random Forest
│   ├── hybrid_model.py       # Stacked ensemble (LightGBM+XGBoost+CatBoost→LR)
│   ├── explainability.py     # SHAP analysis
│   └── save_model.py         # Persist best model + sanity check
├── pyproject.toml
└── README.md
```

---

## ⚙️ Setup & Installation

**Requirements:** Python ≥ 3.11

```bash
# 1. Clone or open the project
cd "Capstone Project"

# 2. Install dependencies
pip install -e .
pip install lightgbm catboost flask shap   # not yet in pyproject.toml

# 3. Train the full pipeline (generates model artifacts)
python notebooks/train.py

# 4. Launch the web application
python app/flask_app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 🌐 REST API Reference

### `POST /predict`

**Request body (JSON):**

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 21.0,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 203.0
}
```

**Response (JSON):**

```json
{
  "success": true,
  "crop": "rice",
  "emoji": "🍚",
  "color": "#1565c0",
  "desc": "Rice requires high humidity...",
  "top5": [
    { "crop": "rice",      "prob": 94.31 },
    { "crop": "jute",      "prob": 3.12  },
    { "crop": "coconut",   "prob": 1.07  },
    { "crop": "maize",     "prob": 0.88  },
    { "crop": "banana",    "prob": 0.62  }
  ]
}
```

**Error response:**
```json
{ "success": false, "error": "missing field: N" }
```

---

## 🔧 Hyperparameter Reference

| Component | Parameter | Value |
|-----------|-----------|-------|
| LightGBM | `n_estimators` | 500 |
| LightGBM | `learning_rate` | 0.05 |
| LightGBM | `num_leaves` | 63 |
| XGBoost | `n_estimators` | 300 |
| XGBoost | `learning_rate` | 0.05 |
| CatBoost | `iterations` | 300 |
| CatBoost | `depth` | 6 |
| Stacking | `cv` | 5-fold |
| RF Tuning | `n_estimators` | [100, 200, 300] |
| RF Tuning | `max_depth` | [None, 10, 20] |

---

## 🗃️ Dataset

| Property | Value |
|----------|-------|
| Source | `data/Crop_recommendation.csv` |
| Rows | 2,200 |
| Classes | 22 crop types (100 samples each) |
| Input features | N, P, K, temperature, humidity, pH, rainfall |
| Train / Test split | 80% / 20% (stratified) |
| Scaling | `StandardScaler` (applied to distance-based models) |

---

## 🧰 Tech Stack

| Category | Libraries |
|----------|-----------|
| **Boosting** | LightGBM, XGBoost, CatBoost |
| **ML Framework** | scikit-learn |
| **Explainability** | SHAP |
| **Web** | Flask |
| **Data & Viz** | pandas, NumPy, Matplotlib, Seaborn |
| **Persistence** | joblib |
| **Runtime** | Python 3.11+ |

