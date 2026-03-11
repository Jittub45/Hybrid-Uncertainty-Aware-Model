"""
flask_app.py — Crop Recommendation System Web Application (Flask)

Run:
    python app/flask_app.py

Visit: http://127.0.0.1:5000
"""

import os
import sys
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TMPL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

app = Flask(__name__, template_folder=TMPL_DIR)

# ── Load model once at startup ────────────────────────────────────────────────
model   = joblib.load(os.path.join(MODELS_DIR, "crop_model.pkl"))
le      = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

# ── Crop details ──────────────────────────────────────────────────────────────
CROP_INFO = {
    "apple":       {"emoji": "🍎", "color": "#e53935",
                    "desc": "Apple grows best in cool climates with well-drained, loamy soil and requires cold winters for dormancy."},
    "banana":      {"emoji": "🍌", "color": "#f9a825",
                    "desc": "Banana thrives in tropical climates with high humidity and consistent rainfall throughout the year."},
    "blackgram":   {"emoji": "🫘", "color": "#4e342e",
                    "desc": "Blackgram is a legume suited to warm, semi-arid conditions and fixes nitrogen in soil."},
    "chickpea":    {"emoji": "🟡", "color": "#f57f17",
                    "desc": "Chickpea grows in dry, cool conditions and is highly drought-tolerant, ideal for Rabi season."},
    "coconut":     {"emoji": "🥥", "color": "#6d4c41",
                    "desc": "Coconut palm requires a tropical climate with sandy, well-drained soil and coastal humid winds."},
    "coffee":      {"emoji": "☕", "color": "#4e342e",
                    "desc": "Coffee needs a subtropical climate, high humidity, moderate rainfall, and well-drained acidic soil."},
    "cotton":      {"emoji": "🌿", "color": "#c8e6c9",
                    "desc": "Cotton requires long warm summers, moderate rainfall, deep fertile soil, and full sunshine."},
    "grapes":      {"emoji": "🍇", "color": "#6a1b9a",
                    "desc": "Grapes thrive in Mediterranean climate with well-drained sandy soil and hot, dry summers."},
    "jute":        {"emoji": "🌾", "color": "#8d6e63",
                    "desc": "Jute grows best in warm, humid climates with loamy, alluvial soil and high monsoonal rainfall."},
    "kidneybeans": {"emoji": "🫘", "color": "#b71c1c",
                    "desc": "Kidney beans prefer well-drained, fertile soil with moderate temperatures between 15–25°C."},
    "lentil":      {"emoji": "🟤", "color": "#795548",
                    "desc": "Lentil is a cool-season pulse crop highly tolerant to drought, suited to semi-arid regions."},
    "maize":       {"emoji": "🌽", "color": "#f9a825",
                    "desc": "Maize requires warm weather, fertile soil with good drainage, and moderate to high rainfall."},
    "mango":       {"emoji": "🥭", "color": "#ff8f00",
                    "desc": "Mango needs a tropical climate, deep well-drained soil, and seasonal dry spells for flowering."},
    "mothbeans":   {"emoji": "🫘", "color": "#827717",
                    "desc": "Moth beans are highly drought-resistant and perfectly suited to the hot arid zones of India."},
    "mungbean":    {"emoji": "🟢", "color": "#2e7d32",
                    "desc": "Mung bean is a short-season legume suited to warm, humid regions and improves soil fertility."},
    "muskmelon":   {"emoji": "🍈", "color": "#f9a825",
                    "desc": "Muskmelon needs warm temperatures, dry weather during fruiting, and light sandy, well-drained soil."},
    "orange":      {"emoji": "🍊", "color": "#e65100",
                    "desc": "Orange grows in subtropical climates with well-drained fertile soil and warm, sunny conditions."},
    "papaya":      {"emoji": "🍑", "color": "#ff6f00",
                    "desc": "Papaya is a fast-growing tropical fruit requiring rich, moist, loamy soil and full sun."},
    "pigeonpeas":  {"emoji": "🫘", "color": "#33691e",
                    "desc": "Pigeon pea is drought-tolerant, nitrogen-fixing, and suits the semi-arid tropical regions."},
    "pomegranate": {"emoji": "🍎", "color": "#c62828",
                    "desc": "Pomegranate thrives in semi-arid to arid regions and tolerates alkaline, well-drained soils."},
    "rice":        {"emoji": "🍚", "color": "#1565c0",
                    "desc": "Rice requires high humidity, standing water during growth, clayey or loamy soil, and warm temperatures."},
    "watermelon":  {"emoji": "🍉", "color": "#388e3c",
                    "desc": "Watermelon needs hot, sunny weather, a long frost-free season, and sandy well-drained soil."},
}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = [
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"]),
        ]
        arr  = np.array([features])
        pred = model.predict(arr)[0]
        crop = le.inverse_transform([pred])[0]

        # Top-5 probabilities
        proba = model.predict_proba(arr)[0]
        top5  = sorted(
            [{"crop": le.classes_[i], "prob": round(float(p) * 100, 2)} for i, p in enumerate(proba)],
            key=lambda x: -x["prob"]
        )[:5]

        info = CROP_INFO.get(crop, {"emoji": "🌱", "color": "#2e7d32", "desc": "A great crop for your conditions."})

        return jsonify({
            "success": True,
            "crop":    crop,
            "emoji":   info["emoji"],
            "color":   info["color"],
            "desc":    info["desc"],
            "top5":    top5,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    print("=" * 55)
    print("  🌾 Crop Recommendation System")
    print("  Open browser: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=False, host="127.0.0.1", port=5000)
