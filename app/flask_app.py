"""
flask_app.py — Crop Recommendation System Web Application (Flask)

Run:
    python app/flask_app.py

Visit: http://127.0.0.1:5000
"""

import os
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
import secrets
import re
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, abort, send_from_directory
from flask import redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from werkzeug.security import check_password_hash, generate_password_hash

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TMPL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__, template_folder=TMPL_DIR)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
SQLITE_DB_PATH = os.path.join(BASE_DIR, "app", "auth.db").replace("\\", "/")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{SQLITE_DB_PATH}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to continue."

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

CROP_LIBRARY = {
    "rice": {"soil": "Clayey or loamy", "ph": "5.5 - 7.0", "rain": "150 - 300 mm", "temp": "20 - 30 C"},
    "maize": {"soil": "Well-drained loam", "ph": "5.5 - 7.5", "rain": "60 - 120 mm", "temp": "18 - 27 C"},
    "cotton": {"soil": "Deep black soil", "ph": "5.8 - 8.0", "rain": "50 - 100 mm", "temp": "21 - 30 C"},
    "mango": {"soil": "Deep loamy soil", "ph": "5.5 - 7.5", "rain": "75 - 250 mm", "temp": "24 - 30 C"},
    "coffee": {"soil": "Acidic loam", "ph": "5.0 - 6.5", "rain": "150 - 250 mm", "temp": "18 - 24 C"},
}

OTP_EXP_MINUTES = 10
OTP_MAX_ATTEMPTS = 5
OTP_RESEND_COOLDOWN_SECONDS = 30
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def _smtp_configured() -> bool:
    return bool(
        os.environ.get("SMTP_HOST")
        and os.environ.get("SMTP_USER")
        and os.environ.get("SMTP_PASSWORD")
    )


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    is_verified = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.utcnow())
    last_login_at = db.Column(db.DateTime, nullable=True)


class OTPChallenge(db.Model):
    __tablename__ = "otp_challenges"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    purpose = db.Column(db.String(32), nullable=False, index=True)
    otp_hash = db.Column(db.String(255), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False, index=True)
    attempts = db.Column(db.Integer, nullable=False, default=0)
    is_used = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.utcnow())


@login_manager.user_loader
def load_user(user_id: str):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None


def _now_utc() -> datetime:
    # Keep timestamps naive UTC because SQLite DateTime values are returned as naive.
    return datetime.utcnow()


def _password_valid(password: str) -> tuple:
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not any(ch.isdigit() for ch in password):
        return False, "Password must contain at least one digit."
    if not any(ch.isalpha() for ch in password):
        return False, "Password must contain at least one letter."
    return True, ""


def _email_valid(email: str) -> bool:
    return bool(EMAIL_RE.match(email or ""))


def _generate_otp_code() -> str:
    return "".join(secrets.choice("0123456789") for _ in range(6))


def _send_otp_email(recipient: str, otp_code: str, purpose: str) -> bool:
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASSWORD")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user or "no-reply@example.com")

    subject = "Your OTP for Precision Crop Recommender"
    purpose_text = "account verification" if purpose in {"signup_verify", "email_verify"} else "login verification"
    body = (
        f"Your OTP for {purpose_text} is: {otp_code}\n"
        f"This OTP expires in {OTP_EXP_MINUTES} minutes."
    )

    if not _smtp_configured():
        print(f"[DEV OTP] {recipient} -> {otp_code} ({purpose})")
        return True

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = recipient
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True
    except Exception as exc:
        print(f"[OTP EMAIL ERROR] {exc}")
        return False


def _create_otp_challenge(user_id: int, purpose: str) -> str:
    # Invalidate previous active OTPs for the same purpose.
    OTPChallenge.query.filter_by(user_id=user_id, purpose=purpose, is_used=False).update({"is_used": True})
    otp_code = _generate_otp_code()
    challenge = OTPChallenge(
        user_id=user_id,
        purpose=purpose,
        otp_hash=generate_password_hash(otp_code),
        expires_at=_now_utc() + timedelta(minutes=OTP_EXP_MINUTES),
        attempts=0,
        is_used=False,
    )
    db.session.add(challenge)
    db.session.commit()
    return otp_code


def _verify_otp(user_id: int, purpose: str, otp_code: str) -> tuple:
    challenge = (
        OTPChallenge.query
        .filter_by(user_id=user_id, purpose=purpose, is_used=False)
        .order_by(OTPChallenge.created_at.desc())
        .first()
    )
    if challenge is None:
        return False, "No active OTP challenge. Please resend OTP."

    if challenge.expires_at < _now_utc():
        challenge.is_used = True
        db.session.commit()
        return False, "OTP expired. Please request a new OTP."

    if challenge.attempts >= OTP_MAX_ATTEMPTS:
        challenge.is_used = True
        db.session.commit()
        return False, "Too many failed attempts. Please request a new OTP."

    if not check_password_hash(challenge.otp_hash, otp_code):
        challenge.attempts += 1
        if challenge.attempts >= OTP_MAX_ATTEMPTS:
            challenge.is_used = True
        db.session.commit()
        remaining = max(0, OTP_MAX_ATTEMPTS - challenge.attempts)
        return False, f"Invalid OTP. Attempts remaining: {remaining}."

    challenge.is_used = True
    db.session.commit()
    return True, "OTP verified successfully."


with app.app_context():
    db.create_all()


def _insight_images() -> list:
    """Return available insight image filenames from models directory."""
    wanted = [
        "feature_importance.png",
        "model_comparison.png",
        "confusion_matrix.png",
        "confusion_matrix_hybrid.png",
        "shap_feature_importance.png",
        "shap_summary_plot.png",
        "correlation_heatmap.png",
        "crop_distribution.png",
    ]
    return [name for name in wanted if os.path.exists(os.path.join(MODELS_DIR, name))]


def _insight_caption(filename: str) -> str:
    if filename == "feature_importance.png":
        return "Ranks raw input features by average importance contribution in the baseline tree model."
    if filename == "model_comparison.png":
        return "Compares baseline model test accuracies to help justify the selected production model."
    if filename == "confusion_matrix.png":
        return "Shows class-by-class prediction correctness for the best baseline model."
    if filename == "confusion_matrix_hybrid.png":
        return "Shows class-by-class prediction correctness for the hybrid stacked ensemble."
    if filename == "shap_feature_importance.png":
        return "Global SHAP impact summary using mean absolute SHAP values across features."
    if filename == "shap_summary_plot.png":
        return "Beeswarm SHAP plot showing direction and magnitude of feature influence per sample."
    if filename == "correlation_heatmap.png":
        return "Highlights linear relationships between numeric agronomic features in the dataset."
    if filename == "crop_distribution.png":
        return "Shows label balance across all crop classes in the training dataset."
    return "Supporting analytical visualization from the training workflow."


def _insight_artifacts() -> list:
    artifacts = []
    for filename in _insight_images():
        artifacts.append(
            {
                "filename": filename,
                "title": filename.replace("_", " ").replace(".png", "").title(),
                "caption": _insight_caption(filename),
            }
        )
    return artifacts


@app.context_processor
def inject_globals():
    return {"app_name": "Precision Crop Recommender", "year": 2026}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    return render_template("home.html")


@app.route("/recommend")
@login_required
def recommend_page():
    return render_template("predict.html")


@app.route("/crops")
@login_required
def crops_page():
    merged = []
    for crop_name, info in sorted(CROP_INFO.items(), key=lambda x: x[0]):
        defaults = {"soil": "Loamy", "ph": "5.5 - 7.5", "rain": "60 - 200 mm", "temp": "18 - 32 C"}
        defaults.update(CROP_LIBRARY.get(crop_name, {}))
        merged.append({"name": crop_name, **info, **defaults})
    return render_template("crops.html", crops=merged)


@app.route("/insights")
@login_required
def insights_page():
    return render_template("insights.html", artifacts=_insight_artifacts())


@app.route("/about")
@login_required
def about_page():
    return render_template("about.html")


@app.route("/register", methods=["GET", "POST"])
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("recommend_page"))

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not email or not password or not confirm_password:
            flash("All fields are required.", "error")
            return render_template("signup.html")

        if not _email_valid(email):
            flash("Please enter a valid email address.", "error")
            return render_template("signup.html")

        if password != confirm_password:
            flash("Password and confirm password do not match.", "error")
            return render_template("signup.html")

        valid, msg = _password_valid(password)
        if not valid:
            flash(msg, "error")
            return render_template("signup.html")

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            if existing_user.is_verified:
                flash("Account already exists. Please log in.", "error")
                return redirect(url_for("login"))

            otp_code = _create_otp_challenge(existing_user.id, "signup_verify")
            sent = _send_otp_email(existing_user.email, otp_code, "signup_verify")
            if not sent:
                flash("Account exists but OTP could not be sent. Please try again.", "error")
                return render_template("signup.html")

            session["pending_user_id"] = existing_user.id
            session["pending_purpose"] = "signup_verify"
            session["last_otp_sent_at"] = int(_now_utc().timestamp())
            flash("Account exists but is not verified. OTP sent again.", "success")
            return redirect(url_for("verify_otp"))

        user = User(
            email=email,
            password_hash=generate_password_hash(password),
            is_verified=False,
        )
        db.session.add(user)
        db.session.commit()

        otp_code = _create_otp_challenge(user.id, "signup_verify")
        sent = _send_otp_email(user.email, otp_code, "signup_verify")
        if not sent:
            OTPChallenge.query.filter_by(user_id=user.id).delete()
            db.session.delete(user)
            db.session.commit()
            flash("Could not send OTP email right now. Please try again.", "error")
            return render_template("signup.html")

        session["pending_user_id"] = user.id
        session["pending_purpose"] = "signup_verify"
        session["last_otp_sent_at"] = int(_now_utc().timestamp())
        flash("OTP sent to your email. Enter it to activate your account.", "success")
        return redirect(url_for("verify_otp"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("recommend_page"))

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        if user.is_verified:
            user.last_login_at = _now_utc()
            db.session.commit()
            login_user(user)
            flash("Login successful.", "success")
            next_url = request.args.get("next")
            if next_url:
                return redirect(next_url)
            return redirect(url_for("recommend_page"))

        # Unverified users must complete email OTP verification.
        purpose = "signup_verify"
        otp_code = _create_otp_challenge(user.id, purpose)
        sent = _send_otp_email(user.email, otp_code, purpose)
        if not sent:
            flash("Could not send OTP email right now. Please try again.", "error")
            return render_template("login.html")

        session["pending_user_id"] = user.id
        session["pending_purpose"] = purpose
        session["last_otp_sent_at"] = int(_now_utc().timestamp())
        flash("Your account is not verified yet. OTP sent to complete verification.", "success")
        return redirect(url_for("verify_otp"))

    return render_template("login.html")


@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    pending_user_id = session.get("pending_user_id")
    pending_purpose = session.get("pending_purpose")

    if not pending_user_id or not pending_purpose:
        flash("No OTP verification in progress.", "error")
        return redirect(url_for("login"))

    user = db.session.get(User, int(pending_user_id))
    if user is None:
        session.pop("pending_user_id", None)
        session.pop("pending_purpose", None)
        session.pop("last_otp_sent_at", None)
        flash("Account not found. Please sign up again.", "error")
        return redirect(url_for("signup"))

    if request.method == "POST":
        otp_code = (request.form.get("otp") or "").strip()
        if not otp_code.isdigit() or len(otp_code) != 6:
            flash("OTP must be a 6-digit number.", "error")
            return render_template("verify_otp.html", email=user.email)

        ok, msg = _verify_otp(user.id, pending_purpose, otp_code)
        if not ok:
            flash(msg, "error")
            return render_template("verify_otp.html", email=user.email)

        if pending_purpose in {"signup_verify", "email_verify"}:
            user.is_verified = True

        user.last_login_at = _now_utc()
        db.session.commit()

        session.pop("pending_user_id", None)
        session.pop("pending_purpose", None)
        session.pop("last_otp_sent_at", None)

        login_user(user)
        flash("Verification successful. Welcome back!", "success")
        return redirect(url_for("recommend_page"))

    return render_template("verify_otp.html", email=user.email)


@app.route("/resend-otp", methods=["POST"])
def resend_otp():
    pending_user_id = session.get("pending_user_id")
    pending_purpose = session.get("pending_purpose")
    if not pending_user_id or not pending_purpose:
        flash("No OTP verification in progress.", "error")
        return redirect(url_for("login"))

    now_ts = int(_now_utc().timestamp())
    last_sent_ts = int(session.get("last_otp_sent_at", 0))
    if now_ts - last_sent_ts < OTP_RESEND_COOLDOWN_SECONDS:
        wait_for = OTP_RESEND_COOLDOWN_SECONDS - (now_ts - last_sent_ts)
        flash(f"Please wait {wait_for} seconds before requesting another OTP.", "error")
        return redirect(url_for("verify_otp"))

    user = db.session.get(User, int(pending_user_id))
    if user is None:
        flash("Account not found.", "error")
        return redirect(url_for("signup"))

    otp_code = _create_otp_challenge(user.id, pending_purpose)
    sent = _send_otp_email(user.email, otp_code, pending_purpose)
    if not sent:
        flash("Could not resend OTP email right now. Please try again.", "error")
        return redirect(url_for("verify_otp"))
    session["last_otp_sent_at"] = now_ts
    flash("A new OTP has been sent.", "success")
    return redirect(url_for("verify_otp"))


@app.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("index"))


@app.route("/artifacts/<path:filename>")
@login_required
def model_artifact(filename: str):
    if not filename.endswith(".png"):
        abort(404)
    full_path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(full_path):
        abort(404)
    return send_from_directory(MODELS_DIR, filename)


@app.route("/predict", methods=["POST"])
@login_required
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
    port = int(os.environ.get("PORT", 5000))
    smtp_state = "configured" if _smtp_configured() else "NOT configured (DEV OTP mode)"
    print("=" * 55)
    print("  🌾 Crop Recommendation System")
    print(f"  SMTP: {smtp_state}")
    print(f"  Open browser: http://0.0.0.0:{port}")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=port)
