"""
MedMate_ML.py
=============
Flask backend for Early-Stage Diabetes & Dementia Detection using ML.

Endpoints
---------
POST /predict/diabetes   – predict diabetes risk
POST /predict/dementia   – predict dementia risk
GET  /health             – health check
GET  /model/info         – model metadata & feature info

Run
---
    python MedMate_ML.py
"""

import os
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
load_dotenv()

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR              = Path(__file__).parent
MODEL_DIR             = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

DIABETES_CSV          = BASE_DIR / os.getenv("DIABETES_CSV",  "diabetes_data_upload.csv")
DEMENTIA_CSV          = BASE_DIR / os.getenv("DEMENTIA_CSV",  "dementia_dataset.csv")
DIABETES_MODEL_PATH   = MODEL_DIR / "diabetes_model.pkl"
DEMENTIA_MODEL_PATH   = MODEL_DIR / "dementia_model.pkl"
DIABETES_ENCODER_PATH = MODEL_DIR / "diabetes_encoders.pkl"
DEMENTIA_ENCODER_PATH = MODEL_DIR / "dementia_encoders.pkl"

# ─── Feature definitions ─────────────────────────────────────────────────────
DIABETES_TARGET  = "class"
DIABETES_DROP    = []                    # nothing to drop – all 16 are useful
DIABETES_BINARY  = [                     # Yes/No columns → 1/0
    "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
    "Polyphagia", "Genital thrush", "visual blurring", "Itching",
    "Irritability", "delayed healing", "partial paresis",
    "muscle stiffness", "Alopecia", "Obesity",
]
DIABETES_FEATURES = [
    "Age", "Gender",
    "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
    "Polyphagia", "Genital thrush", "visual blurring", "Itching",
    "Irritability", "delayed healing", "partial paresis",
    "muscle stiffness", "Alopecia", "Obesity",
]

DEMENTIA_TARGET   = "Group"
DEMENTIA_DROP     = ["Subject ID", "MRI ID", "Hand"]  # IDs / single-value cols
DEMENTIA_FEATURES = [
    "Visit", "MR Delay", "M/F", "Age", "EDUC",
    "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_diabetes(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Label-encode Gender + binary Yes/No columns.
    Returns (X, y, encoders_dict)
    """
    df = df.copy()

    if fit:
        encoders = {}

    # Binary Yes/No → 1/0
    for col in DIABETES_BINARY:
        df[col] = df[col].str.strip().map({"Yes": 1, "No": 0})

    # Gender
    if fit:
        le = LabelEncoder()
        df["Gender"] = le.fit_transform(df["Gender"].str.strip())
        encoders["Gender"] = le
    else:
        df["Gender"] = encoders["Gender"].transform(df["Gender"].str.strip())

    # Target
    y = None
    if DIABETES_TARGET in df.columns:
        if fit:
            le_target = LabelEncoder()
            y = le_target.fit_transform(df[DIABETES_TARGET].str.strip())
            encoders["target"] = le_target
        else:
            y = encoders["target"].transform(df[DIABETES_TARGET].str.strip()) \
                if DIABETES_TARGET in df.columns else None

    X = df[DIABETES_FEATURES]
    return X, y, encoders


def preprocess_dementia(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Encode M/F, impute SES & MMSE, map Group → binary (Demented=1).
    Returns (X, y, encoders_dict)
    """
    df = df.copy()

    if fit:
        encoders = {}

    # Encode M/F
    if fit:
        le = LabelEncoder()
        df["M/F"] = le.fit_transform(df["M/F"].str.strip())
        encoders["M/F"] = le
    else:
        df["M/F"] = encoders["M/F"].transform(df["M/F"].str.strip())

    # Impute missing SES and MMSE with median
    for col in ["SES", "MMSE"]:
        if fit:
            median_val = df[col].median()
            encoders[f"{col}_median"] = median_val
        else:
            median_val = encoders[f"{col}_median"]
        df[col] = df[col].fillna(median_val)

    # Target: Demented=1, Nondemented=0, Converted=1 (already converting → positive)
    y = None
    if DEMENTIA_TARGET in df.columns:
        group_map = {"Demented": 1, "Converted": 1, "Nondemented": 0}
        y = df[DEMENTIA_TARGET].map(group_map).values
        if fit:
            encoders["group_map"] = group_map

    X = df[DEMENTIA_FEATURES]
    return X, y, encoders


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_diabetes_model():
    log.info("Training Diabetes model …")
    df = pd.read_csv(DIABETES_CSV)
    log.info(f"  Dataset shape: {df.shape}")

    X, y, encoders = preprocess_diabetes(df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Model: Random Forest (best for tabular symptom data) ──
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cv  = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
    log.info(f"  Test Accuracy : {acc:.4f}")
    log.info(f"  CV  Accuracy  : {cv:.4f}")
    log.info("\n" + classification_report(y_test, y_pred,
             target_names=encoders["target"].classes_))

    joblib.dump(model,    DIABETES_MODEL_PATH)
    joblib.dump(encoders, DIABETES_ENCODER_PATH)
    log.info(f"  Saved → {DIABETES_MODEL_PATH}")
    return model, encoders


def train_dementia_model():
    log.info("Training Dementia model …")
    df = pd.read_csv(DEMENTIA_CSV)
    log.info(f"  Dataset shape: {df.shape}")

    X, y, encoders = preprocess_dementia(df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Model: Gradient Boosting (handles small MRI datasets well) ──
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cv  = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
    log.info(f"  Test Accuracy : {acc:.4f}")
    log.info(f"  CV  Accuracy  : {cv:.4f}")
    log.info("\n" + classification_report(y_test, y_pred,
             target_names=["Nondemented", "Demented"]))

    joblib.dump(model,    DEMENTIA_MODEL_PATH)
    joblib.dump(encoders, DEMENTIA_ENCODER_PATH)
    log.info(f"  Saved → {DEMENTIA_MODEL_PATH}")
    return model, encoders


def load_or_train(model_path, encoder_path, train_fn):
    if model_path.exists() and encoder_path.exists():
        log.info(f"Loading cached model from {model_path}")
        return joblib.load(model_path), joblib.load(encoder_path)
    return train_fn()


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
CORS(app)


# ─── Serve Frontend ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return "MedMate ML API Server"

# Load / train on startup
diabetes_model, diabetes_encoders = load_or_train(
    DIABETES_MODEL_PATH, DIABETES_ENCODER_PATH, train_diabetes_model
)
dementia_model, dementia_encoders = load_or_train(
    DEMENTIA_MODEL_PATH, DEMENTIA_ENCODER_PATH, train_dementia_model
)


# ─── /health ──────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": ["diabetes", "dementia"]})


# ─── /model/info ──────────────────────────────────────────────────────────────
@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "diabetes": {
            "algorithm": "Random Forest Classifier",
            "features": DIABETES_FEATURES,
            "target_classes": list(diabetes_encoders["target"].classes_),
            "note": "Binary Yes/No symptoms + Age + Gender",
        },
        "dementia": {
            "algorithm": "Gradient Boosting Classifier",
            "features": DEMENTIA_FEATURES,
            "target_classes": ["Nondemented", "Demented"],
            "note": "MRI-derived + cognitive scores (MMSE, CDR)",
        },
    })


# ─── /predict/diabetes ────────────────────────────────────────────────────────
@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    """
    Expected JSON body (all fields required):
    {
        "Age": 45,
        "Gender": "Male",
        "Polyuria": "Yes",
        "Polydipsia": "No",
        "sudden weight loss": "Yes",
        "weakness": "Yes",
        "Polyphagia": "No",
        "Genital thrush": "No",
        "visual blurring": "No",
        "Itching": "Yes",
        "Irritability": "No",
        "delayed healing": "No",
        "partial paresis": "No",
        "muscle stiffness": "No",
        "Alopecia": "No",
        "Obesity": "No"
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        # Build single-row DataFrame
        required = DIABETES_FEATURES
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        row = pd.DataFrame([{f: data[f] for f in required}])

        # Preprocess (no target column)
        X, _, _ = preprocess_diabetes(row, encoders=diabetes_encoders, fit=False)

        pred       = diabetes_model.predict(X)[0]
        prob       = diabetes_model.predict_proba(X)[0]
        label      = diabetes_encoders["target"].inverse_transform([pred])[0]
        confidence = float(round(max(prob) * 100, 2))

        risk_level = (
            "High"   if confidence >= 75 else
            "Medium" if confidence >= 50 else
            "Low"
        )

        return jsonify({
            "prediction":  label,           # "Positive" | "Negative"
            "confidence":  confidence,       # 0-100 %
            "risk_level":  risk_level,
            "probabilities": {
                cls: float(round(p * 100, 2))
                for cls, p in zip(
                    diabetes_encoders["target"].classes_, prob
                )
            },
        })

    except Exception as e:
        log.exception("Diabetes prediction error")
        return jsonify({"error": str(e)}), 500


# ─── /predict/dementia ────────────────────────────────────────────────────────
@app.route("/predict/dementia", methods=["POST"])
def predict_dementia():
    """
    Expected JSON body:
    {
        "Visit":    2,
        "MR Delay": 0,
        "M/F":      "M",
        "Age":      77,
        "EDUC":     14,
        "SES":      2.0,
        "MMSE":     27.0,
        "CDR":      0.0,
        "eTIV":     1987,
        "nWBV":     0.696,
        "ASF":      0.883
    }

    Field notes
    -----------
    MMSE : Mini-Mental State Examination   (0–30; lower = worse cognition)
    CDR  : Clinical Dementia Rating        (0=none, 0.5=very mild, 1=mild …)
    eTIV : Estimated Total Intracranial Volume (mm³)
    nWBV : Normalized Whole Brain Volume
    ASF  : Atlas Scaling Factor
    EDUC : Years of education
    SES  : Socioeconomic status (1=high … 5=low)
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        missing = [f for f in DEMENTIA_FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        row = pd.DataFrame([{f: data[f] for f in DEMENTIA_FEATURES}])

        X, _, _ = preprocess_dementia(row, encoders=dementia_encoders, fit=False)

        pred       = dementia_model.predict(X)[0]
        prob       = dementia_model.predict_proba(X)[0]
        label      = "Demented" if pred == 1 else "Nondemented"
        confidence = float(round(max(prob) * 100, 2))

        risk_level = (
            "High"   if confidence >= 75 and pred == 1 else
            "Medium" if confidence >= 50 and pred == 1 else
            "Low"
        )

        return jsonify({
            "prediction":  label,
            "confidence":  confidence,
            "risk_level":  risk_level,
            "probabilities": {
                "Nondemented": float(round(prob[0] * 100, 2)),
                "Demented":    float(round(prob[1] * 100, 2)),
            },
        })

    except Exception as e:
        log.exception("Dementia prediction error")
        return jsonify({"error": str(e)}), 500


# ─── /retrain ─────────────────────────────────────────────────────────────────
@app.route("/retrain", methods=["POST"])
def retrain():
    """Force-retrain both models (useful after uploading new data)."""
    global diabetes_model, diabetes_encoders, dementia_model, dementia_encoders
    try:
        secret = request.headers.get("X-Admin-Key", "")
        if secret != os.getenv("ADMIN_KEY", "medmate_admin"):
            return jsonify({"error": "Unauthorized"}), 401

        diabetes_model, diabetes_encoders = train_diabetes_model()
        dementia_model, dementia_encoders  = train_dementia_model()
        return jsonify({"status": "retrained successfully"})
    except Exception as e:
        log.exception("Retrain error")
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    log.info(f"MedMate ML server starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)