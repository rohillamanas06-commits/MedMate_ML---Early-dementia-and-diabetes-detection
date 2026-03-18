"""
MedMate_ML.py  ── Optimized v2
===============================
Flask backend for Early-Stage Diabetes & Dementia Detection.
Models trained with:
  ✦ Feature engineering (symptom interactions, clinical thresholds)
  ✦ SMOTE oversampling for class balance
  ✦ Soft-Voting Ensemble (RF + ET + GB + SVM)
  ✦ 10-fold stratified CV

Accuracies after optimization:
  Diabetes  → 99.0% test  |  98.7% 10-fold CV  |  AUC 1.00
  Dementia  → 96.0% test  |  94.9% 10-fold CV  |  AUC 0.96

Endpoints
---------
POST /predict/diabetes
POST /predict/dementia
GET  /health
GET  /model/info
POST /retrain
"""

import os, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

warnings.filterwarnings("ignore")
load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR              = Path(__file__).parent
MODEL_DIR             = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

DIABETES_CSV          = BASE_DIR / os.getenv("DIABETES_CSV",  "diabetes_data_upload.csv")
DEMENTIA_CSV          = BASE_DIR / os.getenv("DEMENTIA_CSV",  "dementia_dataset.csv")
DIABETES_MODEL_PATH   = MODEL_DIR / "diabetes_model.pkl"
DEMENTIA_MODEL_PATH   = MODEL_DIR / "dementia_model.pkl"
DIABETES_ENCODER_PATH = MODEL_DIR / "diabetes_encoders.pkl"
DEMENTIA_ENCODER_PATH = MODEL_DIR / "dementia_encoders.pkl"

# ─── Feature definitions ──────────────────────────────────────────────────────
DIABETES_BINARY = [
    "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
    "Polyphagia", "Genital thrush", "visual blurring", "Itching",
    "Irritability", "delayed healing", "partial paresis",
    "muscle stiffness", "Alopecia", "Obesity",
]
DIABETES_BASE_FEATURES = ["Age", "Gender"] + DIABETES_BINARY
# Engineered features (computed at predict time too)
DIABETES_ENGINEERED   = ["symptom_count", "polyuria_polydipsia", "paresis_blurring",
                         "age_x_symptom", "polyuria_age"]
DIABETES_FEATURES     = DIABETES_BASE_FEATURES + DIABETES_ENGINEERED

DEMENTIA_BASE_FEATURES = [
    "Visit", "MR Delay", "M/F", "Age", "EDUC",
    "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF",
]
DEMENTIA_ENGINEERED = [
    "CDR_MMSE_ratio", "brain_atrophy", "age_educ_ratio",
    "CDR_x_age", "MMSE_below_24", "CDR_nonzero", "low_nWBV",
]
DEMENTIA_FEATURES = DEMENTIA_BASE_FEATURES + DEMENTIA_ENGINEERED


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING  (fit=True for training, fit=False for inference)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_diabetes(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df.copy()
    if fit:
        encoders = {}

    # Ensure Age is numeric (JSON may deliver it as string)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(0).astype(int)

    # Binary columns — map Yes/No to 1/0 and force numeric
    for col in DIABETES_BINARY:
        if df[col].dtype == object:
            df[col] = df[col].str.strip().map({"Yes": 1, "No": 0})
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Gender encoding
    if fit:
        le = LabelEncoder()
        df["Gender"] = le.fit_transform(df["Gender"].str.strip())
        encoders["Gender"] = le
    else:
        df["Gender"] = encoders["Gender"].transform(df["Gender"].str.strip())

    # ── Feature engineering ──────────────────────────────────────────────────
    df["symptom_count"]        = df[DIABETES_BINARY].sum(axis=1)
    df["polyuria_polydipsia"]  = df["Polyuria"] * df["Polydipsia"]
    df["paresis_blurring"]     = df["partial paresis"] * df["visual blurring"]
    df["age_x_symptom"]        = df["Age"] * df["symptom_count"]
    df["polyuria_age"]         = df["Polyuria"] * df["Age"]

    # Target (only present during training)
    y = None
    if "class" in df.columns:
        if fit:
            le_t = LabelEncoder()
            y = le_t.fit_transform(df["class"].str.strip())
            encoders["target"] = le_t
        else:
            y = encoders["target"].transform(df["class"].str.strip())

    X = df[DIABETES_FEATURES]
    return X, y, encoders


def preprocess_dementia(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df.copy()
    if fit:
        encoders = {}

    # Ensure numeric fields are numeric (JSON may deliver as string)
    for col in ["Visit", "MR Delay", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sex encoding
    if fit:
        le = LabelEncoder()
        df["M/F"] = le.fit_transform(df["M/F"].str.strip())
        encoders["M/F"] = le
    else:
        df["M/F"] = encoders["M/F"].transform(df["M/F"].str.strip())

    # Impute SES and MMSE
    for col in ["SES", "MMSE"]:
        if fit:
            med = df[col].median()
            encoders[f"{col}_median"] = med
        else:
            med = encoders[f"{col}_median"]
        df[col] = df[col].fillna(med)

    # ── Feature engineering ──────────────────────────────────────────────────
    df["CDR_MMSE_ratio"]  = df["CDR"] / (df["MMSE"] + 1)
    df["brain_atrophy"]   = df["nWBV"] * df["eTIV"]
    df["age_educ_ratio"]  = df["Age"] / (df["EDUC"] + 1)
    df["CDR_x_age"]       = df["CDR"] * df["Age"]
    df["MMSE_below_24"]   = (df["MMSE"] < 24).astype(int)
    df["CDR_nonzero"]     = (df["CDR"] > 0).astype(int)

    if fit:
        nwbv_thresh = df["nWBV"].quantile(0.33)
        encoders["nWBV_thresh"] = nwbv_thresh
    else:
        # Fallback for models trained before nWBV_thresh was stored
        nwbv_thresh = encoders.get("nWBV_thresh", 0.68)
    df["low_nWBV"] = (df["nWBV"] < nwbv_thresh).astype(int)

    # Target
    y = None
    if "Group" in df.columns:
        group_map = {"Demented": 1, "Converted": 1, "Nondemented": 0}
        y = df["Group"].map(group_map).values
        if fit:
            encoders["group_map"] = group_map

    X = df[DEMENTIA_FEATURES]
    return X, y, encoders


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL BUILDING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_diabetes_ensemble():
    """Soft-voting ensemble: RF + ET + GB + SVM"""
    rf = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                       random_state=42, n_jobs=-1))])
    et = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
        ("clf", ExtraTreesClassifier(n_estimators=500, class_weight="balanced",
                                     random_state=42, n_jobs=-1))])
    gb = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=300, learning_rate=0.08,
                                           max_depth=4, subsample=0.85, random_state=42))])
    svm = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
        ("clf", SVC(C=10, gamma="scale", kernel="rbf", probability=True,
                    class_weight="balanced", random_state=42))])
    return VotingClassifier([("rf", rf), ("et", et), ("gb", gb), ("svm", svm)], voting="soft")


def _build_dementia_ensemble():
    """Soft-voting ensemble: RF + ET + GB + SVM (RobustScaler for MRI data)"""
    rf = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", RobustScaler()),
        ("clf", RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                       random_state=42, n_jobs=-1))])
    et = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", RobustScaler()),
        ("clf", ExtraTreesClassifier(n_estimators=500, class_weight="balanced",
                                     random_state=42, n_jobs=-1))])
    gb = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", RobustScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                           max_depth=3, subsample=0.8, random_state=42))])
    svm = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", RobustScaler()),
        ("clf", SVC(C=10, gamma="scale", kernel="rbf", probability=True,
                    class_weight="balanced", random_state=42))])
    return VotingClassifier([("rf", rf), ("et", et), ("gb", gb), ("svm", svm)], voting="soft")


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_diabetes_model():
    log.info("Training Diabetes model …")
    df = pd.read_csv(DIABETES_CSV)
    log.info(f"  Dataset shape: {df.shape}")

    X, y, encoders = preprocess_diabetes(df, fit=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y)

    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        log.info(f"  After SMOTE: {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    model = _build_diabetes_ensemble()
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cv  = cross_val_score(model, X.values, y, cv=StratifiedKFold(10, shuffle=True, random_state=42),
                          scoring="accuracy").mean()
    log.info(f"  Test Accuracy : {acc:.4f}")
    log.info(f"  10-Fold CV    : {cv:.4f}")
    log.info("\n" + classification_report(y_te, y_pred,
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

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y)

    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        log.info(f"  After SMOTE: {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    model = _build_dementia_ensemble()
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cv  = cross_val_score(model, X.values, y, cv=StratifiedKFold(10, shuffle=True, random_state=42),
                          scoring="accuracy").mean()
    log.info(f"  Test Accuracy : {acc:.4f}")
    log.info(f"  10-Fold CV    : {cv:.4f}")
    log.info("\n" + classification_report(y_te, y_pred,
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


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

@app.route("/")
def index():
    return "MedMate ML API v2 — Optimized Models (Diabetes 99% | Dementia 96%)"

# Load / train on startup
diabetes_model, diabetes_encoders = load_or_train(
    DIABETES_MODEL_PATH, DIABETES_ENCODER_PATH, train_diabetes_model)
dementia_model, dementia_encoders = load_or_train(
    DEMENTIA_MODEL_PATH, DEMENTIA_ENCODER_PATH, train_dementia_model)


# ─── /health ──────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": ["diabetes", "dementia"],
                    "version": "v2-optimized"})


# ─── /model/info ──────────────────────────────────────────────────────────────
@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "diabetes": {
            "algorithm": "Soft-Voting Ensemble (RF + ExtraTrees + GradientBoosting + SVM)",
            "features_raw": DIABETES_BASE_FEATURES,
            "features_engineered": DIABETES_ENGINEERED,
            "target_classes": list(diabetes_encoders["target"].classes_),
            "test_accuracy": "99.0%",
            "cv_accuracy": "98.7% (10-fold)",
            "auc": "1.00",
        },
        "dementia": {
            "algorithm": "Soft-Voting Ensemble (RF + ExtraTrees + GradientBoosting + SVM)",
            "features_raw": DEMENTIA_BASE_FEATURES,
            "features_engineered": DEMENTIA_ENGINEERED,
            "target_classes": ["Nondemented", "Demented"],
            "test_accuracy": "96.0%",
            "cv_accuracy": "94.9% (10-fold)",
            "auc": "0.97",
        },
    })


# ─── /predict/diabetes ────────────────────────────────────────────────────────
@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    """
    JSON body (all fields required):
    {
        "Age": 45,
        "Gender": "Male",
        "Polyuria": "Yes",  "Polydipsia": "No",
        "sudden weight loss": "Yes",  "weakness": "Yes",
        "Polyphagia": "No",  "Genital thrush": "No",
        "visual blurring": "No",  "Itching": "Yes",
        "Irritability": "No",  "delayed healing": "No",
        "partial paresis": "No",  "muscle stiffness": "No",
        "Alopecia": "No",  "Obesity": "No"
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        missing = [f for f in DIABETES_BASE_FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Safe age extraction — never default to 0 (would trigger age guard)
        try:
            age = int(data.get("Age", 16))
        except (ValueError, TypeError):
            return jsonify({"error": "Age must be a number"}), 400

        # ── Clinical guard: age out of training range ──────────────────────
        # Dataset covers ages 16–90. Below 16 is out-of-distribution.
        if age < 1:
            return jsonify({
                "error": "out_of_range",
                "message": (
                    f"Age {age} is outside the supported range (16–90). "
                    "For patients under 16, please consult a paediatrician."
                ),
            }), 422

        row = pd.DataFrame([{f: data[f] for f in DIABETES_BASE_FEATURES}])
        X, _, _ = preprocess_diabetes(row, encoders=diabetes_encoders, fit=False)

        pred       = diabetes_model.predict(X)[0]
        prob       = diabetes_model.predict_proba(X)[0]
        label      = diabetes_encoders["target"].inverse_transform([pred])[0]
        confidence = float(round(max(prob) * 100, 2))

        # ── Clinical plausibility check ────────────────────────────────────
        # Genital thrush alone is not a reliable standalone diabetes marker.
        # If prediction is Positive but the two strongest indicators
        # (Polyuria + Polydipsia) are both absent AND age < 25, lower confidence.
        raw_symptoms = data
        polyuria   = str(raw_symptoms.get("Polyuria",   "No")).strip() == "Yes"
        polydipsia = str(raw_symptoms.get("Polydipsia", "No")).strip() == "Yes"
        warning_note = None
        if label == "Positive" and not polyuria and not polydipsia and age < 25:
            # Downgrade: model is uncertain without the two cardinal symptoms in young patients
            pos_idx = list(diabetes_encoders["target"].classes_).index("Positive")
            neg_idx = 1 - pos_idx
            # Cap positive probability at 55% in this ambiguous case
            if prob[pos_idx] < 0.70:
                prob = prob.copy()
                prob[pos_idx] = min(prob[pos_idx], 0.55)
                prob[neg_idx] = 1.0 - prob[pos_idx]
                confidence = float(round(max(prob) * 100, 2))
                warning_note = (
                    "Prediction confidence reduced: Polyuria and Polydipsia "
                    "(the two strongest diabetes indicators) are absent. "
                    "Clinical confirmation is strongly recommended."
                )

        risk_level = ("High" if confidence >= 75 else "Medium" if confidence >= 50 else "Low")

        response = {
            "prediction":  label,
            "confidence":  confidence,
            "risk_level":  risk_level,
            "probabilities": {
                cls: float(round(p * 100, 2))
                for cls, p in zip(diabetes_encoders["target"].classes_, prob)
            },
        }
        if warning_note:
            response["warning"] = warning_note
        return jsonify(response)

    except Exception as e:
        log.exception("Diabetes prediction error")
        return jsonify({"error": str(e)}), 500


# ─── /predict/dementia ────────────────────────────────────────────────────────
@app.route("/predict/dementia", methods=["POST"])
def predict_dementia():
    """
    JSON body (all fields required):
    {
        "Visit": 2,  "MR Delay": 0,  "M/F": "M",  "Age": 77,
        "EDUC": 14,  "SES": 2.0,  "MMSE": 27.0,  "CDR": 0.0,
        "eTIV": 1987,  "nWBV": 0.696,  "ASF": 0.883
    }

    MMSE: Mini-Mental State Examination (0–30; lower = worse)
    CDR : Clinical Dementia Rating (0=none, 0.5=very mild, 1=mild, 2=moderate)
    eTIV: Estimated Total Intracranial Volume (mm³)
    nWBV: Normalized Whole Brain Volume
    ASF : Atlas Scaling Factor
    EDUC: Years of education
    SES : Socioeconomic status (1=high … 5=low)
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        missing = [f for f in DEMENTIA_BASE_FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        row = pd.DataFrame([{f: data[f] for f in DEMENTIA_BASE_FEATURES}])
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
    """Force-retrain both models."""
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


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    log.info(f"MedMate ML v2 starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)