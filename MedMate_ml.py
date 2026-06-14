"""
MedMate_ML.py  ── v5  (Auth + Dashboard)
==========================================
Flask backend for Early-Stage Diabetes Readmission & Dementia Detection.

NEW in v5
  ─ PostgreSQL user auth  (sign up / sign in via JWT)
  ─ Prediction logging per user
  ─ /dashboard endpoint  (history, stats, last login)

DATASETS
  Diabetes : UCI/Kaggle 130-hospital diabetes readmission dataset
             (diabetic_data.csv — 101 766 encounters, 50 cols)
             Target → readmitted within 30 days  (binary: 0/1)

  Dementia : NACC Uniform Data Set (investigator_nacc73.csv — 215k visits)
             Target → DEMENTED  (binary: 0=no, 1=yes)

ENDPOINTS (public)
  POST /auth/signup
  POST /auth/login
  GET  /health
  GET  /model/info

ENDPOINTS (JWT required)
  GET  /auth/me
  POST /predict/diabetes
  POST /predict/dementia
  GET  /dashboard
  POST /retrain              (admin key + JWT)
"""

import os, sys, warnings, logging, functools, datetime
from pathlib import Path

# ── Windows Store Python fix ───────────────────────────────────────────────
# Must happen BEFORE joblib/sklearn are imported.
# Windows Store Python blocks wmic (used by loky to count physical cores).
# LOKY_MAX_CPU_COUNT tells loky to skip the wmic call entirely.
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# Always use n_jobs=1 — avoids loky worker-pool creation on Windows entirely.
_N_JOBS = 1

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import psycopg2
import psycopg2.extras
import bcrypt
import jwt as pyjwt
from dotenv import load_dotenv

from flask import Flask, request, jsonify, g
from flask_cors import CORS

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

warnings.filterwarnings("ignore")
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR              = Path(__file__).parent
MODEL_DIR             = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

DIABETES_CSV          = BASE_DIR / os.getenv("DIABETES_CSV", "diabetic_data.csv")
DEMENTIA_CSV          = BASE_DIR / os.getenv("DEMENTIA_CSV", "investigator_nacc73.csv")
DIABETES_MODEL_PATH   = MODEL_DIR / "diabetes_model.pkl"
DEMENTIA_MODEL_PATH   = MODEL_DIR / "dementia_model.pkl"
DIABETES_ENCODER_PATH = MODEL_DIR / "diabetes_encoders.pkl"
DEMENTIA_ENCODER_PATH = MODEL_DIR / "dementia_encoders.pkl"

DATABASE_URL          = os.getenv("DATABASE_URL")          # required
JWT_SECRET            = os.getenv("JWT_SECRET", "medmate_jwt_secret_change_me")
JWT_EXPIRY_HOURS      = int(os.getenv("JWT_EXPIRY_HOURS", 24))
ADMIN_KEY             = os.getenv("ADMIN_KEY", "medmate_admin")

# ─── Clinical ranges ──────────────────────────────────────────────────────────
DIABETES_AGE_MIN, DIABETES_AGE_MAX = 0, 100
DEMENTIA_AGE_MIN, DEMENTIA_AGE_MAX = 18, 110
MMSE_MIN,  MMSE_MAX  = 0, 30
CDR_VALID            = {0.0, 0.5, 1.0, 2.0, 3.0}

# ─── Feature definitions ──────────────────────────────────────────────────────
DIABETES_BASE_FEATURES = [
    "age", "gender", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "number_diagnoses",
    "A1Cresult", "max_glu_serum", "insulin", "metformin",
    "change", "diabetesMed", "diag_1", "diag_2", "diag_3",
]
DIABETES_FEATURES = [
    "age_num", "gender_bin", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "number_diagnoses",
    "A1C_ord", "glu_ord", "insulin_ord", "metformin_ord",
    "change_bin", "diabetesMed_bin",
    "diag1_diabetes", "diag2_diabetes", "diag3_diabetes",
    "total_visits", "service_use", "num_meds_x_time",
    # Additional domain features (added in Train_models.py v5.1 — must match saved model)
    "prior_inpatient", "prior_inpatient_cnt",
    "high_emergency", "A1C_tested", "high_glucose",
    "insulin_changed", "high_med_burden", "long_stay", "diag1_circ",
]

DEMENTIA_BASE_FEATURES = [
    "NACCAGE", "SEX", "EDUC", "CDRGLOB", "CDRSUM",
    "NACCMMSE", "NACCGDS", "ANIMALS", "TRAILA", "TRAILB",
    "DIABETES", "HYPERTEN", "NACCDEP", "APOE4", "NACCBMI",
]
DEMENTIA_ENGINEERED = [
    "CDR_MMSE_ratio", "CDR_x_age", "MMSE_below_24", "CDR_nonzero",
    "age_educ_ratio", "trail_ratio", "comorbidity_sum",
    # Additional engineered features (added in Train_models.py v5.1 — must match saved model)
    "CDR_sum_x_mmse", "severe_CDR", "MMSE_quartile",
    "APOE4_x_age", "trail_diff", "functional_cognitive_gap",
]
DEMENTIA_FEATURES = DEMENTIA_BASE_FEATURES + DEMENTIA_ENGINEERED


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    """Return a new psycopg2 connection (caller must close it)."""
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def init_db():
    """Create tables if they don't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS users (
        id            SERIAL PRIMARY KEY,
        email         VARCHAR(255) UNIQUE NOT NULL,
        password_hash TEXT        NOT NULL,
        full_name     VARCHAR(255),
        created_at    TIMESTAMPTZ DEFAULT NOW(),
        last_login    TIMESTAMPTZ
    );

    CREATE TABLE IF NOT EXISTS prediction_logs (
        id            SERIAL PRIMARY KEY,
        user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
        model_type    VARCHAR(50)  NOT NULL,   -- 'diabetes' | 'dementia'
        inputs        JSONB        NOT NULL,
        prediction    VARCHAR(100) NOT NULL,
        confidence    FLOAT        NOT NULL,
        risk_level    VARCHAR(50)  NOT NULL,
        probabilities JSONB        NOT NULL,
        created_at    TIMESTAMPTZ  DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_prediction_logs_user_id
        ON prediction_logs(user_id);
    CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_type
        ON prediction_logs(model_type);
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    log.info("DB tables ready.")


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def check_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))
    except ValueError:
        return False


def create_token(user_id: int, email: str) -> str:
    payload = {
        "sub":   str(user_id),
        "email": email,
        "iat":   datetime.datetime.utcnow(),
        "exp":   datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> dict:
    return pyjwt.decode(token, JWT_SECRET, algorithms=["HS256"])


def require_auth(f):
    """Decorator — validates Bearer JWT and sets g.user_id / g.user_email."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            payload = decode_token(token)
        except pyjwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except pyjwt.InvalidTokenError as e:
            print("InvalidTokenError:", str(e))
            return jsonify({"error": f"Invalid token: {str(e)}"}), 401
        g.user_id    = int(payload["sub"])
        g.user_email = payload["email"]
        return f(*args, **kwargs)
    return wrapper


def log_prediction(user_id: int, model_type: str, inputs: dict,
                   prediction: str, confidence: float,
                   risk_level: str, probabilities: dict):
    """Insert a prediction record into prediction_logs."""
    import json
    sql = """
        INSERT INTO prediction_logs
            (user_id, model_type, inputs, prediction, confidence, risk_level, probabilities)
        VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s::jsonb)
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    user_id, model_type,
                    json.dumps(inputs), prediction, confidence,
                    risk_level, json.dumps(probabilities),
                ))
            conn.commit()
    except Exception as e:
        log.warning(f"Failed to log prediction: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING  (unchanged from v4)
# ══════════════════════════════════════════════════════════════════════════════

AGE_MAP  = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}
A1C_MAP  = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
GLU_MAP  = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
INS_MAP  = {"No": 0, "Steady": 1, "Up": 2, "Down": -1}
METF_MAP = {"No": 0, "Steady": 1, "Up": 2, "Down": -1}
NACC_SENTINELS = [-4, 8, 9, 88, 99, 888, 999]


def _is_diabetes_icd(code):
    try:
        return 1 if str(code).split(".")[0] == "250" else 0
    except Exception:
        return 0


def _is_circulatory(code):
    try:
        c = str(code).split(".")[0]
        return 1 if c.isdigit() and 390 <= int(c) <= 459 else 0
    except Exception:
        return 0


def preprocess_diabetes(df, encoders=None, fit=True):
    df = df.copy()
    if fit:
        encoders = {}
    df["age_num"]         = df["age"].map(AGE_MAP).fillna(45).astype(float)
    df["gender_bin"]      = (df["gender"].astype(str).str.strip() == "Female").astype(int)
    df["A1C_ord"]         = df["A1Cresult"].map(A1C_MAP).fillna(0)
    df["glu_ord"]         = df["max_glu_serum"].map(GLU_MAP).fillna(0)
    df["insulin_ord"]     = df["insulin"].map(INS_MAP).fillna(0)
    df["metformin_ord"]   = df["metformin"].map(METF_MAP).fillna(0)
    df["change_bin"]      = (df["change"].astype(str).str.strip() == "Ch").astype(int)
    df["diabetesMed_bin"] = (df["diabetesMed"].astype(str).str.strip() == "Yes").astype(int)
    df["diag1_diabetes"]  = df["diag_1"].apply(_is_diabetes_icd)
    df["diag2_diabetes"]  = df["diag_2"].apply(_is_diabetes_icd)
    df["diag3_diabetes"]  = df["diag_3"].apply(_is_diabetes_icd)
    df["total_visits"]    = (df["number_outpatient"] + df["number_emergency"]
                              + df["number_inpatient"])
    df["service_use"]     = df["num_lab_procedures"] + df["num_procedures"]
    df["num_meds_x_time"] = df["num_medications"] * df["time_in_hospital"]
    # Additional domain features — must match Train_models.py v5.1 feature set
    df["prior_inpatient"]     = (pd.to_numeric(df["number_inpatient"], errors="coerce").fillna(0) > 0).astype(int)
    df["prior_inpatient_cnt"] = pd.to_numeric(df["number_inpatient"], errors="coerce").fillna(0).clip(0, 10)
    df["high_emergency"]      = (pd.to_numeric(df["number_emergency"], errors="coerce").fillna(0) >= 2).astype(int)
    df["A1C_tested"]          = (df["A1C_ord"] > 0).astype(int)
    df["high_glucose"]        = (df["glu_ord"] >= 2).astype(int)
    df["insulin_changed"]     = (df["insulin_ord"] != 0).astype(int)
    df["high_med_burden"]     = (pd.to_numeric(df["num_medications"], errors="coerce").fillna(0) > 15).astype(int)
    df["long_stay"]           = (pd.to_numeric(df["time_in_hospital"], errors="coerce").fillna(0) > 7).astype(int)
    df["diag1_circ"]          = df["diag_1"].apply(_is_circulatory)
    y = None
    if "readmitted" in df.columns:
        y = (df["readmitted"].astype(str).str.strip() == "<30").astype(int).values
    elif "target" in df.columns:
        y = df["target"].values
    X = df[DIABETES_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, y, encoders


def preprocess_dementia(df, encoders=None, fit=True):
    df = df.copy()
    if fit:
        encoders = {}
    df.replace(NACC_SENTINELS, np.nan, inplace=True)
    df.loc[df["NACCMMSE"] > 30,  "NACCMMSE"] = np.nan
    df.loc[df["NACCBMI"]  > 80,  "NACCBMI"]  = np.nan
    df.loc[df["TRAILA"]   > 500, "TRAILA"]   = np.nan
    df.loc[df["TRAILB"]   > 500, "TRAILB"]   = np.nan
    df.loc[df["ANIMALS"]  > 60,  "ANIMALS"]  = np.nan
    df.loc[df["NACCGDS"]  > 15,  "NACCGDS"]  = np.nan
    if "NACCSEX" in df.columns:
        df["SEX"] = df["NACCSEX"].map({1: 0, 2: 1})
    elif "SEX" not in df.columns:
        df["SEX"] = 0
    if "NACCAPOE" in df.columns:
        df["APOE4"] = df["NACCAPOE"].map({1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1})
    elif "APOE4" not in df.columns:
        df["APOE4"] = np.nan
    for col in DEMENTIA_BASE_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        key = f"{col}_median"
        if fit:
            med = df[col].median()
            encoders[key] = float(med) if not np.isnan(med) else 0.0
        df[col] = df[col].fillna(encoders.get(key, 0.0))
    mmse = df["NACCMMSE"].clip(0, 30)
    df["CDR_MMSE_ratio"] = df["CDRGLOB"] / (mmse + 1)
    df["CDR_x_age"]      = df["CDRGLOB"] * df["NACCAGE"]
    df["MMSE_below_24"]  = (mmse < 24).astype(int)
    df["CDR_nonzero"]    = (df["CDRGLOB"] > 0).astype(int)
    df["age_educ_ratio"] = df["NACCAGE"] / (df["EDUC"] + 1)
    df["trail_ratio"]    = df["TRAILA"] / (df["TRAILB"] + 1)
    df["comorbidity_sum"]= (df["DIABETES"].fillna(0) + df["HYPERTEN"].fillna(0)
                             + df["NACCDEP"].fillna(0))
    # Additional engineered features — must match Train_models.py v5.1 feature set
    df["CDR_sum_x_mmse"]           = df["CDRSUM"] * (30 - mmse)
    df["severe_CDR"]                = (df["CDRGLOB"] >= 2).astype(int)
    df["MMSE_quartile"]             = pd.cut(mmse, bins=[-1, 7.5, 15, 22.5, 30],
                                             labels=[0, 1, 2, 3]).astype(float).fillna(0)
    df["APOE4_x_age"]              = df["APOE4"] * df["NACCAGE"]
    df["trail_diff"]               = (df["TRAILB"] - df["TRAILA"]).clip(0, 500)
    df["functional_cognitive_gap"] = df["CDRSUM"] / (mmse + 1)
    y = None
    if "DEMENTED" in df.columns:
        y = df["DEMENTED"].values.astype(int)
    X = df[DEMENTIA_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, y, encoders


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL BUILDING & TRAINING  (unchanged from v4)
# ══════════════════════════════════════════════════════════════════════════════

def _build_ensemble(robust=False):
    sc_cls = RobustScaler if robust else StandardScaler
    def pipe(clf):
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  sc_cls()), ("clf", clf)])
    rf  = pipe(RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                      random_state=42, n_jobs=_N_JOBS))
    et  = pipe(ExtraTreesClassifier(n_estimators=500, class_weight="balanced",
                                    random_state=42, n_jobs=_N_JOBS))
    gb  = pipe(GradientBoostingClassifier(n_estimators=300, learning_rate=0.08,
                                          max_depth=4, subsample=0.85, random_state=42))
    svm = pipe(SVC(C=10, gamma="scale", kernel="rbf", probability=True,
                   class_weight="balanced", random_state=42))
    return VotingClassifier([("rf", rf), ("et", et), ("gb", gb), ("svm", svm)],
                            voting="soft")


def _safe_smote(X_tr, y_tr):
    if not SMOTE_AVAILABLE:
        return X_tr, y_tr
    min_count = int(np.bincount(y_tr).min())
    k = min(5, min_count - 1)
    if k < 1:
        log.warning("Minority class too small for SMOTE — skipping.")
        return X_tr, y_tr
    X_res, y_res = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_tr, y_tr)
    log.info(f"  SMOTE (k={k}): {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


def train_diabetes_model():
    log.info("Training Diabetes model …")
    df = pd.read_csv(DIABETES_CSV, low_memory=False)
    log.info(f"  Dataset shape: {df.shape}")
    X, y, encoders = preprocess_diabetes(df, fit=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, y_tr = _safe_smote(X_tr, y_tr)
    model = _build_ensemble(robust=False)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    cv  = cross_val_score(model, X.values, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring="roc_auc").mean()
    log.info(f"  Test Accuracy: {acc:.4f}  |  5-Fold CV AUC: {cv:.4f}")
    log.info("\n" + classification_report(y_te, model.predict(X_te),
             target_names=["Not Readmitted <30d", "Readmitted <30d"]))
    joblib.dump(model,    DIABETES_MODEL_PATH)
    joblib.dump(encoders, DIABETES_ENCODER_PATH)
    log.info(f"  Saved → {DIABETES_MODEL_PATH}")
    return model, encoders


def train_dementia_model():
    log.info("Training Dementia model …")
    df = pd.read_csv(DEMENTIA_CSV, low_memory=False,
                     usecols=["NACCAGE", "NACCSEX", "EDUC", "CDRGLOB", "CDRSUM",
                               "NACCMMSE", "NACCGDS", "ANIMALS", "TRAILA", "TRAILB",
                               "DIABETES", "HYPERTEN", "NACCDEP", "NACCAPOE",
                               "NACCBMI", "DEMENTED"],
                     on_bad_lines="skip")
    log.info(f"  Dataset shape: {df.shape}")
    X, y, encoders = preprocess_dementia(df, fit=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, y_tr = _safe_smote(X_tr, y_tr)
    model = _build_ensemble(robust=True)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    cv  = cross_val_score(model, X.values, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring="roc_auc").mean()
    log.info(f"  Test Accuracy: {acc:.4f}  |  5-Fold CV AUC: {cv:.4f}")
    log.info("\n" + classification_report(y_te, model.predict(X_te),
             target_names=["Nondemented", "Demented"]))
    joblib.dump(model,    DEMENTIA_MODEL_PATH)
    joblib.dump(encoders, DEMENTIA_ENCODER_PATH)
    log.info(f"  Saved → {DEMENTIA_MODEL_PATH}")
    return model, encoders


def load_or_train(model_path, encoder_path, train_fn):
    if model_path.exists() and encoder_path.exists():
        log.info(f"Loading cached model from {model_path}")
        try:
            return joblib.load(model_path), joblib.load(encoder_path)
        except Exception as e:
            log.warning(f"Load failed ({type(e).__name__}: {e}) — retraining.")
    return train_fn()


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
}})

# ── DB init + model load ──────────────────────────────────────────────────────
init_db()

# ── Lazy model loader ──────────────────────────────────────────────────────────
import threading
_model_lock = threading.Lock()
_models = {}

def get_model(kind: str):
    if kind not in _models:
        with _model_lock:
            if kind not in _models:
                if kind == "diabetes":
                    _models[kind] = load_or_train(
                        DIABETES_MODEL_PATH, DIABETES_ENCODER_PATH, train_diabetes_model)
                else:
                    _models[kind] = load_or_train(
                        DEMENTIA_MODEL_PATH, DEMENTIA_ENCODER_PATH, train_dementia_model)
    return _models[kind]


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/auth/signup", methods=["POST"])
def signup():
    """
    Body: { "email": "...", "password": "...", "full_name": "..." }
    Returns: { "token": "...", "user": { id, email, full_name, created_at } }
    """
    data = request.get_json(force=True) or {}
    email     = (data.get("email") or "").strip().lower()
    password  = data.get("password") or ""
    full_name = (data.get("full_name") or "").strip()

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    pw_hash = hash_password(password)
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO users (email, password_hash, hashed_password, name, full_name, created_at)
                       VALUES (%s, %s, %s, %s, %s, NOW())
                       RETURNING id, email, full_name, created_at""",
                    (email, pw_hash, pw_hash, full_name, full_name),
                )
                user = dict(cur.fetchone())
                # Record first login time
                cur.execute("UPDATE users SET last_login=NOW() WHERE id=%s", (user["id"],))
            conn.commit()
    except psycopg2.errors.UniqueViolation:
        return jsonify({"error": "Email already registered"}), 409
    except Exception as e:
        log.exception("Signup error")
        return jsonify({"error": str(e)}), 500

    token = create_token(user["id"], user["email"])
    user["created_at"] = user["created_at"].isoformat() if user.get("created_at") else None
    return jsonify({"token": token, "user": user}), 201


@app.route("/auth/login", methods=["POST"])
def login():
    """
    Body: { "email": "...", "password": "..." }
    Returns: { "token": "...", "user": { id, email, full_name, last_login } }
    """
    data     = request.get_json(force=True) or {}
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, email, password_hash, full_name, last_login FROM users WHERE email=%s",
                    (email,),
                )
                row = cur.fetchone()
                if not row or not check_password(password, row["password_hash"]):
                    return jsonify({"error": "Invalid email or password"}), 401
                # Update last_login
                cur.execute("UPDATE users SET last_login=NOW() WHERE id=%s", (row["id"],))
            conn.commit()
    except Exception as e:
        log.exception("Login error")
        return jsonify({"error": str(e)}), 500

    user = {
        "id":         row["id"],
        "email":      row["email"],
        "full_name":  row["full_name"],
        "last_login": row["last_login"].isoformat() if row["last_login"] else None,
    }
    token = create_token(user["id"], user["email"])
    return jsonify({"token": token, "user": user})


@app.route("/auth/me", methods=["GET"])
@require_auth
def me():
    """Returns the current user's profile (JWT required)."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, email, full_name, created_at, last_login FROM users WHERE id=%s",
                    (g.user_id,),
                )
                row = cur.fetchone()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not row:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "id":         row["id"],
        "email":      row["email"],
        "full_name":  row["full_name"],
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "last_login": row["last_login"].isoformat() if row["last_login"] else None,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/dashboard", methods=["GET"])
@require_auth
def dashboard():
    """
    Returns the authenticated user's activity dashboard.

    Query params:
      limit  (int, default 20)  — number of history records to return
      offset (int, default 0)   — for pagination
      model  (str, optional)    — filter by 'diabetes' or 'dementia'

    Response:
    {
      "user": { id, email, full_name, created_at, last_login },
      "stats": {
        "total_predictions": int,
        "diabetes_predictions": int,
        "dementia_predictions": int,
        "high_risk_count": int,
        "medium_risk_count": int,
        "low_risk_count": int,
        "avg_confidence": float
      },
      "history": [
        {
          "id": int,
          "model_type": str,
          "prediction": str,
          "confidence": float,
          "risk_level": str,
          "probabilities": dict,
          "inputs": dict,
          "created_at": str (ISO 8601)
        },
        ...
      ],
      "pagination": { "limit": int, "offset": int, "total": int }
    }
    """
    limit      = min(int(request.args.get("limit",  20)), 100)
    offset     = int(request.args.get("offset", 0))
    model_filter = request.args.get("model", None)

    try:
        with get_db() as conn:
            with conn.cursor() as cur:

                # ── User profile ──────────────────────────────────────────────
                cur.execute(
                    "SELECT id, email, full_name, created_at, last_login FROM users WHERE id=%s",
                    (g.user_id,),
                )
                user_row = cur.fetchone()
                if not user_row:
                    return jsonify({"error": "User not found"}), 404

                # ── Aggregate stats ───────────────────────────────────────────
                cur.execute("""
                    SELECT
                        COUNT(*)                                          AS total_predictions,
                        COUNT(*) FILTER (WHERE model_type='diabetes')    AS diabetes_predictions,
                        COUNT(*) FILTER (WHERE model_type='dementia')    AS dementia_predictions,
                        COUNT(*) FILTER (WHERE risk_level='High')        AS high_risk_count,
                        COUNT(*) FILTER (WHERE risk_level='Medium')      AS medium_risk_count,
                        COUNT(*) FILTER (WHERE risk_level='Low')         AS low_risk_count,
                        ROUND(AVG(confidence)::numeric, 2)               AS avg_confidence
                    FROM prediction_logs
                    WHERE user_id = %s
                """, (g.user_id,))
                stats_row = dict(cur.fetchone())

                # ── History (paginated, optional model filter) ─────────────────
                base_where = "WHERE user_id = %s"
                params     = [g.user_id]
                if model_filter in ("diabetes", "dementia"):
                    base_where += " AND model_type = %s"
                    params.append(model_filter)

                cur.execute(
                    f"SELECT COUNT(*) AS cnt FROM prediction_logs {base_where}",
                    params,
                )
                total = cur.fetchone()["cnt"]

                cur.execute(f"""
                    SELECT id, model_type, prediction, confidence, risk_level,
                           probabilities, inputs, created_at
                    FROM prediction_logs
                    {base_where}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, params + [limit, offset])
                history = []
                for r in cur.fetchall():
                    history.append({
                        "id":           r["id"],
                        "model_type":   r["model_type"],
                        "prediction":   r["prediction"],
                        "confidence":   float(r["confidence"]),
                        "risk_level":   r["risk_level"],
                        "probabilities":r["probabilities"],
                        "inputs":       r["inputs"],
                        "created_at":   r["created_at"].isoformat(),
                    })

    except Exception as e:
        log.exception("Dashboard error")
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "user": {
            "id":         user_row["id"],
            "email":      user_row["email"],
            "full_name":  user_row["full_name"],
            "created_at": user_row["created_at"].isoformat() if user_row.get("created_at") else None,
            "last_login": user_row["last_login"].isoformat() if user_row["last_login"] else None,
        },
        "stats": {
            "total_predictions":   int(stats_row["total_predictions"]),
            "diabetes_predictions":int(stats_row["diabetes_predictions"]),
            "dementia_predictions":int(stats_row["dementia_predictions"]),
            "high_risk_count":     int(stats_row["high_risk_count"]),
            "medium_risk_count":   int(stats_row["medium_risk_count"]),
            "low_risk_count":      int(stats_row["low_risk_count"]),
            "avg_confidence":      float(stats_row["avg_confidence"] or 0),
        },
        "history": history,
        "pagination": {
            "limit":  limit,
            "offset": offset,
            "total":  int(total),
        },
    })



# ══════════════════════════════════════════════════════════════════════════════
#  DELETE HISTORY ENDPOINTS  (JWT required)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/dashboard/history/all", methods=["DELETE"])
@require_auth
def delete_history_all():
    """Delete ALL prediction logs for the authenticated user."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM prediction_logs WHERE user_id = %s",
                    (g.user_id,),
                )
                deleted = cur.rowcount
            conn.commit()
    except Exception as e:
        log.exception("Delete all history error")
        return jsonify({"error": str(e)}), 500
    return jsonify({"deleted": deleted})


@app.route("/dashboard/history/<int:log_id>", methods=["GET"])
@require_auth
def get_history_one(log_id):
    """Return a single prediction log belonging to the authenticated user."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT id, model_type, prediction, confidence, risk_level,
                              probabilities, inputs, created_at
                       FROM prediction_logs
                       WHERE id = %s AND user_id = %s""",
                    (log_id, g.user_id),
                )
                row = cur.fetchone()
    except Exception as e:
        log.exception("Get history item error")
        return jsonify({"error": str(e)}), 500
    if not row:
        return jsonify({"error": "Record not found or not owned by you"}), 404
    return jsonify({
        "id":           row["id"],
        "model_type":   row["model_type"],
        "prediction":   row["prediction"],
        "confidence":   float(row["confidence"]),
        "risk_level":   row["risk_level"],
        "probabilities":row["probabilities"],
        "inputs":       row["inputs"],
        "created_at":   row["created_at"].isoformat(),
    })


@app.route("/dashboard/history/<int:log_id>", methods=["DELETE"])
@require_auth
def delete_history_one(log_id):
    """Delete a single prediction log (must belong to the authenticated user)."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM prediction_logs WHERE id = %s AND user_id = %s",
                    (log_id, g.user_id),
                )
                deleted = cur.rowcount
            conn.commit()
    except Exception as e:
        log.exception("Delete history item error")
        return jsonify({"error": str(e)}), 500
    if deleted == 0:
        return jsonify({"error": "Record not found or not owned by you"}), 404
    return jsonify({"deleted": deleted})





# ══════════════════════════════════════════════════════════════════════════════
#  HEALTH / INFO
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return "MedMate ML API v5 — Diabetes Readmission & Dementia Detection (Auth enabled)"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": ["diabetes", "dementia"],
                    "version": "v5", "auth": "jwt"})

@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "diabetes": {
            "dataset":   "UCI 130-Hospital Diabetes (101 766 encounters)",
            "target":    "Readmitted within 30 days (binary)",
            "algorithm": "Soft-Voting Ensemble (RF + ET + GB + SVM)",
            "features":  DIABETES_FEATURES,
        },
        "dementia": {
            "dataset":   "NACC UDS investigator_nacc73 (~215k visits)",
            "target":    "Dementia diagnosis (DEMENTED binary)",
            "algorithm": "Soft-Voting Ensemble (RF + ET + GB + SVM)",
            "features":  DEMENTIA_FEATURES,
        },
    })


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION ENDPOINTS  (JWT required)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/predict/diabetes", methods=["POST"])
@require_auth
def predict_diabetes():
    """
    Header: Authorization: Bearer <token>
    Body  : same as v4 diabetes prediction JSON
    Logs the result to prediction_logs for the authenticated user.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        missing = [f for f in DIABETES_BASE_FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        age_str = str(data.get("age", "")).strip()
        if age_str not in AGE_MAP:
            return jsonify({"error": f"age must be one of {list(AGE_MAP.keys())}"}), 422

        row = pd.DataFrame([{f: data[f] for f in DIABETES_BASE_FEATURES}])
        diabetes_model, diabetes_encoders = get_model("diabetes")
        X, _, _ = preprocess_diabetes(row, encoders=diabetes_encoders, fit=False)

        pred        = diabetes_model.predict(X.values)[0]
        prob        = diabetes_model.predict_proba(X.values)[0]
        label       = "Readmitted <30d" if pred == 1 else "Not Readmitted"
        confidence  = float(round(max(prob) * 100, 2))
        risk_level  = ("High"   if pred == 1 and confidence >= 70
                        else "Medium" if pred == 1
                        else "Low")
        probabilities = {
            "Not Readmitted":  float(round(prob[0] * 100, 2)),
            "Readmitted <30d": float(round(prob[1] * 100, 2)),
        }

        log_prediction(g.user_id, "diabetes", data, label,
                       confidence, risk_level, probabilities)

        return jsonify({
            "prediction":   label,
            "confidence":   confidence,
            "risk_level":   risk_level,
            "probabilities":probabilities,
        })

    except Exception as e:
        log.exception("Diabetes prediction error")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/dementia", methods=["POST"])
@require_auth
def predict_dementia():
    """
    Header: Authorization: Bearer <token>
    Body  : same as v4 dementia prediction JSON
    Logs the result to prediction_logs for the authenticated user.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        missing = [f for f in DEMENTIA_BASE_FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        try:
            mmse = float(data.get("NACCMMSE", 27))
            cdr  = float(data.get("CDRGLOB",  0.0))
        except (ValueError, TypeError):
            return jsonify({"error": "NACCMMSE and CDRGLOB must be numbers"}), 400

        if not (MMSE_MIN <= mmse <= MMSE_MAX):
            return jsonify({"error": f"NACCMMSE must be 0–30. Got {mmse}."}), 422
        if cdr not in CDR_VALID:
            return jsonify({"error": f"CDRGLOB must be one of {sorted(CDR_VALID)}. Got {cdr}."}), 422

        row = pd.DataFrame([{f: data[f] for f in DEMENTIA_BASE_FEATURES}])
        dementia_model, dementia_encoders = get_model("dementia")
        X, _, _ = preprocess_dementia(row, encoders=dementia_encoders, fit=False)

        pred       = dementia_model.predict(X.values)[0]
        prob       = dementia_model.predict_proba(X.values)[0]
        label      = "Demented" if pred == 1 else "Nondemented"
        confidence = float(round(max(prob) * 100, 2))

        if pred == 1 and confidence >= 75:
            risk_level = "High"
        elif pred == 1:
            risk_level = "Medium"
        elif confidence >= 75:
            risk_level = "Low"
        else:
            risk_level = "Medium"

        probabilities = {
            "Nondemented": float(round(prob[0] * 100, 2)),
            "Demented":    float(round(prob[1] * 100, 2)),
        }

        log_prediction(g.user_id, "dementia", data, label,
                       confidence, risk_level, probabilities)

        return jsonify({
            "prediction":   label,
            "confidence":   confidence,
            "risk_level":   risk_level,
            "probabilities":probabilities,
        })

    except Exception as e:
        log.exception("Dementia prediction error")
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  RETRAIN  (admin)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/retrain", methods=["POST"])
@require_auth
def retrain():
    try:
        if request.headers.get("X-Admin-Key", "") != ADMIN_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        
        # Clear cache and force retrain
        with _model_lock:
            _models["diabetes"] = train_diabetes_model()
            _models["dementia"] = train_dementia_model()
            
        return jsonify({"status": "retrained successfully"})
    except Exception as e:
        log.exception("Retrain error")
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    log.info(f"MedMate ML v5 starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)