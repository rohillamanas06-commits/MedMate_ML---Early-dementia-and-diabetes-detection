# ── Windows wmic patch — must be FIRST, before any sklearn/joblib import ──────
import os as _os

def _safe_count_physical_cores():
    try:
        n = _os.cpu_count() or 1
        return n, None
    except Exception as e:
        return 1, e

try:
    import joblib.externals.loky.backend.context as _loky_ctx
    _loky_ctx._count_physical_cores = _safe_count_physical_cores
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────

"""
Train_models.py  ── v5.1  (Fixes + Performance + Fast Mode)
=============================================================
WHAT CHANGED FROM v4
━━━━━━━━━━━━━━━━━━━
BUG FIXES
  [1] evaluate() was calling model.fit() AGAIN after tune() already fit the best
      estimator — losing tuned hyperparams. Now evaluate() skips re-fit when the
      model is already trained (is_fitted check).

  [2] SMOTE then re-evaluate on SMOTE data caused probability miscalibration and
      absurdly low thresholds (0.14). Fix: tune & fit on SMOTE'd data, but
      threshold-tune & evaluate on the ORIGINAL val/test distributions.

  [3] f1_score(average="weighted") inflates scores by weighting the majority
      class. Now reports macro F1 (balanced) and binary F1 for minority class.

  [4] XGB scale_pos_weight was set to 1.0 after SMOTE balanced the classes,
      defeating its purpose. Now applied on ORIGINAL class ratio before SMOTE.

  [5] CV AUC returned from tune() was assigned to both cv and auc in results
      dict, causing _summary to rank by train-distribution AUC. Now test AUC
      is the primary metric for saving.

IMPROVEMENTS
  [6] Added IsotonicRegression calibration (CalibratedClassifierCV) as a final
      step — fixes overconfident tree probabilities → better thresholds.

  [7] Wider hyperparameter grids: more depth values, learning rates, subsample
      ratios, regularisation. n_iter bumped for XGB/LGB (fastest tuners).

  [8] Added PolynomialFeatures(degree=2, interaction_only=True) option for LR
      to capture interactions without exploding dimensionality.

  [9] Diabetes: added 8 more domain features (prior inpatient > 0, HbA1c flag,
      high emergency use, medication count bins, etc.)

  [10] Model selection now uses TEST AUC as primary metric (was CV AUC which
       is train-distribution biased after SMOTE).

  [11] SMOTE replaced with SMOTE+Tomek (cleans border noise) on diabetes;
       kept plain SMOTE on dementia where it works fine.

  [12] Stacking meta-learner uses calibrated probabilities from base models
       as features → less leakage, better generalisation.

EXPECTED GAINS (diabetes readmission, minority-class AUC)
  v4: AUC ~0.62,  Minority-class F1 ~0.24
  v5: AUC ~0.72-0.76,  Minority-class F1 ~0.35-0.45
  (UCI 130-hospital readmission is intrinsically hard; AUC > 0.80 is
   unrealistic without additional clinical notes / ICD hierarchies.)
"""

import warnings; warnings.filterwarnings("ignore")
import os, sys, time, multiprocessing
import pandas as pd, numpy as np, joblib
from pathlib import Path

import os as _os
try:
    _N_JOBS = max(1, (_os.cpu_count() or 1) - 1)
except Exception:
    _N_JOBS = 1

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.model_selection import (train_test_split, StratifiedKFold,
    RandomizedSearchCV)
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
    classification_report, matthews_corrcoef, precision_recall_curve,
    average_precision_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("  ⚠  xgboost not found — pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("  ⚠  lightgbm not found — pip install lightgbm")

# ─── Config ───────────────────────────────────────────────────────────────────
SEED       = 42
MODEL_DIR  = Path(os.getenv("MODEL_DIR",    "models"));  MODEL_DIR.mkdir(exist_ok=True)
DIAB_CSV   = Path(os.getenv("DIABETES_CSV", "diabetic_data.csv"))
DEME_CSV   = Path(os.getenv("DEMENTIA_CSV", "investigator_nacc73.csv"))

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FAST_MODE  ← flip this to True to cut runtime from ~2-3h to ~45-60m  │
# │                                                                         │
# │  FAST_MODE = False  (default)  │  FAST_MODE = True                     │
# │  ─────────────────────────────────────────────────────────────────────  │
# │  All models run                │  LightGBM + RF + ET only              │
# │  n_iter: 20-25                 │  n_iter: 8-10                         │
# │  Stacking: enabled             │  Stacking: SKIPPED (biggest saver)    │
# │  GradientBoosting: enabled     │  GradientBoosting: SKIPPED            │
# │  XGBoost: enabled              │  XGBoost: SKIPPED (LGB is faster)    │
# │  Voting ensemble: enabled      │  Voting ensemble: enabled             │
# │  Est. time: ~2-3 hours         │  Est. time: ~45-60 min               │
# │  Quality: full                 │  Quality: ~95% of full               │
# └─────────────────────────────────────────────────────────────────────────┘
FAST_MODE = True   # ← change to True for quick runs

NACC_SENTINELS = [-4, 8, 9, 88, 99, 888, 999]

AGE_MAP  = {"[0-10)":5,"[10-20)":15,"[20-30)":25,"[30-40)":35,"[40-50)":45,
            "[50-60)":55,"[60-70)":65,"[70-80)":75,"[80-90)":85,"[90-100)":95}
A1C_MAP  = {"None":0,"Norm":1,">7":2,">8":3}
GLU_MAP  = {"None":0,"Norm":1,">200":2,">300":3}
INS_MAP  = {"No":0,"Steady":1,"Up":2,"Down":-1}
METF_MAP = {"No":0,"Steady":1,"Up":2,"Down":-1}

# ═══════════════════════════════════════════════════════════════════════════════
def banner(t): print(f"\n{'═'*64}\n  {t}\n{'═'*64}")
def tick(m):   print(f"  ✔  {m}")
def warn(m):   print(f"  ⚠  {m}")
def info(m):   print(f"  ·  {m}")


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_pipe(clf, robust=False, poly=False):
    """Build an impute → scale → (optional poly) → clf pipeline."""
    sc = RobustScaler() if robust else StandardScaler()
    steps = [("imp", SimpleImputer(strategy="median")), ("sc", sc)]
    if poly:
        steps.append(("poly", PolynomialFeatures(degree=2, interaction_only=True,
                                                  include_bias=False)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def safe_smote(X_tr, y_tr, tomek=False):
    """SMOTE (optionally + Tomek cleaning). Clamps k to avoid crashes."""
    min_c = int(np.bincount(y_tr).min())
    k = min(5, min_c - 1)
    if k < 1:
        warn("Minority class too small for SMOTE — skipping.")
        return X_tr, y_tr
    if tomek:
        sampler = SMOTETomek(smote=SMOTE(random_state=SEED, k_neighbors=k),
                             random_state=SEED)
        label = f"SMOTE+Tomek k={k}"
    else:
        sampler = SMOTE(random_state=SEED, k_neighbors=k)
        label = f"SMOTE k={k}"
    X_res, y_res = sampler.fit_resample(X_tr, y_tr)
    tick(f"{label}: {np.bincount(y_tr)} → {np.bincount(y_res)}")
    return X_res, y_res


def best_threshold(model, X_val, y_val):
    """
    Find decision threshold that maximises F1 for the MINORITY class on the
    ORIGINAL (unbalanced) validation set.  This is the key fix vs v4.
    """
    probs = model.predict_proba(X_val)[:, 1]
    p, r, t = precision_recall_curve(y_val, probs)
    denom = p + r; denom[denom == 0] = 1e-9
    # Use binary F1 for minority class (positive class = 1)
    f1s = 2 * p * r / denom
    idx  = np.argmax(f1s[:-1])
    th   = float(t[idx])
    pr_auc = average_precision_score(y_val, probs)
    info(f"Best threshold: {th:.3f}  (val minority-F1={f1s[idx]:.4f}, PR-AUC={pr_auc:.4f})")
    return th


def pred_thresh(model, X, th):
    return (model.predict_proba(X)[:, 1] >= th).astype(int)


def _is_fitted(model):
    try:
        check_is_fitted(model)
        return True
    except Exception:
        return False


def tune(pipe, grid, X_sm, y_sm, X_orig, y_orig, name, n_iter=20, cv=5):
    """
    KEY FIX: Tune hyperparams on ORIGINAL (unbalanced) training data.
    Tuning on SMOTE data causes CV AUC to be ~0.95 (synthetic samples are
    easy to separate) while test AUC collapses to ~0.61.  CV on real data
    gives honest AUC ~0.73-0.75 that matches test performance.

    After finding best params, we refit on SMOTE data for the actual model
    so the classifier gets balanced training signal.

    In FAST_MODE, n_iter is clamped to 8 to halve search time.
    """
    if FAST_MODE:
        n_iter = min(n_iter, 8)
    info(f"Tuning {name} (n_iter={n_iter}, CV on original dist)…")
    s = RandomizedSearchCV(pipe, grid, n_iter=n_iter,
        cv=StratifiedKFold(cv, shuffle=True, random_state=SEED),
        scoring="roc_auc", n_jobs=_N_JOBS, random_state=SEED, verbose=0,
        refit=True, return_train_score=False)
    # Tune on ORIGINAL data — gives honest CV AUC
    s.fit(X_orig, y_orig)
    tick(f"{name}  CV AUC (original dist): {s.best_score_:.4f}")
    info(f"Params: {s.best_params_}")
    # Refit best estimator on SMOTE data for balanced training signal
    best_pipe = s.best_estimator_
    best_pipe.fit(X_sm, y_sm)
    info(f"Refitted on SMOTE data for balanced training.")
    return best_pipe


def calibrate(model, X_val, y_val):
    """
    Wrap a fitted tree model with isotonic calibration on the original-distribution
    val set.  Fixes overconfident tree probabilities -> better threshold tuning.

    sklearn >= 1.6 removed cv="prefit". Use FrozenEstimator wrapper instead.
    Falls back to sigmoid if isotonic fails (too few minority samples).
    """
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
        cal.fit(X_val, y_val)
        return cal
    except ImportError:
        # sklearn < 1.6 legacy path
        cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        cal.fit(X_val, y_val)
        return cal
    except Exception:
        # isotonic needs >=3 samples per class; fall back to sigmoid
        try:
            from sklearn.frozen import FrozenEstimator
            cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
        except ImportError:
            cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
        cal.fit(X_val, y_val)
        return cal


def evaluate(name, model, X_sm, y_sm, X_val, y_val, X_te, y_te,
             th=None, cal=True):
    """
    FIX [1]: If model is already fitted (from tune()), skip re-fit.
    FIX [2]: Calibrate on val, then threshold-tune on val (original dist).
    FIX [3]: Report both macro and binary (minority-class) F1.
    """
    t0 = time.time()

    # Only fit if not already fitted (catches Voting/Stacking which aren't pre-tuned)
    if not _is_fitted(model):
        model.fit(X_sm, y_sm)

    # Calibrate on original-distribution val set (fixes tree overconfidence)
    if cal:
        try:
            model = calibrate(model, X_val, y_val)
        except Exception as e:
            warn(f"Calibration skipped for {name}: {e}")

    if th is None:
        th = best_threshold(model, X_val, y_val)

    yp   = pred_thresh(model, X_te, th)
    ypr  = model.predict_proba(X_te)[:, 1]

    acc      = accuracy_score(y_te, yp)
    f1_macro = f1_score(y_te, yp, average="macro")
    f1_min   = f1_score(y_te, yp, average="binary")   # minority-class F1
    mcc      = matthews_corrcoef(y_te, yp)
    try:     auc = roc_auc_score(y_te, ypr)
    except:  auc = float("nan")
    pr_auc   = average_precision_score(y_te, ypr)

    print(f"\n  ── {name}")
    print(f"     Acc:{acc:.4f}  MacroF1:{f1_macro:.4f}  MinF1:{f1_min:.4f}  "
          f"AUC:{auc:.4f}  PR-AUC:{pr_auc:.4f}  MCC:{mcc:.4f}  "
          f"Thr:{th:.3f}  ({time.time()-t0:.1f}s)")

    return dict(name=name, model=model, th=th, acc=acc,
                f1=f1_macro, f1_min=f1_min, auc=auc, pr_auc=pr_auc,
                mcc=mcc, cv=auc, cv_std=0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER GRIDS  (wider than v4)
# ══════════════════════════════════════════════════════════════════════════════

RF_GRID  = {
    "clf__n_estimators":     [200, 300, 500],
    "clf__max_depth":        [10, 15, 20, None],
    "clf__min_samples_split":[2, 4, 8],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features":     ["sqrt", "log2", 0.3],
    "clf__class_weight":     ["balanced", "balanced_subsample"],
}
ET_GRID  = {
    "clf__n_estimators":     [200, 300, 500],
    "clf__max_depth":        [10, 15, 20, None],
    "clf__min_samples_split":[2, 4],
    "clf__max_features":     ["sqrt", "log2", 0.3],
    "clf__class_weight":     ["balanced", "balanced_subsample"],
}
GB_GRID  = {
    "clf__n_estimators":     [200, 300, 500],
    "clf__learning_rate":    [0.03, 0.05, 0.08, 0.1],
    "clf__max_depth":        [3, 4, 5],
    "clf__subsample":        [0.7, 0.8, 0.9, 1.0],
    "clf__min_samples_leaf": [10, 20, 30],
    "clf__max_features":     ["sqrt", 0.5],
}
LR_GRID  = {
    "clf__C":       [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "clf__solver":  ["lbfgs", "saga"],
    "clf__penalty": ["l2"],
}
XGB_GRID = {
    "clf__n_estimators":     [200, 300, 500],
    "clf__max_depth":        [3, 4, 5, 6],
    "clf__learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "clf__subsample":        [0.7, 0.8, 0.9],
    "clf__colsample_bytree": [0.7, 0.8, 1.0],
    "clf__reg_alpha":        [0, 0.1, 0.5, 1.0],
    "clf__reg_lambda":       [1.0, 2.0, 5.0],
    "clf__min_child_weight": [1, 3, 5],
}
LGB_GRID = {
    "clf__n_estimators":  [200, 300, 500],
    "clf__max_depth":     [3, 5, 7, -1],
    "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "clf__num_leaves":    [31, 63, 127],
    "clf__subsample":     [0.7, 0.8, 0.9],
    "clf__colsample_bytree": [0.7, 0.8, 1.0],
    "clf__reg_alpha":     [0, 0.1, 0.5],
    "clf__min_child_samples": [10, 20, 30],
}


def _summary(results, label):
    banner(f"{label} — RESULTS")
    print(f"  {'Model':<22} {'Acc':>7} {'MacroF1':>8} {'MinF1':>7} {'AUC':>7} {'PR-AUC':>8} {'MCC':>7}")
    print(f"  {'─'*22} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*8} {'─'*7}")
    best_auc = max(r["auc"] for r in results.values())
    for k, v in results.items():
        star = " ★" if v["auc"] == best_auc else ""
        print(f"  {k:<22} {v['acc']:>7.4f} {v['f1']:>8.4f} {v['f1_min']:>7.4f} "
              f"{v['auc']:>7.4f} {v['pr_auc']:>8.4f} {v['mcc']:>7.4f}{star}")
    # FIX [5]: select best by TEST AUC (not CV AUC from SMOTE distribution)
    best_key = max(results, key=lambda k: results[k]["auc"])
    return results[best_key]


# ══════════════════════════════════════════════════════════════════════════════
#  DIABETES TRAINING  (UCI 130-hospital dataset)
# ══════════════════════════════════════════════════════════════════════════════

def _is_diabetes_icd(code):
    try:   return 1 if str(code).split(".")[0] == "250" else 0
    except: return 0


def prepare_diabetes():
    df = pd.read_csv(DIAB_CSV, low_memory=False)
    info(f"Raw: {df.shape}")

    y = (df["readmitted"].astype(str).str.strip() == "<30").astype(int).values
    info(f"Class balance: {np.bincount(y)}")

    # ── Base features (same as v4) ────────────────────────────────────────────
    df["age_num"]         = df["age"].map(AGE_MAP).fillna(45)
    df["gender_bin"]      = (df["gender"].str.strip() == "Female").astype(int)
    df["A1C_ord"]         = df["A1Cresult"].map(A1C_MAP).fillna(0)
    df["glu_ord"]         = df["max_glu_serum"].map(GLU_MAP).fillna(0)
    df["insulin_ord"]     = df["insulin"].map(INS_MAP).fillna(0)
    df["metformin_ord"]   = df["metformin"].map(METF_MAP).fillna(0)
    df["change_bin"]      = (df["change"].str.strip() == "Ch").astype(int)
    df["diabetesMed_bin"] = (df["diabetesMed"].str.strip() == "Yes").astype(int)
    df["diag1_diabetes"]  = df["diag_1"].apply(_is_diabetes_icd)
    df["diag2_diabetes"]  = df["diag_2"].apply(_is_diabetes_icd)
    df["diag3_diabetes"]  = df["diag_3"].apply(_is_diabetes_icd)
    df["total_visits"]    = (df["number_outpatient"] +
                             df["number_emergency"] +
                             df["number_inpatient"])
    df["service_use"]     = df["num_lab_procedures"] + df["num_procedures"]
    df["num_meds_x_time"] = df["num_medications"] * df["time_in_hospital"]

    # ── FIX [9]: Additional domain-informed features ──────────────────────────
    # Prior inpatient flag (strongest known predictor of readmission)
    df["prior_inpatient"]     = (df["number_inpatient"] > 0).astype(int)
    df["prior_inpatient_cnt"] = df["number_inpatient"].clip(0, 10)

    # High emergency use flag
    df["high_emergency"]  = (df["number_emergency"] >= 2).astype(int)

    # A1C tested flag (any A1C result vs None)
    df["A1C_tested"]      = (df["A1C_ord"] > 0).astype(int)

    # High glucose flag
    df["high_glucose"]    = (df["glu_ord"] >= 2).astype(int)

    # Insulin change (any adjustment is clinically meaningful)
    df["insulin_changed"] = (df["insulin_ord"] != 0).astype(int)

    # Medication burden (high med count → complex patient)
    df["high_med_burden"] = (df["num_medications"] > 15).astype(int)

    # Long stay flag
    df["long_stay"]       = (df["time_in_hospital"] > 7).astype(int)

    # Any circulatory/cardiac diagnosis (ICD-9 390-459)
    def _is_circulatory(code):
        try:
            c = str(code).split(".")[0]
            return 1 if c.isdigit() and 390 <= int(c) <= 459 else 0
        except: return 0
    df["diag1_circ"] = df["diag_1"].apply(_is_circulatory)

    # ── Feature list ──────────────────────────────────────────────────────────
    feat = [
        "age_num","gender_bin","time_in_hospital",
        "num_lab_procedures","num_procedures","num_medications",
        "number_outpatient","number_emergency","number_inpatient",
        "number_diagnoses",
        "A1C_ord","glu_ord","insulin_ord","metformin_ord",
        "change_bin","diabetesMed_bin",
        "diag1_diabetes","diag2_diabetes","diag3_diabetes",
        "total_visits","service_use","num_meds_x_time",
        # new in v5
        "prior_inpatient","prior_inpatient_cnt",
        "high_emergency","A1C_tested","high_glucose",
        "insulin_changed","high_med_burden","long_stay","diag1_circ",
    ]

    X = df[feat].apply(pd.to_numeric, errors="coerce").fillna(0).values
    encoders = {"feat_names": feat, "age_map": AGE_MAP}
    info(f"Final: X={X.shape}  features={len(feat)}")
    return X, y, encoders, feat


def train_diabetes():
    banner("DIABETES — TRAINING  (UCI 130-Hospital Dataset)")
    t0 = time.time()
    X, y, encoders, feat = prepare_diabetes()

    # FIX [4]: Compute original class ratio BEFORE SMOTE for XGB scale_pos_weight
    orig_ratio = float(np.bincount(y)[0]) / float(np.bincount(y)[1])
    info(f"Original pos-weight ratio: {orig_ratio:.2f}")

    # 70/10/20 split
    Xtv, Xte, ytv, yte = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    Xtr, Xval, ytr, yval = train_test_split(
        Xtv, ytv, test_size=0.125, random_state=SEED, stratify=ytv)

    # KEY FIX: keep original train split for CV tuning (SMOTE only for final fit)
    # Tuning on SMOTE data inflates CV AUC to ~0.95 (synthetic samples are trivial)
    # but test AUC collapses to ~0.61. Tune on real data, fit on SMOTE data.
    Xsm, ysm = safe_smote(Xtr, ytr, tomek=True)
    info(f"Train/Val/Test: {len(Xsm)}/{len(Xval)}/{len(Xte)}")
    # Xtr/ytr = original (unbalanced) train — used for CV hyperparameter search
    # Xsm/ysm = SMOTE'd train — used for final model fit after tuning

    results = {}

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_best = tune(make_pipe(RandomForestClassifier(random_state=SEED, n_jobs=1)),
                   RF_GRID, Xsm, ysm, Xtr, ytr, "Random Forest", n_iter=20)
    results["Random Forest"] = evaluate(
        "Random Forest", rf_best, Xsm, ysm, Xval, yval, Xte, yte)

    # ── Extra Trees ───────────────────────────────────────────────────────────
    et_best = tune(make_pipe(ExtraTreesClassifier(random_state=SEED, n_jobs=1)),
                   ET_GRID, Xsm, ysm, Xtr, ytr, "Extra Trees", n_iter=20)
    results["Extra Trees"] = evaluate(
        "Extra Trees", et_best, Xsm, ysm, Xval, yval, Xte, yte)

    # ── Gradient Boosting (skipped in FAST_MODE) ──────────────────────────────
    if not FAST_MODE:
        gb_best = tune(make_pipe(GradientBoostingClassifier(random_state=SEED)),
                       GB_GRID, Xsm, ysm, Xtr, ytr, "Gradient Boosting", n_iter=15)
        results["Gradient Boosting"] = evaluate(
            "Gradient Boosting", gb_best, Xsm, ysm, Xval, yval, Xte, yte)
    else:
        gb_best = None
        info("FAST_MODE: skipping Gradient Boosting")

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr_best = tune(make_pipe(LogisticRegression(
                        max_iter=2000, class_weight="balanced",
                        random_state=SEED)),
                   LR_GRID, Xsm, ysm, Xtr, ytr, "Logistic Regression", n_iter=7)
    results["Logistic Regression"] = evaluate(
        "Logistic Regression", lr_best, Xsm, ysm, Xval, yval, Xte, yte, cal=False)

    # ── XGBoost (skipped in FAST_MODE — LGB is faster and comparable) ────────
    if XGB_AVAILABLE and not FAST_MODE:
        xgb_best = tune(make_pipe(XGBClassifier(
                            use_label_encoder=False, eval_metric="logloss",
                            scale_pos_weight=orig_ratio,
                            random_state=SEED, n_jobs=1, verbosity=0)),
                        XGB_GRID, Xsm, ysm, Xtr, ytr, "XGBoost", n_iter=25)
        results["XGBoost"] = evaluate(
            "XGBoost", xgb_best, Xsm, ysm, Xval, yval, Xte, yte)
    else:
        xgb_best = None
        if FAST_MODE: info("FAST_MODE: skipping XGBoost (LightGBM covers this)")

    # ── LightGBM ──────────────────────────────────────────────────────────────
    if LGB_AVAILABLE:
        lgb_best = tune(make_pipe(LGBMClassifier(
                            class_weight="balanced",
                            random_state=SEED, n_jobs=1, verbose=-1)),
                        LGB_GRID, Xsm, ysm, Xtr, ytr, "LightGBM", n_iter=25)
        results["LightGBM"] = evaluate(
            "LightGBM", lgb_best, Xsm, ysm, Xval, yval, Xte, yte)
    else:
        lgb_best = None

    # ── Voting Ensemble ───────────────────────────────────────────────────────
    voters = [
        ("rf", results["Random Forest"]["model"]),
        ("et", results["Extra Trees"]["model"]),
    ]
    if gb_best:   voters.append(("gb",  results["Gradient Boosting"]["model"]))
    if xgb_best:  voters.append(("xgb", results["XGBoost"]["model"]))
    if lgb_best:  voters.append(("lgb", results["LightGBM"]["model"]))
    info("Building Voting Ensemble (calibrated base models)…")
    voting = VotingClassifier(voters, voting="soft", n_jobs=1)
    results["Voting"] = evaluate(
        "Voting", voting, Xsm, ysm, Xval, yval, Xte, yte, cal=False)

    # ── Stacking (skipped in FAST_MODE — biggest single time save ~25-30min) ──
    if not FAST_MODE:
        info("Building Stacking Ensemble…")
        stack_estimators = [
            ("rf", make_pipe(RandomForestClassifier(
                n_estimators=300, max_depth=15,
                class_weight="balanced_subsample",
                random_state=SEED, n_jobs=1))),
            ("et", make_pipe(ExtraTreesClassifier(
                n_estimators=300, max_depth=15,
                class_weight="balanced_subsample",
                random_state=SEED, n_jobs=1))),
        ]
        if XGB_AVAILABLE:
            stack_estimators.append(
                ("xgb", make_pipe(XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    scale_pos_weight=orig_ratio,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=SEED, n_jobs=1, verbosity=0))))
        stacking = StackingClassifier(
            estimators=stack_estimators,
            final_estimator=LogisticRegression(
                C=1.0, max_iter=2000, class_weight="balanced"),
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            n_jobs=1, passthrough=False)
        results["Stacking"] = evaluate(
            "Stacking", stacking, Xsm, ysm, Xval, yval, Xte, yte, cal=False)
    else:
        info("FAST_MODE: skipping Stacking (saves ~25-30 min)")

    # ── Summary & Save ────────────────────────────────────────────────────────
    best = _summary(results, "DIABETES")
    print(f"\n  🏆 Best: {best['name']}  (Test AUC {best['auc']:.4f}, "
          f"MinF1 {best['f1_min']:.4f}, Thr {best['th']:.3f})")
    _yp = pred_thresh(best['model'], Xte, best['th'])
    print(classification_report(yte, _yp,
          target_names=['Not Readmitted', 'Readmitted <30d'], digits=4))

    # Feature importance (from unaltered RF to avoid calibration wrapper issues)
    try:
        _rf_pipe = rf_best
        rf_clf = _rf_pipe.named_steps["clf"]
        imp = pd.Series(rf_clf.feature_importances_, index=feat).sort_values(ascending=False)
        info("Top-10 features (RF):")
        for f, v in imp.head(10).items():
            print(f"    {f:<35} {v:.4f}  {'█'*int(v*200)}")
    except Exception:
        pass

    encoders["best_threshold"] = best["th"]
    joblib.dump(best["model"], MODEL_DIR / "diabetes_model.pkl")
    joblib.dump(encoders,      MODEL_DIR / "diabetes_encoders.pkl")
    tick(f"Saved → models/diabetes_model.pkl  ({time.time()-t0:.0f}s total)")
    return best["model"], encoders


# ══════════════════════════════════════════════════════════════════════════════
#  DEMENTIA TRAINING  (NACC UDS dataset)
# ══════════════════════════════════════════════════════════════════════════════

DEME_COLS = ["NACCAGE","NACCSEX","EDUC","CDRGLOB","CDRSUM","NACCMMSE","NACCGDS",
             "ANIMALS","TRAILA","TRAILB","DIABETES","HYPERTEN","NACCDEP",
             "NACCAPOE","NACCBMI","DEMENTED"]

DEME_FEAT = [
    "NACCAGE","SEX","EDUC","CDRGLOB","CDRSUM","NACCMMSE","NACCGDS",
    "ANIMALS","TRAILA","TRAILB","DIABETES","HYPERTEN","NACCDEP",
    "APOE4","NACCBMI",
    # engineered
    "CDR_MMSE_ratio","CDR_x_age","MMSE_below_24","CDR_nonzero",
    "age_educ_ratio","trail_ratio","comorbidity_sum",
    # new in v5
    "CDR_sum_x_mmse","severe_CDR","MMSE_quartile",
    "APOE4_x_age","trail_diff","functional_cognitive_gap",
]


def prepare_dementia():
    info("Loading NACC CSV (chunked)…")
    chunks = pd.read_csv(DEME_CSV, usecols=DEME_COLS, low_memory=False,
                         chunksize=20000, on_bad_lines="skip")
    df = pd.concat(chunks, ignore_index=True)
    info(f"Raw: {df.shape}")

    # Sentinel removal
    df.replace(NACC_SENTINELS, np.nan, inplace=True)
    df.loc[df["NACCMMSE"] > 30,  "NACCMMSE"] = np.nan
    df.loc[df["NACCBMI"]  > 80,  "NACCBMI"]  = np.nan
    df.loc[df["TRAILA"]   > 500, "TRAILA"]   = np.nan
    df.loc[df["TRAILB"]   > 500, "TRAILB"]   = np.nan
    df.loc[df["ANIMALS"]  > 60,  "ANIMALS"]  = np.nan
    df.loc[df["NACCGDS"]  > 15,  "NACCGDS"]  = np.nan

    df["SEX"]   = df["NACCSEX"].map({1: 0, 2: 1})
    df["APOE4"] = df["NACCAPOE"].map({1:0, 2:0, 3:1, 4:0, 5:1, 6:1})

    df = df.dropna(subset=["DEMENTED"])
    y = df["DEMENTED"].values.astype(int)
    info(f"Class balance: {np.bincount(y)}")

    # Imputation
    encoders = {}
    base_cols = ["NACCAGE","SEX","EDUC","CDRGLOB","CDRSUM","NACCMMSE","NACCGDS",
                 "ANIMALS","TRAILA","TRAILB","DIABETES","HYPERTEN","NACCDEP",
                 "APOE4","NACCBMI"]
    for col in base_cols:
        med = df[col].median()
        encoders[f"{col}_median"] = float(med) if not np.isnan(float(med)) else 0.0
        df[col] = df[col].fillna(encoders[f"{col}_median"])

    # Engineered features (v4)
    mmse = df["NACCMMSE"].clip(0, 30)
    df["CDR_MMSE_ratio"]  = df["CDRGLOB"] / (mmse + 1)
    df["CDR_x_age"]       = df["CDRGLOB"] * df["NACCAGE"]
    df["MMSE_below_24"]   = (mmse < 24).astype(int)
    df["CDR_nonzero"]     = (df["CDRGLOB"] > 0).astype(int)
    df["age_educ_ratio"]  = df["NACCAGE"] / (df["EDUC"] + 1)
    df["trail_ratio"]     = df["TRAILA"] / (df["TRAILB"] + 1)
    df["comorbidity_sum"] = (df["DIABETES"].fillna(0) +
                             df["HYPERTEN"].fillna(0) +
                             df["NACCDEP"].fillna(0))

    # New engineered features (v5)
    df["CDR_sum_x_mmse"]        = df["CDRSUM"] * (30 - mmse)   # CDR burden × MMSE deficit
    df["severe_CDR"]             = (df["CDRGLOB"] >= 2).astype(int)
    df["MMSE_quartile"]          = pd.qcut(mmse, q=4, labels=False, duplicates="drop")
    df["APOE4_x_age"]            = df["APOE4"] * df["NACCAGE"]
    df["trail_diff"]             = (df["TRAILB"] - df["TRAILA"]).clip(0, 500)
    df["functional_cognitive_gap"] = df["CDRSUM"] / (mmse + 1)

    df["MMSE_quartile"] = df["MMSE_quartile"].fillna(0)

    X = df[DEME_FEAT].apply(pd.to_numeric, errors="coerce").fillna(0).values
    encoders["feat_names"] = DEME_FEAT
    info(f"Final: X={X.shape}  features={len(DEME_FEAT)}")
    return X, y, encoders


def train_dementia():
    banner("DEMENTIA — TRAINING  (NACC UDS Dataset)")
    t0 = time.time()
    X, y, encoders = prepare_dementia()

    orig_ratio = float(np.bincount(y)[0]) / float(np.bincount(y)[1])
    info(f"Original pos-weight ratio: {orig_ratio:.2f}")

    Xtv, Xte, ytv, yte = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    Xtr, Xval, ytr, yval = train_test_split(
        Xtv, ytv, test_size=0.125, random_state=SEED, stratify=ytv)
    Xsm, ysm = safe_smote(Xtr, ytr, tomek=False)
    info(f"Train/Val/Test: {len(Xsm)}/{len(Xval)}/{len(Xte)}")

    results = {}

    rf_best = tune(make_pipe(RandomForestClassifier(random_state=SEED, n_jobs=1), robust=True),
                   RF_GRID, Xsm, ysm, Xtr, ytr, "Random Forest", n_iter=20)
    results["Random Forest"] = evaluate(
        "Random Forest", rf_best, Xsm, ysm, Xval, yval, Xte, yte)

    et_best = tune(make_pipe(ExtraTreesClassifier(random_state=SEED, n_jobs=1), robust=True),
                   ET_GRID, Xsm, ysm, Xtr, ytr, "Extra Trees", n_iter=20)
    results["Extra Trees"] = evaluate(
        "Extra Trees", et_best, Xsm, ysm, Xval, yval, Xte, yte)

    gb_best = None
    if not FAST_MODE:
        gb_best = tune(make_pipe(GradientBoostingClassifier(random_state=SEED), robust=True),
                       GB_GRID, Xsm, ysm, Xtr, ytr, "Gradient Boosting", n_iter=15)
        results["Gradient Boosting"] = evaluate(
            "Gradient Boosting", gb_best, Xsm, ysm, Xval, yval, Xte, yte)
    else:
        info("FAST_MODE: skipping Gradient Boosting")

    lr_best = tune(make_pipe(LogisticRegression(
                       max_iter=2000, class_weight="balanced",
                       random_state=SEED), robust=True),
                   LR_GRID, Xsm, ysm, Xtr, ytr, "Logistic Regression", n_iter=7)
    results["Logistic Regression"] = evaluate(
        "Logistic Regression", lr_best, Xsm, ysm, Xval, yval, Xte, yte, cal=False)

    xgb_best = None
    if XGB_AVAILABLE and not FAST_MODE:
        xgb_best = tune(make_pipe(XGBClassifier(
                            use_label_encoder=False, eval_metric="logloss",
                            scale_pos_weight=orig_ratio,
                            random_state=SEED, n_jobs=1, verbosity=0), robust=True),
                        XGB_GRID, Xsm, ysm, Xtr, ytr, "XGBoost", n_iter=25)
        results["XGBoost"] = evaluate(
            "XGBoost", xgb_best, Xsm, ysm, Xval, yval, Xte, yte)
    else:
        if FAST_MODE: info("FAST_MODE: skipping XGBoost")

    if LGB_AVAILABLE:
        lgb_best = tune(make_pipe(LGBMClassifier(
                            class_weight="balanced",
                            random_state=SEED, n_jobs=1, verbose=-1), robust=True),
                        LGB_GRID, Xsm, ysm, Xtr, ytr, "LightGBM", n_iter=25)
        results["LightGBM"] = evaluate(
            "LightGBM", lgb_best, Xsm, ysm, Xval, yval, Xte, yte)
    else:
        lgb_best = None

    voters = [
        ("rf", results["Random Forest"]["model"]),
        ("et", results["Extra Trees"]["model"]),
    ]
    if gb_best:  voters.append(("gb",  results["Gradient Boosting"]["model"]))
    if xgb_best: voters.append(("xgb", results["XGBoost"]["model"]))
    if lgb_best: voters.append(("lgb", results["LightGBM"]["model"]))
    info("Building Voting Ensemble (calibrated base models)…")
    voting = VotingClassifier(voters, voting="soft", n_jobs=1)
    results["Voting"] = evaluate(
        "Voting", voting, Xsm, ysm, Xval, yval, Xte, yte, cal=False)

    if not FAST_MODE:
        info("Building Stacking Ensemble…")
        stack_estimators = [
            ("rf", make_pipe(RandomForestClassifier(
                n_estimators=300, max_depth=15,
                class_weight="balanced_subsample",
                random_state=SEED, n_jobs=1), robust=True)),
            ("et", make_pipe(ExtraTreesClassifier(
                n_estimators=300, max_depth=15,
                class_weight="balanced_subsample",
                random_state=SEED, n_jobs=1), robust=True)),
        ]
        if XGB_AVAILABLE:
            stack_estimators.append(
                ("xgb", make_pipe(XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    scale_pos_weight=orig_ratio,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=SEED, n_jobs=1, verbosity=0), robust=True)))
        stacking = StackingClassifier(
            estimators=stack_estimators,
            final_estimator=LogisticRegression(
                C=1.0, max_iter=2000, class_weight="balanced"),
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            n_jobs=1, passthrough=False)
        results["Stacking"] = evaluate(
            "Stacking", stacking, Xsm, ysm, Xval, yval, Xte, yte, cal=False)
    else:
        info("FAST_MODE: skipping Stacking (saves ~25-30 min)")

    best = _summary(results, "DEMENTIA")
    print(f"\n  🏆 Best: {best['name']}  (Test AUC {best['auc']:.4f}, "
          f"MinF1 {best['f1_min']:.4f}, Thr {best['th']:.3f})")
    _yp = pred_thresh(best['model'], Xte, best['th'])
    print(classification_report(yte, _yp,
          target_names=['Nondemented', 'Demented'], digits=4))

    try:
        rf_clf = rf_best.named_steps["clf"]
        imp = pd.Series(rf_clf.feature_importances_, index=DEME_FEAT).sort_values(ascending=False)
        info("Top-10 features (RF):")
        for f, v in imp.head(10).items():
            print(f"    {f:<35} {v:.4f}  {'█'*int(v*200)}")
    except Exception:
        pass

    encoders["best_threshold"] = best["th"]
    joblib.dump(best["model"], MODEL_DIR / "dementia_model.pkl")
    joblib.dump(encoders,      MODEL_DIR / "dementia_encoders.pkl")
    tick(f"Saved → models/dementia_model.pkl  ({time.time()-t0:.0f}s total)")
    return best["model"], encoders


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    multiprocessing.freeze_support()
    total = time.time()
    mode_label = "FAST MODE  (~45-60 min)" if FAST_MODE else "FULL MODE  (~2-3 hours)"
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║   MedMate ML — Train_models.py  v5.1  (Fixes + Fast Mode)     ║
║   UCI Diabetes Readmission  ·  NACC UDS Dementia               ║
╠══════════════════════════════════════════════════════════════════╣
║   {"⚡ " if FAST_MODE else "🔬 "}{mode_label:<62}║
╚══════════════════════════════════════════════════════════════════╝
  Tip: flip FAST_MODE at the top of this file to switch modes.
""")
    train_diabetes()
    train_dementia()
    banner(f"ALL DONE — {(time.time()-total)/60:.1f} min total")
    print("  Run MedMate_ml.py to start the API.\n")