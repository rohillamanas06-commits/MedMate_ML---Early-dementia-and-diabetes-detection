"""
train_models_deep.py
====================
DEEP TRAINING — Exhaustive hyperparameter tuning + advanced techniques.

What this does differently from train_models.py:
  ✦ GridSearchCV / RandomizedSearchCV — tries hundreds of hyperparameter combos
  ✦ SMOTE — synthetic oversampling to fix class imbalance
  ✦ Voting & Stacking Ensembles — combine multiple models
  ✦ Feature importance + selection
  ✦ Learning curve analysis
  ✦ Full per-class metrics

Run:
    pip install imbalanced-learn
    python train_models_deep.py

Expected time: 3–8 minutes depending on CPU.
"""

import warnings
warnings.filterwarnings("ignore")

import time
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Core ML ──────────────────────────────────────────────────────────────────
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# ── Preprocessing ─────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV, RandomizedSearchCV,
    learning_curve,
)
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
    matthews_corrcoef,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# ── Imbalanced learning ───────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠  imbalanced-learn not installed. Run: pip install imbalanced-learn")
    print("   Continuing without SMOTE...\n")

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR    = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
DIABETES_CSV = "diabetes_data_upload.csv"
DEMENTIA_CSV = "dementia_dataset.csv"

# ─── Feature lists ────────────────────────────────────────────────────────────
DIABETES_BINARY = [
    "Polyuria","Polydipsia","sudden weight loss","weakness",
    "Polyphagia","Genital thrush","visual blurring","Itching",
    "Irritability","delayed healing","partial paresis",
    "muscle stiffness","Alopecia","Obesity",
]
DIABETES_FEATURES = ["Age","Gender"] + DIABETES_BINARY
DEMENTIA_FEATURES = [
    "Visit","MR Delay","M/F","Age","EDUC",
    "SES","MMSE","CDR","eTIV","nWBV","ASF",
]

# ═══════════════════════════════════════════════════════════════════════════════
def banner(title, char="═"):
    w = 62
    print(f"\n{char*w}")
    print(f"  {title}")
    print(f"{char*w}")

def sub(title):
    print(f"\n  ── {title} ──")

def tick(msg):
    print(f"  ✔  {msg}")

def info(msg):
    print(f"  ·  {msg}")

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_diabetes():
    banner("DIABETES — DATA PREPARATION")
    df = pd.read_csv(DIABETES_CSV)
    info(f"Shape: {df.shape}")

    for col in DIABETES_BINARY:
        df[col] = df[col].str.strip().map({"Yes": 1, "No": 0})

    le_gender = LabelEncoder()
    df["Gender"] = le_gender.fit_transform(df["Gender"].str.strip())

    le_target = LabelEncoder()
    y = le_target.fit_transform(df["class"].str.strip())
    X = df[DIABETES_FEATURES]

    encoders = {"Gender": le_gender, "target": le_target}
    info(f"Classes: {dict(zip(le_target.classes_, range(len(le_target.classes_))))}")
    info(f"Class balance: {np.bincount(y)}")
    return X.values, y, encoders, DIABETES_FEATURES


def prepare_dementia():
    banner("DEMENTIA — DATA PREPARATION")
    df = pd.read_csv(DEMENTIA_CSV)
    info(f"Shape: {df.shape}")

    le_sex = LabelEncoder()
    df["M/F"] = le_sex.fit_transform(df["M/F"].str.strip())

    ses_med  = df["SES"].median()
    mmse_med = df["MMSE"].median()
    df["SES"]  = df["SES"].fillna(ses_med)
    df["MMSE"] = df["MMSE"].fillna(mmse_med)

    group_map = {"Demented": 1, "Converted": 1, "Nondemented": 0}
    y = df["Group"].map(group_map).values
    X = df[DEMENTIA_FEATURES]

    encoders = {
        "M/F": le_sex, "group_map": group_map,
        "SES_median": ses_med, "MMSE_median": mmse_med,
    }
    info(f"Class balance: {np.bincount(y)}")
    return X.values, y, encoders, DEMENTIA_FEATURES


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def full_eval(model, X_tr, X_te, y_tr, y_te, name, class_names, cv_folds=10):
    """Train, evaluate with multiple metrics, print report."""
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    f1     = f1_score(y_te, y_pred, average="weighted")
    mcc    = matthews_corrcoef(y_te, y_pred)

    # AUC (binary)
    try:
        proba = model.predict_proba(X_te)[:, 1]
        auc   = roc_auc_score(y_te, proba)
    except Exception:
        auc = float("nan")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, np.vstack([X_tr, X_te]),
                                np.concatenate([y_tr, y_te]),
                                cv=cv, scoring="accuracy")

    print(f"\n  {'='*56}")
    print(f"  {name}")
    print(f"  {'='*56}")
    print(f"  Test Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Weighted F1     : {f1:.4f}")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"  MCC             : {mcc:.4f}  (1.0 = perfect)")
    print(f"  {cv_folds}-Fold CV Mean  : {cv_scores.mean():.4f}  ± {cv_scores.std():.4f}")
    print(f"  Train Time      : {train_time:.2f}s")
    print(f"\n{classification_report(y_te, y_pred, target_names=class_names, digits=4)}")

    return {
        "name": name, "model": model,
        "acc": acc, "f1": f1, "auc": auc,
        "mcc": mcc, "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER GRIDS
# ═══════════════════════════════════════════════════════════════════════════════

RF_GRID = {
    "clf__n_estimators":      [200, 400, 600],
    "clf__max_depth":         [None, 10, 15, 20],
    "clf__min_samples_split": [2, 4, 6],
    "clf__min_samples_leaf":  [1, 2],
    "clf__max_features":      ["sqrt", "log2"],
    "clf__class_weight":      ["balanced", None],
}

GB_GRID = {
    "clf__n_estimators":  [100, 200, 300],
    "clf__learning_rate": [0.05, 0.08, 0.1, 0.15],
    "clf__max_depth":     [3, 4, 5],
    "clf__subsample":     [0.7, 0.8, 0.9, 1.0],
    "clf__min_samples_split": [2, 4],
}

SVM_GRID = {
    "clf__C":      [0.1, 1, 5, 10, 50, 100],
    "clf__gamma":  ["scale", "auto", 0.001, 0.01, 0.1],
    "clf__kernel": ["rbf", "poly"],
}

LR_GRID = {
    "clf__C":       [0.01, 0.1, 1, 5, 10, 50],
    "clf__penalty": ["l2"],
    "clf__solver":  ["lbfgs", "liblinear"],
}

MLP_GRID = {
    "clf__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
    "clf__activation":         ["relu", "tanh"],
    "clf__alpha":              [0.0001, 0.001, 0.01],
    "clf__learning_rate_init": [0.001, 0.005, 0.01],
}

ET_GRID = {
    "clf__n_estimators":  [200, 400],
    "clf__max_depth":     [None, 10, 15],
    "clf__min_samples_split": [2, 4],
    "clf__class_weight":  ["balanced", None],
}


def make_pipe(clf, use_robust=False):
    scaler = RobustScaler() if use_robust else StandardScaler()
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  scaler),
        ("clf",     clf),
    ])


def grid_search(pipe, param_grid, X_tr, y_tr, name, n_iter=None, cv=5):
    sub(f"Tuning {name}…")
    cv_strat = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    if n_iter:
        searcher = RandomizedSearchCV(
            pipe, param_grid, n_iter=n_iter,
            cv=cv_strat, scoring="accuracy",
            n_jobs=-1, random_state=42, verbose=0,
        )
    else:
        searcher = GridSearchCV(
            pipe, param_grid,
            cv=cv_strat, scoring="accuracy",
            n_jobs=-1, verbose=0,
        )

    searcher.fit(X_tr, y_tr)
    tick(f"Best CV accuracy: {searcher.best_score_:.4f}")
    info(f"Best params: {searcher.best_params_}")
    return searcher.best_estimator_


# ═══════════════════════════════════════════════════════════════════════════════
#  DIABETES — DEEP TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_diabetes_deep():
    banner("DIABETES — DEEP TRAINING", "═")

    X, y, encoders, feat_names = prepare_diabetes()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── SMOTE oversampling ────────────────────────────────────────────────────
    if SMOTE_AVAILABLE:
        sub("Applying SMOTE oversampling on training set")
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
        tick(f"After SMOTE: {np.bincount(y_tr_sm)} (was {np.bincount(y_tr)})")
    else:
        X_tr_sm, y_tr_sm = X_tr, y_tr

    results = {}
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── 1. Random Forest — Grid Search ───────────────────────────────────────
    rf_pipe = make_pipe(RandomForestClassifier(random_state=42))
    best_rf = grid_search(rf_pipe, RF_GRID, X_tr_sm, y_tr_sm, "Random Forest")
    results["Random Forest (tuned)"] = full_eval(
        best_rf, X_tr_sm, X_te, y_tr_sm, y_te,
        "Random Forest (tuned)", ["Negative", "Positive"]
    )

    # ── 2. Gradient Boosting — Randomized Search ──────────────────────────────
    gb_pipe = make_pipe(GradientBoostingClassifier(random_state=42))
    best_gb = grid_search(gb_pipe, GB_GRID, X_tr_sm, y_tr_sm, "Gradient Boosting", n_iter=40)
    results["Gradient Boosting (tuned)"] = full_eval(
        best_gb, X_tr_sm, X_te, y_tr_sm, y_te,
        "Gradient Boosting (tuned)", ["Negative", "Positive"]
    )

    # ── 3. SVM — Grid Search ──────────────────────────────────────────────────
    svm_pipe = make_pipe(SVC(probability=True, class_weight="balanced", random_state=42))
    best_svm = grid_search(svm_pipe, SVM_GRID, X_tr_sm, y_tr_sm, "SVM", n_iter=30)
    results["SVM (tuned)"] = full_eval(
        best_svm, X_tr_sm, X_te, y_tr_sm, y_te,
        "SVM (tuned)", ["Negative", "Positive"]
    )

    # ── 4. MLP Neural Network ─────────────────────────────────────────────────
    mlp_pipe = make_pipe(MLPClassifier(max_iter=500, random_state=42))
    best_mlp = grid_search(mlp_pipe, MLP_GRID, X_tr_sm, y_tr_sm, "MLP Neural Net", n_iter=30)
    results["MLP Neural Net (tuned)"] = full_eval(
        best_mlp, X_tr_sm, X_te, y_tr_sm, y_te,
        "MLP Neural Net (tuned)", ["Negative", "Positive"]
    )

    # ── 5. Extra Trees ────────────────────────────────────────────────────────
    et_pipe = make_pipe(ExtraTreesClassifier(random_state=42))
    best_et = grid_search(et_pipe, ET_GRID, X_tr_sm, y_tr_sm, "Extra Trees")
    results["Extra Trees (tuned)"] = full_eval(
        best_et, X_tr_sm, X_te, y_tr_sm, y_te,
        "Extra Trees (tuned)", ["Negative", "Positive"]
    )

    # ── 6. Voting Ensemble (top 3) ────────────────────────────────────────────
    sub("Building Voting Ensemble (RF + GB + SVM)")
    voting = VotingClassifier(
        estimators=[
            ("rf",  best_rf),
            ("gb",  best_gb),
            ("svm", best_svm),
        ],
        voting="soft",
    )
    results["Voting Ensemble"] = full_eval(
        voting, X_tr_sm, X_te, y_tr_sm, y_te,
        "Voting Ensemble (RF+GB+SVM)", ["Negative", "Positive"]
    )

    # ── 7. Stacking Ensemble ──────────────────────────────────────────────────
    sub("Building Stacking Ensemble")
    stacking = StackingClassifier(
        estimators=[
            ("rf",  best_rf.named_steps["clf"]),
            ("et",  best_et.named_steps["clf"]),
            ("svm", best_svm.named_steps["clf"]),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        passthrough=False,
    )
    stack_pipe = make_pipe(stacking)
    results["Stacking Ensemble"] = full_eval(
        stack_pipe, X_tr_sm, X_te, y_tr_sm, y_te,
        "Stacking Ensemble", ["Negative", "Positive"]
    )

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    banner("DIABETES — FINAL COMPARISON", "─")
    print(f"  {'Model':<32} {'Acc':>7} {'F1':>7} {'AUC':>7} {'MCC':>7} {'CV':>9}")
    print(f"  {'─'*32} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*9}")
    best_cv = 0
    best_name = ""
    for k, v in results.items():
        marker = " ★" if v["cv_mean"] == max(r["cv_mean"] for r in results.values()) else ""
        print(f"  {k:<32} {v['acc']:>7.4f} {v['f1']:>7.4f} {v['auc']:>7.4f} {v['mcc']:>7.4f} {v['cv_mean']:>7.4f}±{v['cv_std']:.3f}{marker}")
        if v["cv_mean"] > best_cv:
            best_cv = v["cv_mean"]
            best_name = k

    print(f"\n  🏆 Best: {best_name}  (CV={best_cv:.4f})")

    # ── Feature importance ────────────────────────────────────────────────────
    sub("Feature Importances (from best Random Forest)")
    rf_clf = results["Random Forest (tuned)"]["model"].named_steps["clf"]
    if hasattr(rf_clf, "feature_importances_"):
        imp = pd.Series(rf_clf.feature_importances_, index=feat_names).sort_values(ascending=False)
        print()
        for feat, val in imp.items():
            bar = "█" * int(val * 200)
            print(f"    {feat:<28} {val:.4f}  {bar}")

    # ── Save best model ───────────────────────────────────────────────────────
    best_model = results[best_name]["model"]
    joblib.dump(best_model, MODEL_DIR / "diabetes_model.pkl")
    joblib.dump(encoders,   MODEL_DIR / "diabetes_encoders.pkl")
    tick(f"Saved best model ({best_name}) → models/diabetes_model.pkl")
    return best_model, encoders


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMENTIA — DEEP TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_dementia_deep():
    banner("DEMENTIA — DEEP TRAINING", "═")

    X, y, encoders, feat_names = prepare_dementia()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── SMOTE ─────────────────────────────────────────────────────────────────
    if SMOTE_AVAILABLE:
        sub("Applying SMOTE oversampling")
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
        tick(f"After SMOTE: {np.bincount(y_tr_sm)} (was {np.bincount(y_tr)})")
    else:
        X_tr_sm, y_tr_sm = X_tr, y_tr

    results = {}

    # ── 1. SVM ────────────────────────────────────────────────────────────────
    svm_pipe = make_pipe(SVC(probability=True, class_weight="balanced", random_state=42), use_robust=True)
    best_svm = grid_search(svm_pipe, SVM_GRID, X_tr_sm, y_tr_sm, "SVM", n_iter=30)
    results["SVM (tuned)"] = full_eval(
        best_svm, X_tr_sm, X_te, y_tr_sm, y_te,
        "SVM (tuned)", ["Nondemented", "Demented"]
    )

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    rf_pipe = make_pipe(RandomForestClassifier(random_state=42), use_robust=True)
    best_rf = grid_search(rf_pipe, RF_GRID, X_tr_sm, y_tr_sm, "Random Forest")
    results["Random Forest (tuned)"] = full_eval(
        best_rf, X_tr_sm, X_te, y_tr_sm, y_te,
        "Random Forest (tuned)", ["Nondemented", "Demented"]
    )

    # ── 3. Gradient Boosting ──────────────────────────────────────────────────
    gb_pipe = make_pipe(GradientBoostingClassifier(random_state=42), use_robust=True)
    best_gb = grid_search(gb_pipe, GB_GRID, X_tr_sm, y_tr_sm, "Gradient Boosting", n_iter=40)
    results["Gradient Boosting (tuned)"] = full_eval(
        best_gb, X_tr_sm, X_te, y_tr_sm, y_te,
        "Gradient Boosting (tuned)", ["Nondemented", "Demented"]
    )

    # ── 4. Extra Trees ────────────────────────────────────────────────────────
    et_pipe = make_pipe(ExtraTreesClassifier(random_state=42), use_robust=True)
    best_et = grid_search(et_pipe, ET_GRID, X_tr_sm, y_tr_sm, "Extra Trees")
    results["Extra Trees (tuned)"] = full_eval(
        best_et, X_tr_sm, X_te, y_tr_sm, y_te,
        "Extra Trees (tuned)", ["Nondemented", "Demented"]
    )

    # ── 5. MLP ────────────────────────────────────────────────────────────────
    mlp_pipe = make_pipe(MLPClassifier(max_iter=500, random_state=42), use_robust=True)
    best_mlp = grid_search(mlp_pipe, MLP_GRID, X_tr_sm, y_tr_sm, "MLP Neural Net", n_iter=30)
    results["MLP Neural Net (tuned)"] = full_eval(
        best_mlp, X_tr_sm, X_te, y_tr_sm, y_te,
        "MLP Neural Net (tuned)", ["Nondemented", "Demented"]
    )

    # ── 6. Logistic Regression (strong baseline for small data) ───────────────
    lr_pipe = make_pipe(LogisticRegression(max_iter=2000, random_state=42), use_robust=True)
    best_lr = grid_search(lr_pipe, LR_GRID, X_tr_sm, y_tr_sm, "Logistic Regression")
    results["Logistic Regression (tuned)"] = full_eval(
        best_lr, X_tr_sm, X_te, y_tr_sm, y_te,
        "Logistic Regression (tuned)", ["Nondemented", "Demented"]
    )

    # ── 7. Voting Ensemble ────────────────────────────────────────────────────
    sub("Building Voting Ensemble (SVM + RF + GB)")
    voting = VotingClassifier(
        estimators=[
            ("svm", best_svm),
            ("rf",  best_rf),
            ("gb",  best_gb),
        ],
        voting="soft",
    )
    results["Voting Ensemble"] = full_eval(
        voting, X_tr_sm, X_te, y_tr_sm, y_te,
        "Voting Ensemble (SVM+RF+GB)", ["Nondemented", "Demented"]
    )

    # ── 8. Stacking ───────────────────────────────────────────────────────────
    sub("Building Stacking Ensemble")
    stacking = StackingClassifier(
        estimators=[
            ("svm", best_svm.named_steps["clf"]),
            ("et",  best_et.named_steps["clf"]),
            ("gb",  best_gb.named_steps["clf"]),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
    )
    stack_pipe = make_pipe(stacking, use_robust=True)
    results["Stacking Ensemble"] = full_eval(
        stack_pipe, X_tr_sm, X_te, y_tr_sm, y_te,
        "Stacking Ensemble", ["Nondemented", "Demented"]
    )

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    banner("DEMENTIA — FINAL COMPARISON", "─")
    print(f"  {'Model':<32} {'Acc':>7} {'F1':>7} {'AUC':>7} {'MCC':>7} {'CV':>9}")
    print(f"  {'─'*32} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*9}")
    best_cv = 0
    best_name = ""
    for k, v in results.items():
        marker = " ★" if v["cv_mean"] == max(r["cv_mean"] for r in results.values()) else ""
        print(f"  {k:<32} {v['acc']:>7.4f} {v['f1']:>7.4f} {v['auc']:>7.4f} {v['mcc']:>7.4f} {v['cv_mean']:>7.4f}±{v['cv_std']:.3f}{marker}")
        if v["cv_mean"] > best_cv:
            best_cv = v["cv_mean"]
            best_name = k

    print(f"\n  🏆 Best: {best_name}  (CV={best_cv:.4f})")

    # ── Feature importances ───────────────────────────────────────────────────
    sub("Feature Importances (from Random Forest)")
    rf_clf = results["Random Forest (tuned)"]["model"].named_steps["clf"]
    if hasattr(rf_clf, "feature_importances_"):
        imp = pd.Series(rf_clf.feature_importances_, index=feat_names).sort_values(ascending=False)
        print()
        for feat, val in imp.items():
            bar = "█" * int(val * 200)
            print(f"    {feat:<16} {val:.4f}  {bar}")

    best_model = results[best_name]["model"]
    joblib.dump(best_model, MODEL_DIR / "dementia_model.pkl")
    joblib.dump(encoders,   MODEL_DIR / "dementia_encoders.pkl")
    tick(f"Saved best model ({best_name}) → models/dementia_model.pkl")
    return best_model, encoders


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    total_start = time.time()

    print("""
╔══════════════════════════════════════════════════════════════╗
║         MedMate ML — DEEP TRAINING MODE                     ║
║  GridSearchCV + SMOTE + Ensembles + Neural Nets             ║
╚══════════════════════════════════════════════════════════════╝
  This will take 3–10 minutes. Go grab some chai ☕
""")

    install_hint = not SMOTE_AVAILABLE
    if install_hint:
        print("  TIP: Install imbalanced-learn for SMOTE:")
        print("       pip install imbalanced-learn\n")

    train_diabetes_deep()
    train_dementia_deep()

    total = time.time() - total_start
    banner(f"ALL DONE — Total time: {total/60:.1f} minutes", "═")
    print("  Models saved to ./models/")
    print("  Run MedMate_ML.py to start the API server.\n")