"""
train_model.py
─────────────────────────────────────────────────────────────────
Trains a fitness recommendation model using:
  • RandomForestClassifier
  • DecisionTreeClassifier
Automatically picks the best-performing model and saves it.

CSV columns expected:
  Gender | Goal | BMI Category | Exercise Schedule | Meal Plan
─────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

# ── Config ────────────────────────────────────────────────────
CSV_PATH   = "../data/GYM.csv"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fitness_model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "fitness_model_meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Column definitions ────────────────────────────────────────
INPUT_COLS  = ["Gender", "Goal", "BMI Category"]
TARGET_COLS = ["Exercise Schedule", "Meal Plan"]


# ══════════════════════════════════════════════════════════════
#  LOAD & PREPROCESS
# ══════════════════════════════════════════════════════════════

def load_and_preprocess(path: str) -> pd.DataFrame:
    print(f"  Reading CSV from : {path}")
    df = pd.read_csv(path)
    print(f"  Raw shape        : {df.shape}")
    print(f"  Columns          : {list(df.columns)}\n")

    df.columns = df.columns.str.strip()

    required = INPUT_COLS + TARGET_COLS
    df = df.dropna(subset=required)
    print(f"  Rows after dropping nulls: {len(df)}")

    for col in required:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    return df


# ══════════════════════════════════════════════════════════════
#  ENCODE
# ══════════════════════════════════════════════════════════════

def encode_features(df: pd.DataFrame):
    encoders = {}

    # Encode inputs
    X_parts = []
    for col in INPUT_COLS:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col].astype(str))
        X_parts.append(encoded.reshape(-1, 1))
        encoders[f"input_{col}"] = le
        print(f"  [INPUT]  {col:15s} → classes: {list(le.classes_)}")

    X = np.hstack(X_parts)

    # Encode targets
    Y_parts = []
    for col in TARGET_COLS:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col].astype(str))
        Y_parts.append(encoded.reshape(-1, 1))
        encoders[f"target_{col}"] = le
        print(f"  [TARGET] {col:15s} → classes: {list(le.classes_)}")

    y = np.hstack(Y_parts)

    return X, y, encoders


# ══════════════════════════════════════════════════════════════
#  EVALUATE A MODEL
# ══════════════════════════════════════════════════════════════

def evaluate_model(name: str, model, X_test, y_test, encoders) -> float:
    print(f"\n  ── {name} Results ──────────────────────────────")
    y_pred = model.predict(X_test)
    scores = []

    for i, col in enumerate(TARGET_COLS):
        le = encoders[f"target_{col}"]
        true_labels = le.inverse_transform(y_test[:, i])
        pred_labels = le.inverse_transform(y_pred[:, i])
        acc = accuracy_score(true_labels, pred_labels)
        scores.append(acc)

        print(f"\n  ▶  {col}")
        print(f"     Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
        print(classification_report(
            true_labels, pred_labels,
            zero_division=0,
            target_names=le.classes_
        ))

    avg = float(np.mean(scores))
    print(f"  ⭐  {name} — Average Accuracy across targets: {avg:.4f}")
    return avg


# ══════════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════

def train(csv_path: str = CSV_PATH):
    print("=" * 60)
    print("  FITNESS RECOMMENDATION MODEL — TRAINING")
    print("=" * 60)

    df = load_and_preprocess(csv_path)
    X, y, encoders = encode_features(df)

    print(f"\n  Feature matrix : {X.shape}")
    print(f"  Target matrix  : {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Define candidate models ───────────────────────────────
    candidates = {

        # ── 1. Random Forest ──────────────────────────────────
        # Builds many decision trees on random data subsets,
        # then combines their predictions (majority vote).
        # → More accurate, handles noise better than a single tree.
        "RandomForest": MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=200,   # number of trees in the forest
                max_depth=10,       # max depth per tree (None = unlimited)
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",# features considered at each split
                random_state=42,
                n_jobs=-1,
            ),
            n_jobs=-1,
        ),

        # ── 2. Decision Tree ──────────────────────────────────
        # A single tree that splits data by asking yes/no questions.
        # → Simpler, fully interpretable, slightly less accurate.
        "DecisionTree": MultiOutputClassifier(
            DecisionTreeClassifier(
                max_depth=8,        # limit depth to avoid overfitting
                min_samples_split=4,
                min_samples_leaf=2,
                criterion="gini",   # split quality measure (gini or entropy)
                random_state=42,
            ),
            n_jobs=-1,
        ),
    }

    # ── Train & compare ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING & EVALUATION")
    print("=" * 60)

    results = {}
    for name, model in candidates.items():
        print(f"\n  Training {name} ...")
        model.fit(X_train, y_train)
        avg_acc = evaluate_model(name, model, X_test, y_test, encoders)
        results[name] = (model, avg_acc)

    # ── Cross-validation comparison ───────────────────────────
    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION (5-fold) on Exercise Schedule target")
    print("=" * 60)

    y_single = y[:, 0]   # use first target for CV comparison
    for name, (model, _) in results.items():
        base = model.estimators_[0] if hasattr(model, "estimators_") else model
        cv_scores = cross_val_score(base, X, y_single, cv=5, scoring="accuracy")
        print(f"  {name:15s} → CV mean: {cv_scores.mean():.4f}  ± {cv_scores.std():.4f}")

    # ── Pick best model ───────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    best_score = results[best_name][1]

    print("\n" + "=" * 60)
    print(f"  ✅  WINNER: {best_name}  (avg accuracy: {best_score:.4f})")
    print("=" * 60)

    # ── Print Decision Tree rules (if DT won or as bonus info) ─
    if best_name == "DecisionTree":
        print("\n  Decision Tree structure for 'Exercise Schedule':")
        dt = best_model.estimators_[0]
        rules = export_text(dt, feature_names=INPUT_COLS, max_depth=4)
        print(rules)

    # ── Save artifact ─────────────────────────────────────────
    artifact = {
        "model"        : best_model,
        "encoders"     : encoders,
        "input_cols"   : INPUT_COLS,
        "target_cols"  : TARGET_COLS,
        "algorithm"    : best_name,
        "avg_accuracy" : best_score,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n✅  Model saved  → {MODEL_PATH}")

    # ── Save meta JSON ────────────────────────────────────────
    meta = {
        "algorithm"   : best_name,
        "avg_accuracy": round(best_score, 4),
        "input_cols"  : INPUT_COLS,
        "target_cols" : TARGET_COLS,
        "input_classes": {
            col: list(encoders[f"input_{col}"].classes_)
            for col in INPUT_COLS
        },
        "target_classes": {
            col: list(encoders[f"target_{col}"].classes_)
            for col in TARGET_COLS
        },
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅  Meta saved   → {META_PATH}\n")

    return best_model, encoders


# ══════════════════════════════════════════════════════════════
#  INFERENCE HELPER
# ══════════════════════════════════════════════════════════════

def predict(gender: str, goal: str, bmi_category: str,
            model_path: str = MODEL_PATH) -> dict:
    """
    Predict Exercise Schedule and Meal Plan for a single user.

    Example:
        predict("Female", "Weight Loss", "Overweight")
    """
    artifact    = joblib.load(model_path)
    model       = artifact["model"]
    encoders    = artifact["encoders"]
    input_cols  = artifact["input_cols"]
    target_cols = artifact["target_cols"]

    inputs = [gender, goal, bmi_category]
    row = []
    for col, val in zip(input_cols, inputs):
        le = encoders[f"input_{col}"]
        if val not in le.classes_:
            raise ValueError(
                f"Unknown value '{val}' for '{col}'.\n"
                f"  Valid options: {list(le.classes_)}"
            )
        row.append(le.transform([val])[0])

    X_new  = np.array(row).reshape(1, -1)
    y_pred = model.predict(X_new)[0]

    return {
        col: encoders[f"target_{col}"].inverse_transform([y_pred[i]])[0]
        for i, col in enumerate(target_cols)
    }


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()

    print("─" * 60)
    print("  DEMO PREDICTION")
    print("─" * 60)

    with open(META_PATH) as f:
        meta = json.load(f)

    demo_gender = meta["input_classes"]["Gender"][0]
    demo_goal   = meta["input_classes"]["Goal"][0]
    demo_bmi    = meta["input_classes"]["BMI Category"][0]

    print(f"\n  Algorithm used : {meta['algorithm']}")
    print(f"  Avg accuracy   : {meta['avg_accuracy']}")
    print(f"\n  Input  → Gender: {demo_gender} | Goal: {demo_goal} | BMI: {demo_bmi}")

    result = predict(demo_gender, demo_goal, demo_bmi)
    print(f"  Output → {result}\n")