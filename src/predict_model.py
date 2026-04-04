"""
train_model.py
─────────────────────────────────────────────────────────────────
Trains a fitness recommendation model on your CSV dataset and
saves the model artifacts to models/ for use by an inference app.

CSV columns expected:
  Gender | Goal | BMI Category | Exercise Schedule | Meal Plan
─────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import json

# ── Config ────────────────────────────────────────────────────
CSV_PATH   = "../data/GYM.csv"   # ← change to your CSV path
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fitness_model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "fitness_model_meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Column definitions ────────────────────────────────────────
INPUT_COLS  = ["Gender", "Goal", "BMI Category"]      # Features
TARGET_COLS = ["Exercise Schedule", "Meal Plan"]       # What we predict


def load_and_preprocess(path: str):
    """Load CSV, clean, encode, and split into X/y."""
    print(f"  Reading CSV from: {path}")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns  : {list(df.columns)}\n")

    # ── Standardise column names (strip whitespace) ───────────
    df.columns = df.columns.str.strip()

    # ── Drop rows missing any critical column ─────────────────
    required = INPUT_COLS + TARGET_COLS
    df = df.dropna(subset=required)
    print(f"  Rows after dropping nulls in required cols: {len(df)}")

    # ── Strip whitespace from string values ───────────────────
    for col in required:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    return df


def encode_features(df: pd.DataFrame):
    """
    Encode input features and targets.
    Returns encoded X, y arrays plus the fitted encoders.
    """
    encoders = {}

    # ── Encode inputs ─────────────────────────────────────────
    X_parts = []
    for col in INPUT_COLS:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col].astype(str))
        X_parts.append(encoded.reshape(-1, 1))
        encoders[f"input_{col}"] = le
        print(f"  [{col}] classes: {list(le.classes_)}")

    X = np.hstack(X_parts)

    # ── Encode targets ────────────────────────────────────────
    Y_parts = []
    for col in TARGET_COLS:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col].astype(str))
        Y_parts.append(encoded.reshape(-1, 1))
        encoders[f"target_{col}"] = le
        print(f"  [{col}] classes: {list(le.classes_)}")

    y = np.hstack(Y_parts)

    return X, y, encoders


def train(csv_path: str = CSV_PATH):
    print("=" * 60)
    print(" FITNESS RECOMMENDATION MODEL — TRAINING")
    print("=" * 60)

    df = load_and_preprocess(csv_path)
    X, y, encoders = encode_features(df)

    print(f"\n  Feature matrix : {X.shape}")
    print(f"  Target matrix  : {y.shape}")

    # ── Train / test split ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    # ── Model: multi-output gradient boosting ─────────────────
    print("\n  Training MultiOutput GradientBoostingClassifier ...")
    base_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        random_state=42,
    )
    model = MultiOutputClassifier(base_model, n_jobs=-1)
    model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(" EVALUATION RESULTS")
    print("─" * 60)

    y_pred = model.predict(X_test)

    for i, target_col in enumerate(TARGET_COLS):
        le = encoders[f"target_{target_col}"]
        true_labels = le.inverse_transform(y_test[:, i])
        pred_labels = le.inverse_transform(y_pred[:, i])
        acc = accuracy_score(true_labels, pred_labels)

        print(f"\n  ▶  {target_col}")
        print(f"     Accuracy : {acc:.4f} ({acc*100:.1f}%)")
        print(classification_report(
            true_labels, pred_labels,
            zero_division=0,
            target_names=le.classes_
        ))

    # ── Save artifact ─────────────────────────────────────────
    artifact = {
        "model"       : model,
        "encoders"    : encoders,
        "input_cols"  : INPUT_COLS,
        "target_cols" : TARGET_COLS,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n✅  Model saved → {MODEL_PATH}")

    # Save human-readable meta for the inference app
    meta = {
        "input_cols" : INPUT_COLS,
        "target_cols": TARGET_COLS,
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
    print(f"✅  Meta saved  → {META_PATH}\n")

    return model, encoders


# ── Quick inference helper ────────────────────────────────────
def predict(gender: str, goal: str, bmi_category: str,
            model_path: str = MODEL_PATH):
    """
    Example usage:
        predict("Female", "Weight Loss", "Overweight")
    Returns dict with Exercise Schedule and Meal Plan predictions.
    """
    artifact  = joblib.load(model_path)
    model     = artifact["model"]
    encoders  = artifact["encoders"]
    input_cols  = artifact["input_cols"]
    target_cols = artifact["target_cols"]

    inputs = [gender, goal, bmi_category]
    row = []
    for col, val in zip(input_cols, inputs):
        le = encoders[f"input_{col}"]
        if val not in le.classes_:
            raise ValueError(
                f"Unknown value '{val}' for '{col}'. "
                f"Valid options: {list(le.classes_)}"
            )
        row.append(le.transform([val])[0])

    X_new = np.array(row).reshape(1, -1)
    y_pred = model.predict(X_new)[0]

    result = {}
    for i, col in enumerate(target_cols):
        le = encoders[f"target_{col}"]
        result[col] = le.inverse_transform([y_pred[i]])[0]

    return result


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    train()

    # ── Demo prediction (runs after training) ─────────────────
    print("─" * 60)
    print(" DEMO PREDICTION")
    print("─" * 60)

    import json
    with open(META_PATH) as f:
        meta = json.load(f)

    # Pick the first available class for each input as a demo
    demo_gender = meta["input_classes"]["Gender"][0]
    demo_goal   = meta["input_classes"]["Goal"][0]
    demo_bmi    = meta["input_classes"]["BMI Category"][0]

    print(f"\n  Input  → Gender: {demo_gender} | Goal: {demo_goal} | BMI: {demo_bmi}")
    result = predict(demo_gender, demo_goal, demo_bmi)
    print(f"  Output → {result}\n")