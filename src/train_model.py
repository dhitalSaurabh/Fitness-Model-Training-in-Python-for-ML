"""
train_model.py
─────────────────────────────────────────────────────────────────
Trains a TDEE predictor on your CSV dataset and saves the model
to models/tdee_model.pkl for use by app.py.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ── Config ────────────────────────────────────────────────────
CSV_PATH   = "../data/fitness_data.csv"   # ← change to your CSV path
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "tdee_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ── 1. Rename actual CSV columns to standard snake_case ────
    column_mapping = {
        "participant_id": "participant_id", # Will be dropped later
        "height_cm": "height",
        "weight_kg": "weight",
        "duration_minutes": "duration_m",
        "calories_burned": "calories_burn", # Target variable
        "avg_heart_rate": "avg_heartrate",
        "blood_pressure_systolic": "blood_pressure_sys",
        "blood_pressure_diastolic": "blood_pressure_dy",
        "health_condition": "healthcond",
    }
    df.rename(columns=column_mapping, inplace=True)

    # ── 2. Drop rows with missing target ─────────────────────
    df = df.dropna(subset=["calories_burn"])

    # ── 3. Drop unused columns ───────────────────────────────
    if "participant_id" in df.columns:
        df.drop(columns=["participant_id"], inplace=True)
    if "date" in df.columns:
        df.drop(columns=["date"], inplace=True)

    # ── 4. Encode gender ──────────────────────────────────────
    df["gender_male"] = df["gender"].str.lower().map({"male": 1, "m": 1, "female": 0, "f": 0}).fillna(0)

    # ── 5. Encode categorical features ────────────────────────
    cat_cols = ["activity_type", "intensity", "fitness_level", "smoking_status", "healthcond"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            # Fill NaNs with a string so LabelEncoder doesn't crash
            df[col] = df[col].astype(str).fillna("unknown")
            df[col + "_enc"] = le.fit_transform(df[col])

    # ── 6. Fill numeric NaNs with median ─────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


def build_features(df: pd.DataFrame):
    """Select features that map exactly to what app.py sends at inference."""
    feature_cols = [
        "age", "height", "weight", "bmi",
        "gender_male",
        "hours_sleep", "daily_steps",
        "hydration_level", "stress_level",
        "resting_heart_rate",
        "duration_m",           # workout duration
        "avg_heartrate",
        "activity_type_enc",
        "intensity_enc",
        "fitness_level_enc"
    ]
    
    # Keep only columns that exist (safety check)
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols]
    y = df["calories_burn"]
    return X, y, feature_cols


def train(csv_path: str = CSV_PATH):
    print(f"Loading data from {csv_path} ...")
    df = load_and_preprocess(csv_path)
    print(f"  Rows after cleaning: {len(df)}")

    X, y, feature_cols = build_features(df)
    print(f"  Features used ({len(feature_cols)}): {feature_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training GradientBoostingRegressor ...")
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"\n  Test MAE : {mae:.2f} kcal")
    print(f"  Test R²  : {r2:.4f}")

    # Save model + feature list together so app.py knows what to expect
    artifact = {"model": model, "feature_cols": feature_cols}
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n✅  Model saved to {MODEL_PATH}")
    return model, feature_cols


if __name__ == "__main__":
    train()