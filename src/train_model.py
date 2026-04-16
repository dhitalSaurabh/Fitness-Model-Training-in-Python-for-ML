"""
train_tdee_model.py
─────────────────────────────────────────────────────────────────
Trains a TDEE predictor using multiple regression algorithms:
  1. KNeighborsRegressor     (KNN)
  2. RandomForestRegressor   (Ensemble)
  3. DecisionTreeRegressor   (Single Tree)
  4. LinearRegression        (Baseline)
  5. GradientBoostingRegressor (original — kept for comparison)

Automatically picks the best model by R² score and saves it
to models/tdee_model.pkl for use by app.py.
─────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ── Config ────────────────────────────────────────────────────
CSV_PATH   = "../data/fitness_data.csv"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "tdee_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  LOAD & PREPROCESS  (unchanged from original)
# ══════════════════════════════════════════════════════════════

def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    column_mapping = {
        "participant_id"          : "participant_id",
        "height_cm"               : "height",
        "weight_kg"               : "weight",
        "duration_minutes"        : "duration_m",
        "calories_burned"         : "calories_burn",
        "avg_heart_rate"          : "avg_heartrate",
        "blood_pressure_systolic" : "blood_pressure_sys",
        "blood_pressure_diastolic": "blood_pressure_dy",
        "health_condition"        : "healthcond",
    }
    df.rename(columns=column_mapping, inplace=True)

    df = df.dropna(subset=["calories_burn"])

    for drop_col in ["participant_id", "date"]:
        if drop_col in df.columns:
            df.drop(columns=[drop_col], inplace=True)

    # Encode gender
    df["gender_male"] = (
        df["gender"].str.lower()
        .map({"male": 1, "m": 1, "female": 0, "f": 0})
        .fillna(0)
    )

    # Encode categoricals
    cat_cols = ["activity_type", "intensity", "fitness_level",
                "smoking_status", "healthcond"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("unknown")
            df[col + "_enc"] = le.fit_transform(df[col])

    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


def build_features(df: pd.DataFrame):
    feature_cols = [
        "age", "height", "weight", "bmi",
        "gender_male",
        "hours_sleep", "daily_steps",
        "hydration_level", "stress_level",
        "resting_heart_rate",
        "duration_m",
        "avg_heartrate",
        "activity_type_enc",
        "intensity_enc",
        "fitness_level_enc",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["calories_burn"]
    return X, y, feature_cols


# ══════════════════════════════════════════════════════════════
#  CANDIDATE MODELS
# ══════════════════════════════════════════════════════════════

def get_candidates() -> dict:
    """
    Returns all candidate regression models.

    KNN needs feature scaling (distances are sensitive to magnitude),
    so it is wrapped in a Pipeline with StandardScaler.
    All others work fine on raw features.
    """
    return {

        # ── 1. KNN Regressor ──────────────────────────────────
        # Predicts by averaging the k nearest training samples.
        # Simple, non-parametric — no assumptions about data shape.
        # Requires scaling so all features contribute equally.
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(
                n_neighbors=7,      # number of neighbours to average
                weights="distance", # closer neighbours matter more
                metric="euclidean",
                n_jobs=-1,
            )),
        ]),

        # ── 2. Random Forest Regressor ────────────────────────
        # Builds many trees on random data subsets, averages results.
        # Handles non-linear patterns well, robust to outliers.
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),

        # ── 3. Decision Tree Regressor ────────────────────────
        # A single tree — fast and interpretable.
        # max_depth limits overfitting on training data.
        "DecisionTree": DecisionTreeRegressor(
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
        ),

        # ── 4. Linear Regression ──────────────────────────────
        # Fits a straight line through the data.
        # Fast baseline — shows how much non-linear models help.
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),

        # ── 5. Gradient Boosting (original — kept for comparison)
        # Builds trees sequentially, each fixing the previous errors.
        # Usually top performer but slower to train.
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        ),
    }


# ══════════════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════════════

def evaluate(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    """Train, predict, and print metrics for one model."""
    print(f"\n  Training {name} ...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    print(f"  {'MAE':5s}  : {mae:.2f} kcal   (avg error per prediction)")
    print(f"  {'RMSE':5s} : {rmse:.2f} kcal   (penalises large errors more)")
    print(f"  {'R²':5s}   : {r2:.4f}         (1.0 = perfect, >0.85 = good)")

    return {"model": model, "mae": mae, "rmse": rmse, "r2": r2}


# ══════════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════

def train(csv_path: str = CSV_PATH):
    print("=" * 60)
    print("  TDEE MODEL — MULTI-ALGORITHM TRAINING")
    print("=" * 60)

    df = load_and_preprocess(csv_path)
    print(f"\n  Rows after cleaning : {len(df)}")

    X, y, feature_cols = build_features(df)
    print(f"  Features used ({len(feature_cols)}) : {feature_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Train all candidates ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING ALL MODELS")
    print("=" * 60)

    candidates = get_candidates()
    results = {}
    for name, model in candidates.items():
        results[name] = evaluate(name, model, X_train, X_test, y_train, y_test)

    # ── Cross-validation (5-fold) ─────────────────────────────
    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION COMPARISON (5-fold R²)")
    print("=" * 60)
    for name, info in results.items():
        cv = cross_val_score(
            info["model"], X, y, cv=5, scoring="r2", n_jobs=-1
        )
        print(f"  {name:20s} → mean R²: {cv.mean():.4f}  ± {cv.std():.4f}")

    # ── Summary table ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(f"  {'Algorithm':20s}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}")
    print("  " + "-" * 50)
    for name, info in results.items():
        print(
            f"  {name:20s}  "
            f"{info['mae']:>8.2f}  "
            f"{info['rmse']:>8.2f}  "
            f"{info['r2']:>8.4f}"
        )

    # ── Pick best by R² ───────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["r2"])
    best      = results[best_name]

    print("\n" + "=" * 60)
    print(f"  ✅  WINNER : {best_name}")
    print(f"      MAE   : {best['mae']:.2f} kcal")
    print(f"      RMSE  : {best['rmse']:.2f} kcal")
    print(f"      R²    : {best['r2']:.4f}")
    print("=" * 60)

    # ── Save best model ───────────────────────────────────────
    artifact = {
        "model"       : best["model"],
        "feature_cols": feature_cols,
        "algorithm"   : best_name,
        "r2"          : round(best["r2"], 4),
        "mae"         : round(best["mae"], 2),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n✅  Model saved → {MODEL_PATH}")

    return best["model"], feature_cols


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model, feature_cols = train()

    # ── Demo: show a sample prediction ───────────────────────
    print("\n" + "─" * 60)
    print("  DEMO PREDICTION")
    print("─" * 60)
    sample = pd.DataFrame([{col: 0 for col in feature_cols}])
    # Fill with realistic dummy values
    defaults = {
        "age": 30, "height": 175, "weight": 75, "bmi": 24.5,
        "gender_male": 1, "hours_sleep": 7, "daily_steps": 8000,
        "hydration_level": 2.5, "stress_level": 5,
        "resting_heart_rate": 68, "duration_m": 45,
        "avg_heartrate": 130, "activity_type_enc": 1,
        "intensity_enc": 2, "fitness_level_enc": 2,
    }
    for col in feature_cols:
        if col in defaults:
            sample[col] = defaults[col]

    artifact = joblib.load(MODEL_PATH)
    pred = artifact["model"].predict(sample)[0]
    print(f"\n  Algorithm : {artifact['algorithm']}")
    print(f"  Input     : 30yo male, 75kg, 175cm, moderate gym")
    print(f"  Predicted TDEE : {pred:.0f} kcal/day\n")