"""
app.py
─────────────────────────────────────────────────────────────────
Flask inference API combining:
  1. models/tdee_model.pkl        → predicts TDEE calories
  2. models/fitness_model.pkl     → predicts Exercise Schedule + Meal Plan
     (replaces OpenAI — no API key needed)
─────────────────────────────────────────────────────────────────
"""

import pandas as pd
import sys
import os
import traceback
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from datetime import datetime, timedelta
import joblib
import numpy as np
from models.user import User
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# ── Load TDEE model ───────────────────────────────────────────
tdee_artifact  = joblib.load("models/tdee_model.pkl")
tdee_model     = tdee_artifact["model"]
feature_cols   = tdee_artifact["feature_cols"]

# ── Load Fitness Recommendation model ─────────────────────────
fit_artifact      = joblib.load("models/fitness_model.pkl")
fitness_model     = fit_artifact["model"]
fitness_encoders  = fit_artifact["encoders"]
fitness_in_cols   = fit_artifact["input_cols"]   # ["Gender", "Goal", "BMI Category"]
fitness_out_cols  = fit_artifact["target_cols"]  # ["Exercise Schedule", "Meal Plan"]

# ── Load meta for valid class names ───────────────────────────
with open("models/fitness_model_meta.json") as f:
    fitness_meta = json.load(f)

# ── Request schema ────────────────────────────────────────────
class UserRequest(BaseModel):
    message: str
    data: User


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def get_safe_attr(obj, attr, default=None):
    """Safely get an attribute even if the object is None."""
    if obj is None:
        return default
    return getattr(obj, attr, default)


def calc_weeks_to_goal(measurement, goal, target_calories: float, tdee: float) -> int:
    """Estimate realistic weeks to reach goal weight / body-fat target."""
    goal_type = goal.goal_type.lower()

    if goal_type in ("fat_loss", "weight_loss"):
        current_kg = measurement.weight_kg
        target_kg  = goal.target_weight_kg or (current_kg - 5)
        diff_kg    = max(current_kg - target_kg, 0)
        weekly_deficit_kcal = (tdee - target_calories) * 7
        kg_per_week = weekly_deficit_kcal / 7700
        return max(round(diff_kg / kg_per_week), 1) if kg_per_week > 0 else 12

    if goal_type == "muscle_gain":
        current_muscle = measurement.muscle_mass_kg or 0
        target_muscle  = goal.target_muscle_mass_kg or (current_muscle + 3)
        diff_kg        = max(target_muscle - current_muscle, 0)
        kg_per_week    = 0.15
        return max(round(diff_kg / kg_per_week), 4)

    return 8


def map_goal_to_fitness_model(goal_type: str) -> str:
    """
    Map TDEE goal types → fitness model's Goal classes.
    Fitness model knows: ['fat_burn', 'muscle_gain']
    """
    mapping = {
        "fat_loss"    : "fat_burn",
        "weight_loss" : "fat_burn",
        "fat_burn"    : "fat_burn",
        "muscle_gain" : "muscle_gain",
        "maintenance" : "fat_burn",   # closest available class
    }
    return mapping.get(goal_type.lower(), "fat_burn")


def map_bmi_to_category(bmi: float) -> str:
    """
    Convert numeric BMI → fitness model's BMI Category classes.
    Classes: ['Normal weight', 'Obesity', 'Overweight', 'Underweight']
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal weight"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obesity"


def run_fitness_model(gender: str, goal_type: str, bmi: float) -> dict:
    """
    Run the fitness recommendation model.
    Returns {"Exercise Schedule": "...", "Meal Plan": "..."}
    """
    gender_mapped = "Male" if gender.lower() in ("male", "m") else "Female"
    goal_mapped   = map_goal_to_fitness_model(goal_type)
    bmi_mapped    = map_bmi_to_category(bmi)

    inputs = [gender_mapped, goal_mapped, bmi_mapped]
    row = []
    for col, val in zip(fitness_in_cols, inputs):
        le = fitness_encoders[f"input_{col}"]
        if val not in le.classes_:
            val = le.classes_[0]   # safe fallback to first class
        row.append(le.transform([val])[0])

    X_new  = np.array(row).reshape(1, -1)
    y_pred = fitness_model.predict(X_new)[0]

    return {
        col: fitness_encoders[f"target_{col}"].inverse_transform([y_pred[i]])[0]
        for i, col in enumerate(fitness_out_cols)
    }


def build_plans_from_ml(fitness_result: dict, target_calories: float,
                         protein_g: float, carbs_g: float, fat_g: float,
                         lifestyle) -> tuple:
    """
    Convert raw Exercise Schedule + Meal Plan strings into structured lists
    that match the original OpenAI response shape.
    Returns: (meal_plan, workout_plan, habit_tips, supplement_suggestions)
    """
    ex_days  = get_safe_attr(lifestyle, "exercise_days_per_week", 3) or 3
    duration = get_safe_attr(lifestyle, "workout_duration_mins", 45) or 45

    # ── Meal plan ─────────────────────────────────────────────
    raw_meal = fitness_result.get("Meal Plan", "")
    if ":" in raw_meal:
        diet_label, foods_str = raw_meal.split(":", 1)
        foods = [f.strip() for f in foods_str.split(",") if f.strip()]
    else:
        diet_label = raw_meal
        foods = []

    kcal_per_meal = round(target_calories / 3)

    meal_plan = [
        f"Breakfast ({kcal_per_meal} kcal) — {diet_label.strip()}: "
        + (foods[0] if len(foods) > 0 else "balanced portion"),

        f"Lunch ({kcal_per_meal} kcal) — {diet_label.strip()}: "
        + (", ".join(foods[1:3]) if len(foods) > 1 else "balanced portion"),

        f"Dinner ({kcal_per_meal} kcal) — {diet_label.strip()}: "
        + (", ".join(foods[3:]) if len(foods) > 3 else (foods[-1] if foods else "balanced portion")),
    ]

    # ── Workout plan ──────────────────────────────────────────
    raw_exercise = fitness_result.get("Exercise Schedule", "")
    parts = [p.strip() for p in raw_exercise.replace(" and ", ", ").split(",") if p.strip()]

    workout_plan = []
    for part in parts:
        if "steps" in part.lower():
            workout_plan.append(f"Daily goal: {part} every day")
        else:
            workout_plan.append(
                f"{part} — {ex_days}x/week, {duration} min per session"
            )

    # ── Habit tips ────────────────────────────────────────────
    habit_tips = [
        f"Drink 2.5–3 L of water daily (supports your {diet_label.strip()} plan)",
        "Aim for 7–9 hours of sleep to optimise recovery and hormone balance",
        f"Track meals for the first 2 weeks — target {round(target_calories)} kcal "
        f"({round(protein_g)}g protein / {round(carbs_g)}g carbs / {round(fat_g)}g fat)",
    ]

    # ── Supplement suggestions ────────────────────────────────
    supplement_suggestions = [
        "Vitamin D3 (1000–2000 IU daily) — supports muscle function and immunity",
        "Consult a healthcare provider before starting any supplement regimen",
    ]

    return meal_plan, workout_plan, habit_tips, supplement_suggestions


# ══════════════════════════════════════════════════════════════
#  MAIN ENDPOINT
# ══════════════════════════════════════════════════════════════

@app.post("/api/recommendation")
def recommendation():
    try:
        json_data    = request.get_json()
        user_request = UserRequest(**json_data)
        user         = user_request.data

        # ── Latest measurement & active goal ──────────────────
        measurement = sorted(user.body_metrics, key=lambda x: x.measured_at, reverse=True)[0]
        goal        = next((g for g in user.goals if g.is_active), user.goals[-1])
        lifestyle   = user.lifestyle

        # ── Feature engineering for TDEE model ───────────────
        sex_val     = 1 if measurement.sex.lower() == "male" else 0
        sleep_hrs   = getattr(lifestyle, "sleep_hours", 7.0) or 7.0
        daily_steps = getattr(lifestyle, "daily_steps", 8000) or getattr(measurement, "daily_steps", 8000) or 8000
        hydration   = getattr(lifestyle, "water_intake_liters", 2.5) or 2.5
        stress      = getattr(lifestyle, "stress_level", 5) or getattr(measurement, "stress_level", 5) or 5
        resting_hr  = getattr(measurement, "resting_heart_rate", 70) or 70
        duration_m  = getattr(lifestyle, "workout_duration_mins", 45) or 45
        avg_hr      = getattr(measurement, "avg_heart_rate", 80) or 80

        activity_map  = {"cycling": 0, "gym": 1, "running": 2, "swimming": 3, "walking": 4}
        intensity_map = {"high": 0, "low": 1, "moderate": 2, "slow": 3}
        fitness_map   = {"advanced": 0, "beginner": 1, "intermediate": 2, "sedentary": 3}

        act_enc = activity_map.get(getattr(lifestyle, "activity_level",  "gym").lower(),          1)
        int_enc = intensity_map.get(getattr(lifestyle, "intensity_level", "moderate").lower(),    2)
        fit_enc = fitness_map.get(getattr(lifestyle,   "fitness_level",   "intermediate").lower(), 2)

        bmi = measurement.bmi or round(
            measurement.weight_kg / ((measurement.height_cm / 100) ** 2), 2
        )

        features = [[
            measurement.age, measurement.height_cm, measurement.weight_kg, bmi,
            sex_val, sleep_hrs, daily_steps, hydration, stress,
            resting_hr, duration_m, avg_hr, act_enc, int_enc, fit_enc,
        ]]

        features_df    = pd.DataFrame(features, columns=feature_cols)
        predicted_tdee = float(tdee_model.predict(features_df)[0])

        # ── Target calories ───────────────────────────────────
        goal_type = goal.goal_type.lower()
        if goal_type in ("fat_loss", "weight_loss"):
            target_calories = predicted_tdee - 500
        elif goal_type == "muscle_gain":
            target_calories = predicted_tdee + 300
        else:
            target_calories = predicted_tdee
        target_calories = max(target_calories, 1200)

        # ── Macros ────────────────────────────────────────────
        protein_g = measurement.weight_kg * 2.0
        fat_g     = measurement.weight_kg * 0.8
        carbs_g   = max((target_calories - (protein_g * 4 + fat_g * 9)) / 4, 50)

        # ── Run fitness recommendation model ──────────────────
        fitness_result = run_fitness_model(measurement.sex, goal_type, bmi)

        # ── Build structured plans from ML output ─────────────
        meal_plan, workout, habits, supplements = build_plans_from_ml(
            fitness_result, target_calories, protein_g, carbs_g, fat_g, lifestyle
        )

        weeks = calc_weeks_to_goal(measurement, goal, target_calories, predicted_tdee)

        # ── Response ──────────────────────────────────────────
        response = {
            "body_metric_id"        : measurement.id,
            "goal_id"               : goal.id,
            "tdee_calories"         : round(predicted_tdee),
            "target_calories"       : round(target_calories),
            "protein_g"             : round(protein_g),
            "carbs_g"               : round(carbs_g),
            "fat_g"                 : round(fat_g),
            "weeks_to_goal"         : weeks,
            "engine_used"           : "ml_fitness_model",
            "confidence_score"      : 1.0,
            "ml_raw"                : fitness_result,   # raw model output for debugging
            "meal_plan"             : meal_plan,
            "workout_plan"          : workout,
            "habit_tips"            : habits,
            "supplement_suggestions": supplements,
            "is_active"             : True,
            "expires_at"            : (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z",
        }

        return jsonify(response), 200

    except ValidationError as e:
        return jsonify({"error": "Validation error", "details": e.errors()}), 422
    except IndexError:
        return jsonify({"error": "User has no body_metrics or goals"}), 400
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.get("/")
def home():
    return jsonify({"status": "Flask ML API running — TDEE + Fitness Recommendation models active"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)