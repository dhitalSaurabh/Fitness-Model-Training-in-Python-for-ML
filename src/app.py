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
                         lifestyle, measurement, goal) -> tuple:
    """
    Convert raw Exercise Schedule + Meal Plan strings into structured lists.
    Habit tips and supplements are now fully dynamic based on user data.
    Returns: (meal_plan, workout_plan, habit_tips, supplement_suggestions)
    """
    ex_days  = get_safe_attr(lifestyle, "exercise_days_per_week", 3) or 3
    duration = get_safe_attr(lifestyle, "workout_duration_mins", 45) or 45

    # ── Pull user values for dynamic logic ────────────────────
    goal_type    = goal.goal_type.lower()
    bmi          = measurement.bmi or round(
                       measurement.weight_kg / ((measurement.height_cm / 100) ** 2), 2)
    age          = measurement.age
    weight_kg    = measurement.weight_kg
    sleep_hrs    = get_safe_attr(lifestyle, "sleep_hours", 7.0) or 7.0
    stress       = get_safe_attr(lifestyle, "stress_level", 5) or 5
    daily_steps  = get_safe_attr(lifestyle, "daily_steps", 8000) or 8000
    hydration    = get_safe_attr(lifestyle, "water_intake_liters", 2.5) or 2.5
    fitness_lvl  = get_safe_attr(lifestyle, "fitness_level", "intermediate") or "intermediate"
    activity     = get_safe_attr(lifestyle, "activity_level", "gym") or "gym"

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

    # ══════════════════════════════════════════════════════════
    #  DYNAMIC HABIT TIPS
    # ══════════════════════════════════════════════════════════
    habit_tips = []

    # 1. Hydration tip — personalised to body weight
    recommended_water = round(weight_kg * 0.033, 1)  # standard formula
    if hydration < recommended_water:
        habit_tips.append(
            f"Increase water intake to {recommended_water} L/day "
            f"(currently ~{hydration} L) — your body weight of {weight_kg}kg needs more."
        )
    else:
        habit_tips.append(
            f"Good hydration habit! Keep drinking {recommended_water} L/day "
            f"to support your {diet_label.strip()} plan."
        )

    # 2. Sleep tip — based on actual sleep hours
    if sleep_hrs < 6:
        habit_tips.append(
            f"⚠️ You're sleeping only {sleep_hrs}h — aim for 7–9h. "
            f"Poor sleep raises cortisol and can stall "
            + ("fat loss." if goal_type in ("fat_loss", "weight_loss") else "muscle recovery.")
        )
    elif sleep_hrs < 7:
        habit_tips.append(
            f"Try to get 1 more hour of sleep (currently {sleep_hrs}h). "
            f"7–9h optimises hormone balance for {goal_type.replace('_', ' ')}."
        )
    else:
        habit_tips.append(
            f"Great sleep routine ({sleep_hrs}h)! Consistent sleep supports "
            f"recovery and keeps hunger hormones balanced."
        )

    # 3. Steps tip — based on daily_steps
    if daily_steps < 5000:
        habit_tips.append(
            f"Your daily steps ({daily_steps:,}) are low. "
            f"Aim for at least 8,000 steps/day — even short walks add up."
        )
    elif daily_steps < 8000:
        habit_tips.append(
            f"You're at {daily_steps:,} steps/day — push toward 10,000 "
            f"to boost your TDEE by ~{round((10000 - daily_steps) * 0.04)} extra kcal/day."
        )
    else:
        habit_tips.append(
            f"Excellent step count ({daily_steps:,}/day)! "
            f"This contributes significantly to your calorie burn."
        )

    # 4. Stress tip — based on stress level (1–10)
    if stress >= 7:
        habit_tips.append(
            f"Your stress level ({stress}/10) is high. Chronic stress raises "
            f"cortisol which can cause fat storage around the abdomen. "
            f"Try 10 min of daily meditation or breathing exercises."
        )
    elif stress >= 5:
        habit_tips.append(
            f"Moderate stress ({stress}/10) detected. Consider light yoga or "
            f"walking to keep cortisol in check alongside your {activity} sessions."
        )

    # 5. Goal-specific calorie tracking tip
    habit_tips.append(
        f"Track meals for the first 2 weeks — target {round(target_calories)} kcal "
        f"({round(protein_g)}g protein / {round(carbs_g)}g carbs / {round(fat_g)}g fat). "
        + (
            "A 500 kcal deficit is safe for steady fat loss."
            if goal_type in ("fat_loss", "weight_loss") else
            "A 300 kcal surplus helps muscle growth without excess fat gain."
            if goal_type == "muscle_gain" else
            "Staying at maintenance keeps your weight stable."
        )
    )

    # 6. Fitness level tip
    if fitness_lvl.lower() == "beginner":
        habit_tips.append(
            "As a beginner, focus on consistency over intensity. "
            "3 sessions/week is plenty — rest days are when muscles grow."
        )
    elif fitness_lvl.lower() == "advanced":
        habit_tips.append(
            "At advanced level, consider periodisation — cycle high/low "
            "intensity weeks to avoid plateaus and overtraining."
        )

    # 7. Age-specific tip
    if age >= 50:
        habit_tips.append(
            f"At {age}, prioritise resistance training to preserve muscle mass "
            f"(natural muscle loss accelerates after 50). Aim for 2–3 strength sessions/week."
        )
    elif age <= 25:
        habit_tips.append(
            f"At {age}, your metabolism is at its peak — take advantage "
            f"by building solid training habits now that will last long-term."
        )

    # ══════════════════════════════════════════════════════════
    #  DYNAMIC SUPPLEMENT SUGGESTIONS
    # ══════════════════════════════════════════════════════════
    supplement_suggestions = []

    # Goal-based supplements
    if goal_type in ("fat_loss", "weight_loss"):
        supplement_suggestions.append(
            "L-Carnitine (500–2000 mg/day) — may support fat metabolism "
            "during cardio sessions."
        )
        supplement_suggestions.append(
            "Green Tea Extract or Caffeine — can modestly increase calorie "
            "burn; use pre-workout if tolerated."
        )

    elif goal_type == "muscle_gain":
        supplement_suggestions.append(
            f"Creatine Monohydrate (5g/day) — most evidence-backed supplement "
            f"for strength and muscle gain; safe for long-term use."
        )
        supplement_suggestions.append(
            f"Whey Protein — helps hit your {round(protein_g)}g protein target "
            f"if whole food sources fall short."
        )

    else:  # maintenance
        supplement_suggestions.append(
            "Omega-3 Fish Oil (1–3g/day) — supports heart health, "
            "joint recovery, and inflammation control."
        )

    # Sleep-based supplement
    if sleep_hrs < 7:
        supplement_suggestions.append(
            "Magnesium Glycinate (200–400 mg before bed) — helps improve "
            "sleep quality and muscle relaxation."
        )

    # Stress-based supplement
    if stress >= 7:
        supplement_suggestions.append(
            "Ashwagandha (300–600 mg/day) — adaptogen shown to reduce "
            "cortisol levels under chronic stress."
        )

    # Age-based supplement
    if age >= 40:
        supplement_suggestions.append(
            "Vitamin D3 + K2 (2000 IU D3 daily) — bone density and "
            "immune support become more important after 40."
        )
    else:
        supplement_suggestions.append(
            "Vitamin D3 (1000–2000 IU daily) — especially if you train "
            "indoors or live in a low-sunlight region."
        )

    # BMI-based supplement
    if bmi >= 30:
        supplement_suggestions.append(
            "Berberine (500 mg 2–3x/day with meals) — may help improve "
            "insulin sensitivity alongside your fat-loss diet."
        )

    # Always-last disclaimer
    supplement_suggestions.append(
        "⚠️ Consult a healthcare provider before starting any supplement — "
        "especially if you have existing medical conditions."
    )

    return meal_plan, workout_plan, habit_tips, supplement_suggestions

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
            fitness_result, target_calories, protein_g, carbs_g, fat_g, lifestyle, measurement, goal
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
            "engine_used"           : "ml_model",
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