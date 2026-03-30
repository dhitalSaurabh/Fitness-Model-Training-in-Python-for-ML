import pandas as pd 
import sys
import os
import traceback
import openai
import json
# Add project root to sys.path so we can import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from datetime import datetime, timedelta
import joblib
from models.user import User
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("WARNING: OPENAI_API_KEY is missing from .env file")
# ── Load ML model ─────────────────────────────────────────────
artifact = joblib.load("models/tdee_model.pkl")
tdee_model = artifact["model"]
feature_cols = artifact["feature_cols"]

# ── Request schema ────────────────────────────────────────────
class UserRequest(BaseModel):
    message: str
    data: User


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS  (drop-in replacements for hardcoded lists)
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
        kg_per_week = weekly_deficit_kcal / 7700   # ~7700 kcal per kg fat
        return max(round(diff_kg / kg_per_week), 1) if kg_per_week > 0 else 12

    if goal_type == "muscle_gain":
        # Natural muscle gain ≈ 0.5–1 kg/month for intermediate lifters
        current_muscle = measurement.muscle_mass_kg or 0
        target_muscle  = goal.target_muscle_mass_kg or (current_muscle + 3)
        diff_kg        = max(target_muscle - current_muscle, 0)
        kg_per_week    = 0.15   # conservative estimate
        return max(round(diff_kg / kg_per_week), 4)

    return 8   # maintenance / unknown


# open Ai

def generate_dynamic_ai_content(measurement, goal, lifestyle, target_calories, protein_g, carbs_g, fat_g):
    """Sends user context to GPT-4o to generate highly personalized text plans."""
    
    is_vegan = get_safe_attr(lifestyle, "is_vegan", False)
    is_veg = get_safe_attr(lifestyle, "is_vegetarian", False)
    allergies = get_safe_attr(lifestyle, "food_allergies", "None")
    limitations = get_safe_attr(lifestyle, "physical_limitations", "None")
    ex_days = get_safe_attr(lifestyle, "exercise_days_per_week", 3) or 3
    duration = get_safe_attr(lifestyle, "workout_duration_mins", 45) or 45
    
    prompt = f"""
    You are an expert fitness and nutrition AI. Generate a personalized plan based on this exact data:
    
    **User Stats:** {measurement.weight_kg}kg, {measurement.height_cm}cm, Age {measurement.age}
    **Goal:** {goal.goal_type} (Target weight: {get_safe_attr(goal, 'target_weight_kg', 'N/A')}kg)
    **Target Intake:** {round(target_calories)} kcal | {round(protein_g)}g Protein | {round(carbs_g)}g Carbs | {round(fat_g)}g Fat
    **Preferences:** Vegan={is_vegan}, Vegetarian={is_veg}, Allergies={allergies}
    **Constraints:** Workout {ex_days} days/week for {duration} mins. Limitations: {limitations}
    
    Return STRICTLY valid JSON in this exact format, nothing else:
    {{
      "meal_plan": ["meal 1 string", "meal 2 string", "meal 3 string"],
      "workout_plan": ["workout 1 string", "workout 2 string"],
      "habit_tips": ["tip 1", "tip 2", "tip 3"],
      "supplement_suggestions": ["supplement 1", "supplement 2"]
    }}
    """

    try:
        from openai import OpenAI
        client = OpenAI() # Automatically uses OPENAI_API_KEY from .env
        
        response = client.chat.completions.create( # <-- NEW SYNTAX
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        ai_content = json.loads(response.choices[0].message.content)
        return (
            ai_content.get("meal_plan", []),
            ai_content.get("workout_plan", []),
            ai_content.get("habit_tips", []),
            ai_content.get("supplement_suggestions", [])
        )
        
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        # Safe fallback so the app never crashes unpacking None
        return (
            [f"🌅 Breakfast: Balanced macros ({round(target_calories/3)} kcal)", 
             f"☀️  Lunch: High protein ({round(target_calories/3)} kcal)", 
             f"🌙 Dinner: Lean meal ({round(target_calories/3)} kcal)"],
            [f"🏋️ Strength training {ex_days}x/week ({duration} min)", 
             f"🚶 Aim for 8,000 steps daily"],
            ["💧 Drink 2.5-3L of water daily", 
             "😴 Aim for 7-9 hours of sleep", 
             "📓 Track your meals for the first 2 weeks"],
            ["☀️ Vitamin D3 (1000-2000 IU)", 
             "⚠️ Consult a provider before starting supplements"]
        )

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

        # # ── Feature engineering for ML model ─────────────────
        sex_val  = 1 if measurement.sex.lower() == "male" else 0
       
        # 1. Safely fetch data from Lifestyle & Measurement with fallback defaults
        sleep_hrs       = getattr(lifestyle, "sleep_hours", 7.0) or 7.0
        daily_steps     = getattr(lifestyle, "daily_steps", 8000) or getattr(measurement, "daily_steps", 8000) or 8000
        hydration       = getattr(lifestyle, "water_intake_liters", 2.5) or 2.5
        stress          = getattr(lifestyle, "stress_level", 5) or getattr(measurement, "stress_level", 5) or 5
        resting_hr      = getattr(measurement, "resting_heart_rate", 70) or 70
        duration_m      = getattr(lifestyle, "workout_duration_mins", 45) or 45
        avg_hr          = getattr(measurement, "avg_heart_rate", 80) or 80

        # 2. LabelEncoder mapping 
        # (Sklearn's LabelEncoder automatically sorts strings alphabetically during training)
        activity_map   = {"cycling": 0, "gym": 1, "running": 2, "swimming": 3, "walking": 4}
        intensity_map   = {"high": 0, "low": 1, "moderate": 2, "slow": 3}
        fitness_map     = {"advanced": 0, "beginner": 1, "intermediate": 2, "sedentary": 3}

        act_str = getattr(lifestyle, "activity_level", "gym").lower()
        int_str = getattr(lifestyle, "intensity_level", "moderate").lower()
        fit_str = getattr(lifestyle, "fitness_level", "intermediate").lower()

        act_enc = activity_map.get(act_str, 1)       # defaults to 'gym'
        int_enc = intensity_map.get(int_str, 2)       # defaults to 'moderate'
        fit_enc = fitness_map.get(fit_str, 2)         # defaults to 'intermediate'

        # 3. Build the 15-feature array matching train_model.py EXACTLY
        features = [[
            measurement.age,               # 1. age
            measurement.height_cm,         # 2. height
            measurement.weight_kg,         # 3. weight
            measurement.bmi or round(measurement.weight_kg / ((measurement.height_cm / 100) ** 2), 2), # 4. bmi
            sex_val,                       # 5. gender_male
            sleep_hrs,                     # 6. hours_sleep
            daily_steps,                   # 7. daily_steps
            hydration,                     # 8. hydration_level
            stress,                        # 9. stress_level
            resting_hr,                    # 10. resting_heart_rate
            duration_m,                    # 11. duration_m
            avg_hr,                        # 12. avg_heartrate
            act_enc,                       # 13. activity_type_enc
            int_enc,                       # 14. intensity_enc
            fit_enc,                       # 15. fitness_level_enc
    ]]
    
        features_df = pd.DataFrame(features, columns=feature_cols)
        predicted_tdee = float(tdee_model.predict(features_df)[0])
        # ── Target calories ───────────────────────────────────
        goal_type = goal.goal_type.lower()
        if goal_type == "fat_loss":
            target_calories = predicted_tdee - 500
        elif goal_type == "muscle_gain":
            target_calories = predicted_tdee + 300
        else:
            target_calories = predicted_tdee
        target_calories = max(target_calories, 1200)   # safety floor

        # ── Macros ────────────────────────────────────────────
        protein_g = measurement.weight_kg * 2.0
        fat_g     = measurement.weight_kg * 0.8
        carbs_g   = max((target_calories - (protein_g * 4 + fat_g * 9)) / 4, 50)

         # ── Dynamic AI Content ────────────────────────────────
        weeks = calc_weeks_to_goal(measurement, goal, target_calories, predicted_tdee)
        
        # Call the real AI for text generation
        meal_plan, workout, habits, supplements = generate_dynamic_ai_content(
            measurement, goal, lifestyle, target_calories, protein_g, carbs_g, fat_g
        )
       # ── Response ──────────────────────────────────────────
        response = {
            "body_metric_id":        measurement.id,
            "goal_id":               goal.id,
            "tdee_calories":         round(predicted_tdee),
            "target_calories":       round(target_calories),
            "protein_g":             round(protein_g),
            "carbs_g":               round(carbs_g),
            "fat_g":                 round(fat_g),
            "weeks_to_goal":         weeks,
            "engine_used":           "gpt-4o", # Update your Flutter UI to show this!
            "confidence_score":      0.95, # LLMs are highly confident in text formatting
            "meal_plan":             meal_plan,
            "workout_plan":          workout,
            "habit_tips":            habits,
            "supplement_suggestions": supplements,
            "is_active":             True,
            "expires_at":            (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z",
        }

        return jsonify(response), 200

    except ValidationError as e:
        return jsonify({"error": "Validation error", "details": e.errors()}), 422
    except IndexError:
        return jsonify({"error": "User has no body_metrics or goals"}), 400
    except Exception as e:
        print(traceback.format_exc());
        return jsonify({"error": str(e)}), 500


@app.get("/")
def home():
    return jsonify({"status": "Flask ML API running"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)