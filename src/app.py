from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from datetime import datetime, timedelta
import joblib
from models.user import User

app = Flask(__name__)

# ── Load ML model ─────────────────────────────────────────────
tdee_model = joblib.load("src/ml/tdee_model.pkl")


# ── Request schema ────────────────────────────────────────────
class UserRequest(BaseModel):
    message: str
    data: User


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS  (drop-in replacements for hardcoded lists)
# ══════════════════════════════════════════════════════════════

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


def generate_meal_plan(lifestyle, target_calories: float,
                       protein_g: float, carbs_g: float, fat_g: float) -> list[str]:
    """Build a day's meal plan personalised to diet preferences."""
    meals     = getattr(lifestyle, "meals_per_day", 3) or 3
    is_vegan  = getattr(lifestyle, "is_vegan", False)
    is_veg    = getattr(lifestyle, "is_vegetarian", False)
    allergies = (getattr(lifestyle, "food_allergies", "") or "").lower()

    # Protein source
    if is_vegan:
        p_src = "tofu / tempeh / lentils"
    elif is_veg:
        p_src = "paneer / eggs / Greek yogurt"
    else:
        p_src = "chicken breast / fish / eggs"

    # Carb source — avoid gluten if needed
    c_src = "rice cakes / sweet potato" if "gluten" in allergies else "oats / brown rice / whole-wheat bread"

    # Fat source — avoid dairy if needed
    f_src = "avocado / nuts / olive oil"

    cal_each   = round(target_calories / meals)
    p_each     = round(protein_g / meals)

    templates = {
        3: [
            f"🌅 Breakfast  (~{cal_each} kcal | {p_each}g protein): {c_src} + {p_src} + fruit",
            f"☀️  Lunch     (~{cal_each} kcal | {p_each}g protein): {p_src} + salad + {f_src}",
            f"🌙 Dinner    (~{cal_each} kcal | {p_each}g protein): {p_src} + steamed veggies + {c_src}",
        ],
        4: [
            f"🌅 Breakfast  (~{cal_each} kcal | {p_each}g protein): {c_src} + {p_src}",
            f"🍎 Snack      (~{cal_each} kcal): mixed nuts / Greek yogurt",
            f"☀️  Lunch     (~{cal_each} kcal | {p_each}g protein): {p_src} + salad + {f_src}",
            f"🌙 Dinner    (~{cal_each} kcal | {p_each}g protein): {p_src} + veggies + {c_src}",
        ],
        5: [
            f"🌅 Breakfast  (~{cal_each} kcal): {c_src} + {p_src}",
            f"🍎 Mid-morning (~{cal_each} kcal): fruit + handful of nuts",
            f"☀️  Lunch     (~{cal_each} kcal): {p_src} + salad + {f_src}",
            f"🥤 Afternoon  (~{cal_each} kcal): protein shake + banana",
            f"🌙 Dinner    (~{cal_each} kcal): {p_src} + veggies + {c_src}",
        ],
    }
    plan = templates.get(meals, templates[3])

    # Append total macro summary
    plan.append(
        f"📊 Daily totals: {round(target_calories)} kcal | "
        f"{round(protein_g)}g protein | {round(carbs_g)}g carbs | {round(fat_g)}g fat"
    )
    return plan


def generate_workout_plan(measurement, goal, lifestyle) -> list[str]:
    """Return a weekly workout plan based on goal, fitness level and available days."""
    goal_type   = goal.goal_type.lower()
    ex_days     = getattr(lifestyle, "exercise_days_per_week", 3) or 3
    duration    = getattr(lifestyle, "workout_duration_mins", 45) or 45
    limitations = (getattr(lifestyle, "physical_limitations", "") or "").strip()

    plans = {
        "fat_loss": [
            f"🏃 Cardio  {max(ex_days - 1, 2)}x/week — HIIT or brisk walk/cycle ({duration} min)",
            f"🏋️ Strength {min(ex_days, 3)}x/week — full-body circuits, 3×12 reps",
            "🧘 Active rest 1x/week — yoga or light stretching",
            "🚶 Target 8,000–10,000 steps every day",
        ],
        "muscle_gain": [
            f"🏋️ Strength {ex_days}x/week — push / pull / legs split",
            f"   Session length: {duration} min | rest 60–90 s between sets",
            "🚴 Low-intensity cardio 2x/week (20 min) for cardiovascular health",
            "📅 Deload week every 4th week to prevent overtraining",
        ],
        "maintenance": [
            f"⚖️  Mixed cardio + strength {ex_days}x/week",
            f"   Moderate intensity | {duration} min sessions",
            "🧘 Flexibility / mobility work 2x/week",
        ],
    }
    plan = plans.get(goal_type, plans["maintenance"])

    if limitations:
        plan.append(f"⚠️  Modify exercises to accommodate: {limitations}")

    return plan


def generate_habit_tips(measurement, lifestyle) -> list[str]:
    """Personalised habit tips based on current metrics."""
    tips   = []
    water  = getattr(lifestyle, "water_intake_liters", 2.0) or 2.0
    sleep  = getattr(lifestyle, "sleep_hours", 7.0) or 7.0
    stress = getattr(measurement, "stress_level", 5) or 5       # from CSV / measurement extras
    steps  = getattr(measurement, "daily_steps", 6000) or 6000

    tips.append(
        f"💧 Hydration: {'increase to 2.5–3 L/day (currently ' + str(water) + ' L)' if water < 2.5 else 'great — maintain 2.5–3 L/day'}"
    )
    tips.append(
        f"😴 Sleep: {'aim for 7–9 hrs (currently ' + str(sleep) + ' hrs)' if sleep < 7 else 'good — keep consistent sleep/wake times'}"
    )
    if steps < 8000:
        tips.append(f"🚶 Steps: aim for 8,000–10,000/day (currently ~{steps})")
    if int(stress) >= 7:
        tips.append("🧠 High stress detected — try 10 min mindfulness or box-breathing daily")

    tips.append("📓 Track meals for the first 2 weeks to build nutritional awareness")
    tips.append("⚖️  Weigh yourself weekly (same time, same day) — not daily")
    return tips


def generate_supplements(measurement, goal, lifestyle) -> list[str]:
    """Evidence-based supplement suggestions."""
    goal_type = goal.goal_type.lower()
    is_vegan  = getattr(lifestyle, "is_vegan", False)
    protein_g = measurement.weight_kg * 2
    sups      = []

    if protein_g > 150:
        src = "plant-based protein" if is_vegan else "whey protein"
        sups.append(f"🥤 {src.capitalize()} powder to help hit {round(protein_g)}g daily protein target")

    if is_vegan:
        sups.append("💊 Vitamin B12 — essential for vegans (2.4 µg/day)")
        sups.append("🐟 Algae-based Omega-3 (EPA + DHA) — vegan alternative to fish oil")

    if goal_type == "muscle_gain":
        sups.append("💪 Creatine monohydrate 3–5 g/day — best-evidenced supplement for strength")

    sups.append("☀️  Vitamin D3 (1,000–2,000 IU/day) if you have limited sun exposure")
    sups.append("⚠️  Always consult a healthcare provider before starting supplements")
    return sups


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

        # ── Feature engineering for ML model ─────────────────
        sex_val  = 1 if measurement.sex.lower() == "male" else 0
        goal_val = 0 if goal.goal_type == "fat_loss" else 1
        features = [[
            measurement.height_cm,
            measurement.weight_kg,
            measurement.age,
            sex_val,
            measurement.bmi or round(measurement.weight_kg / ((measurement.height_cm / 100) ** 2), 2),
            measurement.body_fat_percent or 0,
            measurement.muscle_mass_kg or 0,
            goal_val,
        ]]

        predicted_tdee = float(tdee_model.predict(features)[0])

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

        # ── Dynamic content ───────────────────────────────────
        weeks       = calc_weeks_to_goal(measurement, goal, target_calories, predicted_tdee)
        meal_plan   = generate_meal_plan(lifestyle, target_calories, protein_g, carbs_g, fat_g)
        workout     = generate_workout_plan(measurement, goal, lifestyle)
        habits      = generate_habit_tips(measurement, lifestyle)
        supplements = generate_supplements(measurement, goal, lifestyle)

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
            "engine_used":           "ml_model",
            "confidence_score":      0.85,
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
        return jsonify({"error": str(e)}), 500


@app.get("/")
def home():
    return jsonify({"status": "Flask ML API running"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)