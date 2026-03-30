mkdir tensorflow_project
cd tensorflow_project 
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install tensorflow
python -c "import tensorflow as tf; print(tf.__version__)"
pip freeze > requirements.txt


python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
Change the port in pip
uvicorn src.app:app --reload --port 5000

Health Recommendation API
Setup
bashpip install flask pandas numpy scikit-learn joblib
1 — Train the model first
bash# Put your CSV at data/fitness_data.csv (or edit CSV_PATH in train_model.py)
python train_model.py
This creates models/tdee_model.pkl.
If the model file is missing, app.py automatically falls back to the Mifflin-St Jeor formula.

2 — Start the Flask server
bashpython app.py
# Server running at http://localhost:5000

3 — POST /api/v1/recommendations
Request body
json{
  "measurement": {
    "id": 1,
    "user_id": 42,
    "height_cm": 175,
    "weight_kg": 80,
    "age": 30,
    "sex": "male",
    "bmi": 26.1,
    "body_fat_percent": 22.0,
    "muscle_mass_kg": 35.0,
    "fitness_level": "intermediate",
    "daily_steps": 7000,
    "stress_level": 6
  },
  "goal": {
    "id": 5,
    "user_id": 42,
    "goal_type": "weight_loss",
    "target_weight_kg": 72,
    "target_body_fat_percent": 15.0,
    "target_muscle_mass_kg": 36.0,
    "intensity_level": "moderate",
    "is_active": true
  },
  "lifestyle": {
    "id": 3,
    "user_id": 42,
    "diet_type": "balanced",
    "meals_per_day": 3,
    "daily_calorie_intake": 2200,
    "is_vegetarian": false,
    "is_vegan": false,
    "is_gluten_free": false,
    "is_dairy_free": false,
    "food_allergies": "",
    "sleep_hours": 6.5,
    "water_intake_liters": 2.0,
    "activity_level": "moderate",
    "exercise_days_per_week": 4,
    "preferred_workout_time": "morning",
    "workout_duration_mins": 50,
    "medical_conditions": "",
    "physical_limitations": ""
  }
}
Response
json{
  "body_metric_id": 1,
  "goal_id": 5,
  "tdee_calories": 2743,
  "target_calories": 2243,
  "protein_g": 160,
  "carbs_g": 224,
  "fat_g": 70,
  "weeks_to_goal": 10,
  "engine_used": "ml_model",
  "confidence_score": 0.85,
  "meal_plan": [
    "Breakfast (~748 kcal): oats/brown rice/fruits, chicken/fish/eggs — 40g protein",
    "Lunch (~748 kcal): salad + chicken/fish/eggs, avocado/nuts/olive oil — 56g protein",
    "Dinner (~748 kcal): chicken/fish/eggs with steamed veggies & oats/brown rice/fruits — 56g protein"
  ],
  "workout_plan": [
    "Cardio 3x/week — HIIT or brisk walk 50 min",
    "Strength training 3x/week — full body circuits",
    "Active rest: yoga or stretching on off days",
    "Aim for 8,000–10,000 daily steps"
  ],
  "habit_tips": [
    "Increase water intake to 2.5–3 L/day (current: 2.0 L)",
    "Aim for 7–9 hours of sleep (current: 6.5 hrs)",
    "Increase daily steps to 8,000–10,000 (current: 7000)",
    "Track meals for the first 2 weeks to build awareness",
    "Weigh yourself weekly (same time/day) rather than daily"
  ],
  "supplement_suggestions": [
    "Whey/plant protein powder to hit daily protein target",
    "Vitamin D3 if limited sun exposure",
    "Always consult a healthcare provider before starting supplements"
  ],
  "is_active": true,
  "expires_at": "2025-08-15T10:23:00.000000Z"
}

File structure
health_recommendation_api/
├── app.py            ← Flask API (main file)
├── train_model.py    ← Train ML model from your CSV
├── data/
│   └── fitness_data.csv   ← your CSV goes here
└── models/
    └── tdee_model.pkl     ← auto-created after training