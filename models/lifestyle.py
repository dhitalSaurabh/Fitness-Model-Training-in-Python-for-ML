from pydantic import BaseModel
from datetime import datetime

class Lifestyle(BaseModel):
    id: int
    user_id: int
    diet_type: str
    meals_per_day: int
    daily_calorie_intake: float
    is_vegetarian: bool
    is_vegan: bool
    is_gluten_free: bool
    is_dairy_free: bool
    food_allergies: str
    sleep_hours: float
    preferred_sleep_time: str
    preferred_wake_time: str
    water_intake_liters: float
    activity_level: str
    exercise_days_per_week: int
    preferred_workout_time: str
    workout_duration_mins: int
    medical_conditions: str
    physical_limitations: str
    created_at: datetime
    updated_at: datetime