from pydantic import BaseModel
from datetime import datetime

class Goal(BaseModel):
    id: int
    user_id: int
    goal_type: str
    target_weight_kg: float
    target_body_fat_percent: float
    target_muscle_mass_kg: float
    intensity_level: str
    target_date: datetime
    is_active: bool
    created_at: datetime
    updated_at: datetime