from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Measurement(BaseModel):
    id: int
    user_id: int
    height_cm: float
    weight_kg: float
    age: int
    sex: str
    bmi: Optional[float]
    body_fat_percent: Optional[float]
    muscle_mass_kg: Optional[float]
    visceral_fat_level: Optional[float]
    bone_mass_kg: Optional[float]
    water_percent: Optional[float]
    measurement_method: str
    measured_at: datetime
    created_at: datetime
    updated_at: datetime