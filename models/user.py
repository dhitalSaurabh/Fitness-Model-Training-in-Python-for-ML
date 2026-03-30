from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from models.goal import Goal
from models.lifestyle import Lifestyle
from models.measurement import Measurement
class User(BaseModel):
    id: int
    name: str
    email: str
    email_verified_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    avatar: Optional[str]

    body_metrics: List[Measurement]
    goals: List[Goal]
    lifestyle: Lifestyle