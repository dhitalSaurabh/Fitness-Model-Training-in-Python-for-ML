import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load historical user data CSV/JSON
df = pd.read_csv("fitness_data.csv")  # make this file

X = df[["height_cm","weight_kg","age","sex","bmi","body_fat","muscle_mass","goal_type"]]
y = df["tdee"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "tdee_model.pkl")
print("Model trained and saved.")