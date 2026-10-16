"""
Optional helper: train a demo K-Means model using synthetic data.
This is only for testing the app interface. Replace with real training
on the India Air Quality dataset before making scientific claims.
Run: python train_demo_model.py
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)
FEATURES = ["PM2.5", "PM10", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]

rng = np.random.default_rng(19)
centers = np.array([
    [18, 42, 12, 20, 10, 0.45, 7, 22],
    [65, 115, 32, 48, 20, 1.15, 15, 35],
    [145, 245, 58, 82, 34, 2.3, 30, 48],
], dtype=float)
sizes = [450, 380, 170]
blocks = []
for center, size in zip(centers, sizes):
    std = np.array([9, 18, 7, 10, 5, .18, 4, 8])
    blocks.append(rng.normal(center, std, size=size))
data = np.vstack(blocks)
data = np.clip(data, [0, 0, 0, 0, 0, .05, 0, 0], None)
df = pd.DataFrame(data, columns=FEATURES)

scaler = StandardScaler()
scaled = scaler.fit_transform(df[FEATURES])
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(scaled)

for filename, obj in [
    ("air_pollution_model.pkl", model),
    ("air_pollution_scaler.pkl", scaler),
    ("feature_columns.pkl", FEATURES),
]:
    with open(MODEL_DIR / filename, "wb") as f:
        pickle.dump(obj, f)

print(f"Demo artifacts saved in: {MODEL_DIR}")
print("WARNING: This model uses synthetic data and is for interface testing only.")
