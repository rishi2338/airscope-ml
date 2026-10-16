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

FEATURES = ["PM2.5", "PM10", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]


def train_demo_model(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(19)
    centers = np.array([
        [18, 42, 12, 20, 10, 0.45, 7, 22],
        [65, 115, 32, 48, 20, 1.15, 15, 35],
        [145, 245, 58, 82, 34, 2.3, 30, 48],
    ], dtype=float)
    std = np.array([9, 18, 7, 10, 5, .18, 4, 8])
    data = np.vstack([rng.normal(center, std, size=(size, len(center)))
                      for center, size in zip(centers, [450, 380, 170])])
    df = pd.DataFrame(np.clip(data, [0, 0, 0, 0, 0, .05, 0, 0], None), columns=FEATURES)
    scaler = StandardScaler().fit(df)
    model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaler.transform(df))
    model._airscope_demo = True
    for filename, obj in [
        ("air_pollution_model.pkl", model),
        ("air_pollution_scaler.pkl", scaler),
        ("feature_columns.pkl", FEATURES),
    ]:
        with (output_dir / filename).open("wb") as handle:
            pickle.dump(obj, handle)
    return output_dir


if __name__ == "__main__":
    target = train_demo_model(Path(__file__).resolve().parent / "models")
    print(f"Demo artifacts saved in: {target}")
    print("WARNING: Synthetic data for interface testing only, not scientific evaluation.")
