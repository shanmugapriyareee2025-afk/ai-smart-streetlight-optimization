import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1200
hour = rng.integers(0, 24, n)
ambient = 70 + 450 * np.sin((hour - 6) * np.pi / 12) + rng.normal(0, 35, n)
ambient = np.clip(ambient, 0, 600)
motion = rng.integers(0, 2, n)
mode = ["OFF" if light >= 250 else ("HIGH" if move == 1 else "DIM")
        for light, move in zip(ambient, motion)]
df = pd.DataFrame({
    "timestamp": pd.date_range("2026-01-01", periods=n, freq="5min"),
    "hour": hour,
    "ambient_light_lux": ambient.round(1),
    "motion_detected": motion,
    "lighting_mode": mode
})
df.to_csv("streetlight_sensor_data.csv", index=False)
print("Created streetlight_sensor_data.csv with", len(df), "rows")
