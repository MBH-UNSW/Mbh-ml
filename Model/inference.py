import numpy as np
import json
import pickle
from tensorflow import keras

FEATURE_COLS = [
    "left_flow_lpm", "right_flow_lpm",
    "left_inlet_pressure_mmHg", "right_inlet_pressure_mmHg",
    "left_upstream_pressure_mmHg", "left_downstream_pressure_mmHg",
    "right_upstream_pressure_mmHg", "right_downstream_pressure_mmHg",
    "pump_rpm", "tah_temperature_c",
]

WINDOW_SIZE = 100

model = keras.models.load_model("artifacts/lstm_autoencoder.keras")
scaler = pickle.load(open("preprocessed/scaler.pkl", "rb"))
threshold = json.load(open("artifacts/threshold.json"))["threshold"]

with open("SDG/anomalous_window.json", "r") as f:
    data = json.load(f)

raw = np.array([
    [row[col] for col in FEATURE_COLS]
    for row in data["measurements"]
])

scaled = scaler.transform(raw)

results = []
i = 0
while i + WINDOW_SIZE <= len(scaled):
    window = scaled[i:i + WINDOW_SIZE].reshape(1, WINDOW_SIZE, len(FEATURE_COLS))
    reconstruction = model.predict(window, verbose=0)
    mae = float(np.mean(np.abs(window - reconstruction)))
    results.append(mae)
    i += WINDOW_SIZE

print(f"Threshold: {threshold:.5f}\n")

for idx, mae in enumerate(results):
    status = "ANOMALY" if mae > threshold else "healthy"
    print(f"Window {idx + 1}: MAE {mae:.5f} -> {status}")