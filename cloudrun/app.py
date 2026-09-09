import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
from tensorflow import keras
from google.cloud import storage

app = Flask(__name__)

FEATURE_COLS = [
    "left_flow_lpm", "right_flow_lpm",
    "left_inlet_pressure_mmHg", "right_inlet_pressure_mmHg",
    "left_upstream_pressure_mmHg", "left_downstream_pressure_mmHg",
    "right_upstream_pressure_mmHg", "right_downstream_pressure_mmHg",
    "pump_rpm", "tah_temperature_c",
]

WINDOW_SIZE = 100
BUCKET_NAME = "ubh-ml-artifacts-mbh2026"

def download_artifacts():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    bucket.blob("lstm_autoencoder.keras").download_to_filename("/tmp/lstm_autoencoder.keras")
    bucket.blob("threshold.json").download_to_filename("/tmp/threshold.json")
    bucket.blob("scaler.pkl").download_to_filename("/tmp/scaler.pkl")

download_artifacts()
model = keras.models.load_model("/tmp/lstm_autoencoder.keras")
scaler = pickle.load(open("/tmp/scaler.pkl", "rb"))
threshold = json.load(open("/tmp/threshold.json"))["threshold"]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    window_number = data.get("window_number")
    measurements = data["measurements"]

    raw = np.array([
        [row[col] for col in FEATURE_COLS]
        for row in measurements
    ])

    scaled = scaler.transform(raw)

    results = []
    i = 0
    while i + WINDOW_SIZE <= len(scaled):
        window = scaled[i:i + WINDOW_SIZE].reshape(1, WINDOW_SIZE, len(FEATURE_COLS))
        reconstruction = model.predict(window, verbose=0)
        mae = float(np.mean(np.abs(window - reconstruction)))
        results.append({"window": i // WINDOW_SIZE + 1, "mae": mae, "anomaly": mae > threshold})
        i += WINDOW_SIZE

    return jsonify({
        "window_number": window_number,
        "anomaly_detected": any(r["anomaly"] for r in results),
        "threshold": threshold,
        "windows": results
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))