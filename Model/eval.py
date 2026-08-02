import numpy as np
import json
import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tensorflow import keras

model = keras.models.load_model("artifacts/lstm_autoencoder.keras")
scaler = pickle.load(open("preprocessed/scaler.pkl", "rb"))
threshold = json.load(open("artifacts/threshold.json"))["threshold"]

FEATURE_COLS = [
    "left_flow_lpm", "right_flow_lpm",
    "left_inlet_pressure_mmHg", "right_inlet_pressure_mmHg",
    "left_upstream_pressure_mmHg", "left_downstream_pressure_mmHg",
    "right_upstream_pressure_mmHg", "right_downstream_pressure_mmHg",
    "pump_rpm", "tah_temperature_c",
]

WINDOW_SIZE = 100

# healthy test windows
X_test = np.load("preprocessed/X_test.npy")
recon = model.predict(X_test, verbose=0)
healthy_mae = np.mean(np.abs(X_test - recon), axis=(1, 2))
healthy_labels = np.zeros(len(healthy_mae))

# build anomalous windows by injecting faults into random windows
df = pd.read_csv("SDG/tah_synthetic_healthy.csv")

anomalous_windows = []
for _ in range(500):
    start = np.random.randint(0, len(df) - WINDOW_SIZE)
    window = df[FEATURE_COLS].iloc[start:start + WINDOW_SIZE].copy()
    col = np.random.choice(FEATURE_COLS[:4])
    window[col] = window[col] * np.random.uniform(0.5, 0.9)
    scaled = scaler.transform(window.values)
    anomalous_windows.append(scaled)

anomalous_windows = np.array(anomalous_windows)
recon = model.predict(anomalous_windows, verbose=0)
anomalous_mae = np.mean(np.abs(anomalous_windows - recon), axis=(1, 2))
anomalous_labels = np.ones(len(anomalous_mae))

all_mae = np.concatenate([healthy_mae, anomalous_mae])
all_labels = np.concatenate([healthy_labels, anomalous_labels])
all_preds = (all_mae > threshold).astype(int)

print(f"Threshold: {threshold:.5f}")
print(f"Precision: {precision_score(all_labels, all_preds):.3f}")
print(f"Recall: {recall_score(all_labels, all_preds):.3f}")
print(f"F1: {f1_score(all_labels, all_preds):.3f}")
print(f"AUROC: {roc_auc_score(all_labels, all_mae):.3f}")
print(f"False positive rate: {(healthy_mae > threshold).mean() * 100:.2f}%")
print(f"True positive rate: {(anomalous_mae > threshold).mean() * 100:.2f}%")