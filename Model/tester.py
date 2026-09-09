import numpy as np
import json
import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tensorflow import keras

# change this to switch between the two methods
COMPARE_MODE = "lstm"  # "lstm" or "static"

FEATURE_COLS = [
    "left_flow_lpm", "right_flow_lpm",
    "left_inlet_pressure_mmHg", "right_inlet_pressure_mmHg",
    "left_upstream_pressure_mmHg", "left_downstream_pressure_mmHg",
    "right_upstream_pressure_mmHg", "right_downstream_pressure_mmHg",
    "pump_rpm", "tah_temperature_c",
]

WINDOW_SIZE = 100
Z_THRESHOLD = 3

scaler = pickle.load(open("preprocessed/scaler.pkl", "rb"))
X_train = np.load("preprocessed/X_train.npy")
X_test = np.load("preprocessed/X_test.npy")

df = pd.read_csv("SDG/tah_synthetic_healthy.csv")

# making 500 fake anomalies by messing with one of the flow/pressure columns
anomalous_windows = []
for i in range(500):
    start = np.random.randint(0, len(df) - WINDOW_SIZE)
    window = df[FEATURE_COLS].iloc[start:start+WINDOW_SIZE].copy()
    col = np.random.choice(FEATURE_COLS[:4])
    window[col] = window[col] * np.random.uniform(0.5, 0.9)
    scaled = scaler.transform(window.values)
    anomalous_windows.append(scaled)

anomalous_windows = np.array(anomalous_windows)

if COMPARE_MODE == "lstm":
    model = keras.models.load_model("artifacts/lstm_autoencoder.keras")
    threshold = json.load(open("artifacts/threshold.json"))["threshold"]

    recon = model.predict(X_test, verbose=0)
    healthy_mae = np.mean(np.abs(X_test - recon), axis=(1,2))

    recon = model.predict(anomalous_windows, verbose=0)
    anomalous_mae = np.mean(np.abs(anomalous_windows - recon), axis=(1,2))

else:
    # static threshold, just using mean/std from training data, no model needed here
    feature_mean = X_train.reshape(-1, X_train.shape[2]).mean(axis=0)
    feature_std = X_train.reshape(-1, X_train.shape[2]).std(axis=0)

    healthy_z = np.abs(X_test - feature_mean) / feature_std
    healthy_mae = healthy_z.max(axis=(1,2))

    anom_z = np.abs(anomalous_windows - feature_mean) / feature_std
    anomalous_mae = anom_z.max(axis=(1,2))

    threshold = Z_THRESHOLD


healthy_labels = np.zeros(len(healthy_mae))
anomalous_labels = np.ones(len(anomalous_mae))

all_mae = np.concatenate([healthy_mae, anomalous_mae])
all_labels = np.concatenate([healthy_labels, anomalous_labels])
all_preds = (all_mae > threshold).astype(int)

print("mode:", COMPARE_MODE)
print("threshold:", threshold)
print("precision:", precision_score(all_labels, all_preds))
print("recall:", recall_score(all_labels, all_preds))
print("f1:", f1_score(all_labels, all_preds))
print("auroc:", roc_auc_score(all_labels, all_mae))
print("false positive rate:", (healthy_mae > threshold).mean() * 100, "%")
print("true positive rate:", (anomalous_mae > threshold).mean() * 100, "%")