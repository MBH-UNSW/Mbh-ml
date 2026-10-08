import numpy as np
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

FEATURE_COLS = [
    "left_flow_lpm", "right_flow_lpm",
    "left_inlet_pressure_mmHg", "right_inlet_pressure_mmHg",
    "left_upstream_pressure_mmHg", "left_downstream_pressure_mmHg",
    "right_upstream_pressure_mmHg", "right_downstream_pressure_mmHg",
    "pump_rpm", "tah_temperature_c",
]

WINDOW_SIZE = 100
STRIDE = 10
HZ = 20
Z_THRESHOLD = 3

FAULT_COL = "left_flow_lpm"
MULTIPLIER = 0.85
FAULT_START = 1000
FAULT_END = 1600

model = keras.models.load_model("artifacts/lstm_autoencoder.keras")
scaler = pickle.load(open("preprocessed/scaler.pkl", "rb"))
threshold = json.load(open("artifacts/threshold.json"))["threshold"]
X_train = np.load("preprocessed/X_train.npy")

df = pd.read_csv("SDG/tah_synthetic_healthy.csv")

# conventional alarm normal range for the plotted sensor (just for the grey band)
train_part = df[FEATURE_COLS].iloc[:int(len(df) * 0.7)]
mean = train_part[FAULT_COL].mean()
std = train_part[FAULT_COL].std()
low, high = mean - 3 * std, mean + 3 * std

# per sensor mean/std for the conventional alarm, same as the eval script
feature_mean = X_train.reshape(-1, X_train.shape[2]).mean(axis=0)
feature_std = X_train.reshape(-1, X_train.shape[2]).std(axis=0)

# 2 minute stretch from the test portion with the fault injected
start = int(len(df) * 0.85)
segment = df[FEATURE_COLS].iloc[start:start + 2400].copy().reset_index(drop=True)
segment.loc[FAULT_START:FAULT_END, FAULT_COL] *= MULTIPLIER

scaled = scaler.transform(segment)

windows = []
times = []
i = 0
while i + WINDOW_SIZE <= len(scaled):
    windows.append(scaled[i:i + WINDOW_SIZE])
    times.append((i + WINDOW_SIZE) / HZ)
    i += STRIDE
windows = np.array(windows)

# AI model score
recon = model.predict(windows, verbose=0)
ai_scores = np.mean(np.abs(windows - recon), axis=(1, 2))

# conventional alarm score (worst sensor in each window)
z = np.abs(windows - feature_mean) / feature_std
conv_scores = z.max(axis=(1, 2))

t = np.arange(len(segment)) / HZ
fault_s, fault_e = FAULT_START / HZ, FAULT_END / HZ

plt.style.use("dark_background")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# top: raw signal
ax1.plot(t, segment[FAULT_COL], color="#4da6ff", linewidth=1)
ax1.axhspan(low, high, color="grey", alpha=0.2, label="Normal range")
ax1.axvspan(fault_s, fault_e, color="#e63946", alpha=0.15, label="Fault")
ax1.set_ylabel("Left flow (L/min)")
ax1.legend(loc="upper right", fontsize=8)

# middle: conventional alarm
ax2.plot(times, conv_scores, color="#f4a261", linewidth=1.5)
ax2.axhline(Z_THRESHOLD, color="#e63946", linestyle="--", label="Alarm threshold")
ax2.axvspan(fault_s, fault_e, color="#e63946", alpha=0.15)
ax2.set_ylabel("Conventional\nalarm score")
ax2.legend(loc="upper right", fontsize=8)

# bottom: AI model
ax3.plot(times, ai_scores, color="white", linewidth=1.5)
ax3.axhline(threshold, color="#e63946", linestyle="--", label="Alarm threshold")
ax3.axvspan(fault_s, fault_e, color="#e63946", alpha=0.15)
ax3.set_ylabel("AI anomaly\nscore")
ax3.set_xlabel("Time (s)")
ax3.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("fault_timeline.png", dpi=300, transparent=True)
plt.show()