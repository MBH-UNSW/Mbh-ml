import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

RAW_DATA_PATH = "SDG/tah_synthetic_healthy.csv"
OUTPUT_DIR = "preprocessed"

WINDOW_SIZE = 100  # 20Hz * 5 seconds
STRIDE = 10

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

FEATURE_COLS = [
    "left_flow_lpm", "right_flow_lpm",
    "left_inlet_pressure_mmHg", "right_inlet_pressure_mmHg",
    "left_upstream_pressure_mmHg", "left_downstream_pressure_mmHg",
    "right_upstream_pressure_mmHg", "right_downstream_pressure_mmHg",
    "pump_rpm", "tah_temperature_c",
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

df = pd.read_csv(RAW_DATA_PATH)
df = df[FEATURE_COLS]

# normalise data to fit within same range (0-1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# make windows (5 sec)
windows = []
i = 0
while i + WINDOW_SIZE <= len(scaled):
    window = scaled[i:i + WINDOW_SIZE]
    windows.append(window)
    i += STRIDE

windows = np.array(windows)

# split into train/val/test in time order
n = len(windows)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

X_train = windows[0:train_end]
X_val = windows[train_end:val_end]
X_test = windows[val_end:n]

np.save(OUTPUT_DIR + "/X_train.npy", X_train)
np.save(OUTPUT_DIR + "/X_val.npy", X_val)
np.save(OUTPUT_DIR + "/X_test.npy", X_test)

with open(OUTPUT_DIR + "/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)