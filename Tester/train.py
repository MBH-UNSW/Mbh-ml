import numpy as np
import pandas as pd
import json
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.utils.set_random_seed(42)   # seeds numpy, python and tensorflow

DATA_PATH = "Tester/data/control.csv"
OUT_DIR = "artifacts_mcl"
DOWNSAMPLE = 5        # 100 Hz -> 20 Hz
SAMPLE_HZ = 20
WINDOW_SIZE = 100     # 5 s at 20 Hz, one pump cycle at 12 BPM
STRIDE = 1
TRAIN_RATIO = 0.8
EPOCHS = 100
BATCH_SIZE = 32

CAL_S = 10            # each run is normalised to its own first 10 s

# augmentation, based on the variation seen between real runs
N_COPIES = 20
SCALE_RANGE = 0.15    # +-15% beat amplitude
WARP_RANGE = 0.05     # +-5% beat period
DRIFT_RANGE = 0.2     # max slow drift across a copy (in baseline std units)
NOISE_STD = 0.02      # small noise (in baseline std units)

os.makedirs(OUT_DIR, exist_ok=True)


def preprocess(path):
    df = pd.read_csv(path)

    # block average then downsample (stops aliasing)
    data = df[["MSQ"]].values
    n = len(data) // DOWNSAMPLE * DOWNSAMPLE
    data = data[:n].reshape(-1, DOWNSAMPLE, 1).mean(axis=1)

    # calibrate against this run's own start
    cal = data[:CAL_S * SAMPLE_HZ]
    data = (data - cal.mean(axis=0)) / cal.std(axis=0)
    return data


def make_windows(x):
    windows = []
    i = 0
    while i + WINDOW_SIZE <= len(x):
        windows.append(x[i:i + WINDOW_SIZE])
        i += STRIDE
    windows = np.array(windows)
    # remove each window's own mean so the model learns the shape, not the level
    windows = windows - windows.mean(axis=1, keepdims=True)
    return windows


def time_warp(x, factor):
    # stretch or squash the signal so beats get slightly slower or faster
    n_old = len(x)
    n_new = int(n_old * factor)
    old_t = np.arange(n_old)
    new_t = np.linspace(0, n_old - 1, n_new)
    cols = []
    for j in range(x.shape[1]):
        cols.append(np.interp(new_t, old_t, x[:, j]))
    return np.column_stack(cols)


def augment(x):
    copy = x.copy()

    # scale the beat amplitude around the mean
    mean = copy.mean(axis=0)
    copy = mean + (copy - mean) * np.random.uniform(1 - SCALE_RANGE, 1 + SCALE_RANGE)

    # change beat timing
    copy = time_warp(copy, np.random.uniform(1 - WARP_RANGE, 1 + WARP_RANGE))

    # slow linear drift across the whole copy
    drift = np.random.uniform(-DRIFT_RANGE, DRIFT_RANGE)
    copy = copy + np.linspace(0, drift, len(copy)).reshape(-1, 1)

    # small noise
    copy = copy + np.random.normal(0, NOISE_STD, copy.shape)
    return copy


data = preprocess(DATA_PATH)
print("samples after downsampling:", len(data))

# split before windowing so val is a later chunk of time
split = int(len(data) * TRAIN_RATIO)
train_data = data[:split]
val_data = data[split - WINDOW_SIZE // 2:]

# augment the training chunk only, val stays as real data
augmented = [train_data]
for k in range(N_COPIES):
    augmented.append(augment(train_data))

# window each copy separately so no window crosses the join between copies
X_train = np.concatenate([make_windows(a) for a in augmented])
X_val = make_windows(val_data)
print("train:", X_train.shape, "val:", X_val.shape)

n_features = X_train.shape[2]

model = Sequential([
    Input(shape=(WINDOW_SIZE, n_features)),
    LSTM(32, return_sequences=True),
    LSTM(16),
    RepeatVector(WINDOW_SIZE),
    LSTM(16, return_sequences=True),
    LSTM(32, return_sequences=True),
    TimeDistributed(Dense(n_features))
])
model.compile(optimizer="adam", loss="mae")

model.fit(X_train, X_train,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_data=(X_val, X_val),
          callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
          shuffle=True)

# threshold from real data the model didnt train on
val_mae = np.mean(np.abs(model.predict(X_val) - X_val), axis=(1, 2))
threshold = float(val_mae.mean() + 3 * val_mae.std())
print("threshold:", threshold)

model.save(os.path.join(OUT_DIR, "lstm_ae_mcl.keras"))
with open(os.path.join(OUT_DIR, "threshold_mcl.json"), "w") as f:
    json.dump({"threshold": threshold,
               "mae_mean": float(val_mae.mean()),
               "mae_std": float(val_mae.std()),
               "cal_s": CAL_S}, f)