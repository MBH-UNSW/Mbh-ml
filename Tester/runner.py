import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

FILES = ["Tester/data/control.csv",
         "Tester/data/exagerated_inlet.csv",
         "Tester/data/exaggerated_43_to_51_outlet.csv",
         "Tester/data/subtle_27_to_33_squeeze.csv"]
DOWNSAMPLE = 5
SAMPLE_HZ = 20
WINDOW_SIZE = 100
WINDOW_S = WINDOW_SIZE / SAMPLE_HZ
CAL_S = 10            # must match the trainer
Z_THRESHOLD = 3
PERSIST = 10          # windows in a row over threshold before alarming (0.5 s)

# fault times read off the raw flow plot (visual labels)
FAULTS = {"exagerated_inlet.csv": (15.0, 19.0)}

model = load_model("artifacts_mcl/lstm_ae_mcl.keras")
with open("artifacts_mcl/threshold_mcl.json") as f:
    threshold = json.load(f)["threshold"]


def preprocess(path):
    df = pd.read_csv(path)

    data = df[["MSQ"]].values
    t = df["time_s"].values
    n = len(data) // DOWNSAMPLE * DOWNSAMPLE
    data = data[:n].reshape(-1, DOWNSAMPLE, 1).mean(axis=1)
    t = t[:n].reshape(-1, DOWNSAMPLE).mean(axis=1)

    # calibrate against this run's own start
    cal = data[:CAL_S * SAMPLE_HZ]
    data = (data - cal.mean(axis=0)) / cal.std(axis=0)
    return t, data


def make_windows(x):
    windows = []
    for i in range(len(x) - WINDOW_SIZE + 1):
        windows.append(x[i:i + WINDOW_SIZE])
    return np.array(windows)


def persistent(flags, k):
    # only alarm once k windows in a row are over threshold
    alarm = np.zeros(len(flags), dtype=bool)
    run = 0
    for i in range(len(flags)):
        if flags[i]:
            run += 1
        else:
            run = 0
        if run >= k:
            alarm[i] = True
    return alarm


fig, axes = plt.subplots(len(FILES), 2, figsize=(16, 12))

for row, path in enumerate(FILES):
    name = path.split("/")[-1]
    t, data = preprocess(path)
    t_end = t[WINDOW_SIZE - 1:]   # each window plotted at the time it finishes

    # lstm, remove each window's mean same as in training
    windows = make_windows(data)
    windows = windows - windows.mean(axis=1, keepdims=True)
    recon = model.predict(windows, verbose=0)
    mae = np.mean(np.abs(recon - windows), axis=(1, 2))
    lstm_alarm = persistent(mae > threshold, PERSIST)

    # static, data is already in baseline std units so this is the z-score
    max_z = np.max(np.abs(make_windows(data)), axis=(1, 2))
    static_alarm = persistent(max_z > Z_THRESHOLD, PERSIST)

    print(path)
    print("  windows:", len(mae))
    print("  LSTM alarm:  ", round(np.mean(lstm_alarm) * 100, 1), "%")
    print("  static alarm:", round(np.mean(static_alarm) * 100, 1), "%")

    if name in FAULTS:
        f_start, f_end = FAULTS[name]
        # window is faulty if it overlaps the fault at all
        truth = (t_end > f_start) & (t_end - WINDOW_S < f_end)

        for label, alarm in [("LSTM", lstm_alarm), ("static", static_alarm)]:
            tp = np.sum(alarm & truth)
            fp = np.sum(alarm & ~truth)
            fn = np.sum(~alarm & truth)
            tn = np.sum(~alarm & ~truth)
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            fpr = fp / (fp + tn) if fp + tn > 0 else 0
            print(f"  {label}: precision {precision:.2f}  recall {recall:.2f}  FPR {fpr:.2f}")

    axes[row, 0].plot(t_end, mae)
    axes[row, 0].axhline(threshold, color="r", linestyle="--")
    axes[row, 0].fill_between(t_end, 0, mae.max(), where=lstm_alarm,
                              color="red", alpha=0.15)
    axes[row, 0].set_title(name + " - LSTM MAE")
    axes[row, 0].set_ylabel("MAE")

    axes[row, 1].plot(t_end, max_z, color="tab:orange")
    axes[row, 1].axhline(Z_THRESHOLD, color="r", linestyle="--")
    axes[row, 1].fill_between(t_end, 0, max_z.max(), where=static_alarm,
                              color="red", alpha=0.15)
    axes[row, 1].set_title(name + " - static max z-score")
    axes[row, 1].set_ylabel("max z")

    for ax in axes[row]:
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("mcl_test.png", dpi=120)
plt.show()