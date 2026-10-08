import pandas as pd
import matplotlib.pyplot as plt

FILES = ["Tester/data/control.csv",
         "Tester/data/exagerated_inlet.csv",
         "Tester/data/exaggerated_43_to_51_outlet.csv",
         "Tester/data/subtle_27_to_33_squeeze.csv"]

fig, axes = plt.subplots(len(FILES), 1, figsize=(14, 12))

for ax, path in zip(axes, FILES):
    df = pd.read_csv(path)

    ax.plot(df["time_s"], df["SQ"], linewidth=0.5, alpha=0.4, label="SQ (raw)")
    ax.plot(df["time_s"], df["MSQ"], linewidth=1.5, label="MSQ (mean)")

    ax.set_title(path.split("/")[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("flow")
    ax.set_xlim(0, df["time_s"].iloc[-1])
    ax.grid(alpha=0.3)

axes[0].legend()
plt.tight_layout()
plt.savefig("mcl_flow.png", dpi=120)
plt.show()