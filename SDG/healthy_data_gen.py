import numpy as np
import pandas as pd

# seed so results are reproducible
SEED = 42
np.random.seed(SEED)

SAMPLE_RATE_HZ = 20 # Honeywell sensors run at 200 SPS but data will be recieved at 20hz
DURATION_HRS = 16
N_SAMPLES = SAMPLE_RATE_HZ * 3600 * DURATION_HRS
OUTPUT_FILE = "SDG/tah_synthetic_healthy.csv"

FLOW_MIN = 4.0 # L/min
FLOW_MAX = 6.0 # L/min

INLET_PRESSURE_MIN = 85.0 # mmHg
INLET_PRESSURE_MAX = 105.0 # mmHg

UPSTREAM_PRESSURE_MIN = 87.0 # mmHg
UPSTREAM_PRESSURE_MAX = 107.0 # mmHg
DOWNSTREAM_PRESSURE_MIN = 80.0 # mmHg
DOWNSTREAM_PRESSURE_MAX = 100.0 # mmHg

PUMP_RPM_MIN = 2700.0
PUMP_RPM_MAX = 3000.0

TEMP_MIN = 36.5 # Celcius
TEMP_MAX = 37.5 # Celcius

FLOW_BASELINE = (FLOW_MIN + FLOW_MAX) / 2
INLET_BASELINE = (INLET_PRESSURE_MIN + INLET_PRESSURE_MAX) / 2
UPSTREAM_BASELINE = (UPSTREAM_PRESSURE_MIN + UPSTREAM_PRESSURE_MAX) / 2
DOWNSTREAM_BASELINE = (DOWNSTREAM_PRESSURE_MIN + DOWNSTREAM_PRESSURE_MAX) / 2
PUMP_RPM_BASELINE = (PUMP_RPM_MIN + PUMP_RPM_MAX) / 2
TEMP_BASELINE = (TEMP_MIN + TEMP_MAX) / 2

FLOW_DRIFT_AMP = 0.4
PRESSURE_DRIFT_AMP = 8.0 # used for inlet/upstream/downstream
PUMP_RPM_DRIFT_AMP = 60.0
TEMP_DRIFT_AMP = 0.3

t = np.arange(N_SAMPLES) / SAMPLE_RATE_HZ

# left/right offset slightly since two physical sensors wont read identically
left_flow_lpm = FLOW_BASELINE + FLOW_DRIFT_AMP * np.sin(2 * np.pi * 0.01 * t) + np.random.normal(0, 0.05, N_SAMPLES)
right_flow_lpm = FLOW_BASELINE + FLOW_DRIFT_AMP * np.sin(2 * np.pi * 0.01 * t + 0.2) + np.random.normal(0, 0.05, N_SAMPLES)

left_inlet_pressure_mmHg = INLET_BASELINE + PRESSURE_DRIFT_AMP * np.sin(2 * np.pi * 0.015 * t) + np.random.normal(0, 0.5, N_SAMPLES)
right_inlet_pressure_mmHg = (INLET_BASELINE - 1.0) + PRESSURE_DRIFT_AMP * np.sin(2 * np.pi * 0.015 * t) + np.random.normal(0, 0.5, N_SAMPLES)

left_upstream_pressure_mmHg = UPSTREAM_BASELINE + PRESSURE_DRIFT_AMP * np.sin(2 * np.pi * 0.015 * t) + np.random.normal(0, 0.5, N_SAMPLES)
right_upstream_pressure_mmHg = (UPSTREAM_BASELINE - 1.0) + PRESSURE_DRIFT_AMP * np.sin(2 * np.pi * 0.015 * t) + np.random.normal(0, 0.5, N_SAMPLES)

# downstream drops off from upstream after passing through the pump
left_downstream_pressure_mmHg = DOWNSTREAM_BASELINE + PRESSURE_DRIFT_AMP * np.sin(2 * np.pi * 0.015 * t) + np.random.normal(0, 0.5, N_SAMPLES)
right_downstream_pressure_mmHg = (DOWNSTREAM_BASELINE - 1.0) + PRESSURE_DRIFT_AMP * np.sin(2 * np.pi * 0.015 * t) + np.random.normal(0, 0.5, N_SAMPLES)

pump_rpm = PUMP_RPM_BASELINE + PUMP_RPM_DRIFT_AMP * np.sin(2 * np.pi * 0.01 * t) + np.random.normal(0, 5, N_SAMPLES)

tah_temperature_c = TEMP_BASELINE + TEMP_DRIFT_AMP * np.sin(2 * np.pi * 0.005 * t) + np.random.normal(0, 0.05, N_SAMPLES)

df = pd.DataFrame({
    "timestamp": t,
    "left_flow_lpm": left_flow_lpm,
    "right_flow_lpm": right_flow_lpm,
    "left_inlet_pressure_mmHg": left_inlet_pressure_mmHg,
    "right_inlet_pressure_mmHg": right_inlet_pressure_mmHg,
    "left_upstream_pressure_mmHg": left_upstream_pressure_mmHg,
    "left_downstream_pressure_mmHg": left_downstream_pressure_mmHg,
    "right_upstream_pressure_mmHg": right_upstream_pressure_mmHg,
    "right_downstream_pressure_mmHg": right_downstream_pressure_mmHg,
    "pump_rpm": pump_rpm,
    "tah_temperature_c": tah_temperature_c,
})

df.to_csv(OUTPUT_FILE, index=False)

print(f"Generated {N_SAMPLES:,} timesteps ({DURATION_HRS} hours at {SAMPLE_RATE_HZ} Hz)")
print(f"Saved to {OUTPUT_FILE}")
print()
print("Summary stats:")
print(df.drop(columns="timestamp").describe().round(3))