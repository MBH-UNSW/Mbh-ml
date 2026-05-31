"""
Synthetic TAH Telemetry Data Generator
Ansh Rawat - UNSW Bionic Hearts
 
I'm generating fake but realistic healthy TAH sensor data to use for training
the LSTM-Autoencoder. The plan is to replace this with real data later once
the electrical team has the sensors running, but the format will stay the same
so swapping it out should be straightforward.
 
The TAH has 6 sensors:
- 2 inlet pressure sensors  (mmHg) - measures pressure coming into the pump
- 2 outlet pressure sensors (mmHg) - measures pressure leaving the pump
- 1 flow sensor             (L/min) - measures how much blood is being pumped
- 1 temperature sensor      (Celcius)   - measures blood temperature
 
I'm generating 8 hours of data at 10 Hz (10 readings per second).
That gives us 288,000 timesteps which should be more than enough to train on.
 
Each signal is modelled as a combination of:
1. A baseline value (what the sensor sits at normally)
2. A slow drift (simulates gradual physiological changes over hours)
3. A pump cycle oscillation (the pump runs at ~1.1 Hz so sensors oscillate slightly)
4. Random gaussian noise (simulates real sensor noise)
 
The specific values (baseline, amplitude, frequency) are based on typical
operating ranges for a rotary blood pump TAH which I got from the electrical team.
"""

# --- Imports ---

import numpy as np
import pandas as pd

# setting a seed so the generated data is the same every time I run this. The purpose is for testing reliability of results
SEED = 42
np.random.seed(SEED)

# --- Parameters ---

SAMPLE_RATE_HZ = 10         # 10 readings/sec
DURATION_HRS = 8            # how long to simulate
N_SAMPLES = SAMPLE_RATE_HZ * 3600 * DURATION_HRS # 288000 samples
OUTPUT_FILE = "SDG/tah_synthetic_healthy.csv"

# Sensor Operating ranges (confirm units with elec team later)
INLET_PRESSURE_MIN = 5.0        # mmHg
INLET_PRESSURE_MAX = 12.0       # mmHg
OUTLET_PRESSURE_MIN = 80.0      # mmHg
OUTLET_PRESSURE_MAX = 120.0     # mmHg
FLOW_RATE_MIN = 4.0             # L/min
FLOW_RATE_MAX = 6.0             # L/min
TEMP_MIN = 36.5                 # Celcius
TEMP_MAX = 37.5                 # Celcius

# Midpoints, used as a baseline for signal generation
INLET_BASELINE = (INLET_PRESSURE_MIN + INLET_PRESSURE_MAX) / 2
OUTLET_BASELINE = (OUTLET_PRESSURE_MIN + OUTLET_PRESSURE_MAX) / 2
FLOW_BASELINE = (FLOW_RATE_MIN + FLOW_RATE_MAX) / 2
TEMP_BASELINE = (TEMP_MIN + TEMP_MAX) / 2


# Drift amplitudes - how much the signal wanders around the baseline
# Keeping these well within the healthy range so generated data stays realistic
INLET_DRIFT_AMP = 1.5
OUTLET_DRIFT_AMP = 10.0
FLOW_DRIFT_AMP = 0.4
TEMP_DRIFT_AMP = 0.3

# time vector in seconds, e.g: [0.0, 0.1, 0.2, ... , 28799.9] (matches w/ sample rate)
t = np.arange(N_SAMPLES) / SAMPLE_RATE_HZ

# --- Logic ---

# Inlet pressure, 2nd inlet manually offset by 0.2 to ensure data read isnt the same across both because practically they wont read the same data.
inlet_pressure_1 = INLET_BASELINE + INLET_DRIFT_AMP * np.sin(2 * np.pi * 0.02 * t) + np.random.normal(0, 0.15, N_SAMPLES)
inlet_pressure_2 = (INLET_BASELINE - 0.2) + INLET_DRIFT_AMP * np.sin(2 * np.pi * 0.02 * t) + np.random.normal(0, 0.15, N_SAMPLES)

# Outlet pressure, same reasoning as inlet, 2nd outlet offset by 2.0
outlet_pressure_1 = OUTLET_BASELINE + OUTLET_DRIFT_AMP * np.sin(2 * np.pi * 0.015 * t) + np.random.normal(0, 0.5, N_SAMPLES)
outlet_pressure_2 = (OUTLET_BASELINE - 2.0) + OUTLET_DRIFT_AMP * np.sin(2 * np.pi * 0.015 * t) + np.random.normal(0, 0.5, N_SAMPLES)

# Flow rate, only one sensor so no offset needed
flow_rate = FLOW_BASELINE + FLOW_DRIFT_AMP * np.sin(2 * np.pi * 0.01 * t) + np.random.normal(0, 0.05, N_SAMPLES)

# Temperature, drift is slower than the rest since blood temp changes very gradually
temperature = TEMP_BASELINE + TEMP_DRIFT_AMP * np.sin(2 * np.pi * 0.005 * t) + np.random.normal(0, 0.05, N_SAMPLES)

# --- Dataframe ---

df = pd.DataFrame({
    "timestamp": t,
    "inlet_pressure_1": inlet_pressure_1,
    "inlet_pressure_2": inlet_pressure_2,
    "outlet_pressure_1": outlet_pressure_1,
    "outlet_pressure_2": outlet_pressure_2,
    "flow_rate": flow_rate,
    "temperature": temperature
})

df.to_csv(OUTPUT_FILE, index=False)

print(f"Generated {N_SAMPLES:,} timesteps ({DURATION_HRS} hours at {SAMPLE_RATE_HZ} Hz)")
print(f"Saved to {OUTPUT_FILE}")
print()
print("Summary stats:")
print(df.drop(columns="timestamp").describe().round(3))