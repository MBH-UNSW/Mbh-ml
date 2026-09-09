#tester file

import numpy as np
import pandas as pd
import json
import requests
import json

RAW_DATA_PATH = "SDG/tah_synthetic_healthy.csv"

TARGET_COLUMN = "left_flow_lpm"
MULTIPLIER = 0.98

WINDOW_SAMPLES = 100  # 20Hz * 5 seconds

df = pd.read_csv(RAW_DATA_PATH)

start = np.random.randint(0, len(df) - WINDOW_SAMPLES)
window = df.iloc[start:start + WINDOW_SAMPLES].copy()
window[TARGET_COLUMN] = window[TARGET_COLUMN] * MULTIPLIER

output = {"measurements": window.drop(columns="timestamp").to_dict(orient="records")}

with open("SDG/anomalous_window.json", "w") as f:
    json.dump(output, f, indent=2)
    
URL = "https://ubh-ml-414777335651.australia-southeast1.run.app/predict"
JSON_PATH = "SDG/anomalous_window.json"

with open(JSON_PATH, "r") as f:
    body = json.load(f)

response = requests.post(URL, json=body)

print(response.status_code)
print(json.dumps(response.json(), indent=2))