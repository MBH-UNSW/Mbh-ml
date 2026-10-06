import numpy as np
from tensorflow import keras
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pickle



FEATURE_COLS = [
    "left_flow_lpm", "right_flow_lpm",
    "left_inlet_pressure_mmHg", "right_inlet_pressure_mmHg",
    "left_upstream_pressure_mmHg", "left_downstream_pressure_mmHg",
    "right_upstream_pressure_mmHg", "right_downstream_pressure_mmHg",
    "pump_rpm", "tah_temperature_c",
]

# # find relative path later
# # maybe set up a file callout
# model = keras.models.load_model()
# scaler = pickle.load(open())
# threshold = json.load(open)
    


# evaluation calculations

