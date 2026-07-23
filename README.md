# UBH Machine Learning - TAH Anomaly Detection

Self-supervised LSTM-Autoencoder for detecting haemodynamic anomalies in the UBH Total Artificial Heart, trained only on healthy baseline telemetry.

## What this does

The TAH streams pressure, flow, and temperature data continuously. Right now there's no automated way to catch abnormal behaviour in real time, monitoring only happens at clinical reviews. This project trains a model that learns what "normal" looks like from healthy data alone, then flags anything that doesn't reconstruct properly as a potential anomaly.

No labelled failure data needed, which matters a lot since a rare device like this doesn't have any failure history to learn from.

## Pipeline

```
ESP32 sensors (200Hz)
    -> downsample + low-pass filter
    -> Firebase Data Connect
    -> preprocessing (scale, window, split)
    -> LSTM-Autoencoder (train on healthy data only)
    -> reconstruction error -> threshold -> alert
```

## Tech stack

- **Data / Model**: Python, TensorFlow/Keras, NumPy, pandas, scikit-learn
- **Backend**: Firebase Data Connect (PostgreSQL) for structured data, Firestore for real-time sensor data
- **Hosting**: Google Cloud Run for inference
- **Hardware**: ESP32-S3

## Current status

- Synthetic data generator built to simulate healthy telemetry while real sensor data is pending
- Preprocessing pipeline handles downsampling, scaling, and windowing
- Model architecture defined and training end to end on synthetic data
- Real deployment pipeline (ESP32 -> cloud -> model) still being worked out with the electrical team

## Folder structure

```
SDG/            synthetic data generator
preprocess.py   scaling, windowing, train/val/test split
model.py        LSTM-Autoencoder architecture
train.py        training + threshold calculation
```