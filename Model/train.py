import numpy as np
import json
from tensorflow import keras
from tensorflow.keras import layers

EPOCHS = 50
BATCH_SIZE = 64

X_train = np.load("preprocessed/X_train.npy")
X_val = np.load("preprocessed/X_val.npy")

window_size = X_train.shape[1]
n_features = X_train.shape[2]

inputs = keras.Input(shape=(window_size, n_features))
x = layers.LSTM(64, return_sequences=True)(inputs)
x = layers.LSTM(32, return_sequences=False)(x)
x = layers.RepeatVector(window_size)(x)
x = layers.LSTM(32, return_sequences=True)(x)
x = layers.LSTM(64, return_sequences=True)(x)
outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="mae")

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
)

model.save("artifacts/1lstm_autoencoder.keras")

recon = model.predict(X_train, batch_size=BATCH_SIZE, verbose=0)
train_mae = np.mean(np.abs(X_train - recon), axis=(1, 2))
threshold = float(train_mae.mean() + 3 * train_mae.std())

with open("artifacts/1threshold.json", "w") as f:
    json.dump({"threshold": threshold, "mae_mean": float(train_mae.mean()), "mae_std": float(train_mae.std())}, f, indent=2)