import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# Load dataset
df = pd.read_csv("dataset.csv")

# Pilih fitur dan label
X = df.drop("median_house_value", axis=1).select_dtypes(include=[np.number])
y = df["median_house_value"]

# Normalisasi input
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Buat model ANN
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # output layer
])

# Kompilasi model - PAKAI OBJEK KERAS LANGSUNG (FIX ERROR)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

# Latih model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=1)

# Simpan model ke file
model.save("model_ann.h5")
