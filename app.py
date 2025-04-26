from flask import Flask, render_template, request, url_for
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Inisialisasi Flask App
app = Flask(__name__)

# ------------------------------------------------
# PREPROCESSING DATA
# ------------------------------------------------

# Load dataset
df = pd.read_csv("dataset.csv")

# Cek missing values
if df.isnull().sum().sum() > 0:
    df.dropna(inplace=True)

# Pilih fitur dan label
X = df.drop("median_house_value", axis=1).select_dtypes(include=[np.number])
y = df["median_house_value"]

# Normalisasi fitur
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Load model
model = tf.keras.models.load_model("model_ann.h5")

# ------------------------------------------------
# ROUTE HALAMAN UTAMA
# ------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None

    if request.method == "POST":
        input_data = [
            float(request.form.get("MedInc")),
            float(request.form.get("HouseAge")),
            float(request.form.get("AveRooms")),
            float(request.form.get("AveBedrms")),
            float(request.form.get("Population")),
            float(request.form.get("AveOccup")),
            float(request.form.get("Latitude")),
            float(request.form.get("Longitude")),
        ]

        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)
        predictions = [round(pred[0], 2) for pred in prediction]

        # Grafik prediksi
        plt.figure(figsize=(6,4))
        plt.scatter(1, predictions[0], color="blue", label="Harga Prediksi")
        plt.title("Prediksi Harga Rumah")
        plt.xlabel("Sample")
        plt.ylabel("Harga Rumah ($)")
        plt.legend()
        plt.grid(True)

        if not os.path.exists('static'):
            os.makedirs('static')
        plt.savefig("static/plot.png")
        plt.close()

    # Tampilkan dataset di web
    df_columns = X.columns.tolist()
    df_values = X.values.tolist()

    return render_template("index.html", predictions=predictions, df_columns=df_columns, df_values=df_values)

# ------------------------------------------------
# RUN APP
# ------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
