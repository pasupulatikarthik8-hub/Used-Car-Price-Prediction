import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
CURRENT_YEAR = 2026

FUEL_TYPES = ["Petrol", "Diesel", "CNG"]
TRANSMISSIONS = ["Manual", "Automatic"]

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


def parse_input(data):
    """Validate form/JSON input and return a one-row DataFrame."""
    try:
        year = int(data["year"])
        kms = int(data["kms_driven"])
        engine = int(data["engine_cc"])
        owners = int(data["owners"])
        fuel = data["fuel_type"]
        transmission = data["transmission"]
    except (KeyError, TypeError, ValueError):
        raise ValueError("Please fill in all fields with valid values.")

    if not 1995 <= year <= CURRENT_YEAR:
        raise ValueError(f"Year must be between 1995 and {CURRENT_YEAR}.")
    if not 0 <= kms <= 500000:
        raise ValueError("Kilometres driven must be between 0 and 500,000.")
    if not 600 <= engine <= 6000:
        raise ValueError("Engine capacity must be between 600 and 6000 cc.")
    if not 1 <= owners <= 6:
        raise ValueError("Number of owners must be between 1 and 6.")
    if fuel not in FUEL_TYPES:
        raise ValueError("Invalid fuel type.")
    if transmission not in TRANSMISSIONS:
        raise ValueError("Invalid transmission.")

    return pd.DataFrame(
        [
            {
                "car_age": CURRENT_YEAR - year,
                "kms_driven": kms,
                "engine_cc": engine,
                "owners": owners,
                "fuel_type": fuel,
                "transmission": transmission,
            }
        ]
    )


def predict_price(df):
    price = float(model.predict(df)[0])
    return round(max(price, 0.3), 2)  # price in lakhs, floored for sanity


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form = {}
    if request.method == "POST":
        form = request.form.to_dict()
        try:
            prediction = predict_price(parse_input(form))
        except ValueError as exc:
            error = str(exc)
    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form=form,
        fuel_types=FUEL_TYPES,
        transmissions=TRANSMISSIONS,
        current_year=CURRENT_YEAR,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        price = predict_price(parse_input(data))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"price_lakhs": price, "price_inr": int(price * 100000)})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
