"""
Flask API — Irish Property Price Prediction
Endpoints: /predict  /health  /metrics
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "model/price_model.pkl")
ENCODER_PATH = os.environ.get("ENCODER_PATH", "data/county_encoder.csv")

model = None
county_encoder = None
request_count = 0
start_time = time.time()


def load_artefacts():
    global model, county_encoder
    model = joblib.load(MODEL_PATH)
    enc = pd.read_csv(ENCODER_PATH, index_col="county")
    county_encoder = enc["price_eur"].to_dict()  # county → log-median price
    app.logger.info(f"Model loaded from {MODEL_PATH}")
    app.logger.info(f"County encoder loaded: {len(county_encoder)} counties")


FEATURES = ["county_encoded", "is_new", "year", "month"]

REQUIRED_FIELDS = {
    "county": str,
    "is_new": int,        # 1 = new build, 0 = second-hand
    "year": int,
    "month": int,         # 1–12
}


def encode_input(data: dict) -> pd.DataFrame:
    global_median = np.median(list(county_encoder.values()))
    county_enc = county_encoder.get(
        data["county"].strip().title(), global_median)
    row = {
        "county_encoded": county_enc,
        "is_new": int(data["is_new"]),
        "year": int(data["year"]),
        "month": int(data["month"]),
    }
    return pd.DataFrame([row])[FEATURES]


@app.route("/predict", methods=["POST"])
def predict():
    global request_count
    request_count += 1

    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        X = encode_input(body)
        log_pred = model.predict(X)[0]
        price = float(np.expm1(log_pred))
        return jsonify({
            "predicted_price_eur": round(price, 2),
            "county": body["county"],
            "is_new": body["is_new"],
            "year": body["year"],
            "month": body["month"],
        })
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    if model is None or county_encoder is None:
        return jsonify({"status": "unhealthy", "reason": "artefacts not loaded"}), 503
    return jsonify({"status": "healthy"}), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    uptime = round(time.time() - start_time, 1)
    return jsonify({
        "uptime_seconds": uptime,
        "total_requests": request_count,
        "model_path": MODEL_PATH,
        "counties_in_encoder": len(county_encoder) if county_encoder else 0,
    })


if __name__ == "__main__":
    load_artefacts()
    app.run(host="0.0.0.0", port=5000)
else:
    # Gunicorn entry point
    with app.app_context():
        load_artefacts()
