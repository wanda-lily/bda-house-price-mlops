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

# PATHS
MODEL_PATH = os.environ.get("MODEL_PATH", "data/price_model.pkl")

model = None
request_count = 0
start_time = time.time()

# LOAD ARTIFACTS


def load_artefacts():
    global model
    bundle = joblib.load(MODEL_PATH)
    # training saves either:
    # - model directly OR
    # - model bundle dict
    if isinstance(bundle, dict):
        model = bundle.get("model", bundle)
    else:
        model = bundle
    app.logger.info(f"Model loaded from {MODEL_PATH}")


FEATURES = ["county_encoded", "is_new", "year", "month"]
REQUIRED_FIELDS = {
    "county_encoded": int,
    "is_new": int,
    "year": int,
    "month": int,
}

# ENCODING


def encode_input(data: dict) -> pd.DataFrame:
    row = {
        "county_encoded": int(data["county_encoded"]),
        "is_new": int(data["is_new"]),
        "year": int(data["year"]),
        "month": int(data["month"]),
    }
    return pd.DataFrame([row])[FEATURES]

# ----------------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------------

# PREDICT ENDPOINT


@app.route("/predict", methods=["POST"])
def predict():
    global request_count
    request_count += 1

    body = request.get_json()
    if not body:
        return jsonify({"error": "Invalid JSON body"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        X = encode_input(body)
        log_pred = model.predict(X)[0]
        price = float(np.expm1(log_pred))
        return jsonify({
            "predicted_price_eur": round(price, 2),
            "county_encoded": body["county_encoded"],
            "is_new": body["is_new"],
            "year": body["year"],
            "month": body["month"],
        })
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# HEALTH CHECK


@app.route("/health", methods=["GET"])
def health():
    if model is None:
        return jsonify({
            "status": "unhealthy",
            "reason": "model not loaded"
        }), 503
    return jsonify({"status": "healthy"}), 200

# METRICS


@app.route("/metrics", methods=["GET"])
def metrics():
    uptime = round(time.time() - start_time, 1)
    return jsonify({
        "uptime_seconds": uptime,
        "total_requests": request_count,
        "model_path": MODEL_PATH,
    })


# STARTUP
# STARTUP
if __name__ == "__main__":
    load_artefacts()
    app.run(host="0.0.0.0", port=5000)
else:
    # for gunicorn
    with app.app_context():
        try:
            load_artefacts()
        except FileNotFoundError:
            app.logger.warning("Model not found - starting without model")
            