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
ENCODER_PATH = os.environ.get("ENCODER_PATH", "data/county_encoder.csv")

model = None
county_encoder = None

request_count = 0
start_time = time.time()


# LOAD ARTIFACTS
def load_artefacts():
    global model, county_encoder

    bundle = joblib.load(MODEL_PATH)

    # training saves either:
    # - model directly OR
    # - model bundle dict
    if isinstance(bundle, dict):
        model = bundle.get("model", bundle)
    else:
        model = bundle

    # load encoder (county → encoded value)
    enc = pd.read_csv(ENCODER_PATH, index_col=0)

    # robust conversion (handles single-column csv)
    county_encoder = enc.iloc[:, 0].to_dict()

    app.logger.info(f"Model loaded from {MODEL_PATH}")
    app.logger.info(f"Encoder loaded: {len(county_encoder)} counties")


FEATURES = ["county_encoded", "is_new", "year", "month"]

REQUIRED_FIELDS = {
    "county": str,
    "is_new": int,
    "year": int,
    "month": int,
}


# ENCODING
def encode_input(data: dict) -> pd.DataFrame:
    global_median = np.median(list(county_encoder.values()))

    county = data["county"].strip().title()

    county_enc = county_encoder.get(county, global_median)

    row = {
        "county_encoded": county_enc,
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
            "county": body["county"],
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
    if model is None or county_encoder is None:
        return jsonify({
            "status": "unhealthy",
            "reason": "artefacts not loaded"
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
        "counties_in_encoder": len(county_encoder) if county_encoder else 0,
    })


# STARTUP
if __name__ == "__main__":
    load_artefacts()
    app.run(host="0.0.0.0", port=5000)

else:
    # for gunicorn
    with app.app_context():
        load_artefacts()
