"""
Tests for Irish Property Price Prediction API.
Covers: data schema, /health, /predict, /metrics
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Data schema tests ────────────────────────────────────────────────────────

PROCESSED_PATH = os.path.join("data", "processed.csv")
REQUIRED_COLUMNS = {
    "county_encoded", "is_new",
    "year", "month", "price_eur",
}


@pytest.mark.skipif(not os.path.exists(PROCESSED_PATH), reason="processed.csv not generated yet")
class TestDataSchema:
    def test_columns(self):
        df = pd.read_csv(PROCESSED_PATH)
        assert REQUIRED_COLUMNS.issubset(set(df.columns)), (
            f"Missing columns: {REQUIRED_COLUMNS - set(df.columns)}"
        )

    def test_no_nulls_in_features(self):
        df = pd.read_csv(PROCESSED_PATH)
        feature_cols = list(REQUIRED_COLUMNS - {"price_eur"})
        assert df[feature_cols].isnull().sum().sum() == 0

    def test_price_positive(self):
        df = pd.read_csv(PROCESSED_PATH)
        assert (df["price_eur"] > 0).all()

    def test_binary_flags(self):
        df = pd.read_csv(PROCESSED_PATH)
        for col in ["is_new"]:
            assert set(df[col].unique()).issubset(
                {0, 1}), f"{col} contains non-binary values"

    def test_year_range(self):
        df = pd.read_csv(PROCESSED_PATH)
        assert df["year"].between(2010, 2030).all()

    def test_month_range(self):
        df = pd.read_csv(PROCESSED_PATH)
        assert df["month"].between(1, 12).all()

    def test_min_rows(self):
        df = pd.read_csv(PROCESSED_PATH)
        assert len(df) >= 1000, f"Expected ≥1000 rows, got {len(df)}"

    def test_county_encoded_is_integer(self):
        df = pd.read_csv(PROCESSED_PATH)
        assert pd.api.types.is_integer_dtype(df["county_encoded"]), \
            "county_encoded should be integer (cat.codes)"


# ── API tests ────────────────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "county_encoded": 5,
    "is_new": 0,
    "year": 2023,
    "month": 6,
}


@pytest.fixture(scope="module")
def client():
    """
    Create a test client with stub model artefacts so API tests
    run in CI without a real trained model.
    """
    from unittest.mock import MagicMock, patch
    import numpy as np

    stub_model = MagicMock()
    stub_model.predict.return_value = np.array([np.log1p(300_000)])

    with patch("joblib.load", return_value=stub_model):
        import importlib
        import api.app
        importlib.reload(api.app)
        api.app.app.config["TESTING"] = True
        with api.app.app.test_client() as c:
            yield c


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "healthy"


class TestPredictEndpoint:
    def test_predict_returns_price(self, client):
        resp = client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "predicted_price_eur" in data
        assert data["predicted_price_eur"] > 0

    def test_predict_echoes_inputs(self, client):
        resp = client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data["county_encoded"] == 5
        assert data["year"] == 2023

    def test_predict_missing_field(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k !=
                   "county_encoded"}
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "Missing fields" in resp.get_json()["error"]

    def test_predict_empty_body(self, client):
        resp = client.post("/predict", data="",
                           content_type="application/json")
        assert resp.status_code == 400


class TestMetricsEndpoint:
    def test_metrics_fields(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "uptime_seconds" in data
        assert "total_requests" in data
        assert "model_path" in data
