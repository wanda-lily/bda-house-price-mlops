import joblib
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import mlflow
import mlflow.sklearn


# LOAD DATA
df = pd.read_csv("data/processed.csv")


# FEATURE ENGINEERING
df["log_price"] = np.log1p(df["price_eur"])

features_base = ["county", "is_new", "year", "month"]
df = df.dropna(subset=features_base + ["log_price"])


# TRAIN / TEST SPLIT
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)


# TARGET ENCODING (SMOOTHED)
global_mean = train_df["log_price"].mean()

county_stats = train_df.groupby("county")["log_price"].agg(["mean", "count"])
alpha = 10

county_map = (
    (county_stats["mean"] * county_stats["count"] + global_mean * alpha)
    / (county_stats["count"] + alpha)
)


def encode_county(data):
    return data["county"].map(county_map).fillna(global_mean)


train_df["county_encoded"] = encode_county(train_df)
test_df["county_encoded"] = encode_county(test_df)


# FEATURES
features = [
    "county_encoded",
    "is_new",
    "year",
    "month",
]

X_train = train_df[features]
X_test = test_df[features]

y_train = train_df["log_price"]
y_test = test_df["log_price"]


# BASELINE (dumb model)
baseline_pred = np.full_like(y_test, y_train.mean())
baseline_mae_log = mean_absolute_error(y_test, baseline_pred)


# MODEL
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)


# ----------------------------------------------------------
# MLflow
# ----------------------------------------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("irish-house-prices")


with mlflow.start_run():

    # log params
    mlflow.log_params({
        "model": "RandomForestRegressor",
        "n_estimators": 200,
        "features": features,
        "alpha_smoothing": alpha
    })

    # train
    model.fit(X_train, y_train)

    # predictions
    preds_log = model.predict(X_test)

    # metrics
    rmse_log = np.sqrt(mean_squared_error(y_test, preds_log))
    mae_log = mean_absolute_error(y_test, preds_log)
    r2 = r2_score(y_test, preds_log)

    preds_eur = np.expm1(preds_log)
    actual_eur = np.expm1(y_test)

    mae_eur = mean_absolute_error(actual_eur, preds_eur)

    # log metrics
    mlflow.log_metrics({
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2": r2,
        "mae_eur": mae_eur,
        "baseline_mae_log": baseline_mae_log
    })

    # feature importance
    fi = dict(zip(features, model.feature_importances_))
    mlflow.log_dict(fi, "feature_importance.json")

    # print results
    print("\n=== MODEL RESULTS ===")
    print(f"RMSE (log): {rmse_log:.4f}")
    print(f"MAE  (log): {mae_log:.4f}")
    print(f"R²         : {r2:.4f}")
    print(f"MAE (EUR)  : €{mae_eur:,.0f}")
    print(f"Baseline MAE (log): {baseline_mae_log:.4f}")

    # SAVE MODEL + ENCODER
    model_bundle = {
        "model": model,
        "county_map": county_map,
        "features": features
    }

    joblib.dump(model_bundle, "data/price_model.pkl")
    mlflow.sklearn.log_model(model, "model")


# ----------------------------------------------------------
# DEPLOYMENT LOGIC (CD)
# ----------------------------------------------------------
MODEL_PATH = "data/price_model.pkl"
IMPROVEMENT_THRESHOLD = 0.005


def get_baseline_rmse(model_path, X_test, y_test):
    if not os.path.exists(model_path):
        return None

    try:
        bundle = joblib.load(model_path)
        model = bundle["model"]
        preds = model.predict(X_test)
        return float(np.sqrt(mean_squared_error(y_test, preds)))
    except Exception as e:
        print(f"Baseline error: {e}")
        return None


baseline_rmse = get_baseline_rmse(MODEL_PATH, X_test, y_test)


if baseline_rmse is None:
    decision = "DEPLOY"
    reason = "No existing model — deploying new model."

elif rmse_log < baseline_rmse * (1 - IMPROVEMENT_THRESHOLD):
    improvement = (baseline_rmse - rmse_log) / baseline_rmse * 100
    decision = "DEPLOY"
    reason = f"Improved by {improvement:.2f}%"

else:
    decision = "SKIP"
    reason = "No sufficient improvement"


# save decision
with open("data/deploy_decision.txt", "w") as f:
    f.write(f"{decision}\n{reason}\n")

print(f"\nDeploy decision: {decision}")
print(reason)


# save model
joblib.dump(model_bundle, "data/price_model.pkl")
print("Model saved.")
