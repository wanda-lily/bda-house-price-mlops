import numpy as np
import os
import pandas as pd

"""
Data ingestion & preprocessing for Irish Property Price Register.
"""

RAW_PATH = os.environ.get("RAW_DATA_PATH", "property_price_register.csv")
OUT_DIR = os.environ.get("OUTPUT_DIR", "data")

# Load raw data
df = pd.read_csv(RAW_PATH)


def clean(df):
    df = df[~df["not_full_market_price"]].copy()

    print(f"After dropping non-full-market-price rows: {len(df):,}")

    lo = df["price_eur"].quantile(0.005)
    hi = df["price_eur"].quantile(0.995)

    df = df[
        (df["price_eur"] >= lo) &
        (df["price_eur"] <= hi)
    ].copy()

    print(
        f"After price outlier removal "
        f"(€{lo:,.0f}–€{hi:,.0f}): {len(df):,}"
    )

    return df


def engineer_features(df):
    df["date_of_sale"] = pd.to_datetime(
        df["date_of_sale"],
        errors="coerce"
    )

    df["year"] = df["date_of_sale"].dt.year
    df["month"] = df["date_of_sale"].dt.month

    desc = df["description"].str.lower()
    df["is_new"] = desc.str.contains("new").astype(int)

    df["county"] = (
        df["county"]
        .str.strip()
        .str.title()
    )

    return df


FEATURES = [
    "county_encoded",
    "is_new",
    "year",
    "month",
]

TARGET = "price_eur"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = clean(df)
    data = engineer_features(data)

    cols = FEATURES + [TARGET]

    data = data[cols].dropna()

    clean_path = os.path.join(OUT_DIR, "processed.csv")

    data.to_csv(clean_path, index=False)

    print(f"Clean Dataset: {len(data):,} rows → {clean_path}")

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
