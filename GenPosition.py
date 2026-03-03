import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os


def generate_position_data(rows=200000):
    np.random.seed(42)

    suffix = "+01:00[Europe/Paris]"

    # ---------------------------
    # Allowed Portfolios
    # ---------------------------
    portfolios = ["CTM_SINGLFUT", "CTM_EQUITY", "CTM_DERIV"]

    # Only stock & future
    position_categories = ["stockPosition", "futurePosition"]

    stocks = ["BARC.L", "BATS.L", "BAY.L", "DTE.DE", "AIR.PA"]
    futures = ["FDAX-NS", "FESX-NS", "FTSE-NS"]

    currencies = ["EUR", "USD", "GBP"]
    underlying_ids = ["DAX_X", "CAC40_X", "FTSE_X"]

    base_date = datetime(2024, 1, 1)

    # Random base dates
    random_days = np.random.randint(0, 60, rows)

    def random_timestamp():
        d = base_date + timedelta(days=int(np.random.randint(0, 60)))
        d = d.replace(
            hour=np.random.randint(8, 18),
            minute=np.random.randint(0, 60),
            second=np.random.randint(0, 60),
            microsecond=np.random.randint(0, 1000) * 1000,
        )
        return (
            d.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{d.microsecond // 1000:03d}"
            + suffix
        )

    # ---------------------------
    # Create base dataframe
    # ---------------------------
    df = pd.DataFrame({
        "position_category": np.random.choice(position_categories, rows),
        "positionDateTime": [random_timestamp() for _ in range(rows)],
        "portfolioId": np.random.choice(portfolios, rows),
        "positionDate": f"2024-01-01{suffix}",
        "ventilationDate": f"1970-01-01{suffix}",
        "positionType": "0",
        "marginCallTimestamp": f"1970-01-01T00:00:00{suffix}",
        "configName": "Prod",
        "isCompressed": "False",
        "isMarginCall": "False",
        "clpType": "CLP",
        "isCLPShadow": "False",
        "isPayoffShadow": "False",
    })

    # ---------------------------
    # Stock logic
    # ---------------------------
    stock_mask = df["position_category"] == "stockPosition"

    df.loc[stock_mask, "stockId"] = np.random.choice(stocks, stock_mask.sum())
    df.loc[stock_mask, "contractId"] = df.loc[stock_mask, "stockId"]
    df.loc[stock_mask, "futureContractId"] = "NA_FUTURE"
    df.loc[stock_mask, "underlyingId"] = df.loc[stock_mask, "stockId"]
    df.loc[stock_mask, "underlyingType"] = "S"
    df.loc[stock_mask, "pointValue"] = 1
    df.loc[stock_mask, "closingSpot"] = np.random.uniform(10, 500, stock_mask.sum())

    # ---------------------------
    # Future logic
    # ---------------------------
    future_mask = df["position_category"] == "futurePosition"

    df.loc[future_mask, "stockId"] = "NA_STOCK"
    df.loc[future_mask, "futureContractId"] = np.random.choice(futures, future_mask.sum())
    df.loc[future_mask, "contractId"] = df.loc[future_mask, "futureContractId"]
    df.loc[future_mask, "underlyingId"] = np.random.choice(underlying_ids, future_mask.sum())
    df.loc[future_mask, "underlyingType"] = "I"
    df.loc[future_mask, "pointValue"] = np.random.choice([10, 25, 50], future_mask.sum())
    df.loc[future_mask, "closingSpot"] = np.random.uniform(3000, 18000, future_mask.sum())

    # ---------------------------
    # Common numeric fields
    # ---------------------------
    df["position"] = np.random.uniform(-5_000_000, 5_000_000, rows).round(0)
    df["buySellAmount"] = (df["position"] * df["closingSpot"] * df["pointValue"]).round(2)

    df["productId"] = np.random.randint(100000, 9999999, rows)
    df["productType"] = np.where(df["position_category"] == "futurePosition", "F", "L")

    # Maturity
    maturities = [
        (base_date + timedelta(days=int(np.random.randint(30, 365))))
        .strftime(f"%Y-%m-%d{suffix}")
        for _ in range(rows)
    ]
    df["maturity"] = maturities

    df["month"] = pd.to_datetime(df["maturity"].str[:10]).dt.month
    df["year"] = pd.to_datetime(df["maturity"].str[:10]).dt.year

    # Additional required columns (no nulls)
    df["strike"] = 0.0
    df["typeListedOption"] = "NA"
    df["positionMC"] = 0.0
    df["marginCalls"] = "[]"
    df["clpFlows"] = "[]"
    df["descriptiveFlows"] = "[]"
    df["flows"] = "[]"
    df["deltasByDeal"] = "[]"
    df["histo"] = "[]"
    df["deltas"] = "[]"
    df["closingTime"] = 1600
    df["payCurId"] = np.random.choice(currencies, rows)
    df["currencyId"] = df["payCurId"]
    df["clpId"] = np.random.randint(1000, 999999, rows)
    df["fundId"] = df["portfolioId"]
    df["warrantId"] = 0
    df["valueDate"] = df["positionDate"]
    df["dbcType"] = 0.0
    df["settlementType"] = 0.0
    df["type"] = "F"
    df["indexFutureType"] = "I"

    return df


# ---------------------------
# Generate & Save
# ---------------------------
df_positions = generate_position_data(200000)

os.makedirs("output", exist_ok=True)
df_positions.to_csv("output/synthetic_position_data_clean.csv", index=False)

print("✅ Clean synthetic stock & future dataset generated.")
print("No null values:", df_positions.isna().sum().sum())