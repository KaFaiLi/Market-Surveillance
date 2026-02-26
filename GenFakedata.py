import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_market_data(rows=100000):
    # Categorical setup
    portfolios = ["CTM_SINGLFUT", "CTM_EQUITY", "CTM_DERIV"]
    accounts = ["CTM_IM", "CTM_MARGIN"]
    assets = ["BANCA-INTESA_X", "ENI_S.p.A", "UNICREDIT_X"]
    currencies = ["EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD"]

    # Common Suffix
    suffix = "+01:00[Europe/Paris]"

    # Generate random dates over a 30-day period for 2024
    base_dates = [
        datetime(2024, 1, 16) + timedelta(days=np.random.randint(0, 30))
        for _ in range(rows)
    ]

    # Generate random times (hours, minutes, seconds, milliseconds) for execTime
    rand_hours = np.random.randint(0, 24, rows)
    rand_minutes = np.random.randint(0, 60, rows)
    rand_seconds = np.random.randint(0, 60, rows)
    rand_millis = np.random.randint(0, 1000, rows)

    exec_times = [
        d.replace(hour=h, minute=m, second=s, microsecond=ms * 1000)
        for d, h, m, s, ms in zip(
            base_dates, rand_hours, rand_minutes, rand_seconds, rand_millis
        )
    ]

    data = {
        "dealNature": "REGULAR_MARKET",
        "dealId": np.random.randint(100000000, 999999999, rows).astype(np.int64),
        "dealType": np.random.choice(["SHA", "FUT"], rows),
        # Formatting all dates to YYYY-MM-DD+01:00[Europe/Paris]
        "inputDate": [d.strftime(f"%Y-%m-%d{suffix}") for d in base_dates],
        "tradeDate": [d.strftime(f"%Y-%m-%d{suffix}") for d in base_dates],
        "valueDate": [
            (d + timedelta(days=2)).strftime(f"%Y-%m-%d{suffix}") for d in base_dates
        ],
        "portfolioId": np.random.choice(portfolios, rows),
        "accountId": np.random.choice(accounts, rows),
        "way": np.random.choice(["Buy", "Sell"], rows),
        "quantity": np.random.uniform(1000, 5000000, rows).round(0),
        "assetName": np.random.choice(assets, rows),
        "broker_type": "LCHLTDBAFIGB",
        "currencyId": np.random.choice(currencies, rows),
        "maturity": f"1970-01-01{suffix}",
        "premium": np.random.uniform(1.0, 10.0, rows).round(4),
        "status": "Initial",
        "underlyingId": np.random.choice(["12620", "13440", "15500"], rows),
        # Format: 2024-01-16T17:17:10.206+01:00[Europe/Paris]
        "execTime": [
            t.strftime(f"%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}{suffix}"
            for t in exec_times
        ],
        "underlyingCurrency": np.random.choice(currencies, rows),
        "underlyingType": np.random.choice(["Stock", "Index"], rows),
        "underlyingName": "",  # Populated below
        "GPRC2": np.random.choice(["ASI", "EUR"], rows),
    }

    # Add new column for future point value as random float
    data["futurePointValue"] = np.random.uniform(0.1, 100.0, rows).round(4)

    df = pd.DataFrame(data)
    df["underlyingName"] = df["assetName"]

    return df


df_generated = generate_market_data(100000)
print("Data Generation Complete. All dates formatted.")
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
summary_csv = os.path.join(output_dir, "synthetic_trading_data.csv")
df_generated.to_csv(summary_csv, index=False)
