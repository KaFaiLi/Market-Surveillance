import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_market_data(rows=100000):
    # Categorical setup
    portfolios = ['CTM_SINGLFUT', 'CTM_EQUITY', 'CTM_DERIV']
    accounts = ['CTM_IM', 'CTM_MARGIN']
    assets = ['BANCA-INTESA_X', 'ENI_S.p.A', 'UNICREDIT_X']
    
    # Common Suffix
    suffix = "+01:00[Europe/Paris]"
    
    # Generate random dates over a 30-day period for 2024
    base_dates = [datetime(2024, 1, 16) + timedelta(days=np.random.randint(0, 30)) for _ in range(rows)]
    
    data = {
        'dealNature': 'REGULAR_MARKET',
        'dealId': np.random.randint(100000000, 999999999, rows).astype(np.int64),
        'dealType': 'SHA',
        # Formatting all dates to YYYY-MM-DD+01:00[Europe/Paris]
        'inputDate': [d.strftime(f"%Y-%m-%d{suffix}") for d in base_dates],
        'tradeDate': [d.strftime(f"%Y-%m-%d{suffix}") for d in base_dates],
        'valueDate': [(d + timedelta(days=2)).strftime(f"%Y-%m-%d{suffix}") for d in base_dates],
        'portfolioId': np.random.choice(portfolios, rows),
        'accountId': np.random.choice(accounts, rows),
        'way': np.random.choice(['Buy', 'Sell'], rows),
        'quantity': np.random.uniform(1000, 5000000, rows).round(0),
        'assetName': np.random.choice(assets, rows),
        'broker_type': 'LCHLTDBAFIGB',
        'currencyId': 'EUR',
        'maturity': f"1970-01-01{suffix}",
        'premium': np.random.uniform(1.0, 10.0, rows).round(4),
        'status': 'Initial',
        'underlyingId': np.random.choice(['12620', '13440', '15500'], rows),
        'execTime': [d.strftime(f"%Y-%m-%d{suffix}") for d in base_dates],
        'underlyingCurrency': 'EUR',
        'underlyingType': 'Stock',
        'underlyingName': '' # Populated below
    }

    df = pd.DataFrame(data)
    df['underlyingName'] = df['assetName']
    
    # CRITICAL: We add a hidden "trade_sequence" to maintain order 
    # since execTime no longer has seconds/minutes.
    df['trade_sequence'] = np.arange(len(df))
    
    return df

df_generated = generate_market_data(100000)
print("Data Generation Complete. All dates formatted.")
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
summary_csv = os.path.join(output_dir, 'synthetic_trading_data.csv')
df_generated.to_csv(summary_csv, index=False)