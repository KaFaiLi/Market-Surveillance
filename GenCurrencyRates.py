import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_currency_rates(start_date="2024-01-01", num_days=700):
    """
    Generate currency exchange rates (base currency: EUR)
    
    Returns a DataFrame with REQUEST_DATE and MID rates for each currency to EUR
    """
    
    # Currency rates to EUR (approximate mid-rates for realism)
    currency_rates = {
        "USD_MID": (0.92, 0.95),      # USD to EUR range
        "JPY_MID": (0.0065, 0.0075),  # JPY to EUR range
        "GBP_MID": (1.15, 1.20),      # GBP to EUR range
        "CHF_MID": (1.05, 1.10),      # CHF to EUR range
        "CAD_MID": (0.68, 0.72),      # CAD to EUR range
        "AUD_MID": (0.60, 0.65),      # AUD to EUR range
    }
    
    # Generate dates
    base_date = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [base_date + timedelta(days=i) for i in range(num_days)]
    
    # Generate constant REQUEST_DATE format (assuming daily rates at 00:00:00)
    data = {
        "REQUEST_DATE": [d.strftime("%Y-%m-%d 00:00:00") for d in dates]
    }
    
    # Generate rates for each currency with some random variation
    for currency, (min_rate, max_rate) in currency_rates.items():
        # Add some realistic daily variation
        rates = []
        current_rate = np.random.uniform(min_rate, max_rate)
        for _ in range(num_days):
            # Add small random daily change (±2%)
            daily_change = np.random.uniform(-0.02, 0.02)
            current_rate = current_rate * (1 + daily_change)
            # Keep within reasonable bounds
            current_rate = np.clip(current_rate, min_rate, max_rate)
            rates.append(current_rate)
        data[currency] = [round(rate, 5) for rate in rates]
    
    df = pd.DataFrame(data)
    return df


# Generate the currency rates
df_rates = generate_currency_rates(start_date="2024-01-01", num_days=700)

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Save to Excel
excel_file = os.path.join(output_dir, "currency_rates.xlsx")
df_rates.to_excel(excel_file, index=False, sheet_name="Rates")

print("Currency Rates Generation Complete.")
print(f"Saved to: {excel_file}")
print(f"\nFirst few rows:")
print(df_rates.head())
