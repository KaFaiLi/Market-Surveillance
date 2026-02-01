import os
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import zscore

def analyze_risk_and_export(df):
    df = df.copy()
    
    # 1. Reconstruct Trade Order from execTime
    # Standardizing format: stripping timezone for conversion
    df['execTime_dt'] = pd.to_datetime(df['execTime'].str.split('+').str[0])
    df = df.sort_values(by=['execTime_dt', 'dealId']).reset_index(drop=True)
    df['trade_order'] = np.arange(len(df))
    
    # 2. Risk Logic Calculations
    group_cols = ['tradeDate', 'portfolioId', 'underlyingId', 'maturity']
    
    # Time diff and Position tracking
    df['time_diff_min'] = df.groupby(group_cols)['execTime_dt'].diff().dt.total_seconds() / 60
    df['signed_qty'] = np.where(df['way'].str.upper() == 'BUY', df['quantity'], -df['quantity'])
    df['cumulative_pos'] = df.groupby(group_cols)['signed_qty'].cumsum()
    
    # 3. Create Group Summary for Z-Score
    summary = df.groupby(group_cols).agg(
        max_intraday_exposure=('cumulative_pos', lambda x: x.abs().max()),
        eod_position=('cumulative_pos', 'last'),
        trade_count=('dealId', 'count')
    ).reset_index()
    
    # Calculate Risk Ratio and Z-Score
    summary['risk_ratio'] = summary['max_intraday_exposure'] / (summary['eod_position'].abs() + 0.1)
    summary['z_score'] = zscore(summary['risk_ratio'])
    
    # 4. Filter Outliers (Z-Score > 2)
    outlier_summary = summary[summary['z_score'] > 2].sort_values(by='z_score', ascending=False)
    
    # 5. Extract Detailed Trade Logs for these specific outliers only
    # This helps in auditing exactly what happened on those risky days
    outlier_details = df.merge(
        outlier_summary[group_cols + ['z_score', 'risk_ratio']],
        on=group_cols,
        how='inner'
    )
    
    # 6. Save to two CSV files in output folder
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, 'Outlier_Summary.csv')
    details_csv = os.path.join(output_dir, 'Detailed_Outlier.csv')
    if outlier_summary.empty:
        print("No outlier groups found; no CSVs written.")
        return outlier_summary
    
    outlier_summary.to_csv(summary_csv, index=False)
    outlier_details.to_csv(details_csv, index=False)
    
    print(f"Analysis complete. {len(outlier_summary)} outlier groups found.")
    print(f"Results saved to: {summary_csv} and {details_csv}")
    
    return outlier_summary

# --- EXECUTION ---
# Assuming your CSV exists from the previous steps
try:
    df_trades = pd.read_csv('output/synthetic_trading_data.csv')
    outliers = analyze_risk_and_export(df_trades)
except FileNotFoundError:
    print("Error: 'synthetic_trading_data.csv' not found. Please ensure the file exists.")