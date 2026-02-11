import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def analyze_intraday_leakage_hourly(df, output_folder='hourly_risk_analysis'):
    """
    Analyzes intraday risk using Hourly Buckets.
    Visualizes: Bars = Cumulative Position per Hour | Line = SOD to EOD Reference.
    """
    os.makedirs(output_folder, exist_ok=True)
    df = df.copy()

    # 1. Parse Time and Bucket by Hour
    # Handles timestamps with timezone info like "2024-02-09T10:00:00+01:00"
    df['execTime_dt'] = pd.to_datetime(df['execTime'].astype(str).str.split('[').str[0])
    df['hour_bucket'] = df['execTime_dt'].dt.floor('h') # Rounds down to nearest hour

    # 2. Calculate Signed Quantity
    df['signed_qty'] = np.where(df['way'].str.upper() == 'BUY', df['quantity'], -df['quantity'])

    # 3. Aggregate to Hourly Level
    # We group by the standard keys AND the hour bucket
    keys = ['tradeDate', 'portfolioId', 'underlyingId', 'maturity']
    
    # First, get the net change per hour
    hourly_net = df.groupby(keys + ['hour_bucket'])['signed_qty'].sum().reset_index()
    
    # Sort is crucial for cumsum
    hourly_net = hourly_net.sort_values(by=keys + ['hour_bucket'])

    # 4. Calculate Cumulative Position (The "Bar" Value)
    # Assumes starting position is 0
    hourly_net['cumulative_pos'] = hourly_net.groupby(keys)['signed_qty'].cumsum()

    # 5. Analyze for Leakage and Generate CSV
    summary_data = []
    
    # Iterate through each unique group (Portfolio/Underlying/Day)
    for group_ids, group_df in hourly_net.groupby(keys):
        trade_date, port, und, mat = group_ids
        
        # Define Reference Points
        # Point 1: SOD (Start of Day) -> Position 0
        sod_pos = 0
        sod_time = group_df['hour_bucket'].min()
        
        # Point 2: EOD (End of Day) -> Final Cumulative Position
        eod_pos = group_df['cumulative_pos'].iloc[-1]
        eod_time = group_df['hour_bucket'].max()

        # Risk Logic: Did we exceed the final EOD position significantly?
        # We look at the absolute max exposure during the day vs absolute final position
        max_exposure = group_df['cumulative_pos'].abs().max()
        final_exposure = abs(eod_pos)
        
        # Leakage definition: Intraday peak is higher than EOD result
        # We add a small buffer (e.g. 1% or 100 units) to ignore noise
        is_leakage = max_exposure > (final_exposure + 100) 

        summary_data.append({
            'TradeDate': trade_date,
            'Portfolio': port,
            'Underlying': und,
            'Maturity': mat,
            'Max_Intraday_Pos': max_exposure,
            'EOD_Pos_Abs': final_exposure,
            'Net_EOD_Pos': eod_pos,
            'Leakage_Detected': is_leakage,
            'Leakage_Gap': max_exposure - final_exposure
        })

        # 6. Visualization (Strictly per your requirement)
        if is_leakage:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # BAR: Cumulative Position per Hour
            # Note: We align 'edge' to align the bar starting at the hour
            ax.bar(group_df['hour_bucket'], group_df['cumulative_pos'], 
                   width=0.03, # Width approx 45 mins
                   color='#1f77b4', alpha=0.8, label='Hourly Position (Cumulative)')
            
            # LINE: Only Two Points (SOD -> EOD)
            # Plotting from First Hour timestamp to Last Hour timestamp
            # Y values: 0 -> EOD Position
            ax.plot([sod_time, eod_time], [0, eod_pos], 
                    color='red', linewidth=3, linestyle='--', marker='o', 
                    label='Net Flow Reference (SOD -> EOD)')

            # Formatting
            ax.set_title(f"Risk Leakage: {und} | {port} | {trade_date}\nIntraday Peak: {max_exposure:,.0f} vs EOD: {eod_pos:,.0f}", fontsize=11)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_ylabel("Cumulative Position")
            ax.legend()
            ax.grid(axis='y', linestyle=':', alpha=0.5)
            
            # Save
            safe_name = f"{trade_date}_{port}_{und}".replace(':','').replace('/','-')
            plt.savefig(os.path.join(output_folder, f"Leakage_{safe_name}.png"))
            plt.close()

    # 7. Export Master CSV
    results_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_folder, 'Full_Leakage_Report.csv')
    results_df.to_csv(csv_path, index=False)
    
    print(f"Processing complete.")
    print(f"Total Groups: {len(results_df)}")
    print(f"Leakage Cases Detected: {results_df['Leakage_Detected'].sum()}")
    print(f"Report saved to: {csv_path}")

    return results_df

# --- How to run ---
# df = pd.read_csv('your_data.csv')
# analyze_intraday_leakage_hourly(df)
df_trades = pd.read_csv("output/synthetic_trading_data.csv")
analyze_intraday_leakage_hourly(df_trades)
