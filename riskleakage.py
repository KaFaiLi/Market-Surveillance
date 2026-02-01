import os
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def analyze_risk_and_export(df):
    df = df.copy()
    
    # 1. Reconstruct Trade Order from execTime
    # Parse format: 2024-02-09T17:00:29.719+01:00[Europe/Paris]
    # Extract datetime part before timezone (before '+' or '-' in offset)
    df['execTime_dt'] = pd.to_datetime(df['execTime'].str.extract(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)')[0])
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
    
    # 7. Create visualizations for each outlier group
    visualize_outlier_groups(outlier_details, outlier_summary, output_dir)
    
    return outlier_summary

def visualize_outlier_groups(outlier_details, outlier_summary, output_dir):
    """
    Create bar plot visualizations for each outlier group showing:
    - Quantity traded over time (bars)
    - Trending cumulative position line
    """
    if outlier_details.empty:
        return
    
    group_cols = ['tradeDate', 'portfolioId', 'underlyingId', 'maturity']
    
    # Iterate through each outlier group
    for idx, row in outlier_summary.iterrows():
        # Filter data for this specific group
        mask = True
        for col in group_cols:
            mask &= (outlier_details[col] == row[col])
        group_data = outlier_details[mask].copy()
        
        # Sort by execution time
        group_data = group_data.sort_values('execTime_dt')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Create bar plot - color by BUY (green) vs SELL (red)
        colors = ['green' if w.upper() == 'BUY' else 'red' for w in group_data['way']]
        bars = ax.bar(group_data['execTime_dt'], group_data['signed_qty'], 
                      color=colors, alpha=0.7, width=0.0003, edgecolor='black', linewidth=0.5)
        
        # Add trending cumulative position line
        ax.plot(group_data['execTime_dt'], group_data['cumulative_pos'], 
                color='blue', linestyle='-', linewidth=2.5, marker='o', markersize=4,
                label='Cumulative Position (Trending)', zorder=5)
        
        # Add horizontal reference line for EOD position
        final_position = row['eod_position']
        ax.axhline(y=final_position, color='orange', linestyle='--', linewidth=2, 
                   label=f'EOD Position: {final_position:.0f}', zorder=4)
        
        # Add zero line for reference
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Format x-axis to show time
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45, ha='right')
        
        # Labels and title
        ax.set_xlabel('Time of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Position / Quantity', fontsize=12, fontweight='bold')
        
        title = f"Outlier Group: {row['tradeDate']} | Portfolio: {row['portfolioId']} | "
        title += f"Underlying: {row['underlyingId']} | Maturity: {row['maturity']}\n"
        title += f"Risk Ratio: {row['risk_ratio']:.2f} | Z-Score: {row['z_score']:.2f}"
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='BUY'),
            Patch(facecolor='red', alpha=0.7, label='SELL'),
            plt.Line2D([0], [0], color='blue', linestyle='-', linewidth=2.5, marker='o',
                      label='Cumulative Position (Trending)'),
            plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2,
                      label=f'EOD Position: {final_position:.0f}')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':', axis='y')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure with descriptive filename
        filename = f"outlier_{row['tradeDate']}_{row['portfolioId']}_{row['underlyingId']}_{row['maturity']}.png"
        filename = filename.replace('/', '-').replace(':', '-')  # Clean filename
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {filepath}")

# --- EXECUTION ---
# Assuming your CSV exists from the previous steps
try:
    df_trades = pd.read_csv('output/synthetic_trading_data.csv')
    outliers = analyze_risk_and_export(df_trades)
except FileNotFoundError:
    print("Error: 'synthetic_trading_data.csv' not found. Please ensure the file exists.")