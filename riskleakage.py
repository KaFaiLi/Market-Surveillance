import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


def analyze_intraday_leakage_continuous(
    df,
    output_folder='hourly_risk_analysis_continuous',
    leakage_buffer=100
):

    os.makedirs(output_folder, exist_ok=True)
    df = df.copy()

    # ---------------------------------------------------------
    # 1. Parse timestamps and hourly buckets
    # ---------------------------------------------------------
    df['execTime_dt'] = pd.to_datetime(
        df['execTime'].astype(str).str.split('[').str[0]
    )
    df['hour_bucket'] = df['execTime_dt'].dt.floor('h')

    # ---------------------------------------------------------
    # 2. Signed quantity
    # ---------------------------------------------------------
    df['signed_qty'] = np.where(
        df['way'].str.upper() == 'BUY',
        df['quantity'],
        -df['quantity']
    )

    # ---------------------------------------------------------
    # 3. Hourly aggregation (continuous positions)
    # ---------------------------------------------------------
    position_keys = ['portfolioId', 'underlyingId', 'maturity']

    hourly_net = (
        df.groupby(position_keys + ['hour_bucket'])['signed_qty']
          .sum()
          .reset_index()
          .sort_values(by=position_keys + ['hour_bucket'])
    )

    # ---------------------------------------------------------
    # 4. Continuous cumulative position
    # ---------------------------------------------------------
    hourly_net['cumulative_pos'] = (
        hourly_net
        .groupby(position_keys)['signed_qty']
        .cumsum()
    )

    # ✅ FIX: restore tradeDate AFTER aggregation
    hourly_net['tradeDate'] = hourly_net['hour_bucket'].dt.date

    # ---------------------------------------------------------
    # 5. Daily leakage analysis
    # ---------------------------------------------------------
    summary_data = []
    daily_keys = ['tradeDate', 'portfolioId', 'underlyingId', 'maturity']

    for group_ids, group_df in hourly_net.groupby(daily_keys):

        trade_date, port, und, mat = group_ids
        group_df = group_df.sort_values('hour_bucket')

        sod_time = group_df['hour_bucket'].iloc[0]
        eod_time = group_df['hour_bucket'].iloc[-1]

        sod_pos = group_df['cumulative_pos'].iloc[0]
        eod_pos = group_df['cumulative_pos'].iloc[-1]

        max_exposure = group_df['cumulative_pos'].abs().max()
        eod_exposure = abs(eod_pos)

        is_leakage = max_exposure > (eod_exposure + leakage_buffer)

        summary_data.append({
            'TradeDate': trade_date,
            'Portfolio': port,
            'Underlying': und,
            'Maturity': mat,
            'SOD_Position': sod_pos,
            'EOD_Position': eod_pos,
            'Max_Intraday_Position': max_exposure,
            'Leakage_Gap': max_exposure - eod_exposure,
            'Leakage_Detected': is_leakage
        })

        # -----------------------------------------------------
        # 6. Visualization (only when leakage exists)
        # -----------------------------------------------------
        if is_leakage:
            fig, ax = plt.subplots(figsize=(11, 6))

            ax.bar(
                group_df['hour_bucket'],
                group_df['cumulative_pos'],
                width=0.03,
                alpha=0.8,
                label='Hourly Cumulative Position'
            )

            ax.plot(
                [sod_time, eod_time],
                [sod_pos, eod_pos],
                color='red',
                linestyle='--',
                linewidth=3,
                marker='o',
                label='Net Flow (SOD → EOD)'
            )

            ax.set_title(
                f"Intraday Risk Leakage\n"
                f"{und} | {port} | {trade_date}\n"
                f"Peak: {max_exposure:,.0f} | EOD: {eod_pos:,.0f}",
                fontsize=11
            )

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_ylabel("Position")
            ax.legend()
            ax.grid(axis='y', linestyle=':', alpha=0.5)

            safe_name = f"{trade_date}_{port}_{und}".replace(':', '').replace('/', '-')
            plt.savefig(os.path.join(output_folder, f"Leakage_{safe_name}.png"))
            plt.close()

    # ---------------------------------------------------------
    # 7. Export report
    # ---------------------------------------------------------
    results_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_folder, 'Full_Leakage_Report_Continuous.csv')
    results_df.to_csv(csv_path, index=False)

    print("Processing complete.")
    print(f"Total Daily Groups: {len(results_df)}")
    print(f"Leakage Cases Detected: {results_df['Leakage_Detected'].sum()}")
    print(f"Report saved to: {csv_path}")

    return results_df


if __name__ == "__main__":
    df_trades = pd.read_csv("output/synthetic_trading_data.csv")
    analyze_intraday_leakage_continuous(df_trades)
