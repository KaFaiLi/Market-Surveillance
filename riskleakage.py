import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from market_close_times import MARKET_CLOSE_TIMES


def _currency_to_timezone(currency):
    """Return IANA timezone for an underlying currency, or None if unknown."""
    info = MARKET_CLOSE_TIMES.get(str(currency).upper())
    return info["timezone"] if info else None


def _parse_exec_time_to_utc(exec_time_value):
    """Parse `execTime` string to timezone-aware UTC Timestamp.

    Input examples: `2024-02-09T17:00:29.719+01:00[Europe/Paris]`.
    """
    if pd.isna(exec_time_value):
        return pd.NaT
    # Drop optional bracketed zone suffix, keep ISO datetime + offset.
    raw_value = str(exec_time_value).split("[", 1)[0]
    return pd.to_datetime(raw_value, errors="coerce", utc=True)


def _to_local_exec_time(exec_time_utc, local_timezone):
    """Convert UTC timestamp to local market timezone when available."""
    if pd.isna(exec_time_utc):
        return pd.NaT
    if not local_timezone:
        return exec_time_utc
    try:
        return exec_time_utc.tz_convert(local_timezone)
    except Exception:
        return exec_time_utc


def analyze_intraday_leakage_continuous(
    df,
    output_folder="hourly_risk_analysis_continuous",
    plot_top_pct=5,
    plot_metric="Leakage_Gap",
    max_plots=None,
):
    """Analyze intraday position leakage from execution-level trades.

    Parameters:
    df: Input trades DataFrame. Expected columns include `execTime`, `way`,
        `quantity`, `portfolioId`, `underlyingId`, `maturity`, and
        `underlyingCurrency`.
    output_folder: Folder for report CSV and leakage plots.
    plot_top_pct: Percent (0-100) of flagged leakage groups to plot.
    plot_metric: Ranking metric for plot selection. One of
        `Leakage_Gap`, `Max_Intraday_Position`, `Max_to_EOD_Ratio`.
    max_plots: Optional hard cap on the number of plots to generate.

    Returns:
    pd.DataFrame with one row per grouped exec-date/position key.
    """
    os.makedirs(output_folder, exist_ok=True)
    df = df.copy()
    if plot_top_pct < 0 or plot_top_pct > 100:
        raise ValueError("plot_top_pct must be between 0 and 100.")
    if max_plots is not None and max_plots < 0:
        raise ValueError("max_plots must be >= 0 when provided.")

    # 1) Parse execution timestamp and group by parsed-hour bucket.
    df["market_timezone"] = df.get(
        "underlyingCurrency", pd.Series(index=df.index)
    ).apply(_currency_to_timezone)
    df["execTime_parsed_utc"] = df["execTime"].apply(_parse_exec_time_to_utc)
    df["execTime_parsed"] = df.apply(
        lambda row: _to_local_exec_time(
            row["execTime_parsed_utc"], row["market_timezone"]
        ),
        axis=1,
    )
    df["hour_bucket"] = df["execTime_parsed"].apply(
        lambda ts: ts.floor("h") if pd.notna(ts) else pd.NaT
    )

    # 2) Signed quantity.
    df["signed_qty"] = np.where(
        df["way"].str.upper() == "BUY", df["quantity"], -df["quantity"]
    )

    # 3) Aggregate signed flow by local-hour bucket per position key.
    position_keys = ["portfolioId", "underlyingId", "maturity"]
    hourly_net = (
        df.groupby(position_keys + ["hour_bucket"])["signed_qty"]
        .sum()
        .reset_index()
        .sort_values(by=position_keys + ["hour_bucket"])
    )

    # 4) Rebuild position path from hourly net flow.
    hourly_net["cumulative_pos"] = hourly_net.groupby(position_keys)[
        "signed_qty"
    ].cumsum()
    hourly_net["execDate"] = hourly_net["hour_bucket"].dt.date

    # 5) Evaluate leakage per exec-date and position key.
    summary_data = []
    group_frames = {}
    daily_keys = ["execDate", "portfolioId", "underlyingId", "maturity"]

    for group_ids, group_df in hourly_net.groupby(daily_keys):
        exec_date, port, und, mat = group_ids
        group_df = group_df.sort_values("hour_bucket")
        group_frames[group_ids] = group_df.copy()

        sod_pos = group_df["cumulative_pos"].iloc[0]
        eod_pos = group_df["cumulative_pos"].iloc[-1]

        max_exposure = group_df["cumulative_pos"].abs().max()
        eod_exposure = abs(eod_pos)
        leakage_gap = max_exposure - eod_exposure
        max_to_eod_ratio = max_exposure / (eod_exposure + 1e-9)

        is_leakage = (max_exposure > (eod_exposure)) and (max_exposure != abs(sod_pos))

        summary_data.append(
            {
                "ExecDate": exec_date,
                "Portfolio": port,
                "Underlying": und,
                "Maturity": mat,
                "SOD_Position": sod_pos,
                "EOD_Position": eod_pos,
                "Max_Intraday_Position": max_exposure,
                "Leakage_Gap": leakage_gap,
                "Max_to_EOD_Ratio": max_to_eod_ratio,
                "Leakage_Detected": is_leakage,
            }
        )

    results_df = pd.DataFrame(summary_data)

    # 6) Plot only the highest-ranked flagged leakage groups.
    metric_candidates = {"Leakage_Gap", "Max_Intraday_Position", "Max_to_EOD_Ratio"}
    if plot_metric not in metric_candidates:
        raise ValueError(
            f"Invalid plot_metric '{plot_metric}'. Use one of: {sorted(metric_candidates)}"
        )

    leakage_df = results_df[results_df["Leakage_Detected"]].copy()
    plotted_count = 0

    if not leakage_df.empty and plot_top_pct > 0:
        leakage_df = leakage_df.sort_values(plot_metric, ascending=False)
        top_n = max(1, int(np.ceil(len(leakage_df) * (plot_top_pct / 100.0))))
        if max_plots is not None:
            top_n = min(top_n, int(max_plots))
        to_plot_df = leakage_df.head(top_n)

        for _, row in to_plot_df.iterrows():
            group_ids = (
                row["ExecDate"],
                row["Portfolio"],
                row["Underlying"],
                row["Maturity"],
            )
            group_df = group_frames.get(group_ids)
            if group_df is None or group_df.empty:
                continue

            sod_time = group_df["hour_bucket"].iloc[0]
            eod_time = group_df["hour_bucket"].iloc[-1]
            sod_pos = group_df["cumulative_pos"].iloc[0]
            eod_pos = group_df["cumulative_pos"].iloc[-1]
            max_exposure = group_df["cumulative_pos"].abs().max()

            fig, ax = plt.subplots(figsize=(11, 6))

            ax.bar(
                group_df["hour_bucket"],
                group_df["cumulative_pos"],
                width=0.03,
                alpha=0.8,
                label="Hourly Cumulative Position",
            )

            ax.plot(
                [sod_time, eod_time],
                [sod_pos, eod_pos],
                color="red",
                linestyle="--",
                linewidth=3,
                marker="o",
                label="Net Flow (SOD to EOD)",
            )

            ax.set_title(
                f"Intraday Risk Leakage\n"
                f"{row['Underlying']} | {row['Portfolio']} | {row['ExecDate']}\n"
                f"Peak: {max_exposure:,.0f} | EOD: {eod_pos:,.0f} | {plot_metric}: {row[plot_metric]:,.2f}",
                fontsize=11,
            )

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Position")
            ax.legend()
            ax.grid(axis="y", linestyle=":", alpha=0.5)

            safe_name = (
                f"{row['ExecDate']}_{row['Portfolio']}_{row['Underlying']}".replace(
                    ":", ""
                ).replace("/", "-")
            )
            plt.savefig(os.path.join(output_folder, f"Leakage_{safe_name}.png"))
            plt.close()
            plotted_count += 1

    # 7) Export report.
    csv_path = os.path.join(output_folder, "Full_Leakage_Report_Continuous.csv")
    results_df.to_csv(csv_path, index=False)

    print("Processing complete.")
    print(f"Total Daily Groups: {len(results_df)}")
    print(f"Leakage Cases Detected: {results_df['Leakage_Detected'].sum()}")
    print(f"Plots Generated: {plotted_count}")
    print(f"Report saved to: {csv_path}")

    return results_df


if __name__ == "__main__":
    df_trades = pd.read_csv("output/synthetic_trading_data.csv")
    analyze_intraday_leakage_continuous(
        df_trades, plot_top_pct=5, plot_metric="Max_to_EOD_Ratio", max_plots=20
    )
