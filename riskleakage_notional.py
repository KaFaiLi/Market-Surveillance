import os
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta
from prettytable import PrettyTable
from market_close_times import MARKET_CLOSE_TIMES


def _currency_to_timezone(currency):
    """Return IANA timezone for an underlying currency, or None if unknown."""
    info = MARKET_CLOSE_TIMES.get(str(currency).upper())
    return info["timezone"] if info else None


def _parse_exec_time_to_utc(exec_time_value):
    """Parse `execTime` string to native timezone-aware UTC datetime.

    Input examples: `2024-02-09T17:00:29.719+01:00[Europe/Paris]`.
    """
    if pd.isna(exec_time_value):
        return None
    # Drop optional bracketed zone suffix, keep ISO datetime + offset.
    raw_value = str(exec_time_value).split("[", 1)[0]
    parsed_ts = pd.to_datetime(raw_value, errors="coerce", utc=True)
    if pd.isna(parsed_ts):
        return None
    return parsed_ts.to_pydatetime()


def _to_local_exec_time(exec_time_utc, local_timezone):
    """Convert UTC datetime to local market native timezone-aware datetime."""
    if exec_time_utc is None:
        return None
    if not local_timezone:
        return exec_time_utc
    try:
        return exec_time_utc.astimezone(ZoneInfo(local_timezone))
    except Exception:
        return exec_time_utc


def _first_non_null_timezone(values):
    """Return first non-null timezone string from a sequence, else None."""
    for value in values:
        if pd.notna(value):
            return value
    return None


def _load_currency_rates(rate_path):
    """Load currency rates and return a long-form DataFrame for merging."""
    if not os.path.exists(rate_path):
        raise FileNotFoundError(f"Currency rates not found: {rate_path}")

    rates_df = pd.read_excel(rate_path, sheet_name="Rates")
    rates_df["rate_date"] = pd.to_datetime(
        rates_df["REQUEST_DATE"], errors="coerce"
    ).dt.date

    rate_cols = [col for col in rates_df.columns if col.endswith("_MID")]
    rates_long = rates_df.melt(
        id_vars=["rate_date"],
        value_vars=rate_cols,
        var_name="currency_mid",
        value_name="rate_to_eur",
    )
    rates_long["currency"] = rates_long["currency_mid"].str.replace(
        "_MID", "", regex=False
    )

    return rates_long[["rate_date", "currency", "rate_to_eur"]]


def _write_audit_report(
    results_df: pd.DataFrame,
    flagged_trades_df: pd.DataFrame,
    report_path: str,
) -> None:
    """Write a formatted auditor-ready report to *report_path* (plain text).

    Sections
    --------
    1. Surveillance Overview – headline counts and rates.
    2. Leakage Metric Statistics – min/max/mean/median for key risk measures.
    3. Flagged Cases Detail – one row per leakage group, sorted by Leakage_Gap.
    4. Flagged Trades Summary – trade count and gross signed qty per leakage group.
    """
    leakage_df = results_df[results_df["Leakage_Detected"]].copy()
    total_groups = len(results_df)
    total_leakage = len(leakage_df)
    leakage_rate = (total_leakage / total_groups * 100) if total_groups else 0.0

    sep = "=" * 90

    with open(report_path, "w", encoding="utf-8") as fh:

        def w(*args, **kwargs):
            kwargs.setdefault("file", fh)
            print(*args, **kwargs)

        # ------------------------------------------------------------------
        # Section 1: Surveillance Overview
        # ------------------------------------------------------------------
        w(f"\n{sep}")
        w("  INTRADAY RISK LEAKAGE – AUDIT REPORT")
        w(sep)

        t1 = PrettyTable()
        t1.field_names = ["Metric", "Value"]
        t1.align["Metric"] = "l"
        t1.align["Value"] = "r"
        t1.add_rows([
            ["Total Daily Position Groups",         f"{total_groups:,}"],
            ["Leakage Cases Detected",               f"{total_leakage:,}"],
            ["Leakage Rate",                         f"{leakage_rate:.2f} %"],
            ["Portfolios Affected",                  str(leakage_df["Portfolio"].nunique()) if not leakage_df.empty else "0"],
            ["Underlyings Affected",                 str(leakage_df["Underlying"].nunique()) if not leakage_df.empty else "0"],
            ["Earliest Leakage Date",                str(leakage_df["ExecDate"].min()) if not leakage_df.empty else "N/A"],
            ["Latest Leakage Date",                  str(leakage_df["ExecDate"].max()) if not leakage_df.empty else "N/A"],
            ["Flagged Original Trades",              f"{len(flagged_trades_df):,}"],
        ])
        w(t1)

        if leakage_df.empty:
            w("  No leakage cases – nothing further to report.")
            w(sep)
            return

        # ------------------------------------------------------------------
        # Section 2: Leakage Metric Statistics
        # ------------------------------------------------------------------
        w(f"\n{sep}")
        w("  LEAKAGE METRIC STATISTICS  (flagged groups only)")
        w(sep)

        metrics = [
            "Prior_EOD_Position",
            "SOD_Position",
            "EOD_Position",
            "Max_Intraday_Position",
            "Leakage_Gap",
            "Max_to_EOD_Ratio",
            "Max_to_Prior_EOD_Ratio",
            "Max_to_Baseline_EOD_Ratio",
        ]
        t2 = PrettyTable()
        t2.field_names = ["Metric", "Min", "Max", "Mean", "Median"]
        for col in ["Min", "Max", "Mean", "Median"]:
            t2.align[col] = "r"
        t2.align["Metric"] = "l"
        for m in metrics:
            series = leakage_df[m]
            t2.add_row([
                m,
                f"{series.min():>15,.2f}",
                f"{series.max():>15,.2f}",
                f"{series.mean():>15,.2f}",
                f"{series.median():>15,.2f}",
            ])
        w(t2)

        # ------------------------------------------------------------------
        # Section 3: Flagged Cases Detail
        # ------------------------------------------------------------------
        w(f"\n{sep}")
        w("  FLAGGED CASES DETAIL  (sorted by Leakage_Gap descending)")
        w(sep)

        sorted_leakage = leakage_df.sort_values("Leakage_Gap", ascending=False).reset_index(drop=True)
        t3 = PrettyTable()
        t3.field_names = [
            "#", "ExecDate", "Portfolio", "Underlying", "Maturity", "Currency",
            "Pre-Market", "SOD_Pos", "EOD_Pos", "Max_Intraday", "Leakage_Gap", "Max/EOD Ratio", "Max/PriorEOD Ratio", "Max/BaselineEOD Ratio",
        ]
        for col in ["Pre-Market", "SOD_Pos", "EOD_Pos", "Max_Intraday", "Leakage_Gap", "Max/EOD Ratio", "Max/PriorEOD Ratio", "Max/BaselineEOD Ratio"]:
            t3.align[col] = "r"
        for i, row in sorted_leakage.iterrows():
            t3.add_row([
                i + 1,
                row["ExecDate"],
                row["Portfolio"],
                row["Underlying"],
                row["Maturity"],
                row["Currency"],
                f"{row['Prior_EOD_Position']:>12,.0f}",
                f"{row['SOD_Position']:>12,.0f}",
                f"{row['EOD_Position']:>12,.0f}",
                f"{row['Max_Intraday_Position']:>12,.0f}",
                f"{row['Leakage_Gap']:>12,.2f}",
                f"{row['Max_to_EOD_Ratio']:>10,.4f}",
                f"{row['Max_to_Prior_EOD_Ratio']:>15,.4f}",
                f"{row['Max_to_Baseline_EOD_Ratio']:>18,.4f}",
            ])
        w(t3)

        # ------------------------------------------------------------------
        # Section 4: Flagged Trades Summary per Leakage Group
        # ------------------------------------------------------------------
        if not flagged_trades_df.empty:
            w(f"\n{sep}")
            w("  FLAGGED TRADES SUMMARY  (original trades mapped to leakage groups)")
            w(sep)

            grp_summary = (
                flagged_trades_df.groupby(["ExecDate", "Portfolio", "Underlying", "Maturity", "Currency"])
                .agg(
                    Trade_Count=("signed_qty", "count"),
                    Gross_Buy_Qty=("signed_qty", lambda x: x[x > 0].sum()),
                    Gross_Sell_Qty=("signed_qty", lambda x: x[x < 0].sum()),
                    Net_Qty=("signed_qty", "sum"),
                )
                .reset_index()
                .sort_values("Trade_Count", ascending=False)
            )

            t4 = PrettyTable()
            t4.field_names = [
                "ExecDate", "Portfolio", "Underlying", "Maturity", "Currency",
                "Trades", "Gross Buy", "Gross Sell", "Net Qty",
            ]
            for col in ["Trades", "Gross Buy", "Gross Sell", "Net Qty"]:
                t4.align[col] = "r"
            for _, row in grp_summary.iterrows():
                t4.add_row([
                    row["ExecDate"],
                    row["Portfolio"],
                    row["Underlying"],
                    row["Maturity"],
                    row["Currency"],
                    f"{int(row['Trade_Count']):,}",
                    f"{row['Gross_Buy_Qty']:>14,.0f}",
                    f"{row['Gross_Sell_Qty']:>14,.0f}",
                    f"{row['Net_Qty']:>14,.0f}",
                ])
            w(t4)

        w(f"\n{sep}")
        w("  END OF AUDIT REPORT")
        w(sep + "\n")


def analyze_intraday_leakage_continuous(
    df,
    output_folder="hourly_risk_analysis_continuous",
    plot_top_pct=5,
    plot_metric="Leakage_Gap",
    max_plots=None,
    currency_rates_path="output/currency_rates.xlsx",
    debug_sorting=False,
    expected_start_hour=9,
):
    """Analyze intraday position leakage from execution-level trades.

    Parameters:
    df: Input trades DataFrame. Expected columns include `execTime`, `way`,
        `quantity`, `premium`, `portfolioId`, `underlyingId`, `maturity`, and
        `underlyingCurrency`.
    output_folder: Folder for report CSV and leakage plots.
    plot_top_pct: Percent (0-100) of flagged leakage groups to plot.
    plot_metric: Ranking metric for plot selection. One of
        `Leakage_Gap`, `Max_Intraday_Position`, `Max_to_EOD_Ratio`,
        `Max_to_Prior_EOD_Ratio`, `Max_to_Baseline_EOD_Ratio`.
    max_plots: Optional hard cap on the number of plots to generate.
    currency_rates_path: Path to FX rates workbook used to convert premium
        into EUR notionals by date/currency.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]:
        - results_df: one row per grouped exec-date/position key with leakage metrics.
        - flagged_trades_df: original trade rows that belong to flagged leakage groups,
          enriched with leakage metrics and exec-date columns.
    """
    os.makedirs(output_folder, exist_ok=True)
    df = df.copy()
    if plot_top_pct < 0 or plot_top_pct > 100:
        raise ValueError("plot_top_pct must be between 0 and 100.")
    if max_plots is not None and max_plots < 0:
        raise ValueError("max_plots must be >= 0 when provided.")

    position_keys = ["portfolioId", "underlyingId", "maturity", "underlyingCurrency"]

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
        lambda ts: ts.replace(minute=0, second=0, microsecond=0, tzinfo=None)
        if ts is not None
        else None
    )
    df = df.sort_values("hour_bucket", kind="mergesort").reset_index(drop=True)

    # 2) Convert premium to EUR and derive signed notional.
    df["execDate"] = df["execTime_parsed"].apply(
        lambda ts: ts.date() if ts is not None else None
    )
    df["underlyingCurrency"] = df["underlyingCurrency"].astype("string").str.upper()

    rates_long = _load_currency_rates(currency_rates_path)
    df = df.merge(
        rates_long,
        left_on=["execDate", "underlyingCurrency"],
        right_on=["rate_date", "currency"],
        how="left",
    )

    df.loc[df["underlyingCurrency"] == "EUR", "rate_to_eur"] = 1.0
    missing_rates = df[
        df["rate_to_eur"].isna()
        & df["underlyingCurrency"].notna()
        & (df["underlyingCurrency"] != "EUR")
    ]
    if not missing_rates.empty:
        missing_ccy = sorted(missing_rates["underlyingCurrency"].dropna().unique())
        raise ValueError(
            "Missing FX rates for currencies: "
            f"{', '.join(missing_ccy)} in {currency_rates_path}"
        )

    df["premium_eur"] = pd.to_numeric(df["premium"], errors="coerce") * df["rate_to_eur"]
    df["notional_eur"] = df["premium_eur"] * pd.to_numeric(df["quantity"], errors="coerce")

    # Signed notional: BUY positive, SELL negative.
    df["signed_qty"] = np.where(
        df["way"].str.upper() == "BUY", df["notional_eur"], -df["notional_eur"]
    )

    # 3) Aggregate signed flow by local-hour bucket per position key.
    hourly_net = (
        df.groupby(position_keys + ["hour_bucket"])["signed_qty"]
        .sum()
        .reset_index()
        .sort_values(by=position_keys + ["hour_bucket"], kind="mergesort")
    )

    # 4) Rebuild position path from hourly net flow.
    hourly_net["cumulative_pos"] = hourly_net.groupby(position_keys)[
        "signed_qty"
    ].cumsum()
    hourly_net["execDate"] = hourly_net["hour_bucket"].apply(
        lambda ts: ts.date() if ts is not None else None
    )

    # 5) Evaluate leakage per exec-date and position key.
    summary_data = []
    group_frames = {}
    daily_keys = ["execDate", "portfolioId", "underlyingId", "maturity", "underlyingCurrency"]

    for group_ids, group_df in hourly_net.groupby(daily_keys):
        exec_date, port, und, mat, ccy = group_ids
        group_df = group_df.sort_values("hour_bucket", kind="mergesort")
        if debug_sorting:
            first_bucket = group_df["hour_bucket"].iloc[0]
            if expected_start_hour is not None and first_bucket is not None:
                if first_bucket.hour != expected_start_hour:
                    print(
                        "Warning: first hour_bucket is not the expected start hour for "
                        f"{exec_date} | {port} | {und} | {mat} | {ccy}. "
                        f"First bucket: {first_bucket}"
                    )
                    print(group_df["hour_bucket"].head(10).to_string(index=False))
        group_frames[group_ids] = group_df.copy()
        bin_count = len(group_df)

        sod_pos = group_df["cumulative_pos"].iloc[0]
        eod_pos = group_df["cumulative_pos"].iloc[-1]
        # Position entering the day before any first-hour trading (prior EOD carry-over).
        prior_eod_pos = sod_pos - group_df["signed_qty"].iloc[0]

        max_exposure = group_df["cumulative_pos"].abs().max()
        sod_exposure = abs(sod_pos)
        prior_eod_exposure = abs(prior_eod_pos)
        eod_exposure = abs(eod_pos)
        baseline_eod_exposure = max(sod_exposure, eod_exposure)
        leakage_gap = max_exposure - eod_exposure
        max_to_eod_ratio = max_exposure / (eod_exposure + 1e-9)
        max_to_prior_eod_ratio = max_exposure / (prior_eod_exposure + 1e-9)
        max_to_baseline_eod_ratio = max_exposure / (baseline_eod_exposure + 1e-9)

        # Ignore mixed-zero edge cases (only one of SOD/EOD is zero),
        # but keep groups where both are zero.
        mixed_zero_sod_eod = (prior_eod_exposure == 0) ^ (eod_exposure == 0)

        # Exclude groups with <=2 hourly bins (SOD/EOD only, no intraday path).
        enough_intraday_bins = bin_count > 2

        # Flag leakage only when peak intraday exposure exceeds both prior and current EOD.
        is_leakage = (
            enough_intraday_bins
            and
            (not mixed_zero_sod_eod)
            and
            (max_exposure > eod_exposure)
            and (max_exposure > prior_eod_exposure)
        )

        summary_data.append(
            {
                "ExecDate": exec_date,
                "Portfolio": port,
                "Underlying": und,
                "Maturity": mat,
                "Currency": ccy,
                "Bin_Count": bin_count,
                "Prior_EOD_Position": prior_eod_pos,
                "SOD_Position": sod_pos,
                "EOD_Position": eod_pos,
                "Max_Intraday_Position": max_exposure,
                "Leakage_Gap": leakage_gap,
                "Max_to_EOD_Ratio": max_to_eod_ratio,
                "Max_to_Prior_EOD_Ratio": max_to_prior_eod_ratio,
                "Max_to_Baseline_EOD_Ratio": max_to_baseline_eod_ratio,
                "Leakage_Detected": is_leakage,
            }
        )

    results_df = pd.DataFrame(summary_data)

    # 6) Plot only the highest-ranked flagged leakage groups.
    metric_candidates = {"Leakage_Gap", "Max_Intraday_Position", "Max_to_EOD_Ratio", "Max_to_Prior_EOD_Ratio", "Max_to_Baseline_EOD_Ratio"}
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
                row["Currency"],
            )
            group_df = group_frames.get(group_ids)
            if group_df is None or group_df.empty:
                continue

            group_df = group_df.sort_values("hour_bucket", kind="mergesort").reset_index(drop=True)

            if debug_sorting:
                print(
                    f"\n[PLOT DEBUG] Group: {row['ExecDate']} | {row['Portfolio']} | "
                    f"{row['Underlying']} | {row['Maturity']} | {row['Currency']}"
                )
                print(f"  Buckets ({len(group_df)} total, earliest 5):")
                print(group_df["hour_bucket"].head(5).to_string(index=False))
                print(f"  First bucket : {group_df['hour_bucket'].iloc[0]}")
                print(f"  Last  bucket : {group_df['hour_bucket'].iloc[-1]}")

            sod_time = group_df["hour_bucket"].iloc[0]
            eod_time = group_df["hour_bucket"].iloc[-1]
            sod_pos = group_df["cumulative_pos"].iloc[0]
            eod_pos = group_df["cumulative_pos"].iloc[-1]
            max_exposure = group_df["cumulative_pos"].abs().max()

            fig, ax = plt.subplots(figsize=(11, 6))

            hour_bin_starts = group_df["hour_bucket"]
            hour_bin_centers = hour_bin_starts + timedelta(minutes=30)
            sod_time_center = sod_time + timedelta(minutes=30)
            eod_time_center = eod_time + timedelta(minutes=30)
            local_tz_name = _currency_to_timezone(row["Currency"]) or "UTC"

            ax.bar(
                hour_bin_starts,
                group_df["cumulative_pos"],
                width=timedelta(hours=1),
                align="edge",
                alpha=0.85,
                edgecolor="#1f1f1f",
                linewidth=0.6,
                label="Hourly Cumulative Position",
            )

            ax.plot(
                [sod_time_center, eod_time_center],
                [sod_pos, eod_pos],
                color="red",
                linestyle="--",
                linewidth=3,
                marker="o",
                label="Net Flow (SOD to EOD)",
            )

            ax.set_title(
                f"Intraday Risk Leakage\n"
                f"{row['Underlying']} ({row['Currency']}) | {row['Portfolio']} | {row['ExecDate']}\n"
                f"Peak: {max_exposure:,.0f} | EOD: {eod_pos:,.0f} | {plot_metric}: {row[plot_metric]:,.2f}",
                fontsize=11,
            )

            x_start = group_df["hour_bucket"].iloc[0]
            x_end = group_df["hour_bucket"].iloc[-1] + timedelta(hours=1)
            ax.set_xlim(x_start, x_end)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel(f"Hour ({local_tz_name})")
            ax.set_ylabel("Position")
            ax.tick_params(axis="x", labelrotation=45)
            ax.legend()
            ax.grid(axis="y", linestyle=":", alpha=0.5)

            safe_name = (
                f"{row['ExecDate']}_{row['Portfolio']}_{row['Underlying']}_{row['Currency']}".replace(
                    ":", ""
                ).replace("/", "-")
            )
            plt.savefig(os.path.join(output_folder, f"Leakage_{safe_name}.png"))
            plt.close()
            plotted_count += 1

    # 7) Map flagged leakage cases back to the original trade-level DataFrame.
    leakage_summary = results_df[results_df["Leakage_Detected"]].copy()

    # Enrich the working df with execDate (already has execTime_parsed & signed_qty).
    df["execDate"] = df["execTime_parsed"].apply(
        lambda ts: ts.date() if ts is not None else None
    )

    # Filter original trades that belong to a flagged group via fast merge.
    leakage_flag_keys = leakage_summary.rename(
        columns={
            "ExecDate": "execDate",
            "Portfolio": "portfolioId",
            "Underlying": "underlyingId",
            "Maturity": "maturity",
            "Currency": "underlyingCurrency",
        }
    )[["execDate", "portfolioId", "underlyingId", "maturity", "underlyingCurrency"]].copy()
    leakage_flag_keys["_flagged"] = True
    df_flagged = df.merge(leakage_flag_keys, on=["execDate", "portfolioId", "underlyingId", "maturity", "underlyingCurrency"], how="left")
    flagged_trades_df = df_flagged[df_flagged["_flagged"] == True].drop(columns=["_flagged"]).copy()

    # Merge leakage metrics into the flagged trades for full context.
    leakage_merge = leakage_summary.rename(
        columns={
            "ExecDate": "execDate",
            "Portfolio": "portfolioId",
            "Underlying": "underlyingId",
            "Maturity": "maturity",
            "Currency": "underlyingCurrency",
        }
    )[["execDate", "portfolioId", "underlyingId", "maturity", "underlyingCurrency",
       "Prior_EOD_Position", "SOD_Position", "EOD_Position",
         "Max_Intraday_Position", "Leakage_Gap", "Max_to_EOD_Ratio", "Max_to_Prior_EOD_Ratio", "Max_to_Baseline_EOD_Ratio"]]

    flagged_trades_df = flagged_trades_df.merge(
        leakage_merge,
        on=["execDate", "portfolioId", "underlyingId", "maturity", "underlyingCurrency"],
        how="left",
    )

    # Rename for report-friendly column names.
    flagged_trades_df = flagged_trades_df.rename(
        columns={
            "execDate": "ExecDate",
            "portfolioId": "Portfolio",
            "underlyingId": "Underlying",
            "maturity": "Maturity",
            "underlyingCurrency": "Currency",
        }
    )

    # 8) Export reports.
    csv_path = os.path.join(output_folder, "Full_Leakage_Report_Continuous.csv")
    results_df.to_csv(csv_path, index=False)

    flagged_trades_csv = os.path.join(output_folder, "Leakage_Flagged_Trades.csv")
    flagged_trades_df.to_csv(flagged_trades_csv, index=False)

    # 9) Write auditor report to text file.
    audit_report_path = os.path.join(output_folder, "Audit_Report.txt")
    _write_audit_report(results_df, flagged_trades_df, audit_report_path)

    print(f"Plots Generated              : {plotted_count}")
    print(f"Full Report                  : {csv_path}")
    print(f"Flagged Trades CSV           : {flagged_trades_csv}")
    print(f"Audit Report TXT             : {audit_report_path}")

    return results_df, flagged_trades_df


if __name__ == "__main__":
    df_trades = pd.read_csv("output/synthetic_trading_data.csv")
    results, flagged = analyze_intraday_leakage_continuous(
        df_trades, plot_top_pct=5, plot_metric="Max_to_Baseline_EOD_Ratio", max_plots=20
    )
