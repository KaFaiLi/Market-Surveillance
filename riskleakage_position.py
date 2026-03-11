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


def _write_audit_report(
    results_df: pd.DataFrame,
    flagged_trades_df: pd.DataFrame,
    report_path: str,
    hourly_flagged_df: pd.DataFrame | None = None,
) -> None:
    """Write a formatted auditor-ready report to *report_path* (plain text).

    Sections
    --------
    1. Surveillance Overview – headline counts and rates.
    2. Leakage Metric Statistics – min/max/mean/median for key risk measures.
    3. Flagged Cases Detail – one row per leakage group, sorted by Leakage_Gap.
    4. Flagged Trades Summary – trade count and gross signed qty per leakage group.
    5. Hourly Buy/Sell Nominal – hourly sum of buy and sell nominal per flagged group.
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
            "Prior_EOD_Nominal",
            "SOD_Nominal",
            "EOD_Nominal",
            "Max_Intraday_Nominal",
            "Leakage_Gap",
            "Max_to_EOD_Ratio",
            "Max_to_Prior_EOD_Ratio",
            "Max_to_Baseline_EOD_Ratio",
            "Baseline_Deviation_Ratio",
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
            "Pre-Market", "SOD_Nom", "EOD_Nom", "Max_Intraday", "Leakage_Gap",
            "Max/EOD Ratio", "Max/PriorEOD Ratio", "Max/BaselineEOD Ratio", "Baseline Dev Ratio",
        ]
        for col in ["Pre-Market", "SOD_Nom", "EOD_Nom", "Max_Intraday", "Leakage_Gap",
                     "Max/EOD Ratio", "Max/PriorEOD Ratio", "Max/BaselineEOD Ratio", "Baseline Dev Ratio"]:
            t3.align[col] = "r"
        for i, row in sorted_leakage.iterrows():
            t3.add_row([
                i + 1,
                row["ExecDate"],
                row["Portfolio"],
                row["Underlying"],
                row["Maturity"],
                row["Currency"],
                f"{row['Prior_EOD_Nominal']:>12,.0f}",
                f"{row['SOD_Nominal']:>12,.0f}",
                f"{row['EOD_Nominal']:>12,.0f}",
                f"{row['Max_Intraday_Nominal']:>12,.0f}",
                f"{row['Leakage_Gap']:>12,.2f}",
                f"{row['Max_to_EOD_Ratio']:>10,.4f}",
                f"{row['Max_to_Prior_EOD_Ratio']:>15,.4f}",
                f"{row['Max_to_Baseline_EOD_Ratio']:>18,.4f}",
                f"{row['Baseline_Deviation_Ratio']:>15,.4f}",
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
                    Trade_Count=("signed_nominal", "count"),
                    Gross_Buy_Nominal=("buy_nominal", "sum"),
                    Gross_Sell_Nominal=("sell_nominal", "sum"),
                    Net_Nominal=("signed_nominal", "sum"),
                )
                .reset_index()
                .sort_values("Trade_Count", ascending=False)
            )

            t4 = PrettyTable()
            t4.field_names = [
                "ExecDate", "Portfolio", "Underlying", "Maturity", "Currency",
                "Trades", "Gross Buy Nom", "Gross Sell Nom", "Net Nominal",
            ]
            for col in ["Trades", "Gross Buy Nom", "Gross Sell Nom", "Net Nominal"]:
                t4.align[col] = "r"
            for _, row in grp_summary.iterrows():
                t4.add_row([
                    row["ExecDate"],
                    row["Portfolio"],
                    row["Underlying"],
                    row["Maturity"],
                    row["Currency"],
                    f"{int(row['Trade_Count']):,}",
                    f"{row['Gross_Buy_Nominal']:>14,.0f}",
                    f"{row['Gross_Sell_Nominal']:>14,.0f}",
                    f"{row['Net_Nominal']:>14,.0f}",
                ])
            w(t4)

        # ------------------------------------------------------------------
        # Section 5: Hourly Buy/Sell Nominal per Flagged Group
        # ------------------------------------------------------------------
        if hourly_flagged_df is not None and not hourly_flagged_df.empty:
            w(f"\n{sep}")
            w("  HOURLY BUY / SELL NOMINAL  (flagged leakage groups)")
            w(sep)

            t5 = PrettyTable()
            t5.field_names = [
                "ExecDate", "Portfolio", "Underlying", "Maturity", "Currency",
                "Hour_Bucket", "Sum_Buy_Nominal", "Sum_Sell_Nominal", "Net_Nominal", "Trade_Count",
            ]
            for col in ["Sum_Buy_Nominal", "Sum_Sell_Nominal", "Net_Nominal", "Trade_Count"]:
                t5.align[col] = "r"
            for _, row in hourly_flagged_df.iterrows():
                t5.add_row([
                    row["ExecDate"],
                    row["Portfolio"],
                    row["Underlying"],
                    row["Maturity"],
                    row["Currency"],
                    str(row["Hour_Bucket"]),
                    f"{row['Sum_Buy_Nominal']:>14,.0f}",
                    f"{row['Sum_Sell_Nominal']:>14,.0f}",
                    f"{row['Net_Nominal']:>14,.0f}",
                    f"{int(row['Trade_Count']):,}",
                ])
            w(t5)

        w(f"\n{sep}")
        w("  END OF AUDIT REPORT")
        w(sep + "\n")


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


def analyze_intraday_leakage_continuous(
    df,
    output_folder="hourly_risk_analysis_continuous",
    currency_rates_path="output/currency_rates.xlsx",
    plot_top_pct=5,
    plot_metric="Leakage_Gap",
    max_plots=None,
    plot_exposure_threshold=None,
    debug_sorting=False,
    expected_start_hour=9,
):
    """Analyze intraday nominal leakage from execution-level trades.

    Nominal is computed per trade as:
        premium_EUR × quantity × futurePointValue
    where premium_EUR = premium × FX rate to EUR.

    Parameters:
    df: Input trades DataFrame. Expected columns include `execTime`, `way`,
        `quantity`, `portfolioId`, `underlyingId`, `maturity`,
        `underlyingCurrency`, `premium`, and `futurePointValue`.
    output_folder: Folder for report CSV and leakage plots.
    currency_rates_path: Path to an Excel file with FX rates to EUR.
    plot_top_pct: Percent (0-100) of flagged leakage groups to plot.
    plot_metric: Ranking metric for plot selection. One of
        `Leakage_Gap`, `Max_Intraday_Nominal`, `Max_to_EOD_Ratio`,
        `Max_to_Prior_EOD_Ratio`, `Max_to_Baseline_EOD_Ratio`,
        `Baseline_Deviation_Ratio`.
    max_plots: Optional hard cap on the number of plots to generate.
    plot_exposure_threshold: Optional minimum Max_Intraday_Nominal (EUR)
        required for a leakage group to be plotted. Groups below this
        threshold are skipped. Example: 10_000_000 (10 million EUR).

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        - results_df: one row per grouped exec-date/position key with leakage metrics.
        - flagged_trades_df: original trade rows that belong to flagged leakage groups,
          enriched with leakage metrics and exec-date columns.
        - hourly_flagged_df: hourly sum of buy and sell nominal per flagged leakage group,
          with columns Hour_Bucket, Sum_Buy_Nominal, Sum_Sell_Nominal, Net_Nominal, Trade_Count.
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

    # 2) FX rates and nominal computation.
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
            f"Missing FX rates for currencies: {', '.join(missing_ccy)} "
            f"in {currency_rates_path}"
        )

    df["premium_numeric"] = pd.to_numeric(df["premium"], errors="coerce")
    df["premium_eur"] = df["premium_numeric"] * df["rate_to_eur"]
    df["futurePointValue_numeric"] = pd.to_numeric(
        df.get("futurePointValue", pd.Series(dtype="float64", index=df.index)),
        errors="coerce",
    ).fillna(1.0)

    # Nominal = premium_EUR × quantity × futurePointValue.
    # Use abs(premium_eur) so the sign is solely determined by buy/sell way,
    # since premiums can be negative in some source data.
    df["nominal"] = (
        df["premium_eur"].abs() * df["quantity"] * df["futurePointValue_numeric"]
    )
    df["signed_qty"] = np.where(
        df["way"].str.upper() == "BUY", df["quantity"], -df["quantity"]
    )
    df["signed_nominal"] = np.where(
        df["way"].str.upper() == "BUY", df["nominal"], -df["nominal"]
    )
    df["buy_nominal"] = np.where(
        df["way"].str.upper() == "BUY", df["nominal"], 0.0
    )
    df["sell_nominal"] = np.where(
        df["way"].str.upper() == "SELL", df["nominal"], 0.0
    )

    # 3) Aggregate by local-hour bucket per position key.
    hourly_net = (
        df.groupby(position_keys + ["hour_bucket"])
        .agg(
            signed_nominal=("signed_nominal", "sum"),
            buy_nominal=("buy_nominal", "sum"),
            sell_nominal=("sell_nominal", "sum"),
            trade_count=("signed_nominal", "count"),
        )
        .reset_index()
        .sort_values(by=position_keys + ["hour_bucket"], kind="mergesort")
    )

    # 4) Rebuild nominal position path from hourly net nominal flow.
    hourly_net["cumulative_nominal"] = hourly_net.groupby(position_keys)[
        "signed_nominal"
    ].cumsum()
    hourly_net["execDate"] = hourly_net["hour_bucket"].apply(
        lambda ts: ts.date() if ts is not None else None
    )

    # 5) Evaluate leakage per exec-date and position key.
    # Build a mapping from underlyingId to assetName for use in reports and plots.
    # Use the most frequent assetName per underlyingId to handle any inconsistencies.
    if "assetName" in df.columns:
        asset_name_map = (
            df.groupby("underlyingId")["assetName"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
            .to_dict()
        )
    else:
        asset_name_map = {}

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

        sod_nominal = group_df["cumulative_nominal"].iloc[0]
        eod_nominal = group_df["cumulative_nominal"].iloc[-1]
        # Nominal entering the day before any first-hour trading.
        prior_eod_nominal = sod_nominal - group_df["signed_nominal"].iloc[0]

        max_exposure = group_df["cumulative_nominal"].abs().max()
        sod_exposure = abs(sod_nominal)
        prior_eod_exposure = abs(prior_eod_nominal)
        eod_exposure = abs(eod_nominal)
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

        # Raw max nominal (without abs) for the baseline deviation metric.
        max_nominal_raw = group_df["cumulative_nominal"].max()
        baseline_deviation_ratio = (
            (baseline_eod_exposure - max_nominal_raw) / (baseline_eod_exposure + 1e-9)
        )

        summary_data.append(
            {
                "ExecDate": exec_date,
                "Portfolio": port,
                "Underlying": und,
                "AssetName": asset_name_map.get(und, str(und)),
                "Maturity": mat,
                "Currency": ccy,
                "Bin_Count": bin_count,
                "Prior_EOD_Nominal": prior_eod_nominal,
                "SOD_Nominal": sod_nominal,
                "EOD_Nominal": eod_nominal,
                "Max_Intraday_Nominal": max_exposure,
                "Leakage_Gap": leakage_gap,
                "Max_to_EOD_Ratio": max_to_eod_ratio,
                "Max_to_Prior_EOD_Ratio": max_to_prior_eod_ratio,
                "Max_to_Baseline_EOD_Ratio": max_to_baseline_eod_ratio,
                "Baseline_Deviation_Ratio": baseline_deviation_ratio,
                "Leakage_Detected": is_leakage,
            }
        )

    results_df = pd.DataFrame(summary_data)

    # 6) Plot only the highest-ranked flagged leakage groups.
    metric_candidates = {"Leakage_Gap", "Max_Intraday_Nominal", "Max_to_EOD_Ratio", "Max_to_Prior_EOD_Ratio", "Max_to_Baseline_EOD_Ratio", "Baseline_Deviation_Ratio"}
    if plot_metric not in metric_candidates:
        raise ValueError(
            f"Invalid plot_metric '{plot_metric}'. Use one of: {sorted(metric_candidates)}"
        )

    leakage_df = results_df[results_df["Leakage_Detected"]].copy()
    plotted_count = 0

    if not leakage_df.empty and plot_top_pct > 0:
        # Apply exposure threshold filter before ranking.
        if plot_exposure_threshold is not None:
            leakage_df = leakage_df[
                leakage_df["Max_Intraday_Nominal"] >= plot_exposure_threshold
            ]
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
            sod_nominal = group_df["cumulative_nominal"].iloc[0]
            eod_nominal = group_df["cumulative_nominal"].iloc[-1]
            max_exposure = group_df["cumulative_nominal"].abs().max()

            local_tz_name = _currency_to_timezone(row["Currency"]) or "UTC"

            # Build a complete hourly range (fill hours with no trades).
            all_hours = pd.date_range(start=sod_time, end=eod_time, freq="h")
            full_df = (
                pd.DataFrame({"hour_bucket": all_hours})
                .merge(group_df, on="hour_bucket", how="left")
            )
            full_df["signed_nominal"] = full_df["signed_nominal"].fillna(0.0)
            full_df["buy_nominal"] = full_df["buy_nominal"].fillna(0.0)
            full_df["sell_nominal"] = full_df["sell_nominal"].fillna(0.0)
            full_df["trade_count"] = full_df["trade_count"].fillna(0).astype(int)
            # Forward-fill cumulative nominal so gap hours carry the last position.
            full_df["cumulative_nominal"] = full_df["cumulative_nominal"].ffill()
            # If leading hours had no trades, back-fill with SOD position.
            full_df["cumulative_nominal"] = full_df["cumulative_nominal"].bfill()

            hour_bin_starts = full_df["hour_bucket"]
            hour_bin_centers = hour_bin_starts + timedelta(minutes=30)
            sod_time_center = sod_time + timedelta(minutes=30)
            eod_time_center = eod_time + timedelta(minutes=30)

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(13, 10), sharex=True,
                gridspec_kw={"height_ratios": [1.2, 1]},
            )

            # ── Chart 1: Cumulative Position (bars) + SOD→EOD net flow line ──
            bar_width = timedelta(minutes=50)
            ax1.bar(
                hour_bin_starts,
                full_df["cumulative_nominal"],
                width=bar_width,
                align="edge",
                alpha=0.65,
                color="#1f77b4",
                edgecolor="#0d4a8a",
                linewidth=0.5,
                label="Cumulative Position (EUR)",
            )

            # SOD → EOD net flow line.
            ax1.plot(
                [sod_time_center, eod_time_center],
                [sod_nominal, eod_nominal],
                color="red",
                linestyle="--",
                linewidth=3,
                marker="o",
                markersize=6,
                label="Net Flow (SOD to EOD)",
            )

            ax1.set_ylabel("Nominal (EUR)")
            ax1.axhline(0, color="black", linewidth=0.5)
            ax1.legend(loc="upper left", fontsize=8)
            ax1.grid(axis="y", linestyle=":", alpha=0.5)
            ax1.set_title(
                f"Intraday Risk Leakage – Cumulative Position\n"
                f"{row['AssetName']} [{row['Underlying']}] ({row['Currency']}) | {row['Portfolio']} | {row['ExecDate']}\n"
                f"Peak: {max_exposure:,.0f} EUR | EOD: {eod_nominal:,.0f} EUR | {plot_metric}: {row[plot_metric]:,.2f}",
                fontsize=11,
            )

            # ── Chart 2: Buy/Sell Nominal (bars) + Trade Count (line) ──
            nom_bar_width = timedelta(minutes=25)
            sell_bar_offset = timedelta(minutes=25)

            ax2.bar(
                hour_bin_starts,
                full_df["buy_nominal"],
                width=nom_bar_width,
                align="edge",
                alpha=0.65,
                color="#2ca02c",
                edgecolor="#1a6e1a",
                linewidth=0.5,
                label="Buy Nominal (EUR)",
            )
            ax2.bar(
                hour_bin_starts + sell_bar_offset,
                -full_df["sell_nominal"],
                width=nom_bar_width,
                align="edge",
                alpha=0.65,
                color="#d62728",
                edgecolor="#8b1a1a",
                linewidth=0.5,
                label="Sell Nominal (EUR)",
            )
            ax2.set_ylabel("Nominal (EUR)")
            ax2.axhline(0, color="black", linewidth=0.5)

            # Trade count as a line on secondary y-axis.
            ax2b = ax2.twinx()
            ax2b.plot(
                hour_bin_centers,
                full_df["trade_count"],
                color="#ff7f0e",
                linewidth=1.5,
                marker="o",
                markersize=4,
                label="Trade Count",
            )
            ax2b.set_ylabel("Trade Count", color="#ff7f0e")
            ax2b.tick_params(axis="y", labelcolor="#ff7f0e")

            ax2.set_title("Buy / Sell Nominal & Trade Count", fontsize=10)

            # Combine legends from both axes of Chart 2.
            lines_2a, labels_2a = ax2.get_legend_handles_labels()
            lines_2b, labels_2b = ax2b.get_legend_handles_labels()
            ax2.legend(
                lines_2a + lines_2b, labels_2a + labels_2b,
                loc="upper left", fontsize=8,
            )
            ax2.grid(axis="y", linestyle=":", alpha=0.5)

            # Shared x-axis formatting.
            x_start = hour_bin_starts.iloc[0]
            x_end = hour_bin_starts.iloc[-1] + timedelta(hours=1)
            ax2.set_xlim(x_start, x_end)
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax2.set_xlabel(f"Hour ({local_tz_name})")
            ax2.tick_params(axis="x", labelrotation=45)

            safe_name = (
                f"{row['ExecDate']}_{row['Portfolio']}_{row['Underlying']}_{row['Currency']}".replace(
                    ":", ""
                ).replace("/", "-")
            )
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"Leakage_{safe_name}.png"))
            plt.close()
            plotted_count += 1

    # 7) Map flagged leakage cases back to the original trade-level DataFrame.
    leakage_summary = results_df[results_df["Leakage_Detected"]].copy()

    # Enrich the working df with execDate (already computed above).

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
    )[[
        "execDate", "portfolioId", "underlyingId", "maturity", "underlyingCurrency",
        "Prior_EOD_Nominal", "SOD_Nominal", "EOD_Nominal",
        "Max_Intraday_Nominal", "Leakage_Gap", "Max_to_EOD_Ratio",
        "Max_to_Prior_EOD_Ratio", "Max_to_Baseline_EOD_Ratio",
        "Baseline_Deviation_Ratio",
    ]]

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

    # Build hourly buy/sell nominal summary for flagged leakage groups.
    hourly_flagged_df = (
        hourly_net.merge(
            leakage_flag_keys,
            on=["execDate", "portfolioId", "underlyingId", "maturity", "underlyingCurrency"],
            how="inner",
        )
        [[
            "execDate", "portfolioId", "underlyingId", "maturity", "underlyingCurrency",
            "hour_bucket", "buy_nominal", "sell_nominal", "signed_nominal", "trade_count",
        ]]
        .rename(columns={
            "execDate": "ExecDate",
            "portfolioId": "Portfolio",
            "underlyingId": "Underlying",
            "maturity": "Maturity",
            "underlyingCurrency": "Currency",
            "hour_bucket": "Hour_Bucket",
            "buy_nominal": "Sum_Buy_Nominal",
            "sell_nominal": "Sum_Sell_Nominal",
            "signed_nominal": "Net_Nominal",
            "trade_count": "Trade_Count",
        })
        .sort_values(["ExecDate", "Portfolio", "Underlying", "Maturity", "Currency", "Hour_Bucket"])
        .reset_index(drop=True)
    )

    # 8) Export reports.
    csv_path = os.path.join(output_folder, "Full_Leakage_Report_Continuous.csv")
    results_df.to_csv(csv_path, index=False)

    flagged_trades_csv = os.path.join(output_folder, "Leakage_Flagged_Trades.csv")
    flagged_trades_df.to_csv(flagged_trades_csv, index=False)

    hourly_flagged_csv = os.path.join(output_folder, "Leakage_Flagged_Trades_Hourly.csv")
    hourly_flagged_df.to_csv(hourly_flagged_csv, index=False)

    # 9) Write auditor report to text file.
    audit_report_path = os.path.join(output_folder, "Audit_Report.txt")
    _write_audit_report(results_df, flagged_trades_df, audit_report_path, hourly_flagged_df)

    print(f"Plots Generated              : {plotted_count}")
    print(f"Full Report                  : {csv_path}")
    print(f"Flagged Trades CSV           : {flagged_trades_csv}")
    print(f"Flagged Trades Hourly CSV    : {hourly_flagged_csv}")
    print(f"Audit Report TXT             : {audit_report_path}")

    return results_df, flagged_trades_df, hourly_flagged_df


if __name__ == "__main__":
    df_trades = pd.read_csv("output/synthetic_trading_data.csv")
    results, flagged, hourly_flagged = analyze_intraday_leakage_continuous(
        df_trades,
        currency_rates_path="output/currency_rates.xlsx",
        plot_top_pct=5,
        plot_metric="Max_to_Baseline_EOD_Ratio",
        max_plots=20,
        plot_exposure_threshold=10_000_000,
    )
