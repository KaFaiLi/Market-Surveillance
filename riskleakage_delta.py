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
    portfolio_results_df: pd.DataFrame = None,
) -> None:
    """Write a formatted auditor-ready report to *report_path* (plain text).

    Sections
    --------
    1. Surveillance Overview – headline counts and rates.
    2. Leakage Metric Statistics – min/max/mean/median for key risk measures.
    3. Flagged Cases Detail – one row per leakage group, sorted by Delta_Leakage_Gap.
    4. Flagged Trades Summary – trade count and gross signed qty per leakage group.
    5. Portfolio Delta Exposure Summary – portfolio-level delta leakage detail.
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
            "Prior_EOD_Delta_Exposure",
            "SOD_Delta_Exposure",
            "EOD_Delta_Exposure",
            "Max_Intraday_Delta_Exposure",
            "Delta_Leakage_Gap",
            "Delta_Max_to_EOD_Ratio",
            "Delta_Max_to_Prior_EOD_Ratio",
            "Delta_Max_to_Baseline_EOD_Ratio",
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
        w("  FLAGGED CASES DETAIL  (sorted by Delta_Leakage_Gap descending)")
        w(sep)

        sorted_leakage = leakage_df.sort_values("Delta_Leakage_Gap", ascending=False).reset_index(drop=True)
        t3 = PrettyTable()
        t3.field_names = [
            "#", "ExecDate", "Portfolio", "Underlying", "Maturity", "Currency",
            "Pre-Market", "SOD_Pos", "EOD_Pos", "Max_Intraday", "Leakage_Gap",
            "SOD_Delta", "EOD_Delta", "Max_Delta", "Delta_Gap",
        ]
        for col in ["Pre-Market", "SOD_Pos", "EOD_Pos", "Max_Intraday", "Leakage_Gap",
                     "SOD_Delta", "EOD_Delta", "Max_Delta", "Delta_Gap"]:
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
                f"{row['SOD_Delta_Exposure']:>14,.0f}",
                f"{row['EOD_Delta_Exposure']:>14,.0f}",
                f"{row['Max_Intraday_Delta_Exposure']:>14,.0f}",
                f"{row['Delta_Leakage_Gap']:>14,.0f}",
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

        # ------------------------------------------------------------------
        # Section 5: Portfolio-Level Delta Exposure Summary
        # ------------------------------------------------------------------
        if portfolio_results_df is not None and not portfolio_results_df.empty:
            pf_leakage = portfolio_results_df[portfolio_results_df["Leakage_Detected"]].copy()
            w(f"\n{sep}")
            w("  PORTFOLIO-LEVEL DELTA EXPOSURE SUMMARY")
            w(sep)

            t5 = PrettyTable()
            t5.field_names = [
                "#", "ExecDate", "Portfolio", "Bins",
                "Prior_EOD_Δ", "SOD_Δ", "EOD_Δ", "Max_Δ",
                "Δ_Gap", "Δ_Max/EOD", "Leakage?",
            ]
            for col in ["Prior_EOD_Δ", "SOD_Δ", "EOD_Δ", "Max_Δ",
                         "Δ_Gap", "Δ_Max/EOD"]:
                t5.align[col] = "r"

            display_df = portfolio_results_df.sort_values(
                "Delta_Leakage_Gap", ascending=False
            ).reset_index(drop=True)

            for i, prow in display_df.iterrows():
                t5.add_row([
                    i + 1,
                    prow["ExecDate"],
                    prow["Portfolio"],
                    prow["Bin_Count"],
                    f"{prow['Prior_EOD_Delta_Exposure']:>14,.0f}",
                    f"{prow['SOD_Delta_Exposure']:>14,.0f}",
                    f"{prow['EOD_Delta_Exposure']:>14,.0f}",
                    f"{prow['Max_Intraday_Delta_Exposure']:>14,.0f}",
                    f"{prow['Delta_Leakage_Gap']:>14,.0f}",
                    f"{prow['Delta_Max_to_EOD_Ratio']:>10,.4f}",
                    "YES" if prow["Leakage_Detected"] else "no",
                ])
            w(t5)

            if not pf_leakage.empty:
                w(f"\n  Portfolio leakage detected: {len(pf_leakage)} of "
                  f"{len(portfolio_results_df)} portfolio-date groups.")
            else:
                w("  No portfolio-level leakage detected.")

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
    debug_sorting=False,
    expected_start_hour=9,
):
    """Analyze intraday position leakage and delta exposure from execution-level trades.

    Computes EUR-denominated delta exposure for each trade based on deal type:
      - FUT: signed_qty × price × futurePointValue × FX rate to EUR
      - SHA: beta × price × signed_qty × FX rate to EUR (beta=1.0 placeholder)

    Leakage detection uses delta exposure as the primary signal.  Position-based
    metrics are retained as informational columns.  Portfolio-level delta exposure
    is aggregated across all underlyings per portfolio per hour.

    Parameters:
    df: Input trades DataFrame. Expected columns include `execTime`, `way`,
        `quantity`, `portfolioId`, `underlyingId`, `maturity`,
        `underlyingCurrency`, `dealType`, `futurePointValue`, and `premium`.
    output_folder: Folder for report CSV and leakage plots.
    plot_top_pct: Percent (0-100) of flagged leakage groups to plot.
    plot_metric: Ranking metric for plot selection. Supports both position-based
        metrics (e.g. `Leakage_Gap`) and delta-based metrics
        (e.g. `Delta_Leakage_Gap`, `Delta_Max_to_Baseline_EOD_Ratio`).
    max_plots: Optional hard cap on the number of plots to generate.
    currency_rates_path: Path to FX rates workbook used to convert premium
        into EUR notionals by date/currency.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        - results_df: one row per grouped exec-date/position key with leakage
          and delta exposure metrics.
        - flagged_trades_df: original trade rows that belong to flagged leakage
          groups, enriched with leakage and delta exposure metrics.
        - portfolio_results_df: one row per portfolio × exec-date with
          portfolio-level delta exposure metrics and leakage flags.
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

    df["premium_numeric"] = pd.to_numeric(df["premium"], errors="coerce")
    df["premium_eur"] = df["premium_numeric"] * df["rate_to_eur"]
    df["futurePointValue_numeric"] = pd.to_numeric(
        df.get("futurePointValue", pd.Series(dtype="float64", index=df.index)),
        errors="coerce",
    ).fillna(1.0)
    df["dealType_upper"] = df["dealType"].astype("string").str.upper()

    # Beta placeholder – hardcoded to 1.0 for SHA; replace with external beta dataset.
    df["beta"] = 1.0

    # 2) Signed quantity.
    df["signed_qty"] = np.where(
        df["way"].str.upper() == "BUY", df["quantity"], -df["quantity"]
    )
    df["signed_qty_fut"] = np.where(df["dealType_upper"] == "FUT", df["signed_qty"], 0.0)
    df["signed_qty_sha"] = np.where(df["dealType_upper"] == "SHA", df["signed_qty"], 0.0)

    # Per-trade delta notional (EUR):
    #   FUT: signed_qty × price × futurePointValue × FX
    #   SHA: beta × price × signed_qty × FX  (beta = 1.0 placeholder)
    df["delta_notional"] = np.where(
        df["dealType_upper"] == "FUT",
        df["signed_qty"] * df["premium_numeric"] * df["futurePointValue_numeric"] * df["rate_to_eur"],
        df["beta"] * df["premium_numeric"] * df["signed_qty"] * df["rate_to_eur"],
    )
    df["delta_notional_fut"] = np.where(df["dealType_upper"] == "FUT", df["delta_notional"], 0.0)
    df["delta_notional_sha"] = np.where(df["dealType_upper"] == "SHA", df["delta_notional"], 0.0)

    # 3) Aggregate signed flow by local-hour bucket per position key.
    #    Sort by execution time so 'last' picks the latest traded price per hour.
    df = df.sort_values("execTime_parsed", kind="mergesort").reset_index(drop=True)
    hourly_net = (
        df.groupby(position_keys + ["hour_bucket"])
        .agg(
            signed_qty=("signed_qty", "sum"),
            signed_qty_fut=("signed_qty_fut", "sum"),
            signed_qty_sha=("signed_qty_sha", "sum"),
            delta_notional=("delta_notional", "sum"),
            delta_notional_fut=("delta_notional_fut", "sum"),
            delta_notional_sha=("delta_notional_sha", "sum"),
            latest_price=("premium_numeric", "last"),
            latest_fpv=("futurePointValue_numeric", "last"),
            rate_to_eur=("rate_to_eur", "first"),
            dealType=("dealType_upper", "first"),
        )
        .reset_index()
        .sort_values(by=position_keys + ["hour_bucket"], kind="mergesort")
    )

    # 4) Rebuild position path from hourly net flow.
    hourly_net["cumulative_pos"] = hourly_net.groupby(position_keys)[
        "signed_qty"
    ].cumsum()
    hourly_net["cumulative_delta_exposure_fut"] = hourly_net.groupby(position_keys)[
        "delta_notional_fut"
    ].cumsum()
    hourly_net["cumulative_delta_exposure_sha"] = hourly_net.groupby(position_keys)[
        "delta_notional_sha"
    ].cumsum()

    # Forward-fill latest price and FPV within each position group across hours.
    hourly_net["latest_price"] = hourly_net.groupby(position_keys)["latest_price"].ffill()
    hourly_net["latest_fpv"] = hourly_net.groupby(position_keys)["latest_fpv"].ffill()

    # 4b) Cumulative delta exposure (snapshot: net position × latest price × multipliers × FX).
    hourly_net["cumulative_delta_exposure"] = np.where(
        hourly_net["dealType"] == "FUT",
        hourly_net["cumulative_pos"] * hourly_net["latest_price"] * hourly_net["latest_fpv"] * hourly_net["rate_to_eur"],
        1.0 * hourly_net["latest_price"] * hourly_net["cumulative_pos"] * hourly_net["rate_to_eur"],
    )

    hourly_net["execDate"] = hourly_net["hour_bucket"].apply(
        lambda ts: ts.date() if ts is not None else None
    )

    # 5) Evaluate leakage per exec-date and position key.
    summary_data = []
    group_frames = {}
    underlying_prior_eod_deltas = {}  # Store per-underlying prior-EOD delta for portfolio agg.
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

        # -- Position-based metrics (informational) --
        sod_pos = group_df["cumulative_pos"].iloc[0]
        eod_pos = group_df["cumulative_pos"].iloc[-1]
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

        # -- Delta exposure metrics (primary leakage signal) --
        sod_delta = group_df["cumulative_delta_exposure"].iloc[0]
        eod_delta = group_df["cumulative_delta_exposure"].iloc[-1]
        prior_eod_delta = sod_delta - group_df["delta_notional"].iloc[0]
        underlying_prior_eod_deltas[group_ids] = prior_eod_delta

        max_delta_exp = group_df["cumulative_delta_exposure"].abs().max()
        sod_delta_abs = abs(sod_delta)
        eod_delta_abs = abs(eod_delta)
        prior_eod_delta_abs = abs(prior_eod_delta)
        baseline_eod_delta = max(sod_delta_abs, eod_delta_abs)
        delta_leakage_gap = max_delta_exp - eod_delta_abs
        delta_max_to_eod = max_delta_exp / (eod_delta_abs + 1e-9)
        delta_max_to_prior_eod = max_delta_exp / (prior_eod_delta_abs + 1e-9)
        delta_max_to_baseline_eod = max_delta_exp / (baseline_eod_delta + 1e-9)

        # Leakage flag based on delta exposure.
        mixed_zero_delta = (prior_eod_delta_abs == 0) ^ (eod_delta_abs == 0)
        enough_intraday_bins = bin_count > 2
        is_leakage = (
            enough_intraday_bins
            and (not mixed_zero_delta)
            and (max_delta_exp > eod_delta_abs)
            and (max_delta_exp > prior_eod_delta_abs)
        )

        summary_data.append(
            {
                "ExecDate": exec_date,
                "Portfolio": port,
                "Underlying": und,
                "Maturity": mat,
                "Currency": ccy,
                "Bin_Count": bin_count,
                # Position metrics (informational).
                "Prior_EOD_Position": prior_eod_pos,
                "SOD_Position": sod_pos,
                "EOD_Position": eod_pos,
                "Max_Intraday_Position": max_exposure,
                "Leakage_Gap": leakage_gap,
                "Max_to_EOD_Ratio": max_to_eod_ratio,
                "Max_to_Prior_EOD_Ratio": max_to_prior_eod_ratio,
                "Max_to_Baseline_EOD_Ratio": max_to_baseline_eod_ratio,
                # Delta exposure metrics (primary).
                "Prior_EOD_Delta_Exposure": prior_eod_delta,
                "SOD_Delta_Exposure": sod_delta,
                "EOD_Delta_Exposure": eod_delta,
                "Max_Intraday_Delta_Exposure": max_delta_exp,
                "Delta_Leakage_Gap": delta_leakage_gap,
                "Delta_Max_to_EOD_Ratio": delta_max_to_eod,
                "Delta_Max_to_Prior_EOD_Ratio": delta_max_to_prior_eod,
                "Delta_Max_to_Baseline_EOD_Ratio": delta_max_to_baseline_eod,
                "Leakage_Detected": is_leakage,
            }
        )

    results_df = pd.DataFrame(summary_data)

    # 5b) Portfolio-level delta exposure aggregation and leakage detection.
    portfolio_data = []
    portfolio_group_frames = {}

    for (exec_date, port), _ in hourly_net.groupby(["execDate", "portfolioId"]):
        mask = (hourly_net["execDate"] == exec_date) & (hourly_net["portfolioId"] == port)
        pdf = hourly_net[mask].copy()
        all_hours = sorted(pdf["hour_bucket"].dropna().unique())
        if len(all_hours) == 0:
            continue

        # Forward-fill each underlying's delta exposure into the full hour grid, then sum.
        underlying_cols = []
        for _, udf in pdf.groupby(["underlyingId", "maturity", "underlyingCurrency"]):
            udf_indexed = udf.set_index("hour_bucket")["cumulative_delta_exposure"]
            udf_reindexed = udf_indexed.reindex(all_hours).ffill().fillna(0)
            underlying_cols.append(udf_reindexed)

        portfolio_exposure = pd.concat(underlying_cols, axis=1).sum(axis=1)
        pf_df = pd.DataFrame({
            "hour_bucket": all_hours,
            "portfolio_delta_exposure": portfolio_exposure.values,
        }).sort_values("hour_bucket")
        portfolio_group_frames[(exec_date, port)] = pf_df.copy()

        pf_bin_count = len(pf_df)
        pf_sod_delta = pf_df["portfolio_delta_exposure"].iloc[0]
        pf_eod_delta = pf_df["portfolio_delta_exposure"].iloc[-1]
        pf_max_delta = pf_df["portfolio_delta_exposure"].abs().max()

        # Portfolio prior-EOD = sum of per-underlying prior-EOD deltas.
        pf_prior_eod_delta = sum(
            underlying_prior_eod_deltas.get((exec_date, port, u, m, c), 0)
            for (u, m, c) in pdf.groupby(
                ["underlyingId", "maturity", "underlyingCurrency"]
            ).groups.keys()
        )

        pf_sod_abs = abs(pf_sod_delta)
        pf_eod_abs = abs(pf_eod_delta)
        pf_prior_eod_abs = abs(pf_prior_eod_delta)
        pf_baseline = max(pf_sod_abs, pf_eod_abs)
        pf_delta_gap = pf_max_delta - pf_eod_abs
        pf_max_to_eod = pf_max_delta / (pf_eod_abs + 1e-9)
        pf_max_to_prior_eod = pf_max_delta / (pf_prior_eod_abs + 1e-9)
        pf_max_to_baseline = pf_max_delta / (pf_baseline + 1e-9)

        pf_mixed_zero = (pf_prior_eod_abs == 0) ^ (pf_eod_abs == 0)
        pf_enough_bins = pf_bin_count > 2
        pf_is_leakage = (
            pf_enough_bins
            and (not pf_mixed_zero)
            and (pf_max_delta > pf_eod_abs)
            and (pf_max_delta > pf_prior_eod_abs)
        )

        portfolio_data.append({
            "ExecDate": exec_date,
            "Portfolio": port,
            "Bin_Count": pf_bin_count,
            "Prior_EOD_Delta_Exposure": pf_prior_eod_delta,
            "SOD_Delta_Exposure": pf_sod_delta,
            "EOD_Delta_Exposure": pf_eod_delta,
            "Max_Intraday_Delta_Exposure": pf_max_delta,
            "Delta_Leakage_Gap": pf_delta_gap,
            "Delta_Max_to_EOD_Ratio": pf_max_to_eod,
            "Delta_Max_to_Prior_EOD_Ratio": pf_max_to_prior_eod,
            "Delta_Max_to_Baseline_EOD_Ratio": pf_max_to_baseline,
            "Leakage_Detected": pf_is_leakage,
        })

    portfolio_results_df = pd.DataFrame(portfolio_data)

    # 6) Plot only the highest-ranked flagged leakage groups.
    metric_candidates = {
        "Leakage_Gap", "Max_Intraday_Position", "Max_to_EOD_Ratio",
        "Max_to_Prior_EOD_Ratio", "Max_to_Baseline_EOD_Ratio",
        "Delta_Leakage_Gap", "Max_Intraday_Delta_Exposure",
        "Delta_Max_to_EOD_Ratio", "Delta_Max_to_Prior_EOD_Ratio",
        "Delta_Max_to_Baseline_EOD_Ratio",
    }
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
            max_delta = group_df["cumulative_delta_exposure"].abs().max()
            eod_delta = group_df["cumulative_delta_exposure"].iloc[-1]

            fig, ax = plt.subplots(figsize=(12, 6))

            hour_bin_starts = group_df["hour_bucket"]
            hour_bin_centers = hour_bin_starts + timedelta(minutes=30)
            hour_bin_centers_num = mdates.date2num(hour_bin_centers)
            cluster_bar_width_days = (1.0 / 24.0) * 0.35
            sod_time_center = sod_time + timedelta(minutes=30)
            eod_time_center = eod_time + timedelta(minutes=30)
            local_tz_name = _currency_to_timezone(row["Currency"]) or "UTC"

            ax.bar(
                hour_bin_centers_num - (cluster_bar_width_days / 2.0),
                group_df["signed_qty_sha"],
                width=cluster_bar_width_days,
                align="center",
                alpha=0.65,
                edgecolor="#1f1f1f",
                linewidth=0.6,
                color="#6baed6",
                label="Share Qty (SHA)",
            )

            ax.bar(
                hour_bin_centers_num + (cluster_bar_width_days / 2.0),
                group_df["signed_qty_fut"],
                width=cluster_bar_width_days,
                align="center",
                alpha=0.65,
                edgecolor="#1f1f1f",
                linewidth=0.6,
                color="#9ecae1",
                label="Future Qty (FUT)",
            )

            ax.plot(
                [sod_time_center, eod_time_center],
                [sod_pos, eod_pos],
                color="red",
                linestyle="--",
                linewidth=2,
                marker="o",
                label="Position SOD→EOD",
            )

            ax.set_ylabel("Position (qty)", color="#1f77b4")
            ax.tick_params(axis="y", labelcolor="#1f77b4")

            # Secondary axis: delta exposure (EUR).
            ax2 = ax.twinx()
            ax2.plot(
                hour_bin_centers,
                group_df["cumulative_delta_exposure"],
                color="#d62728",
                linewidth=2.5,
                marker="s",
                markersize=4,
                label="Total Delta Exposure (EUR)",
            )
            ax2.plot(
                hour_bin_centers,
                group_df["cumulative_delta_exposure_fut"],
                color="#ff7f0e",
                linewidth=1.8,
                marker="^",
                markersize=3,
                label="Future Delta Exposure (EUR)",
            )
            ax2.plot(
                hour_bin_centers,
                group_df["cumulative_delta_exposure_sha"],
                color="#9467bd",
                linewidth=1.8,
                marker="v",
                markersize=3,
                label="Share Delta Exposure (EUR)",
            )
            ax2.set_ylabel("Delta Exposure (EUR)", color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728")

            ax.set_title(
                f"Intraday Risk Leakage – Position & Delta Exposure\n"
                f"{row['Underlying']} ({row['Currency']}) | {row['Portfolio']} | {row['ExecDate']}\n"
                f"Peak Pos: {max_exposure:,.0f} | Peak Δ-Exp: {max_delta:,.0f} | "
                f"EOD Δ-Exp: {eod_delta:,.0f} | {plot_metric}: {row[plot_metric]:,.2f}",
                fontsize=10,
            )

            x_start = group_df["hour_bucket"].iloc[0]
            x_end = group_df["hour_bucket"].iloc[-1] + timedelta(hours=1)
            ax.set_xlim(x_start, x_end)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel(f"Hour ({local_tz_name})")
            ax.tick_params(axis="x", labelrotation=45)

            # Combined legend.
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
            ax.grid(axis="y", linestyle=":", alpha=0.5)

            safe_name = (
                f"{row['ExecDate']}_{row['Portfolio']}_{row['Underlying']}_{row['Currency']}".replace(
                    ":", ""
                ).replace("/", "-")
            )
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"Leakage_{safe_name}.png"))
            plt.close()
            plotted_count += 1

    # 6b) Portfolio-level delta exposure plots.
    pf_leakage_df = portfolio_results_df[portfolio_results_df["Leakage_Detected"]].copy()
    pf_plotted_count = 0

    if not pf_leakage_df.empty and plot_top_pct > 0:
        pf_plot_metric = (
            plot_metric if plot_metric in pf_leakage_df.columns else "Delta_Leakage_Gap"
        )
        pf_leakage_df = pf_leakage_df.sort_values(pf_plot_metric, ascending=False)
        pf_top_n = max(1, int(np.ceil(len(pf_leakage_df) * (plot_top_pct / 100.0))))
        if max_plots is not None:
            pf_top_n = min(pf_top_n, int(max_plots))
        pf_to_plot = pf_leakage_df.head(pf_top_n)

        for _, prow in pf_to_plot.iterrows():
            pf_key = (prow["ExecDate"], prow["Portfolio"])
            pf_df = portfolio_group_frames.get(pf_key)
            if pf_df is None or pf_df.empty:
                continue

            pf_df = pf_df.sort_values("hour_bucket").reset_index(drop=True)
            pf_sod = pf_df["portfolio_delta_exposure"].iloc[0]
            pf_eod = pf_df["portfolio_delta_exposure"].iloc[-1]
            pf_peak = pf_df["portfolio_delta_exposure"].abs().max()

            fig, ax = plt.subplots(figsize=(12, 6))
            h_starts = pf_df["hour_bucket"]
            h_centers = h_starts + timedelta(minutes=30)

            ax.bar(
                h_starts,
                pf_df["portfolio_delta_exposure"],
                width=timedelta(hours=1),
                align="edge",
                alpha=0.85,
                edgecolor="#1f1f1f",
                linewidth=0.6,
                color="#2ca02c",
                label="Portfolio Delta Exposure (EUR)",
            )
            ax.plot(
                [h_centers.iloc[0], h_centers.iloc[-1]],
                [pf_sod, pf_eod],
                color="red", linestyle="--", linewidth=2, marker="o",
                label="SOD→EOD",
            )
            ax.set_title(
                f"Portfolio Delta Exposure – Intraday Path\n"
                f"{prow['Portfolio']} | {prow['ExecDate']}\n"
                f"Peak: {pf_peak:,.0f} EUR | EOD: {pf_eod:,.0f} EUR | "
                f"Gap: {prow['Delta_Leakage_Gap']:,.0f}",
                fontsize=10,
            )
            ax.set_xlim(h_starts.iloc[0], h_starts.iloc[-1] + timedelta(hours=1))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Hour (UTC)")
            ax.set_ylabel("Delta Exposure (EUR)")
            ax.tick_params(axis="x", labelrotation=45)
            ax.legend(fontsize=8)
            ax.grid(axis="y", linestyle=":", alpha=0.5)

            pf_safe = f"{prow['ExecDate']}_{prow['Portfolio']}".replace(":", "").replace("/", "-")
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"Portfolio_Delta_{pf_safe}.png"))
            plt.close()
            pf_plotted_count += 1

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
       "Max_Intraday_Position", "Leakage_Gap", "Max_to_EOD_Ratio",
       "Max_to_Prior_EOD_Ratio", "Max_to_Baseline_EOD_Ratio",
       "Prior_EOD_Delta_Exposure", "SOD_Delta_Exposure", "EOD_Delta_Exposure",
       "Max_Intraday_Delta_Exposure", "Delta_Leakage_Gap",
       "Delta_Max_to_EOD_Ratio", "Delta_Max_to_Prior_EOD_Ratio",
       "Delta_Max_to_Baseline_EOD_Ratio"]]

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

    portfolio_csv = os.path.join(output_folder, "Portfolio_Delta_Exposure_Report.csv")
    portfolio_results_df.to_csv(portfolio_csv, index=False)

    # 9) Write auditor report to text file.
    audit_report_path = os.path.join(output_folder, "Audit_Report.txt")
    _write_audit_report(results_df, flagged_trades_df, audit_report_path, portfolio_results_df)

    print(f"Plots Generated (underlying) : {plotted_count}")
    print(f"Plots Generated (portfolio)  : {pf_plotted_count}")
    print(f"Full Report                  : {csv_path}")
    print(f"Flagged Trades CSV           : {flagged_trades_csv}")
    print(f"Portfolio Delta Report        : {portfolio_csv}")
    print(f"Audit Report TXT             : {audit_report_path}")

    return results_df, flagged_trades_df, portfolio_results_df


if __name__ == "__main__":
    df_trades = pd.read_csv("output/synthetic_trading_data.csv")
    results, flagged, portfolio = analyze_intraday_leakage_continuous(
        df_trades, plot_top_pct=5, plot_metric="Delta_Max_to_Baseline_EOD_Ratio", max_plots=20
    )
