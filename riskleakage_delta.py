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
    portfolio_results_df: pd.DataFrame,
    flagged_trades_df: pd.DataFrame,
    report_path: str,
) -> None:
    """Write a formatted auditor-ready report to *report_path* (plain text).

    Sections
    --------
    1. Surveillance Overview – headline counts and rates.
    2. Delta Leakage Metric Statistics – min/max/mean/median.
    3. Portfolio Flagged Cases Detail – sorted by Delta_Leakage_Gap.
    4. Flagged Trades Summary – trade count and gross signed qty per portfolio.
    """
    leakage_df = portfolio_results_df[portfolio_results_df["Leakage_Detected"]].copy()
    total_groups = len(portfolio_results_df)
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
        w("  INTRADAY DELTA RISK LEAKAGE – AUDIT REPORT")
        w(sep)

        t1 = PrettyTable()
        t1.field_names = ["Metric", "Value"]
        t1.align["Metric"] = "l"
        t1.align["Value"] = "r"
        t1.add_rows([
            ["Total Portfolio-Date Groups",   f"{total_groups:,}"],
            ["Leakage Cases Detected",         f"{total_leakage:,}"],
            ["Leakage Rate",                   f"{leakage_rate:.2f} %"],
            ["Portfolios Affected",            str(leakage_df["Portfolio"].nunique()) if not leakage_df.empty else "0"],
            ["Earliest Leakage Date",          str(leakage_df["ExecDate"].min()) if not leakage_df.empty else "N/A"],
            ["Latest Leakage Date",            str(leakage_df["ExecDate"].max()) if not leakage_df.empty else "N/A"],
            ["Flagged Original Trades",        f"{len(flagged_trades_df):,}"],
        ])
        w(t1)

        if leakage_df.empty:
            w("  No leakage cases – nothing further to report.")
            w(sep)
            return

        # ------------------------------------------------------------------
        # Section 2: Delta Leakage Metric Statistics
        # ------------------------------------------------------------------
        w(f"\n{sep}")
        w("  DELTA LEAKAGE METRIC STATISTICS  (flagged portfolio-dates only)")
        w(sep)

        metrics = [
            "Prior_EOD_Delta",
            "SOD_Delta",
            "EOD_Delta",
            "Max_Intraday_Delta",
            "Delta_Leakage_Gap",
            "Delta_Max_to_EOD_Ratio",
            "Delta_Max_to_Prior_EOD_Ratio",
            "Delta_Max_to_Baseline_Ratio",
        ]
        t2 = PrettyTable()
        t2.field_names = ["Metric", "Min", "Max", "Mean", "Median"]
        for col in ["Min", "Max", "Mean", "Median"]:
            t2.align[col] = "r"
        t2.align["Metric"] = "l"
        for m in metrics:
            if m not in leakage_df.columns:
                continue
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
        # Section 3: Portfolio Flagged Cases Detail
        # ------------------------------------------------------------------
        w(f"\n{sep}")
        w("  PORTFOLIO FLAGGED CASES  (sorted by Delta_Leakage_Gap descending)")
        w(sep)

        display_df = leakage_df.sort_values(
            "Delta_Leakage_Gap", ascending=False
        ).reset_index(drop=True)
        t3 = PrettyTable()
        t3.field_names = [
            "#", "ExecDate", "Portfolio", "Bins",
            "Prior_EOD_Δ", "SOD_Δ", "EOD_Δ", "Max_Δ",
            "Δ_Gap", "Δ_Max/EOD", "Leakage?",
        ]
        for col in ["Prior_EOD_Δ", "SOD_Δ", "EOD_Δ", "Max_Δ",
                     "Δ_Gap", "Δ_Max/EOD"]:
            t3.align[col] = "r"
        for i, prow in display_df.iterrows():
            t3.add_row([
                i + 1,
                prow["ExecDate"],
                prow["Portfolio"],
                prow["Bin_Count"],
                f"{prow['Prior_EOD_Delta']:>14,.0f}",
                f"{prow['SOD_Delta']:>14,.0f}",
                f"{prow['EOD_Delta']:>14,.0f}",
                f"{prow['Max_Intraday_Delta']:>14,.0f}",
                f"{prow['Delta_Leakage_Gap']:>14,.0f}",
                f"{prow['Delta_Max_to_EOD_Ratio']:>10,.4f}",
                "YES" if prow["Leakage_Detected"] else "no",
            ])
        w(t3)

        # ------------------------------------------------------------------
        # Section 4: Flagged Trades Summary
        # ------------------------------------------------------------------
        if not flagged_trades_df.empty:
            w(f"\n{sep}")
            w("  FLAGGED TRADES SUMMARY  (original trades in flagged portfolios)")
            w(sep)

            grp_summary = (
                flagged_trades_df.groupby(["ExecDate", "Portfolio"])
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
                "ExecDate", "Portfolio",
                "Trades", "Gross Buy", "Gross Sell", "Net Qty",
            ]
            for col in ["Trades", "Gross Buy", "Gross Sell", "Net Qty"]:
                t4.align[col] = "r"
            for _, row in grp_summary.iterrows():
                t4.add_row([
                    row["ExecDate"],
                    row["Portfolio"],
                    f"{int(row['Trade_Count']):,}",
                    f"{row['Gross_Buy_Qty']:>14,.0f}",
                    f"{row['Gross_Sell_Qty']:>14,.0f}",
                    f"{row['Net_Qty']:>14,.0f}",
                ])
            w(t4)

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
    plot_metric="Delta_Leakage_Gap",
    max_plots=None,
    debug_sorting=False,
    expected_start_hour=9,
):
    """Analyze intraday portfolio delta nominal exposure leakage.

    For each hourly bucket the function computes:
      1. FUT delta nominal per product (underlyingId × maturity):
           net_qty × latest_premium_EUR × futurePointValue
      2. SHA delta nominal per product (underlyingId):
           net_qty × latest_premium_EUR × beta  (beta = 1.0 placeholder)
      3. Portfolio delta = Σ FUT deltas + Σ SHA deltas

    Leakage detection uses portfolio-level delta nominal exposure.

    Parameters
    ----------
    df : DataFrame
        Input trades with columns: execTime, way, quantity, portfolioId,
        underlyingId, maturity, underlyingCurrency, dealType,
        futurePointValue, premium.
    output_folder : str
    currency_rates_path : str
    plot_top_pct : float  (0–100)
    plot_metric : str
    max_plots : int or None
    debug_sorting : bool
    expected_start_hour : int

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        portfolio_results_df, flagged_trades_df
    """
    os.makedirs(output_folder, exist_ok=True)
    df = df.copy()
    if plot_top_pct < 0 or plot_top_pct > 100:
        raise ValueError("plot_top_pct must be between 0 and 100.")
    if max_plots is not None and max_plots < 0:
        raise ValueError("max_plots must be >= 0 when provided.")

    # ── 1. Parse execution timestamps ──────────────────────────────────
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

    # ── 2. FX rates & numeric fields ───────────────────────────────────
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
    df["dealType_upper"] = df["dealType"].astype("string").str.upper()
    df["beta"] = 1.0  # placeholder – replace with external beta dataset

    # ── 3. Signed quantity ─────────────────────────────────────────────
    df["signed_qty"] = np.where(
        df["way"].str.upper() == "BUY", df["quantity"], -df["quantity"]
    )

    # ── 4. Hourly aggregation per product ──────────────────────────────
    #   FUT product key = (portfolioId, underlyingId, maturity, underlyingCurrency)
    #   SHA product key = (portfolioId, underlyingId, underlyingCurrency)
    df = df.sort_values("execTime_parsed", kind="mergesort").reset_index(drop=True)

    fut_keys = ["portfolioId", "underlyingId", "maturity", "underlyingCurrency"]
    sha_keys = ["portfolioId", "underlyingId", "underlyingCurrency"]

    df_fut = df[df["dealType_upper"] == "FUT"].copy()
    df_sha = df[df["dealType_upper"] == "SHA"].copy()

    # --- FUT hourly aggregation ---
    hourly_fut = pd.DataFrame()
    if not df_fut.empty:
        hourly_fut = (
            df_fut.groupby(fut_keys + ["hour_bucket"])
            .agg(
                signed_qty=("signed_qty", "sum"),
                latest_price=("premium_numeric", "last"),
                latest_fpv=("futurePointValue_numeric", "last"),
                rate_to_eur=("rate_to_eur", "first"),
            )
            .reset_index()
            .sort_values(by=fut_keys + ["hour_bucket"], kind="mergesort")
        )
        hourly_fut["cumulative_pos"] = hourly_fut.groupby(fut_keys)["signed_qty"].cumsum()
        hourly_fut["latest_price"] = hourly_fut.groupby(fut_keys)["latest_price"].ffill()
        hourly_fut["latest_fpv"] = hourly_fut.groupby(fut_keys)["latest_fpv"].ffill()
        hourly_fut["latest_price_eur"] = hourly_fut["latest_price"] * hourly_fut["rate_to_eur"]
        hourly_fut["delta_nominal"] = (
            hourly_fut["cumulative_pos"]
            * hourly_fut["latest_price_eur"]
            * hourly_fut["latest_fpv"]
        )
        hourly_fut["execDate"] = hourly_fut["hour_bucket"].apply(
            lambda ts: ts.date() if ts is not None else None
        )

    # --- SHA hourly aggregation ---
    hourly_sha = pd.DataFrame()
    if not df_sha.empty:
        hourly_sha = (
            df_sha.groupby(sha_keys + ["hour_bucket"])
            .agg(
                signed_qty=("signed_qty", "sum"),
                latest_price=("premium_numeric", "last"),
                rate_to_eur=("rate_to_eur", "first"),
                beta=("beta", "first"),
            )
            .reset_index()
            .sort_values(by=sha_keys + ["hour_bucket"], kind="mergesort")
        )
        hourly_sha["cumulative_pos"] = hourly_sha.groupby(sha_keys)["signed_qty"].cumsum()
        hourly_sha["latest_price"] = hourly_sha.groupby(sha_keys)["latest_price"].ffill()
        hourly_sha["latest_price_eur"] = hourly_sha["latest_price"] * hourly_sha["rate_to_eur"]
        hourly_sha["delta_nominal"] = (
            hourly_sha["cumulative_pos"]
            * hourly_sha["latest_price_eur"]
            * hourly_sha["beta"]
        )
        hourly_sha["execDate"] = hourly_sha["hour_bucket"].apply(
            lambda ts: ts.date() if ts is not None else None
        )

    # ── 5. Portfolio-level delta aggregation per hour ───────────────────
    portfolio_data = []
    portfolio_group_frames = {}

    # Collect all (execDate, portfolioId) combinations.
    all_portfolio_dates = set()
    if not hourly_fut.empty:
        for key in hourly_fut.groupby(["execDate", "portfolioId"]).groups.keys():
            all_portfolio_dates.add(key)
    if not hourly_sha.empty:
        for key in hourly_sha.groupby(["execDate", "portfolioId"]).groups.keys():
            all_portfolio_dates.add(key)

    for exec_date, port in sorted(all_portfolio_dates):
        # Collect all hours across all products in this portfolio-date.
        all_hours_set = set()

        fut_pdf = pd.DataFrame()
        if not hourly_fut.empty:
            fut_mask = (hourly_fut["execDate"] == exec_date) & (hourly_fut["portfolioId"] == port)
            fut_pdf = hourly_fut[fut_mask].copy()
            if not fut_pdf.empty:
                all_hours_set.update(fut_pdf["hour_bucket"].dropna().unique())

        sha_pdf = pd.DataFrame()
        if not hourly_sha.empty:
            sha_mask = (hourly_sha["execDate"] == exec_date) & (hourly_sha["portfolioId"] == port)
            sha_pdf = hourly_sha[sha_mask].copy()
            if not sha_pdf.empty:
                all_hours_set.update(sha_pdf["hour_bucket"].dropna().unique())

        all_hours = sorted(all_hours_set)
        if not all_hours:
            continue

        if debug_sorting:
            first_bucket = all_hours[0]
            if expected_start_hour is not None and first_bucket is not None:
                if first_bucket.hour != expected_start_hour:
                    print(
                        f"Warning: first hour_bucket is not the expected start hour for "
                        f"{exec_date} | {port}. First bucket: {first_bucket}"
                    )

        # Reindex each product's delta & qty onto the full hour grid.
        fut_delta_cols = []
        fut_qty_cols = []
        fut_prior_eod_deltas = []

        if not fut_pdf.empty:
            for _, udf in fut_pdf.groupby(["underlyingId", "maturity", "underlyingCurrency"]):
                udf = udf.sort_values("hour_bucket")
                delta_s = udf.set_index("hour_bucket")["delta_nominal"].reindex(all_hours).ffill().fillna(0)
                qty_s = udf.set_index("hour_bucket")["signed_qty"].reindex(all_hours).fillna(0)
                fut_delta_cols.append(delta_s)
                fut_qty_cols.append(qty_s)
                # Prior-EOD delta for this FUT product.
                first_row = udf.iloc[0]
                prior_pos = first_row["cumulative_pos"] - first_row["signed_qty"]
                prior_delta = prior_pos * first_row["latest_price_eur"] * first_row["latest_fpv"]
                fut_prior_eod_deltas.append(prior_delta)

        sha_delta_cols = []
        sha_qty_cols = []
        sha_prior_eod_deltas = []

        if not sha_pdf.empty:
            for _, udf in sha_pdf.groupby(["underlyingId", "underlyingCurrency"]):
                udf = udf.sort_values("hour_bucket")
                delta_s = udf.set_index("hour_bucket")["delta_nominal"].reindex(all_hours).ffill().fillna(0)
                qty_s = udf.set_index("hour_bucket")["signed_qty"].reindex(all_hours).fillna(0)
                sha_delta_cols.append(delta_s)
                sha_qty_cols.append(qty_s)
                first_row = udf.iloc[0]
                prior_pos = first_row["cumulative_pos"] - first_row["signed_qty"]
                prior_delta = prior_pos * first_row["latest_price_eur"] * first_row["beta"]
                sha_prior_eod_deltas.append(prior_delta)

        # Sum across products per deal type.
        total_delta_fut = (
            pd.concat(fut_delta_cols, axis=1).sum(axis=1)
            if fut_delta_cols
            else pd.Series(0.0, index=all_hours)
        )
        total_delta_sha = (
            pd.concat(sha_delta_cols, axis=1).sum(axis=1)
            if sha_delta_cols
            else pd.Series(0.0, index=all_hours)
        )
        total_qty_fut = (
            pd.concat(fut_qty_cols, axis=1).sum(axis=1)
            if fut_qty_cols
            else pd.Series(0.0, index=all_hours)
        )
        total_qty_sha = (
            pd.concat(sha_qty_cols, axis=1).sum(axis=1)
            if sha_qty_cols
            else pd.Series(0.0, index=all_hours)
        )
        portfolio_delta = total_delta_fut + total_delta_sha

        pf_df = pd.DataFrame({
            "hour_bucket": all_hours,
            "delta_fut": total_delta_fut.values,
            "delta_sha": total_delta_sha.values,
            "qty_fut": total_qty_fut.values,
            "qty_sha": total_qty_sha.values,
            "portfolio_delta": portfolio_delta.values,
        }).sort_values("hour_bucket").reset_index(drop=True)

        portfolio_group_frames[(exec_date, port)] = pf_df.copy()

        # ── Metrics ────────────────────────────────────────────────────
        bin_count = len(pf_df)
        sod_delta = pf_df["portfolio_delta"].iloc[0]
        eod_delta = pf_df["portfolio_delta"].iloc[-1]
        max_delta = pf_df["portfolio_delta"].abs().max()
        prior_eod_delta = sum(fut_prior_eod_deltas) + sum(sha_prior_eod_deltas)

        sod_abs = abs(sod_delta)
        eod_abs = abs(eod_delta)
        prior_eod_abs = abs(prior_eod_delta)
        baseline = max(sod_abs, eod_abs)
        delta_gap = max_delta - eod_abs
        max_to_eod = max_delta / (eod_abs + 1e-9)
        max_to_prior_eod = max_delta / (prior_eod_abs + 1e-9)
        max_to_baseline = max_delta / (baseline + 1e-9)

        mixed_zero = (prior_eod_abs == 0) ^ (eod_abs == 0)
        enough_bins = bin_count > 2
        is_leakage = (
            enough_bins
            and (not mixed_zero)
            and (max_delta > eod_abs)
            and (max_delta > prior_eod_abs)
        )

        portfolio_data.append({
            "ExecDate": exec_date,
            "Portfolio": port,
            "Bin_Count": bin_count,
            "Prior_EOD_Delta": prior_eod_delta,
            "SOD_Delta": sod_delta,
            "EOD_Delta": eod_delta,
            "Max_Intraday_Delta": max_delta,
            "Delta_Leakage_Gap": delta_gap,
            "Delta_Max_to_EOD_Ratio": max_to_eod,
            "Delta_Max_to_Prior_EOD_Ratio": max_to_prior_eod,
            "Delta_Max_to_Baseline_Ratio": max_to_baseline,
            "Leakage_Detected": is_leakage,
        })

    portfolio_results_df = pd.DataFrame(portfolio_data)

    # ── 6. Plot flagged portfolio-dates ────────────────────────────────
    metric_candidates = {
        "Delta_Leakage_Gap",
        "Max_Intraday_Delta",
        "Delta_Max_to_EOD_Ratio",
        "Delta_Max_to_Prior_EOD_Ratio",
        "Delta_Max_to_Baseline_Ratio",
    }
    if plot_metric not in metric_candidates:
        raise ValueError(
            f"Invalid plot_metric '{plot_metric}'. "
            f"Use one of: {sorted(metric_candidates)}"
        )

    pf_leakage_df = portfolio_results_df[
        portfolio_results_df["Leakage_Detected"]
    ].copy()
    plotted_count = 0

    if not pf_leakage_df.empty and plot_top_pct > 0:
        pf_leakage_df = pf_leakage_df.sort_values(plot_metric, ascending=False)
        top_n = max(1, int(np.ceil(len(pf_leakage_df) * (plot_top_pct / 100.0))))
        if max_plots is not None:
            top_n = min(top_n, int(max_plots))
        to_plot = pf_leakage_df.head(top_n)

        for _, prow in to_plot.iterrows():
            pf_key = (prow["ExecDate"], prow["Portfolio"])
            pf_df = portfolio_group_frames.get(pf_key)
            if pf_df is None or pf_df.empty:
                continue

            pf_df = pf_df.sort_values("hour_bucket").reset_index(drop=True)
            h_starts = pf_df["hour_bucket"]
            h_centers = h_starts + timedelta(minutes=30)
            h_centers_num = mdates.date2num(h_centers)
            bar_width = (1.0 / 24.0) * 0.35

            pf_peak = pf_df["portfolio_delta"].abs().max()
            pf_eod = pf_df["portfolio_delta"].iloc[-1]

            fig, ax = plt.subplots(figsize=(12, 6))

            # Bars: hourly net quantity by deal type.
            ax.bar(
                h_centers_num - (bar_width / 2.0),
                pf_df["qty_sha"],
                width=bar_width,
                align="center",
                alpha=0.65,
                edgecolor="#1f1f1f",
                linewidth=0.6,
                color="#1f77b4",
                label="Share Qty (SHA)",
            )
            ax.bar(
                h_centers_num + (bar_width / 2.0),
                pf_df["qty_fut"],
                width=bar_width,
                align="center",
                alpha=0.65,
                edgecolor="#1f1f1f",
                linewidth=0.6,
                color="#ff7f0e",
                label="Future Qty (FUT)",
            )
            ax.set_ylabel("Net Quantity (contracts / shares)", color="#333333")
            ax.tick_params(axis="y", labelcolor="#333333")

            # Lines: delta exposure on secondary axis.
            ax2 = ax.twinx()
            ax2.plot(
                h_centers,
                pf_df["delta_sha"],
                color="#1f77b4",
                linewidth=1.8,
                marker="v",
                markersize=3,
                label="SHA Delta (EUR)",
            )
            ax2.plot(
                h_centers,
                pf_df["delta_fut"],
                color="#ff7f0e",
                linewidth=1.8,
                marker="^",
                markersize=3,
                label="FUT Delta (EUR)",
            )
            ax2.plot(
                h_centers,
                pf_df["portfolio_delta"],
                color="#d62728",
                linewidth=2.5,
                marker="s",
                markersize=4,
                label="Portfolio Delta (EUR)",
            )
            ax2.set_ylabel("Delta Nominal (EUR)", color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728")

            ax.set_title(
                f"Portfolio Delta Nominal Exposure – Intraday Path\n"
                f"{prow['Portfolio']} | {prow['ExecDate']}\n"
                f"Peak: {pf_peak:,.0f} EUR | EOD: {pf_eod:,.0f} EUR | "
                f"{plot_metric}: {prow[plot_metric]:,.2f}",
                fontsize=10,
            )

            x_start = h_starts.iloc[0]
            x_end = h_starts.iloc[-1] + timedelta(hours=1)
            ax.set_xlim(x_start, x_end)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Hour (local)")
            ax.tick_params(axis="x", labelrotation=45)

            # Combined legend.
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(
                lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8
            )
            ax.grid(axis="y", linestyle=":", alpha=0.5)

            safe_name = (
                f"{prow['ExecDate']}_{prow['Portfolio']}"
                .replace(":", "")
                .replace("/", "-")
            )
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"Portfolio_Delta_{safe_name}.png"))
            plt.close()
            plotted_count += 1

    # ── 7. Map flagged trades ──────────────────────────────────────────
    leakage_summary = portfolio_results_df[
        portfolio_results_df["Leakage_Detected"]
    ].copy()

    df["execDate"] = df["execTime_parsed"].apply(
        lambda ts: ts.date() if ts is not None else None
    )

    flag_keys = leakage_summary.rename(columns={
        "ExecDate": "execDate",
        "Portfolio": "portfolioId",
    })[["execDate", "portfolioId"]].copy()
    flag_keys["_flagged"] = True

    df_flagged = df.merge(
        flag_keys, on=["execDate", "portfolioId"], how="left"
    )
    flagged_trades_df = df_flagged[
        df_flagged["_flagged"] == True
    ].drop(columns=["_flagged"]).copy()

    # Enrich flagged trades with portfolio leakage metrics.
    leakage_merge = leakage_summary.rename(columns={
        "ExecDate": "execDate",
        "Portfolio": "portfolioId",
    })[[
        "execDate", "portfolioId",
        "Prior_EOD_Delta", "SOD_Delta", "EOD_Delta", "Max_Intraday_Delta",
        "Delta_Leakage_Gap", "Delta_Max_to_EOD_Ratio",
        "Delta_Max_to_Prior_EOD_Ratio", "Delta_Max_to_Baseline_Ratio",
    ]]
    flagged_trades_df = flagged_trades_df.merge(
        leakage_merge, on=["execDate", "portfolioId"], how="left",
    )
    flagged_trades_df = flagged_trades_df.rename(columns={
        "execDate": "ExecDate",
        "portfolioId": "Portfolio",
        "underlyingId": "Underlying",
        "maturity": "Maturity",
        "underlyingCurrency": "Currency",
    })

    # ── 8. Export ──────────────────────────────────────────────────────
    portfolio_csv = os.path.join(output_folder, "Portfolio_Delta_Exposure_Report.csv")
    portfolio_results_df.to_csv(portfolio_csv, index=False)

    flagged_csv = os.path.join(output_folder, "Leakage_Flagged_Trades.csv")
    flagged_trades_df.to_csv(flagged_csv, index=False)

    audit_path = os.path.join(output_folder, "Audit_Report.txt")
    _write_audit_report(portfolio_results_df, flagged_trades_df, audit_path)

    print(f"Plots Generated              : {plotted_count}")
    print(f"Portfolio Delta Report        : {portfolio_csv}")
    print(f"Flagged Trades CSV           : {flagged_csv}")
    print(f"Audit Report TXT             : {audit_path}")

    return portfolio_results_df, flagged_trades_df


if __name__ == "__main__":
    df_trades = pd.read_csv("output/synthetic_trading_data.csv")
    portfolio, flagged = analyze_intraday_leakage_continuous(
        df_trades,
        plot_top_pct=5,
        plot_metric="Delta_Max_to_Baseline_Ratio",
        max_plots=20,
    )
