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
                flagged_trades_df.groupby(["execDate", "portfolioId"])
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
                    row["execDate"],
                    row["portfolioId"],
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


def _load_initial_positions(path):
    """Load prior-EOD positions and return FUT and SHA lookup DataFrames.

    Matching keys
    -------------
    FUT : (portfolioId, assetName, maturity)
          assetName = position-file futureContractId
                      (falls back to stockId if futureContractId is absent)
          maturity  = as-is (e.g. "2024-06-21+01:00[Europe/Paris]")
    SHA : (portfolioId, assetName)
          assetName = position-file stockId

    Rows with position == 0 are excluded (no impact on seeding).

    Returns
    -------
    fut_pos : DataFrame  [portfolioId, assetName, maturity, initial_pos]
    sha_pos : DataFrame  [portfolioId, assetName, initial_pos]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Initial positions file not found: {path}")

    pos_df = pd.read_parquet(path)

    # Cast key join columns to str so they match the string-typed trade columns.
    for _col in ["portfolioId", "stockId", "futureContractId", "maturity", "position_category"]:
        if _col in pos_df.columns:
            pos_df[_col] = pos_df[_col].astype(str)

    # Convert position to numeric and drop zero-position rows.
    pos_df["initial_pos"] = pd.to_numeric(pos_df["position"], errors="coerce").fillna(0.0)
    pos_df = pos_df[pos_df["initial_pos"] != 0.0].copy()

    # For futures: assetName = futureContractId (the contract-level identifier).
    # For shares:  assetName = stockId.
    fut_mask = pos_df["position_category"] == "futurePosition"
    sha_mask = pos_df["position_category"] == "stockPosition"

    fut_rows = pos_df[fut_mask].copy()
    if "futureContractId" in fut_rows.columns:
        fut_rows["assetName"] = fut_rows["futureContractId"]
    else:
        # Fallback to stockId if futureContractId column is absent.
        fut_rows["assetName"] = fut_rows["stockId"]

    sha_rows = pos_df[sha_mask].copy()
    sha_rows["assetName"] = sha_rows["stockId"]

    fut_pos = (
        fut_rows[["portfolioId", "assetName", "maturity", "initial_pos"]]
        .copy()
    )

    sha_pos = (
        sha_rows[["portfolioId", "assetName", "initial_pos"]]
        .copy()
    )

    print(f"Initial positions loaded     : {len(pos_df):,} rows "
          f"({len(fut_pos):,} FUT, {len(sha_pos):,} SHA) from {path}")

    return fut_pos, sha_pos


def analyze_intraday_leakage_continuous(
    df,
    output_folder="hourly_risk_analysis_continuous",
    currency_rates_path="output/currency_rates.xlsx",
    initial_positions_path=None,
    plot_top_pct=5,
    plot_metric="Delta_Leakage_Gap",
    max_plots=None,
    plot_exposure_threshold=None,
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
    initial_positions_path : str or None
        Path to the prior-EOD position CSV.  When provided the cumulative
        position for every product is seeded from this file instead of zero.
        Matched on (portfolioId, underlyingId, underlyingCurrency) for SHA and
        (portfolioId, underlyingId, maturity_key, underlyingCurrency) for FUT.
    plot_top_pct : float  (0–100)
    plot_metric : str
    max_plots : int or None
    plot_exposure_threshold : float or None
        Minimum Max_Intraday_Delta (EUR) required for a leakage group
        to be plotted.  Groups below this threshold are skipped.
    debug_sorting : bool
    expected_start_hour : int

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        portfolio_results_df, flagged_trades_df
    """
    os.makedirs(output_folder, exist_ok=True)
    df = df.copy()

    # ── 0. Load initial (prior-EOD) positions ──────────────────────────
    fut_init_pos = pd.DataFrame()
    sha_init_pos = pd.DataFrame()
    if initial_positions_path is not None:
        fut_init_pos, sha_init_pos = _load_initial_positions(initial_positions_path)

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

    # ── 3. Signed quantity ─────────────────────────────────────────────
    df["signed_qty"] = np.where(
        df["way"].str.upper() == "BUY", df["quantity"], -df["quantity"]
    )

    # Buy / sell nominal (sign-aware, for per-type bar charts).
    df["nominal"] = (
        df["premium_eur"].abs() * df["quantity"] * df["futurePointValue_numeric"]
    )
    df["buy_nominal"] = np.where(
        df["way"].str.upper() == "BUY", df["nominal"], 0.0
    )
    df["sell_nominal"] = np.where(
        df["way"].str.upper() == "SELL", df["nominal"], 0.0
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
        # Pre-join initial position onto df_fut before groupby (assetName is only on the
        # raw trade rows and is not preserved through the groupby product keys).
        if not fut_init_pos.empty:
            df_fut["portfolioId"] = df_fut["portfolioId"].astype(str)
            df_fut["assetName"] = df_fut["assetName"].astype(str)
            df_fut["maturity"] = df_fut["maturity"].astype(str)
            df_fut = df_fut.merge(
                fut_init_pos,
                on=["portfolioId", "assetName", "maturity"],
                how="left",
            )
            df_fut["initial_pos"] = df_fut["initial_pos"].fillna(0.0)

            # ── Match diagnostics (FUT) ────────────────────────────────
            _fut_keys_df = (
                df_fut[["portfolioId", "assetName", "maturity"]]
                .drop_duplicates()
            )
            _fut_matched = (df_fut["initial_pos"] != 0.0)
            _fut_matched_keys = (
                df_fut.loc[_fut_matched, ["portfolioId", "assetName", "maturity"]]
                .drop_duplicates()
            )
            _fut_total   = len(_fut_keys_df)
            _fut_n_match = len(_fut_matched_keys)
            _fut_rate    = (_fut_n_match / _fut_total * 100) if _fut_total else 0.0
            print(
                f"\nFUT position match           : "
                f"{_fut_n_match}/{_fut_total} unique products "
                f"({_fut_rate:.1f} %)"
            )
            _fut_unmatched = _fut_keys_df[
                ~_fut_keys_df.set_index(["portfolioId", "assetName", "maturity"])
                .index.isin(
                    _fut_matched_keys.set_index(["portfolioId", "assetName", "maturity"]).index
                )
            ]
            if not _fut_unmatched.empty:
                print("  Unmatched FUT products (no prior position):")
                for _, _r in _fut_unmatched.iterrows():
                    print(f"    portfolio={_r['portfolioId']}  asset={_r['assetName']}  maturity={_r['maturity']}")
            # ──────────────────────────────────────────────────────────
        else:
            df_fut["initial_pos"] = 0.0

        hourly_fut = (
            df_fut.groupby(fut_keys + ["hour_bucket"])
            .agg(
                signed_qty=("signed_qty", "sum"),
                buy_nominal=("buy_nominal", "sum"),
                sell_nominal=("sell_nominal", "sum"),
                trade_count=("signed_qty", "count"),
                latest_price=("premium_numeric", "last"),
                latest_fpv=("futurePointValue_numeric", "last"),
                rate_to_eur=("rate_to_eur", "first"),
                initial_pos=("initial_pos", "first"),
            )
            .reset_index()
            .sort_values(by=fut_keys + ["hour_bucket"], kind="mergesort")
        )
        hourly_fut["cumulative_pos"] = (
            hourly_fut.groupby(fut_keys)["signed_qty"].cumsum()
            + hourly_fut["initial_pos"]
        )

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
        # Pre-join initial position onto df_sha before groupby.
        if not sha_init_pos.empty:
            df_sha["portfolioId"] = df_sha["portfolioId"].astype(str)
            df_sha["assetName"] = df_sha["assetName"].astype(str)
            df_sha = df_sha.merge(
                sha_init_pos,
                on=["portfolioId", "assetName"],
                how="left",
            )
            df_sha["initial_pos"] = df_sha["initial_pos"].fillna(0.0)

            # ── Match diagnostics (SHA) ────────────────────────────────
            _sha_keys_df = (
                df_sha[["portfolioId", "assetName"]]
                .drop_duplicates()
            )
            _sha_matched = (df_sha["initial_pos"] != 0.0)
            _sha_matched_keys = (
                df_sha.loc[_sha_matched, ["portfolioId", "assetName"]]
                .drop_duplicates()
            )
            _sha_total   = len(_sha_keys_df)
            _sha_n_match = len(_sha_matched_keys)
            _sha_rate    = (_sha_n_match / _sha_total * 100) if _sha_total else 0.0
            print(
                f"\nSHA position match           : "
                f"{_sha_n_match}/{_sha_total} unique products "
                f"({_sha_rate:.1f} %)"
            )
            _sha_unmatched = _sha_keys_df[
                ~_sha_keys_df.set_index(["portfolioId", "assetName"])
                .index.isin(
                    _sha_matched_keys.set_index(["portfolioId", "assetName"]).index
                )
            ]
            if not _sha_unmatched.empty:
                print("  Unmatched SHA products (no prior position):")
                for _, _r in _sha_unmatched.iterrows():
                    print(f"    portfolio={_r['portfolioId']}  asset={_r['assetName']}")
            # ──────────────────────────────────────────────────────────
        else:
            df_sha["initial_pos"] = 0.0

        hourly_sha = (
            df_sha.groupby(sha_keys + ["hour_bucket"])
            .agg(
                signed_qty=("signed_qty", "sum"),
                buy_nominal=("buy_nominal", "sum"),
                sell_nominal=("sell_nominal", "sum"),
                trade_count=("signed_qty", "count"),
                latest_price=("premium_numeric", "last"),
                rate_to_eur=("rate_to_eur", "first"),
                initial_pos=("initial_pos", "first"),
            )
            .reset_index()
            .sort_values(by=sha_keys + ["hour_bucket"], kind="mergesort")
        )
        hourly_sha["cumulative_pos"] = (
            hourly_sha.groupby(sha_keys)["signed_qty"].cumsum()
            + hourly_sha["initial_pos"]
        )

        hourly_sha["latest_price"] = hourly_sha.groupby(sha_keys)["latest_price"].ffill()
        hourly_sha["latest_price_eur"] = hourly_sha["latest_price"] * hourly_sha["rate_to_eur"]
        hourly_sha["delta_nominal"] = (
            hourly_sha["cumulative_pos"]
            * hourly_sha["latest_price_eur"]
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
        fut_cum_pos_cols = []
        fut_buy_nom_cols = []
        fut_sell_nom_cols = []
        fut_count_cols = []
        fut_prior_eod_deltas = []

        if not fut_pdf.empty:
            for _, udf in fut_pdf.groupby(["underlyingId", "maturity", "underlyingCurrency"]):
                udf = udf.sort_values("hour_bucket")
                idx = udf.set_index("hour_bucket")
                delta_s = idx["delta_nominal"].reindex(all_hours).ffill().fillna(0)
                qty_s = idx["signed_qty"].reindex(all_hours).fillna(0)
                buy_s = idx["buy_nominal"].reindex(all_hours).fillna(0)
                sell_s = idx["sell_nominal"].reindex(all_hours).fillna(0)
                cnt_s = idx["trade_count"].reindex(all_hours).fillna(0)
                cum_pos_s = idx["cumulative_pos"].reindex(all_hours).ffill().fillna(0)
                fut_delta_cols.append(delta_s)
                fut_qty_cols.append(qty_s)
                fut_cum_pos_cols.append(cum_pos_s)
                fut_buy_nom_cols.append(buy_s)
                fut_sell_nom_cols.append(sell_s)
                fut_count_cols.append(cnt_s)
                # Prior-EOD delta for this FUT product.
                first_row = udf.iloc[0]
                prior_pos = first_row["cumulative_pos"] - first_row["signed_qty"]
                prior_delta = prior_pos * first_row["latest_price_eur"] * first_row["latest_fpv"]
                fut_prior_eod_deltas.append(prior_delta)

        sha_delta_cols = []
        sha_qty_cols = []
        sha_cum_pos_cols = []
        sha_buy_nom_cols = []
        sha_sell_nom_cols = []
        sha_count_cols = []
        sha_prior_eod_deltas = []

        if not sha_pdf.empty:
            for _, udf in sha_pdf.groupby(["underlyingId", "underlyingCurrency"]):
                udf = udf.sort_values("hour_bucket")
                idx = udf.set_index("hour_bucket")
                delta_s = idx["delta_nominal"].reindex(all_hours).ffill().fillna(0)
                qty_s = idx["signed_qty"].reindex(all_hours).fillna(0)
                buy_s = idx["buy_nominal"].reindex(all_hours).fillna(0)
                sell_s = idx["sell_nominal"].reindex(all_hours).fillna(0)
                cnt_s = idx["trade_count"].reindex(all_hours).fillna(0)
                cum_pos_s = idx["cumulative_pos"].reindex(all_hours).ffill().fillna(0)
                sha_delta_cols.append(delta_s)
                sha_qty_cols.append(qty_s)
                sha_cum_pos_cols.append(cum_pos_s)
                sha_buy_nom_cols.append(buy_s)
                sha_sell_nom_cols.append(sell_s)
                sha_count_cols.append(cnt_s)
                first_row = udf.iloc[0]
                prior_pos = first_row["cumulative_pos"] - first_row["signed_qty"]
                prior_delta = prior_pos * first_row["latest_price_eur"]
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
        total_cum_pos_fut = (
            pd.concat(fut_cum_pos_cols, axis=1).sum(axis=1)
            if fut_cum_pos_cols
            else pd.Series(0.0, index=all_hours)
        )
        total_cum_pos_sha = (
            pd.concat(sha_cum_pos_cols, axis=1).sum(axis=1)
            if sha_cum_pos_cols
            else pd.Series(0.0, index=all_hours)
        )
        # Buy / sell / count per deal type.
        total_buy_fut = (
            pd.concat(fut_buy_nom_cols, axis=1).sum(axis=1)
            if fut_buy_nom_cols
            else pd.Series(0.0, index=all_hours)
        )
        total_sell_fut = (
            pd.concat(fut_sell_nom_cols, axis=1).sum(axis=1)
            if fut_sell_nom_cols
            else pd.Series(0.0, index=all_hours)
        )
        total_count_fut = (
            pd.concat(fut_count_cols, axis=1).sum(axis=1)
            if fut_count_cols
            else pd.Series(0, index=all_hours)
        )
        total_buy_sha = (
            pd.concat(sha_buy_nom_cols, axis=1).sum(axis=1)
            if sha_buy_nom_cols
            else pd.Series(0.0, index=all_hours)
        )
        total_sell_sha = (
            pd.concat(sha_sell_nom_cols, axis=1).sum(axis=1)
            if sha_sell_nom_cols
            else pd.Series(0.0, index=all_hours)
        )
        total_count_sha = (
            pd.concat(sha_count_cols, axis=1).sum(axis=1)
            if sha_count_cols
            else pd.Series(0, index=all_hours)
        )
        portfolio_delta = total_delta_fut + total_delta_sha

        pf_df = pd.DataFrame({
            "hour_bucket": all_hours,
            "delta_fut": total_delta_fut.values,
            "delta_sha": total_delta_sha.values,
            "qty_fut": total_qty_fut.values,
            "qty_sha": total_qty_sha.values,
            "cum_pos_fut": total_cum_pos_fut.values,
            "cum_pos_sha": total_cum_pos_sha.values,
            "buy_nom_fut": total_buy_fut.values,
            "sell_nom_fut": total_sell_fut.values,
            "count_fut": total_count_fut.values.astype(int),
            "buy_nom_sha": total_buy_sha.values,
            "sell_nom_sha": total_sell_sha.values,
            "count_sha": total_count_sha.values.astype(int),
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
        # Apply exposure threshold filter before ranking.
        if plot_exposure_threshold is not None:
            pf_leakage_df = pf_leakage_df[
                pf_leakage_df["Max_Intraday_Delta"] >= plot_exposure_threshold
            ]
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

            # Build a complete hourly range (fill hours with no trades).
            sod_time = pf_df["hour_bucket"].iloc[0]
            eod_time = pf_df["hour_bucket"].iloc[-1]
            all_plot_hours = pd.date_range(start=sod_time, end=eod_time, freq="h")
            full_df = (
                pd.DataFrame({"hour_bucket": all_plot_hours})
                .merge(pf_df, on="hour_bucket", how="left")
            )
            for _col in ["buy_nom_fut", "sell_nom_fut", "buy_nom_sha", "sell_nom_sha",
                         "qty_fut", "qty_sha"]:
                full_df[_col] = full_df[_col].fillna(0.0)
            for _col in ["count_fut", "count_sha"]:
                full_df[_col] = full_df[_col].fillna(0).astype(int)
            # Forward-fill cumulative columns then back-fill leading gaps.
            for _col in ["delta_fut", "delta_sha", "portfolio_delta",
                         "cum_pos_fut", "cum_pos_sha"]:
                full_df[_col] = full_df[_col].ffill().bfill().fillna(0.0)

            h_starts = full_df["hour_bucket"]
            h_centers = h_starts + timedelta(minutes=30)

            pf_peak = full_df["portfolio_delta"].abs().max()
            pf_eod = full_df["portfolio_delta"].iloc[-1]
            pf_sod = full_df["portfolio_delta"].iloc[0]
            sod_center = sod_time + timedelta(minutes=30)
            eod_center = eod_time + timedelta(minutes=30)

            bar_width = timedelta(minutes=50)
            nom_bar_width = timedelta(minutes=25)
            sell_bar_offset = timedelta(minutes=25)

            fig, (ax_delta, ax_fut, ax_sha) = plt.subplots(
                3, 1,
                figsize=(14, 12),
                sharex=True,
                gridspec_kw={"height_ratios": [1.4, 1, 1]},
            )

            # ── Chart 1: Cumulative Delta (bars) + SOD→EOD line ────────
            ax_delta.bar(
                h_starts,
                full_df["portfolio_delta"],
                width=bar_width,
                align="edge",
                alpha=0.55,
                color="#d62728",
                edgecolor="#8b1a1a",
                linewidth=0.5,
                label="Portfolio Delta (EUR)",
            )
            # Overlay SHA and FUT delta as lines for decomposition.
            ax_delta.plot(
                h_centers,
                full_df["delta_sha"],
                color="#1f77b4",
                linewidth=1.5,
                marker="v",
                markersize=3,
                label="SHA Delta (EUR)",
            )
            ax_delta.plot(
                h_centers,
                full_df["delta_fut"],
                color="#ff7f0e",
                linewidth=1.5,
                marker="^",
                markersize=3,
                label="FUT Delta (EUR)",
            )
            # SOD → EOD net flow line.
            ax_delta.plot(
                [sod_center, eod_center],
                [pf_sod, pf_eod],
                color="black",
                linestyle="--",
                linewidth=3,
                marker="o",
                markersize=6,
                label="Net Flow (SOD to EOD)",
            )
            ax_delta.set_ylabel("Delta Nominal (EUR)", fontsize=9)
            ax_delta.axhline(0, color="black", linewidth=0.5)
            ax_delta.legend(loc="upper left", fontsize=8)
            ax_delta.grid(axis="y", linestyle=":", alpha=0.5)
            ax_delta.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
            )
            ax_delta.set_title(
                f"Portfolio Delta Nominal Exposure – Intraday Path\n"
                f"{prow['Portfolio']} | {prow['ExecDate']}\n"
                f"Peak: {pf_peak:,.0f} EUR | EOD: {pf_eod:,.0f} EUR | "
                f"{plot_metric}: {prow[plot_metric]:,.2f}",
                fontsize=10,
            )

            # ── Chart 2: FUT Cumulative Position (bars) + Trade Count (line)
            ax_fut.bar(
                h_starts,
                full_df["cum_pos_fut"],
                width=bar_width,
                align="edge",
                alpha=0.55,
                color="#ff7f0e",
                edgecolor="#b25600",
                linewidth=0.5,
                label="FUT Cumulative Position (qty)",
            )
            ax_fut.set_ylabel("FUT Cumulative Position (qty)", fontsize=9)
            ax_fut.axhline(0, color="black", linewidth=0.5)
            ax_fut.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
            )
            # Trade count as line on secondary axis.
            ax_fut2 = ax_fut.twinx()
            ax_fut2.plot(
                h_centers,
                full_df["count_fut"],
                color="#d62728",
                linewidth=1.5,
                marker="o",
                markersize=3,
                label="FUT Trade Count",
            )
            ax_fut2.set_ylabel("Trade Count", color="#d62728", fontsize=9)
            ax_fut2.tick_params(axis="y", labelcolor="#d62728")
            lines_f1, labels_f1 = ax_fut.get_legend_handles_labels()
            lines_f2, labels_f2 = ax_fut2.get_legend_handles_labels()
            ax_fut.legend(lines_f1 + lines_f2, labels_f1 + labels_f2, loc="upper left", fontsize=8)
            ax_fut.grid(axis="y", linestyle=":", alpha=0.5)
            ax_fut.set_title("Futures – Cumulative Position & Trade Count", fontsize=10)

            # ── Chart 3: SHA Cumulative Position (bars) + Trade Count (line)
            ax_sha.bar(
                h_starts,
                full_df["cum_pos_sha"],
                width=bar_width,
                align="edge",
                alpha=0.55,
                color="#1f77b4",
                edgecolor="#0d4f8b",
                linewidth=0.5,
                label="SHA Cumulative Position (qty)",
            )
            ax_sha.set_ylabel("SHA Cumulative Position (qty)", fontsize=9)
            ax_sha.axhline(0, color="black", linewidth=0.5)
            ax_sha.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
            )
            # Trade count as line on secondary axis.
            ax_sha2 = ax_sha.twinx()
            ax_sha2.plot(
                h_centers,
                full_df["count_sha"],
                color="#d62728",
                linewidth=1.5,
                marker="o",
                markersize=3,
                label="SHA Trade Count",
            )
            ax_sha2.set_ylabel("Trade Count", color="#d62728", fontsize=9)
            ax_sha2.tick_params(axis="y", labelcolor="#d62728")
            lines_s1, labels_s1 = ax_sha.get_legend_handles_labels()
            lines_s2, labels_s2 = ax_sha2.get_legend_handles_labels()
            ax_sha.legend(lines_s1 + lines_s2, labels_s1 + labels_s2, loc="upper left", fontsize=8)
            ax_sha.grid(axis="y", linestyle=":", alpha=0.5)
            ax_sha.set_title("Shares – Cumulative Position & Trade Count", fontsize=10)

            # ── Shared X-axis formatting ───────────────────────────────
            x_start = h_starts.iloc[0]
            x_end = h_starts.iloc[-1] + timedelta(hours=1)
            ax_sha.set_xlim(x_start, x_end)
            ax_sha.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax_sha.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax_sha.set_xlabel("Hour (local)")
            ax_sha.tick_params(axis="x", labelrotation=45)

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
        initial_positions_path="output/synthetic_position_data_clean.parquet",
        plot_top_pct=5,
        plot_metric="Delta_Max_to_Baseline_Ratio",
        max_plots=20,
        plot_exposure_threshold=10_000_000,
    )
