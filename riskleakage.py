import os
import logging
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.ensemble import IsolationForest
from market_close_times import MARKET_CLOSE_TIMES


LOGGER = logging.getLogger(__name__)


def _configure_logging(level=logging.INFO):
    """Configure module logging once (safe for repeated calls)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    LOGGER.setLevel(level)


def _log_progress(step, total_steps, message):
    LOGGER.info("[%s/%s] %s", step, total_steps, message)


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


def _build_fp_config(fp_config=None):
    """Return statistical triage configuration with optional overrides."""
    config = {
        "contamination": 0.10,
        "p_suppress": 0.75,
        "p_review": 0.40,
        "model_random_state": 42,
        "min_training_samples": 8,
        "epsilon": 1e-9,
    }
    if fp_config:
        config.update(fp_config)

    if not (0.0 <= float(config["p_review"]) < float(config["p_suppress"]) <= 1.0):
        raise ValueError("Require 0 <= p_review < p_suppress <= 1.")

    contamination = float(config["contamination"])
    if contamination <= 0.0 or contamination > 0.5:
        raise ValueError("contamination must be in (0, 0.5].")

    return config


def _minmax_scale(values):
    """Scale an array-like to [0, 1]. If flat, return all 0.5."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if np.isclose(vmax, vmin):
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - vmin) / (vmax - vmin)


def _compute_normalized_metrics(group_df, sod_pos, eod_pos, max_exposure, eps=1e-9):
    """Compute dimensionless leakage descriptors for statistical scoring."""
    abs_sod = abs(float(sod_pos))
    abs_eod = abs(float(eod_pos))
    max_exposure = float(max_exposure)

    eod_retention = abs_eod / (max_exposure + eps)

    avg_book_size = (abs_sod + abs_eod) / 2.0
    excess_excursion = max(0.0, max_exposure - abs_sod) / (avg_book_size + eps)

    total_hours = int(len(group_df))
    if total_hours <= 1:
        flow_asymmetry = 0.0
    else:
        peak_pos = int(np.argmax(group_df["cumulative_pos"].abs().to_numpy()))
        flow_asymmetry = peak_pos / float(total_hours)

    hourly_flow = group_df["signed_qty"].astype(float)
    cumulative_abs = group_df["cumulative_pos"].abs().astype(float)
    intraday_volatility = hourly_flow.std(ddof=0) / (cumulative_abs.mean() + eps)

    return {
        "eod_retention": eod_retention,
        "excess_excursion": excess_excursion,
        "flow_asymmetry": flow_asymmetry,
        "intraday_volatility": intraday_volatility,
    }


def _score_leakage_cases(leakage_df, config):
    """Score leakage cases with Isolation Forest; returns fp_score in [0, 1]."""
    if leakage_df.empty:
        return pd.Series(dtype=float)

    feature_cols = [
        "eod_retention",
        "excess_excursion",
        "flow_asymmetry",
        "intraday_volatility",
    ]
    feature_df = leakage_df[feature_cols].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.apply(
        lambda col: col.fillna(float(col.median()) if not col.dropna().empty else 0.0)
    )

    if len(feature_df) < int(config["min_training_samples"]):
        return pd.Series(0.5, index=leakage_df.index, dtype=float)

    if feature_df.nunique(dropna=False).sum() <= len(feature_cols):
        return pd.Series(0.5, index=leakage_df.index, dtype=float)

    model = IsolationForest(
        contamination=float(config["contamination"]),
        random_state=int(config["model_random_state"]),
    )
    model.fit(feature_df)
    raw_scores = model.decision_function(feature_df)
    scaled_scores = _minmax_scale(raw_scores)
    return pd.Series(scaled_scores, index=leakage_df.index, dtype=float)

def _triage_from_fp_score(fp_score, p_review, p_suppress):
    """Map statistical fp_score into suppress/review/investigate triage."""
    if pd.isna(fp_score):
        return "operational-review", False, "score_unavailable"
    if fp_score >= p_suppress:
        return "suppress", True, "high_fp_score"
    if fp_score >= p_review:
        return "operational-review", False, "mid_fp_score"
    return "investigate", False, "low_fp_score"


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

    suppressed_df = leakage_df[leakage_df["Action"] == "suppress"].copy() if "Action" in leakage_df.columns else pd.DataFrame()
    operational_df = leakage_df[leakage_df["Action"] == "operational-review"].copy() if "Action" in leakage_df.columns else pd.DataFrame()
    investigate_df = leakage_df[leakage_df["Action"] == "investigate"].copy() if "Action" in leakage_df.columns else leakage_df.copy()
    suppression_rate = (len(suppressed_df) / total_leakage * 100) if total_leakage else 0.0

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
            ["Suppressed (Likely FP)",               f"{len(suppressed_df):,}"],
            ["Suppression Rate",                     f"{suppression_rate:.2f} %"],
            ["Operational Review Cases",             f"{len(operational_df):,}"],
            ["Investigate Cases",                    f"{len(investigate_df):,}"],
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
            "#", "ExecDate", "Portfolio", "Underlying", "Maturity",
            "Pre-Market", "SOD_Pos", "EOD_Pos", "Max_Intraday", "Leakage_Gap", "Max/EOD Ratio",
            "FP_Score", "Likely_FP", "Action", "FP_Reasons",
        ]
        for col in ["Pre-Market", "SOD_Pos", "EOD_Pos", "Max_Intraday", "Leakage_Gap", "Max/EOD Ratio"]:
            t3.align[col] = "r"
        t3.align["FP_Score"] = "r"
        t3.align["FP_Reasons"] = "l"
        for i, row in sorted_leakage.iterrows():
            t3.add_row([
                i + 1,
                row["ExecDate"],
                row["Portfolio"],
                row["Underlying"],
                row["Maturity"],
                f"{row['Prior_EOD_Position']:>12,.0f}",
                f"{row['SOD_Position']:>12,.0f}",
                f"{row['EOD_Position']:>12,.0f}",
                f"{row['Max_Intraday_Position']:>12,.0f}",
                f"{row['Leakage_Gap']:>12,.2f}",
                f"{row['Max_to_EOD_Ratio']:>10,.4f}",
                f"{float(row.get('fp_score', np.nan)):.4f}" if not pd.isna(row.get("fp_score", np.nan)) else "N/A",
                "Y" if bool(row.get("Likely_FP", False)) else "N",
                row.get("Action", "investigate"),
                row.get("FP_Reasons", ""),
            ])
        w(t3)

        # ------------------------------------------------------------------
        # Section 3b: FP Reason Distribution (governance KPI)
        # ------------------------------------------------------------------
        if "FP_Reasons" in leakage_df.columns:
            reason_tokens = (
                leakage_df["FP_Reasons"]
                .fillna("")
                .str.split("|")
                .explode()
                .str.strip()
            )
            reason_tokens = reason_tokens[reason_tokens != ""]
            w(f"\n{sep}")
            w("  FP REASON DISTRIBUTION")
            w(sep)

            if reason_tokens.empty:
                w("  No FP reason codes assigned.")
            else:
                reason_dist = reason_tokens.value_counts().reset_index()
                reason_dist.columns = ["Reason", "Count"]
                t3b = PrettyTable()
                t3b.field_names = ["Reason", "Count", "Pct of Leakage"]
                t3b.align["Reason"] = "l"
                t3b.align["Count"] = "r"
                t3b.align["Pct of Leakage"] = "r"
                for _, r in reason_dist.iterrows():
                    pct = (r["Count"] / total_leakage * 100) if total_leakage else 0.0
                    t3b.add_row([r["Reason"], f"{int(r['Count']):,}", f"{pct:.2f} %"])
                w(t3b)

        # ------------------------------------------------------------------
        # Section 4: Flagged Trades Summary per Leakage Group
        # ------------------------------------------------------------------
        if not flagged_trades_df.empty:
            w(f"\n{sep}")
            w("  FLAGGED TRADES SUMMARY  (original trades mapped to leakage groups)")
            w(sep)

            grp_summary = (
                flagged_trades_df.groupby(["ExecDate", "Portfolio", "Underlying", "Maturity"])
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
                "ExecDate", "Portfolio", "Underlying", "Maturity",
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
    fp_config=None,
    log_progress=True,
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
    fp_config: Optional dictionary for statistical triage config. Supported
        keys include `contamination`, `p_suppress`, `p_review`,
        `model_random_state`, and `min_training_samples`.
    log_progress: If True, emits step-by-step progress logs.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]:
        - results_df: one row per grouped exec-date/position key with leakage metrics.
        - flagged_trades_df: original trade rows that belong to flagged leakage groups,
          enriched with leakage metrics and exec-date columns.
    """
    if log_progress:
        _configure_logging()

    total_steps = 9
    _log_progress(1, total_steps, "Starting intraday leakage analysis")

    os.makedirs(output_folder, exist_ok=True)
    df = df.copy()
    _log_progress(1, total_steps, f"Input trades: {len(df):,}")

    if plot_top_pct < 0 or plot_top_pct > 100:
        raise ValueError("plot_top_pct must be between 0 and 100.")
    if max_plots is not None and max_plots < 0:
        raise ValueError("max_plots must be >= 0 when provided.")
    config = _build_fp_config(fp_config)

    # 1) Parse execution timestamp and group by parsed-hour bucket.
    _log_progress(2, total_steps, "Parsing execution timestamps and deriving hour buckets")
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
        lambda ts: ts.replace(minute=0, second=0, microsecond=0)
        if ts is not None
        else None
    )

    # 2) Signed quantity.
    _log_progress(3, total_steps, "Calculating signed quantities")
    df["signed_qty"] = np.where(
        df["way"].str.upper() == "BUY", df["quantity"], -df["quantity"]
    )

    # 3) Aggregate signed flow by local-hour bucket per position key.
    _log_progress(4, total_steps, "Aggregating hourly net flow by position key")
    position_keys = ["portfolioId", "underlyingId", "maturity"]
    hourly_net = (
        df.groupby(position_keys + ["hour_bucket"])["signed_qty"]
        .sum()
        .reset_index()
        .sort_values(by=position_keys + ["hour_bucket"])
    )
    _log_progress(4, total_steps, f"Hourly buckets generated: {len(hourly_net):,}")

    # 4) Rebuild position path from hourly net flow.
    hourly_net["cumulative_pos"] = hourly_net.groupby(position_keys)[
        "signed_qty"
    ].cumsum()
    hourly_net["execDate"] = hourly_net["hour_bucket"].apply(
        lambda ts: ts.date() if ts is not None else None
    )

    # 5) Evaluate leakage per exec-date and position key.
    _log_progress(5, total_steps, "Evaluating leakage metrics per daily position group")
    summary_data = []
    group_frames = {}
    daily_keys = ["execDate", "portfolioId", "underlyingId", "maturity"]

    for group_ids, group_df in hourly_net.groupby(daily_keys):
        exec_date, port, und, mat = group_ids
        group_df = group_df.sort_values("hour_bucket")
        group_frames[group_ids] = group_df.copy()

        sod_pos = group_df["cumulative_pos"].iloc[0]
        eod_pos = group_df["cumulative_pos"].iloc[-1]
        # Position entering the day before any first-hour trading (prior EOD carry-over).
        prior_eod_pos = sod_pos - group_df["signed_qty"].iloc[0]

        max_exposure = group_df["cumulative_pos"].abs().max()
        eod_exposure = abs(eod_pos)
        leakage_gap = max_exposure - eod_exposure
        max_to_eod_ratio = max_exposure / (eod_exposure + 1e-9)
        normalized_metrics = _compute_normalized_metrics(
            group_df,
            sod_pos=sod_pos,
            eod_pos=eod_pos,
            max_exposure=max_exposure,
            eps=float(config["epsilon"]),
        )

        # Exclude fully flattened books at EOD (treated as intentional portfolio clearing).
        is_leakage = (
            eod_exposure != 0
            and (max_exposure > eod_exposure)
            and (max_exposure != abs(sod_pos))
        )

        summary_data.append(
            {
                "ExecDate": exec_date,
                "Portfolio": port,
                "Underlying": und,
                "Maturity": mat,
                "Prior_EOD_Position": prior_eod_pos,
                "SOD_Position": sod_pos,
                "EOD_Position": eod_pos,
                "Max_Intraday_Position": max_exposure,
                "Leakage_Gap": leakage_gap,
                "Max_to_EOD_Ratio": max_to_eod_ratio,
                **normalized_metrics,
                "Leakage_Detected": is_leakage,
            }
        )

    results_df = pd.DataFrame(summary_data)
    _log_progress(5, total_steps, f"Daily groups evaluated: {len(results_df):,}")

    if results_df.empty:
        results_df["Likely_FP"] = []
        results_df["FP_Reasons"] = []
        results_df["Action"] = []
        results_df["Incremental_Intraday_Risk"] = []
        results_df["fp_score"] = []
    else:
        _log_progress(6, total_steps, "Scoring leakage cases and assigning triage actions")
        results_df["Incremental_Intraday_Risk"] = (
            results_df["Max_Intraday_Position"] - results_df["SOD_Position"].abs()
        ).clip(lower=0.0)
        results_df["fp_score"] = np.nan
        results_df["Likely_FP"] = False
        results_df["FP_Reasons"] = ""
        results_df["Action"] = "none"

        leakage_mask = results_df["Leakage_Detected"]
        leakage_cases = results_df[leakage_mask].copy()
        _log_progress(6, total_steps, f"Leakage cases detected: {len(leakage_cases):,}")
        if not leakage_cases.empty:
            leakage_scores = _score_leakage_cases(leakage_cases, config)
            results_df.loc[leakage_scores.index, "fp_score"] = leakage_scores.values

            p_review = float(config["p_review"])
            p_suppress = float(config["p_suppress"])
            triage_out = results_df.loc[leakage_mask, "fp_score"].apply(
                lambda s: _triage_from_fp_score(s, p_review, p_suppress)
            )
            triage_df = pd.DataFrame(
                triage_out.tolist(),
                columns=["Action", "Likely_FP", "FP_Reasons"],
                index=results_df.index[leakage_mask],
            )
            results_df.loc[triage_df.index, ["Action", "Likely_FP", "FP_Reasons"]] = triage_df

        results_df.loc[leakage_mask & results_df["fp_score"].isna(), "fp_score"] = 0.5
        results_df.loc[leakage_mask & results_df["Action"].eq("none"), "Action"] = "operational-review"
        results_df.loc[leakage_mask & results_df["FP_Reasons"].eq(""), "FP_Reasons"] = "mid_fp_score"

        results_df.loc[~leakage_mask, ["Action", "Likely_FP", "FP_Reasons"]] = ["none", False, ""]
        results_df.loc[~leakage_mask, "Incremental_Intraday_Risk"] = 0.0
        results_df.loc[~leakage_mask, "fp_score"] = np.nan

        results_df["fp_score"] = results_df["fp_score"].astype(float).clip(lower=0.0, upper=1.0)
        results_df["Likely_FP"] = results_df["Likely_FP"].astype(bool)
        results_df["FP_Reasons"] = results_df["FP_Reasons"].astype(str)
        results_df["Action"] = results_df["Action"].astype(str)

    # 6) Plot only the highest-ranked flagged leakage groups.
    _log_progress(7, total_steps, "Selecting and generating leakage plots")
    metric_candidates = {"Leakage_Gap", "Max_Intraday_Position", "Max_to_EOD_Ratio"}
    if plot_metric not in metric_candidates:
        raise ValueError(
            f"Invalid plot_metric '{plot_metric}'. Use one of: {sorted(metric_candidates)}"
        )

    leakage_df = results_df[
        (results_df["Leakage_Detected"]) & (results_df["Action"] == "investigate")
    ].copy()
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

            _log_progress(7, total_steps, f"Plots generated: {plotted_count}")

    # 7) Map flagged leakage cases back to the original trade-level DataFrame.
            _log_progress(8, total_steps, "Mapping escalated leakage groups back to original trades")
    leakage_summary = results_df[results_df["Leakage_Detected"]].copy()
    escalated_summary = leakage_summary[leakage_summary["Action"] != "suppress"].copy()
    suppressed_candidates = leakage_summary[leakage_summary["Action"] == "suppress"].copy()

    escalated_keys = set(
        zip(
            escalated_summary["ExecDate"],
            escalated_summary["Portfolio"],
            escalated_summary["Underlying"],
            escalated_summary["Maturity"],
        )
    )

    # Enrich the working df with execDate (already has execTime_parsed & signed_qty).
    df["execDate"] = df["execTime_parsed"].apply(
        lambda ts: ts.date() if ts is not None else None
    )

    # Filter original trades that belong to an escalated group.
    flagged_mask = df.apply(
        lambda row: (
            row["execDate"],
            row["portfolioId"],
            row["underlyingId"],
            row["maturity"],
        )
        in escalated_keys,
        axis=1,
    )
    flagged_trades_df = df[flagged_mask].copy()

    # Merge leakage metrics into the flagged trades for full context.
    leakage_merge = leakage_summary.rename(
        columns={
            "ExecDate": "execDate",
            "Portfolio": "portfolioId",
            "Underlying": "underlyingId",
            "Maturity": "maturity",
        }
    )[["execDate", "portfolioId", "underlyingId", "maturity",
       "Prior_EOD_Position", "SOD_Position", "EOD_Position",
       "Max_Intraday_Position", "Leakage_Gap", "Max_to_EOD_Ratio",
       "Incremental_Intraday_Risk", "fp_score", "Likely_FP", "FP_Reasons", "Action"]]

    flagged_trades_df = flagged_trades_df.merge(
        leakage_merge,
        on=["execDate", "portfolioId", "underlyingId", "maturity"],
        how="left",
    )

    # Rename for report-friendly column names.
    flagged_trades_df = flagged_trades_df.rename(
        columns={
            "execDate": "ExecDate",
            "portfolioId": "Portfolio",
            "underlyingId": "Underlying",
            "maturity": "Maturity",
        }
    )
    _log_progress(8, total_steps, f"Flagged trade rows exported: {len(flagged_trades_df):,}")

    # 8) Export reports.
    _log_progress(9, total_steps, "Writing CSV and TXT reports")
    csv_path = os.path.join(output_folder, "Full_Leakage_Report_Continuous.csv")
    results_df.to_csv(csv_path, index=False)

    flagged_trades_csv = os.path.join(output_folder, "Leakage_Flagged_Trades.csv")
    flagged_trades_df.to_csv(flagged_trades_csv, index=False)

    suppressed_csv = os.path.join(output_folder, "Suppressed_Leakage_Candidates.csv")
    suppressed_candidates.to_csv(suppressed_csv, index=False)

    # 9) Write auditor report to text file.
    audit_report_path = os.path.join(output_folder, "Audit_Report.txt")
    _write_audit_report(results_df, flagged_trades_df, audit_report_path)

    print(f"Plots Generated              : {plotted_count}")
    print(f"Full Report                  : {csv_path}")
    print(f"Flagged Trades CSV           : {flagged_trades_csv}")
    print(f"Suppressed Candidates CSV    : {suppressed_csv}")
    print(f"Audit Report TXT             : {audit_report_path}")
    _log_progress(9, total_steps, "Analysis complete")

    return results_df, flagged_trades_df


if __name__ == "__main__":
    df_trades = pd.read_csv("output/synthetic_trading_data.csv")
    results, flagged = analyze_intraday_leakage_continuous(
        df_trades, plot_top_pct=5, plot_metric="Max_to_EOD_Ratio", max_plots=20
    )
