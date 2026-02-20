import os
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def _build_fp_config(fp_config=None):
    """Return false-positive handling configuration with optional overrides."""
    config = {
        "eod_abs_min": 1_000.0,
        "eod_to_sod_ratio_min": 0.05,
        "peak_first_hour_tolerance_hours": 1.0,
        "peak_equals_sod_rel_tol": 0.02,
        "decreasing_flow_min_reduction": 0.20,
        "new_risk_abs_min": 5_000.0,
        "new_risk_to_sod_min": 0.10,
        "operational_review_gap_max": 25_000.0,
        "operational_review_ratio_max": 2.0,
        "adaptive_scale_floor": 0.5,
        "adaptive_scale_cap": 3.0,
    }
    if fp_config:
        config.update(fp_config)
    return config


def _clip_scale(value, lower, upper):
    return float(np.clip(value, lower, upper))


def _compute_adaptive_scales(results_df, config):
    """Compute per-row adaptive scaling based on underlying and portfolio size."""
    if results_df.empty:
        return pd.Series(dtype=float)

    under_avg = (
        results_df.groupby("Underlying")["EOD_Position"]
        .apply(lambda x: x.abs().mean())
        .replace(0, np.nan)
    )
    port_avg = (
        results_df.groupby("Portfolio")["Max_Intraday_Position"]
        .mean()
        .replace(0, np.nan)
    )

    under_median = float(under_avg.median()) if not under_avg.dropna().empty else 1.0
    port_median = float(port_avg.median()) if not port_avg.dropna().empty else 1.0
    if under_median == 0:
        under_median = 1.0
    if port_median == 0:
        port_median = 1.0

    floor = config["adaptive_scale_floor"]
    cap = config["adaptive_scale_cap"]

    def _scale_row(row):
        u_avg = under_avg.get(row["Underlying"], under_median)
        p_avg = port_avg.get(row["Portfolio"], port_median)

        if pd.isna(u_avg):
            u_avg = under_median
        if pd.isna(p_avg):
            p_avg = port_median

        scale = np.sqrt((u_avg / under_median) * (p_avg / port_median))
        return _clip_scale(scale, floor, cap)

    return results_df.apply(_scale_row, axis=1)


def _classify_leakage_false_positive(row, group_df, config):
    """Classify a leakage case into suppress/operational-review/investigate."""
    leakage_detected = bool(row["Leakage_Detected"])
    if not leakage_detected:
        return {
            "Likely_FP": False,
            "FP_Reasons": "",
            "Action": "none",
            "Incremental_Intraday_Risk": 0.0,
        }

    abs_sod = abs(float(row["SOD_Position"]))
    abs_eod = abs(float(row["EOD_Position"]))
    max_exposure = float(row["Max_Intraday_Position"])
    incremental_risk = max(0.0, max_exposure - abs_sod)

    eod_abs_min = float(row["effective_eod_abs_min"])
    new_risk_abs_min = float(row["effective_new_risk_abs_min"])
    operational_review_gap_max = float(row["effective_operational_review_gap_max"])

    eod_to_sod_ratio = abs_eod / (abs_sod + 1e-9)
    low_eod = (abs_eod <= eod_abs_min) or (
        eod_to_sod_ratio <= config["eod_to_sod_ratio_min"]
    )

    peak_idx = group_df["cumulative_pos"].abs().idxmax()
    peak_time = group_df.loc[peak_idx, "hour_bucket"]
    first_time = group_df["hour_bucket"].iloc[0]
    peak_at_first_hour = bool(
        abs((peak_time - first_time).total_seconds())
        <= (config["peak_first_hour_tolerance_hours"] * 3600)
    )

    peak_equals_sod = bool(
        np.isclose(
            max_exposure,
            abs_sod,
            rtol=config["peak_equals_sod_rel_tol"],
            atol=1e-9,
        )
    )

    decreasing_net_flow = bool(
        abs_eod <= abs_sod * (1.0 - config["decreasing_flow_min_reduction"])
    )

    new_risk_ratio = incremental_risk / (abs_sod + 1e-9)
    tiny_new_risk = (incremental_risk <= new_risk_abs_min) or (
        new_risk_ratio <= config["new_risk_to_sod_min"]
    )

    reason_flags = {
        "low_eod": low_eod,
        "peak_at_first_hour": peak_at_first_hour,
        "peak_equals_sod": peak_equals_sod,
        "decreasing_net_flow": decreasing_net_flow,
        "tiny_new_risk": tiny_new_risk,
    }

    behavioral_confirmations = any(
        reason_flags[k]
        for k in [
            "peak_at_first_hour",
            "peak_equals_sod",
            "decreasing_net_flow",
            "tiny_new_risk",
        ]
    )

    likely_fp = low_eod and behavioral_confirmations
    reasons = [code for code, enabled in reason_flags.items() if enabled]

    if likely_fp:
        action = "suppress"
    else:
        low_materiality = (
            float(row["Leakage_Gap"]) <= operational_review_gap_max
            or float(row["Max_to_EOD_Ratio"]) <= config["operational_review_ratio_max"]
            or incremental_risk <= (new_risk_abs_min * 1.5)
        )
        action = "operational-review" if low_materiality else "investigate"
        if low_materiality:
            reasons.append("low_materiality")

    return {
        "Likely_FP": likely_fp,
        "FP_Reasons": "|".join(reasons),
        "Action": action,
        "Incremental_Intraday_Risk": incremental_risk,
    }


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
            "Likely_FP", "Action", "FP_Reasons",
        ]
        for col in ["Pre-Market", "SOD_Pos", "EOD_Pos", "Max_Intraday", "Leakage_Gap", "Max/EOD Ratio"]:
            t3.align[col] = "r"
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
    enable_adaptive_thresholds=True,
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
    fp_config: Optional dictionary of false-positive thresholds.
    enable_adaptive_thresholds: If True, scales absolute thresholds by
        underlying and portfolio historical size proxies.

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
    config = _build_fp_config(fp_config)

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
        lambda ts: ts.replace(minute=0, second=0, microsecond=0)
        if ts is not None
        else None
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
    hourly_net["execDate"] = hourly_net["hour_bucket"].apply(
        lambda ts: ts.date() if ts is not None else None
    )

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
        # Position entering the day before any first-hour trading (prior EOD carry-over).
        prior_eod_pos = sod_pos - group_df["signed_qty"].iloc[0]

        max_exposure = group_df["cumulative_pos"].abs().max()
        eod_exposure = abs(eod_pos)
        leakage_gap = max_exposure - eod_exposure
        max_to_eod_ratio = max_exposure / (eod_exposure + 1e-9)

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
                "Leakage_Detected": is_leakage,
            }
        )

    results_df = pd.DataFrame(summary_data)

    if results_df.empty:
        results_df["Likely_FP"] = []
        results_df["FP_Reasons"] = []
        results_df["Action"] = []
        results_df["Incremental_Intraday_Risk"] = []
        results_df["Adaptive_Threshold_Scale"] = []
        results_df["effective_eod_abs_min"] = []
        results_df["effective_new_risk_abs_min"] = []
        results_df["effective_operational_review_gap_max"] = []
    else:
        if enable_adaptive_thresholds:
            scale_series = _compute_adaptive_scales(results_df, config)
        else:
            scale_series = pd.Series(1.0, index=results_df.index)

        results_df["Adaptive_Threshold_Scale"] = scale_series
        results_df["effective_eod_abs_min"] = (
            config["eod_abs_min"] * results_df["Adaptive_Threshold_Scale"]
        )
        results_df["effective_new_risk_abs_min"] = (
            config["new_risk_abs_min"] * results_df["Adaptive_Threshold_Scale"]
        )
        results_df["effective_operational_review_gap_max"] = (
            config["operational_review_gap_max"] * results_df["Adaptive_Threshold_Scale"]
        )

        classifications = []
        for idx, row in results_df.iterrows():
            group_ids = (
                row["ExecDate"],
                row["Portfolio"],
                row["Underlying"],
                row["Maturity"],
            )
            group_df = group_frames.get(group_ids)
            if group_df is None or group_df.empty:
                classifications.append(
                    {
                        "Likely_FP": False,
                        "FP_Reasons": "",
                        "Action": "none" if not bool(row["Leakage_Detected"]) else "investigate",
                        "Incremental_Intraday_Risk": max(
                            0.0,
                            float(row["Max_Intraday_Position"]) - abs(float(row["SOD_Position"])),
                        ),
                    }
                )
                continue

            classifications.append(_classify_leakage_false_positive(row, group_df, config))

        class_df = pd.DataFrame(classifications, index=results_df.index)
        results_df = pd.concat([results_df, class_df], axis=1)

    # 6) Plot only the highest-ranked flagged leakage groups.
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

    # 7) Map flagged leakage cases back to the original trade-level DataFrame.
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
       "Incremental_Intraday_Risk", "Likely_FP", "FP_Reasons", "Action"]]

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

    # 8) Export reports.
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

    return results_df, flagged_trades_df


if __name__ == "__main__":
    df_trades = pd.read_csv("output/synthetic_trading_data.csv")
    results, flagged = analyze_intraday_leakage_continuous(
        df_trades, plot_top_pct=5, plot_metric="Max_to_EOD_Ratio", max_plots=20
    )
