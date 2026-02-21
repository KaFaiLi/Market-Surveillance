import os
import logging
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
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
    """Return path-risk statistical configuration with optional overrides."""
    config = {
        "resample_points": 24,
        "path_variance_retained": 0.90,
        "min_training_samples": 12,
        "mcd_support_fraction": None,
        "epsilon": 1e-9,
    }
    if fp_config:
        config.update(fp_config)

    resample_points = int(config["resample_points"])
    if resample_points < 4:
        raise ValueError("resample_points must be >= 4.")

    path_variance_retained = float(config["path_variance_retained"])
    if path_variance_retained <= 0.0 or path_variance_retained > 1.0:
        raise ValueError("path_variance_retained must be in (0, 1].")

    min_training_samples = int(config["min_training_samples"])
    if min_training_samples < 6:
        raise ValueError("min_training_samples must be >= 6.")

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


def _resample_path(group_df, n_points=24):
    """Resample cumulative intraday position path to fixed-length vector."""
    position_values = group_df["cumulative_pos"].astype(float).to_numpy()
    if position_values.size == 0:
        return np.zeros(int(n_points), dtype=float)
    if position_values.size == 1:
        return np.full(int(n_points), float(position_values[0]), dtype=float)
    t = np.linspace(0.0, 1.0, position_values.size)
    t_uniform = np.linspace(0.0, 1.0, int(n_points))
    return np.interp(t_uniform, t, position_values)


def _normalized_path_from_group(group_df, sod_pos, eod_pos, n_points=24):
    """Return scale-normalized resampled path using SOD/EOD anchored scale."""
    resampled_path = _resample_path(group_df, n_points=int(n_points))
    scale = max(abs(float(sod_pos)), abs(float(eod_pos)), 1.0)
    return resampled_path / scale


def _compute_path_feature_set(normalized_path):
    """Compute path-abnormality descriptors from a normalized path."""
    if len(normalized_path) == 0:
        return 0.0, 0.0, 0.0
    linear_path = np.linspace(normalized_path[0], normalized_path[-1], len(normalized_path))
    excess_intraday_area = float(
        np.mean(np.maximum(np.abs(normalized_path) - np.abs(linear_path), 0.0))
    )
    peak_time = float(np.argmax(np.abs(normalized_path)) / max(len(normalized_path) - 1, 1))
    flow_volatility = float(np.std(np.diff(normalized_path), ddof=0)) if len(normalized_path) > 1 else 0.0
    return excess_intraday_area, peak_time, flow_volatility


def _compute_path_risk_scores(leakage_df, group_frames, config):
    """Compute FPCA residuals + robust Mahalanobis path-risk scores."""
    if leakage_df.empty:
        return pd.DataFrame(index=leakage_df.index)

    n_points = int(config["resample_points"])
    paths = []
    missing_indices = []
    for idx, row in leakage_df.iterrows():
        group_ids = (row["ExecDate"], row["Portfolio"], row["Underlying"], row["Maturity"])
        group_df = group_frames.get(group_ids)
        if group_df is None or group_df.empty:
            paths.append(np.zeros(n_points, dtype=float))
            missing_indices.append(idx)
            continue
        normalized_path = _normalized_path_from_group(
            group_df,
            sod_pos=row["SOD_Position"],
            eod_pos=row["EOD_Position"],
            n_points=n_points,
        )
        paths.append(normalized_path)

    X_path = np.vstack(paths)
    n_samples = X_path.shape[0]

    explained_var = float(config["path_variance_retained"])
    pca = PCA(n_components=explained_var, svd_solver="full")

    if n_samples >= int(config["min_training_samples"]):
        pca.fit(X_path)
        transformed = pca.transform(X_path)
        reconstructed = pca.inverse_transform(transformed)
        fpca_residual = np.linalg.norm(X_path - reconstructed, axis=1)
        components_used = int(pca.n_components_)
    else:
        centered = X_path - X_path.mean(axis=0, keepdims=True)
        fpca_residual = np.linalg.norm(centered, axis=1)
        components_used = 0

    excess_area = np.zeros(n_samples, dtype=float)
    peak_time = np.zeros(n_samples, dtype=float)
    flow_volatility = np.zeros(n_samples, dtype=float)
    for i, path in enumerate(X_path):
        eia, ptime, fvol = _compute_path_feature_set(path)
        excess_area[i] = eia
        peak_time[i] = ptime
        flow_volatility[i] = fvol

    X_risk = np.column_stack([fpca_residual, excess_area, peak_time, flow_volatility])
    feature_df = pd.DataFrame(
        X_risk,
        columns=["FPCA_Residual", "Excess_Intraday_Area", "Peak_Time", "Flow_Volatility"],
        index=leakage_df.index,
    ).replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.apply(
        lambda col: col.fillna(float(col.median()) if not col.dropna().empty else 0.0)
    )

    try:
        if len(feature_df) >= int(config["min_training_samples"]):
            mcd = MinCovDet(support_fraction=config.get("mcd_support_fraction"))
            mcd.fit(feature_df.to_numpy(dtype=float))
            distances = mcd.mahalanobis(feature_df.to_numpy(dtype=float))
        else:
            centered = feature_df.to_numpy(dtype=float) - feature_df.to_numpy(dtype=float).mean(axis=0)
            distances = np.linalg.norm(centered, axis=1)
    except Exception:
        centered = feature_df.to_numpy(dtype=float) - feature_df.to_numpy(dtype=float).mean(axis=0)
        distances = np.linalg.norm(centered, axis=1)

    rank = pd.Series(distances, index=leakage_df.index).rank(method="average", pct=True)
    risk_percentile = (rank * 100.0).astype(float)

    scores_df = pd.DataFrame(index=leakage_df.index)
    scores_df["Excess_Intraday_Area"] = feature_df["Excess_Intraday_Area"].astype(float)
    scores_df["FPCA_Residual"] = feature_df["FPCA_Residual"].astype(float)
    scores_df["Peak_Time"] = feature_df["Peak_Time"].astype(float)
    scores_df["Flow_Volatility"] = feature_df["Flow_Volatility"].astype(float)
    scores_df["Risk_Distance"] = pd.Series(distances, index=leakage_df.index).astype(float)
    scores_df["Risk_Percentile"] = risk_percentile.astype(float)
    scores_df["FPCA_Components_Used"] = float(components_used)
    if missing_indices:
        scores_df.loc[missing_indices, "Risk_Percentile"] = 50.0
    return scores_df


def _triage_from_percentile(risk_percentile):
    """Policy-banded triage from statistical percentile rank."""
    if pd.isna(risk_percentile):
        return "operational-review"
    if float(risk_percentile) >= 95.0:
        return "investigate"
    if float(risk_percentile) >= 80.0:
        return "operational-review"
    return "de-prioritize"


def _build_explainability_statement(row):
    """Create audit-ready explainability text for a scored leakage case."""
    pct = float(row.get("Risk_Percentile", 0.0))
    excess_area = float(row.get("Excess_Intraday_Area", 0.0))
    peak_time = float(row.get("Peak_Time", 0.0))
    peak_desc = "late-day" if peak_time >= 0.66 else "mid-day" if peak_time >= 0.33 else "early-day"
    return (
        f"This case is in the {pct:.1f}th percentile of abnormal intraday paths, "
        f"driven by excess exposure area ({excess_area:.4f}) and {peak_desc} peak positioning."
    )


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

    suppressed_df = leakage_df[leakage_df["Action"] == "de-prioritize"].copy() if "Action" in leakage_df.columns else pd.DataFrame()
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
            ["De-prioritized Cases",                 f"{len(suppressed_df):,}"],
            ["De-prioritization Rate",               f"{suppression_rate:.2f} %"],
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
            "Excess_Intraday_Area",
            "FPCA_Residual",
            "Risk_Distance",
            "Risk_Percentile",
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
            "Pre-Market", "SOD_Pos", "EOD_Pos", "Max_Intraday", "Leakage_Gap",
            "EIA", "FPCA_Residual", "Risk_Distance", "Risk_Pct", "Action",
        ]
        for col in ["Pre-Market", "SOD_Pos", "EOD_Pos", "Max_Intraday", "Leakage_Gap", "EIA", "FPCA_Residual", "Risk_Distance", "Risk_Pct"]:
            t3.align[col] = "r"
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
                f"{float(row.get('Excess_Intraday_Area', np.nan)):.4f}" if not pd.isna(row.get("Excess_Intraday_Area", np.nan)) else "N/A",
                f"{float(row.get('FPCA_Residual', np.nan)):.4f}" if not pd.isna(row.get("FPCA_Residual", np.nan)) else "N/A",
                f"{float(row.get('Risk_Distance', np.nan)):.4f}" if not pd.isna(row.get("Risk_Distance", np.nan)) else "N/A",
                f"{float(row.get('Risk_Percentile', np.nan)):.2f}" if not pd.isna(row.get("Risk_Percentile", np.nan)) else "N/A",
                row.get("Action", "investigate"),
            ])
        w(t3)

        # ------------------------------------------------------------------
        # Section 3b: Explainability Statements
        # ------------------------------------------------------------------
        if "Explainability_Statement" in sorted_leakage.columns:
            w(f"\n{sep}")
            w("  EXPLAINABILITY STATEMENTS")
            w(sep)
            for _, row in sorted_leakage.head(25).iterrows():
                w(
                    f"- {row['ExecDate']} | {row['Portfolio']} | {row['Underlying']} | "
                    f"{row.get('Explainability_Statement', '')}"
                )

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
    plot_actions=("investigate",),
    plot_min_gap=0.0,
    align_plots_with_exports=True,
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
        `Risk_Percentile`, `Risk_Distance`, `FPCA_Residual`,
        `Excess_Intraday_Area`, `Leakage_Gap`.
    max_plots: Optional hard cap on the number of plots to generate.
    plot_actions: Leakage triage actions eligible for plotting.
    plot_min_gap: Minimum Leakage_Gap required to be plot-eligible.
    align_plots_with_exports: If True, `Leakage_Flagged_Trades.csv`
        contains only plotted-group trades; if False, exports all
        non-de-prioritized leakage groups.
    fp_config: Optional dictionary for path-statistical config. Supported
        keys include `resample_points`, `path_variance_retained`,
        `min_training_samples`, `mcd_support_fraction`, and `epsilon`.
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
    if plot_min_gap < 0:
        raise ValueError("plot_min_gap must be >= 0.")
    config = _build_fp_config(fp_config)
    plot_actions = tuple(str(action) for action in plot_actions)
    if not plot_actions:
        raise ValueError("plot_actions must contain at least one action.")

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
                "Leakage_Detected": is_leakage,
            }
        )

    results_df = pd.DataFrame(summary_data)
    _log_progress(5, total_steps, f"Daily groups evaluated: {len(results_df):,}")

    if results_df.empty:
        results_df["Action"] = []
        results_df["Incremental_Intraday_Risk"] = []
        results_df["Excess_Intraday_Area"] = []
        results_df["FPCA_Residual"] = []
        results_df["Peak_Time"] = []
        results_df["Flow_Volatility"] = []
        results_df["Risk_Distance"] = []
        results_df["Risk_Percentile"] = []
        results_df["FPCA_Components_Used"] = []
        results_df["Explainability_Statement"] = []
    else:
        _log_progress(6, total_steps, "Scoring leakage paths and assigning percentile triage")
        results_df["Incremental_Intraday_Risk"] = (
            results_df["Max_Intraday_Position"] - results_df["SOD_Position"].abs()
        ).clip(lower=0.0)
        results_df["Excess_Intraday_Area"] = np.nan
        results_df["FPCA_Residual"] = np.nan
        results_df["Peak_Time"] = np.nan
        results_df["Flow_Volatility"] = np.nan
        results_df["Risk_Distance"] = np.nan
        results_df["Risk_Percentile"] = np.nan
        results_df["FPCA_Components_Used"] = np.nan
        results_df["Explainability_Statement"] = ""
        results_df["Action"] = "none"

        leakage_mask = results_df["Leakage_Detected"]
        leakage_cases = results_df[leakage_mask].copy()
        _log_progress(6, total_steps, f"Leakage cases detected: {len(leakage_cases):,}")
        if not leakage_cases.empty:
            risk_scores = _compute_path_risk_scores(leakage_cases, group_frames, config)
            score_cols = [
                "Excess_Intraday_Area",
                "FPCA_Residual",
                "Peak_Time",
                "Flow_Volatility",
                "Risk_Distance",
                "Risk_Percentile",
                "FPCA_Components_Used",
            ]
            results_df.loc[risk_scores.index, score_cols] = risk_scores[score_cols]
            results_df.loc[risk_scores.index, "Action"] = risk_scores["Risk_Percentile"].apply(
                _triage_from_percentile
            )

        results_df.loc[leakage_mask & results_df["Risk_Percentile"].isna(), "Risk_Percentile"] = 50.0
        results_df.loc[leakage_mask & results_df["Action"].eq("none"), "Action"] = "operational-review"

        results_df.loc[~leakage_mask, ["Action", "Explainability_Statement"]] = ["none", ""]
        results_df.loc[~leakage_mask, "Incremental_Intraday_Risk"] = 0.0
        results_df.loc[~leakage_mask, [
            "Excess_Intraday_Area",
            "FPCA_Residual",
            "Peak_Time",
            "Flow_Volatility",
            "Risk_Distance",
            "Risk_Percentile",
            "FPCA_Components_Used",
        ]] = np.nan

        results_df["Risk_Percentile"] = results_df["Risk_Percentile"].astype(float).clip(lower=0.0, upper=100.0)
        results_df["Risk_Distance"] = results_df["Risk_Distance"].astype(float)
        results_df["Excess_Intraday_Area"] = results_df["Excess_Intraday_Area"].astype(float)
        results_df["FPCA_Residual"] = results_df["FPCA_Residual"].astype(float)
        results_df["Peak_Time"] = results_df["Peak_Time"].astype(float)
        results_df["Flow_Volatility"] = results_df["Flow_Volatility"].astype(float)
        results_df["FPCA_Components_Used"] = results_df["FPCA_Components_Used"].astype(float)

        leakage_idx = results_df[results_df["Leakage_Detected"]].index
        results_df.loc[leakage_idx, "Explainability_Statement"] = results_df.loc[
            leakage_idx
        ].apply(_build_explainability_statement, axis=1)
        results_df["Action"] = results_df["Action"].astype(str)

    # 6) Plot only the highest-ranked flagged leakage groups.
    _log_progress(7, total_steps, "Selecting and generating leakage plots")
    metric_candidates = {
        "Risk_Percentile",
        "Risk_Distance",
        "FPCA_Residual",
        "Excess_Intraday_Area",
        "Leakage_Gap",
    }
    if plot_metric not in metric_candidates:
        raise ValueError(
            f"Invalid plot_metric '{plot_metric}'. Use one of: {sorted(metric_candidates)}"
        )

    leakage_df = results_df[
        (results_df["Leakage_Detected"])
        & (results_df["Action"].isin(plot_actions))
        & (results_df["Leakage_Gap"] >= float(plot_min_gap))
    ].copy()
    plotted_count = 0
    selected_keys = set()
    plot_manifest_records = []

    plot_selection_df = leakage_df.sort_values(plot_metric, ascending=False).copy()
    plot_selection_df["Plot_Rank"] = np.arange(1, len(plot_selection_df) + 1)
    plot_selection_df["Selected_For_Plot"] = False

    if not plot_selection_df.empty and plot_top_pct > 0:
        top_n = max(1, int(np.ceil(len(plot_selection_df) * (plot_top_pct / 100.0))))
        if max_plots is not None:
            top_n = min(top_n, int(max_plots))
        to_plot_df = plot_selection_df.head(top_n).copy()
        plot_selection_df.loc[to_plot_df.index, "Selected_For_Plot"] = True

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
                f"Peak: {max_exposure:,.0f} | EOD: {eod_pos:,.0f} | {plot_metric}: {float(row[plot_metric]):,.2f}",
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
            png_name = f"Leakage_{safe_name}.png"
            plt.savefig(os.path.join(output_folder, png_name))
            plt.close()
            plotted_count += 1
            selected_keys.add(group_ids)
            plot_manifest_records.append(
                {
                    "ExecDate": row["ExecDate"],
                    "Portfolio": row["Portfolio"],
                    "Underlying": row["Underlying"],
                    "Maturity": row["Maturity"],
                    "Action": row["Action"],
                    "plot_metric": plot_metric,
                    "plot_metric_value": float(row[plot_metric]),
                    "plot_rank": int(row["Plot_Rank"]),
                    "png_filename": png_name,
                }
            )

            _log_progress(7, total_steps, f"Plots generated: {plotted_count}")

    # 7) Map selected/escalated leakage cases back to original trade-level DataFrame.
    _log_progress(8, total_steps, "Mapping selected leakage groups back to original trades")
    leakage_summary = results_df[results_df["Leakage_Detected"]].copy()
    escalated_summary = leakage_summary[
        leakage_summary["Action"].isin(["investigate", "operational-review"])
    ].copy()
    suppressed_candidates = leakage_summary[leakage_summary["Action"] == "de-prioritize"].copy()

    escalated_keys = set(
        zip(
            escalated_summary["ExecDate"],
            escalated_summary["Portfolio"],
            escalated_summary["Underlying"],
            escalated_summary["Maturity"],
        )
    )
    export_keys = selected_keys if align_plots_with_exports else escalated_keys
    if not export_keys and not align_plots_with_exports:
        export_keys = escalated_keys

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
        in export_keys,
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
         "Max_Intraday_Position", "Leakage_Gap",
         "Incremental_Intraday_Risk", "Excess_Intraday_Area", "FPCA_Residual",
         "Peak_Time", "Flow_Volatility", "Risk_Distance", "Risk_Percentile",
         "FPCA_Components_Used", "Explainability_Statement", "Action"]]

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

    plot_selection_csv = os.path.join(output_folder, "Plot_Selection_Manifest.csv")
    manifest_df = plot_selection_df[
        [
            "ExecDate",
            "Portfolio",
            "Underlying",
            "Maturity",
            "Action",
            "Leakage_Gap",
            "Incremental_Intraday_Risk",
            "Excess_Intraday_Area",
            "FPCA_Residual",
            "Risk_Distance",
            "Risk_Percentile",
            "Plot_Rank",
            "Selected_For_Plot",
        ]
    ].copy()
    if plot_manifest_records:
        png_map = pd.DataFrame(plot_manifest_records)[
            ["ExecDate", "Portfolio", "Underlying", "Maturity", "png_filename", "plot_rank"]
        ].rename(columns={"plot_rank": "Plot_Rank"})
        manifest_df = manifest_df.merge(
            png_map,
            on=["ExecDate", "Portfolio", "Underlying", "Maturity", "Plot_Rank"],
            how="left",
        )
    else:
        manifest_df["png_filename"] = ""
    manifest_df.to_csv(plot_selection_csv, index=False)

    suppressed_csv = os.path.join(output_folder, "Deprioritized_Leakage_Candidates.csv")
    suppressed_candidates.to_csv(suppressed_csv, index=False)

    # 9) Write auditor report to text file.
    audit_report_path = os.path.join(output_folder, "Audit_Report.txt")
    _write_audit_report(results_df, flagged_trades_df, audit_report_path)

    print(f"Plots Generated              : {plotted_count}")
    print(f"Plot Candidates              : {len(plot_selection_df)}")
    print(f"Plot Groups Selected         : {len(selected_keys)}")
    print(f"Plot Selection Manifest CSV  : {plot_selection_csv}")
    print(f"Export Scope                 : {'plotted-groups' if align_plots_with_exports else 'all-escalated'}")
    print(f"Full Report                  : {csv_path}")
    print(f"Flagged Trades CSV           : {flagged_trades_csv}")
    print(f"De-prioritized Cases CSV     : {suppressed_csv}")
    print(f"Audit Report TXT             : {audit_report_path}")

    if plotted_count != len(selected_keys):
        LOGGER.warning(
            "Reconciliation mismatch: selected_groups=%s, plotted_count=%s",
            len(selected_keys),
            plotted_count,
        )
    _log_progress(9, total_steps, "Analysis complete")

    return results_df, flagged_trades_df


if __name__ == "__main__":
    df_trades = pd.read_csv("output/synthetic_trading_data.csv")
    results, flagged = analyze_intraday_leakage_continuous(
        df_trades, plot_top_pct=100, plot_metric="Risk_Percentile", max_plots=20
    )
