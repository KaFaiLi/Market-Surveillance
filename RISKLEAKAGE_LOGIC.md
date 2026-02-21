# Intraday Risk Leakage Logic (Detailed)

## Purpose

The script in `riskleakage.py` detects intraday risk leakage events, triages them statistically, and now **aligns plots and exports through one explicit selection pipeline**.

Primary triage outcomes:

- `suppress` (likely false positive),
- `operational-review` (ambiguous),
- `investigate` (higher risk).

Key governance additions in the current logic:

- configurable plot eligibility by action (`plot_actions`),
- optional minimum leakage threshold for plotting (`plot_min_gap`),
- risk-priority ranking metric (`Risk_Priority_Score`),
- plot selection manifest (`Plot_Selection_Manifest.csv`),
- optional strict alignment of plotted groups and trade export (`align_plots_with_exports`).

---

## End-to-End Pipeline

`analyze_intraday_leakage_continuous()` runs in nine logged steps:

1. **Input validation + setup**
   - Creates output folder.
   - Validates `plot_top_pct`, `max_plots`, `plot_min_gap`, and `plot_actions`.
   - Builds FP scoring configuration.

2. **Timestamp normalization**
   - Parses `execTime` into UTC.
   - Converts timestamps to local market timezone from `underlyingCurrency`.
   - Buckets executions into local hourly buckets.

3. **Signed flow construction**
   - BUY → positive quantity, SELL → negative quantity.

4. **Hourly aggregation**
   - Aggregates signed flow by `portfolioId`, `underlyingId`, `maturity`, `hour_bucket`.

5. **Position path + leakage metrics**
   - Reconstructs cumulative position path.
   - Computes group metrics (`Prior_EOD_Position`, `SOD_Position`, `EOD_Position`, `Max_Intraday_Position`, `Leakage_Gap`, `Max_to_EOD_Ratio`).
   - Computes normalized model features.

6. **Statistical scoring + triage**
   - Scores leakage groups with Isolation Forest.
   - Produces `fp_score ∈ [0,1]`.
   - Maps to `Action`, `Likely_FP`, and `FP_Reasons`.

7. **Plot selection + generation**
   - Computes `Risk_Priority_Score`.
   - Filters candidates by (`Leakage_Detected`, `Action ∈ plot_actions`, `Leakage_Gap >= plot_min_gap`).
   - Ranks by selected `plot_metric`.
   - Marks `Selected_For_Plot` and generates PNGs.

8. **Trade mapping with alignment policy**
   - Builds `selected_keys` from plotted groups.
   - If `align_plots_with_exports=True`, flagged trades are limited to plotted groups.
   - If `False`, flagged trades include all non-suppressed groups.

9. **Exports + reconciliation**
   - Writes full report, flagged trades, suppressed candidates, plot-selection manifest, and audit report.
   - Prints candidate/selected/plotted counts and export scope.
   - Warns if selected groups and generated plots diverge.

---

## Leakage Detection Rule (Before Triage)

A group is `Leakage_Detected=True` when all are true:

1. EOD exposure is non-zero,
2. intraday peak exposure exceeds EOD exposure,
3. peak is not equal to opening (SOD) exposure.

This identifies candidate events before FP triage.

---

## Scale-Free Features Used for FP Scoring

For each daily group:

1. **`eod_retention`**
   - \( |EOD| / |Max\_Intraday| \)

2. **`excess_excursion`**
   - \( (Max - |SOD|) / ((|SOD| + |EOD|)/2) \)

3. **`flow_asymmetry`**
   - \( Peak\_Hour\_Index / Total\_Hours \)

4. **`intraday_volatility`**
   - \( std(hourly\_flow) / mean(|cumulative\_position|) \)

---

## Isolation Forest Scoring and Triage

### Model behavior

- Model: `IsolationForest`.
- Input: leakage-detected groups only.
- Raw model output is min-max scaled to `fp_score` in `[0,1]`.

### Fallback behavior

If training data is too small or degenerate, leakage cases get neutral score `0.5`.

### Action mapping

- `fp_score >= p_suppress` → `suppress`
- `p_review <= fp_score < p_suppress` → `operational-review`
- `fp_score < p_review` → `investigate`

Reason codes:

- `high_fp_score`, `mid_fp_score`, `low_fp_score`, `score_unavailable`.

---

## Risk-Priority Ranking for Plotting

The function computes:

- `Risk_Priority_Score = 0.45*norm(Leakage_Gap) + 0.30*norm(Max_to_EOD_Ratio) + 0.25*norm(Incremental_Intraday_Risk) - 0.20*fp_score`

Implementation notes:

- each risk input is min-max normalized over current results,
- `fp_score` is used as a penalty,
- final score is clipped at zero,
- this metric can be selected via `plot_metric="Risk_Priority_Score"`.

---

## Plot Candidate and Selection Logic

Candidate groups are filtered by:

- `Leakage_Detected == True`,
- `Action` in `plot_actions` (default: `("investigate",)`),
- `Leakage_Gap >= plot_min_gap` (default `0.0`).

Selection and plotting:

- candidates are sorted by `plot_metric`,
- `Plot_Rank` is assigned,
- top `ceil(N * plot_top_pct/100)` groups are selected,
- optional `max_plots` cap is applied,
- selected rows are marked `Selected_For_Plot=True`,
- one PNG per selected group is generated.

---

## Plot/CSV Alignment Policy

`align_plots_with_exports` controls trade-level export scope:

- `True` (default): `Leakage_Flagged_Trades.csv` contains **only plotted groups**,
- `False`: `Leakage_Flagged_Trades.csv` contains **all escalated groups** (`Action != suppress`).

This prevents silent drift between what is plotted and what is exported.

---

## Outputs

Generated in output folder:

- `Full_Leakage_Report_Continuous.csv` (all group metrics + triage),
- `Leakage_Flagged_Trades.csv` (trade-level rows based on alignment policy),
- `Suppressed_Leakage_Candidates.csv`,
- `Plot_Selection_Manifest.csv` (candidate/selection transparency),
- `Audit_Report.txt`,
- `Leakage_*.png` (for selected groups).

### `Plot_Selection_Manifest.csv` fields

Includes, at minimum:

- group keys (`ExecDate`, `Portfolio`, `Underlying`, `Maturity`),
- `Action`, leakage metrics, `fp_score`, `Risk_Priority_Score`,
- `Plot_Rank`, `Selected_For_Plot`,
- `png_filename` for selected rows.

---

## Main Parameters

### Function plotting/alignment parameters

- `plot_top_pct` (0–100),
- `plot_metric` in `{Leakage_Gap, Max_Intraday_Position, Max_to_EOD_Ratio, Risk_Priority_Score}`,
- `max_plots` (optional cap),
- `plot_actions` (tuple of eligible actions),
- `plot_min_gap` (minimum `Leakage_Gap`),
- `align_plots_with_exports` (bool).

### `fp_config` parameters

- `contamination` (default `0.10`),
- `p_suppress` (default `0.75`),
- `p_review` (default `0.40`),
- `model_random_state` (default `42`),
- `min_training_samples` (default `8`),
- `epsilon` (default `1e-9`).

Validation:

- `0 <= p_review < p_suppress <= 1`,
- `0 < contamination <= 0.5`,
- `plot_min_gap >= 0`,
- non-empty `plot_actions`.

---

## Progress and Reconciliation Logging

With `log_progress=True`, logs include:

- input size,
- hourly bucket count,
- daily group count,
- leakage case count,
- generated plot count,
- flagged trade count,
- completion signal.

Final console summary also prints:

- plot candidate count,
- selected plot-group count,
- manifest path,
- export scope mode,
- reconciliation warning if selected groups != plotted files.
