# Intraday Risk Leakage Logic (Path-Based Statistical Design)

## Purpose

The script in `riskleakage.py` detects intraday risk leakage and triages cases using a pure statistical, path-based method.

The redesign removes dependence on endpoint-ratio heuristics and fixed model thresholds, while preserving full audit traceability.

Primary triage outcomes:

- `investigate`
- `operational-review`
- `de-prioritize`

---

## Architecture Summary

Current implementation flow:

1. Hourly position reconstruction
2. Fixed-length path resampling
3. SOD/EOD-anchored path normalization
4. FPCA-based path reconstruction residual
5. Additional statistical path descriptors
6. Robust multivariate distance scoring
7. Empirical percentile ranking
8. Policy-band triage (by percentile)

No labels are required. No contamination or score-threshold tuning is required.

---

## End-to-End Pipeline

`analyze_intraday_leakage_continuous()` runs in nine logged steps:

1. **Input validation + setup**
   - Creates output folder.
   - Validates `plot_top_pct`, `max_plots`, `plot_min_gap`, and `plot_actions`.
   - Builds path-statistical configuration (`fp_config`).

2. **Timestamp normalization**
   - Parses `execTime` into UTC.
   - Converts to local market timezone from `underlyingCurrency`.
   - Buckets executions by local hour.

3. **Signed flow construction**
   - BUY → positive quantity, SELL → negative quantity.

4. **Hourly aggregation**
   - Aggregates signed flow by `portfolioId`, `underlyingId`, `maturity`, `hour_bucket`.

5. **Position path + leakage candidate detection**
   - Rebuilds cumulative position path.
   - Computes core metrics:
     - `Prior_EOD_Position`
     - `SOD_Position`
     - `EOD_Position`
     - `Max_Intraday_Position`
     - `Leakage_Gap`
   - Marks `Leakage_Detected` candidates.

6. **Path-risk scoring + triage**
   - Resamples each path to fixed length.
   - Normalizes by `max(|SOD|, |EOD|, 1.0)`.
   - Computes FPCA residual and path descriptors.
   - Computes robust distance and empirical percentile.
   - Maps percentile to `Action`.

7. **Plot selection + generation**
   - Filters candidates by leakage flag, action policy, and minimum gap.
   - Ranks by selected `plot_metric`.
   - Selects top percentage (plus optional `max_plots` cap).
   - Generates one PNG per selected group.

8. **Trade mapping with alignment policy**
   - If `align_plots_with_exports=True`, exports only plotted groups.
   - Else exports all escalated groups (`investigate` + `operational-review`).

9. **Exports + reconciliation**
   - Writes full report, flagged trades, de-prioritized candidates,
     selection manifest, and audit report.
   - Prints plot/export reconciliation summary.

---

## Leakage Detection Rule (Before Statistical Scoring)

A group is `Leakage_Detected=True` when all are true:

1. EOD exposure is non-zero.
2. Intraday peak exposure exceeds EOD exposure.
3. Intraday peak is not the same as opening (SOD) exposure.

This remains a candidate-generation rule. Statistical scoring is applied afterwards.

---

## Path Representation and Normalization

### 1) Resampling

Each group path is resampled to a fixed length (`resample_points`, default 24) via linear interpolation.

### 2) Scale normalization

The resampled path is normalized by:

\[
	ext{scale} = \max(|SOD|, |EOD|, 1.0)
\]

\[
P_{norm}(t) = \frac{P(t)}{\text{scale}}
\]

This suppresses small-EOD denominator artifacts while preserving execution shape.

---

## Statistical Features

For each normalized path:

1. **Excess Intraday Area** (`Excess_Intraday_Area`)

   \[
   L(t) = \text{linear path from } P_{norm}(0) \text{ to } P_{norm}(1)
   \]
   \[
   EIA = \operatorname{mean}(\max(|P_{norm}(t)| - |L(t)|, 0))
   \]

2. **FPCA Residual** (`FPCA_Residual`)
   - PCA is fitted to normalized paths (variance retained = `path_variance_retained`, default 0.90).
   - Residual is reconstruction error norm.

3. **Peak Time** (`Peak_Time`)
   - Relative timing of max absolute exposure in normalized path.

4. **Flow Volatility** (`Flow_Volatility`)
   - Standard deviation of first differences in normalized path.

---

## Multivariate Statistical Scoring

The feature vector per leakage candidate is:

\[
X = [\text{FPCA Residual}, \text{Excess Area}, \text{Peak Time}, \text{Flow Volatility}]
\]

Scoring method:

- Robust covariance estimator: `MinCovDet`
- Distance metric: robust Mahalanobis distance (`Risk_Distance`)
- Rank conversion: empirical percentile (`Risk_Percentile` in 0–100)

Fallback behavior:

- If sample size is too small or robust fit fails, Euclidean distance from centered feature space is used.

---

## Statistical Decisioning Policy

Actions are assigned from `Risk_Percentile`:

- `>= 95` → `investigate`
- `>= 80 and < 95` → `operational-review`
- `< 80` → `de-prioritize`

These are policy bands on percentile rank, not model-fit thresholds.

---

## Explainability and Audit Traceability

Each scored leakage case includes:

- `Excess_Intraday_Area`
- `FPCA_Residual`
- `Peak_Time`
- `Flow_Volatility`
- `Risk_Distance`
- `Risk_Percentile`
- `FPCA_Components_Used`
- `Explainability_Statement`

The explainability statement is auto-generated and references percentile abnormality plus key path drivers.

---

## Plot Candidate and Selection Logic

Candidate filter:

- `Leakage_Detected == True`
- `Action` in `plot_actions` (default: `("investigate",)`)
- `Leakage_Gap >= plot_min_gap`

Selection:

- Sort by `plot_metric`.
- Assign `Plot_Rank`.
- Select top `ceil(N * plot_top_pct/100)`.
- Apply optional `max_plots` cap.
- Mark `Selected_For_Plot=True` and render PNG.

Allowed plot metrics:

- `Risk_Percentile`
- `Risk_Distance`
- `FPCA_Residual`
- `Excess_Intraday_Area`
- `Leakage_Gap`

---

## Plot/CSV Alignment Policy

`align_plots_with_exports` controls trade-level export scope:

- `True` (default): `Leakage_Flagged_Trades.csv` contains plotted groups only.
- `False`: `Leakage_Flagged_Trades.csv` contains all escalated groups (`investigate`, `operational-review`).

---

## Outputs

Generated in the configured output folder:

- `Full_Leakage_Report_Continuous.csv`
- `Leakage_Flagged_Trades.csv`
- `Deprioritized_Leakage_Candidates.csv`
- `Plot_Selection_Manifest.csv`
- `Audit_Report.txt`
- `Leakage_*.png` for selected groups

### `Plot_Selection_Manifest.csv` fields

Includes at minimum:

- `ExecDate`, `Portfolio`, `Underlying`, `Maturity`
- `Action`
- `Leakage_Gap`
- `Incremental_Intraday_Risk`
- `Excess_Intraday_Area`
- `FPCA_Residual`
- `Risk_Distance`
- `Risk_Percentile`
- `Plot_Rank`, `Selected_For_Plot`
- `png_filename` (selected rows)

---

## Main Parameters

### Function plotting/alignment parameters

- `plot_top_pct` (0–100)
- `plot_metric` in `{Risk_Percentile, Risk_Distance, FPCA_Residual, Excess_Intraday_Area, Leakage_Gap}`
- `max_plots` (optional)
- `plot_actions` (eligible actions)
- `plot_min_gap` (minimum leakage gap)
- `align_plots_with_exports` (bool)

### `fp_config` parameters

- `resample_points` (default `24`)
- `path_variance_retained` (default `0.90`)
- `min_training_samples` (default `12`)
- `mcd_support_fraction` (default `None`)
- `epsilon` (default `1e-9`)

Validation:

- `resample_points >= 4`
- `path_variance_retained` in `(0, 1]`
- `min_training_samples >= 6`
- `plot_min_gap >= 0`
- non-empty `plot_actions`

---

## Progress and Reconciliation Logging

With `log_progress=True`, logs include:

- input size
- hourly bucket count
- daily group count
- leakage case count
- generated plot count
- flagged trade count
- completion signal

Final console summary prints:

- plot candidate count
- selected plot-group count
- manifest path
- export scope mode
- reconciliation warning if selected groups != plotted files
