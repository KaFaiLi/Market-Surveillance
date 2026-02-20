# Intraday Risk Leakage Logic (Detailed)

## Purpose

The script in `riskleakage.py` detects cases where intraday risk peaked materially above end-of-day (EOD) exposure and then triages those cases into:

- `suppress` (likely false positive),
- `operational-review` (ambiguous),
- `investigate` (more anomalous).

It also exports:

- group-level leakage report,
- trade-level flagged rows,
- suppressed candidates,
- audit-ready text report,
- optional plots for top-ranked investigate cases.

---

## End-to-End Pipeline

`analyze_intraday_leakage_continuous()` runs in nine logical stages:

1. **Input validation + setup**
   - Creates output folder.
   - Validates plotting parameters.
   - Builds FP scoring configuration.

2. **Timestamp normalization**
   - Parses `execTime` into UTC.
   - Converts each trade to the local market timezone inferred from `underlyingCurrency`.
   - Buckets execution times to local hourly buckets.

3. **Signed flow construction**
   - BUY → positive quantity.
   - SELL → negative quantity.

4. **Hourly aggregation**
   - Aggregates signed quantities by:
     - `portfolioId`, `underlyingId`, `maturity`, `hour_bucket`.

5. **Path reconstruction and leakage metrics**
   - Rebuilds cumulative position path from hourly net flow.
   - Computes daily group metrics (`Prior_EOD_Position`, `SOD_Position`, `EOD_Position`, `Max_Intraday_Position`, `Leakage_Gap`, `Max_to_EOD_Ratio`).
   - Computes normalized statistical features (below).

6. **Statistical triage scoring**
   - Runs Isolation Forest on leakage groups using normalized features.
   - Converts model decision function into `fp_score ∈ [0,1]` with min-max scaling.
   - Maps score to action by cutoffs.

7. **Plot generation (investigate only)**
   - Selects top groups by chosen metric and percent/cap.
   - Produces bar+line intraday leakage plots.

8. **Trade-level mapping**
   - Maps escalated groups (`operational-review` + `investigate`) back to original trades.
   - Enriches rows with group leakage metrics and triage output.

9. **Exports and audit report**
   - Writes CSVs and formatted TXT audit report.

---

## Leakage Detection Rule (Before Triage)

A group is marked as `Leakage_Detected=True` when:

1. EOD exposure is non-zero,
2. intraday peak exposure is above EOD exposure,
3. peak is not exactly the opening (SOD) exposure.

This stage identifies candidate leakage events. Statistical triage happens after this.

---

## Normalized Features (Scale-Free)

For each daily position group, the script computes:

1. **`eod_retention`**
   - \( |EOD| / |Max\_Intraday| \)
   - Fraction of peak risk still held at close.

2. **`excess_excursion`**
   - \( (Max - |SOD|) / ((|SOD| + |EOD|)/2) \)
   - Intraday risk build-up relative to book size.

3. **`flow_asymmetry`**
   - \( Peak\_Hour\_Index / Total\_Hours \)
   - Where in the day the peak occurs (early vs late).

4. **`intraday_volatility`**
   - \( std(hourly\_net\_flow) / mean(|hourly\_cumpos|) \)
   - Noise/churn relative to typical absolute position level.

These ratios avoid hard notional thresholds and generalize better across books/instruments.

---

## Isolation Forest Scoring

### Features

- `eod_retention`
- `excess_excursion`
- `flow_asymmetry`
- `intraday_volatility`

### Behavior

- Model: `IsolationForest`.
- Input rows: leakage-detected groups.
- Output: decision function (higher = more normal), then min-max scaled to `fp_score` in `[0,1]`.

### Fallbacks

If there are too few samples or features are degenerate:

- score defaults to `0.5` (neutral),
- triage naturally falls into `operational-review` unless cutoffs are altered.

---

## Triage Policy

Configured by two cutoffs:

- `p_suppress` (default `0.75`)
- `p_review` (default `0.40`)

Action mapping:

- `fp_score >= p_suppress` → `suppress`
- `p_review <= fp_score < p_suppress` → `operational-review`
- `fp_score < p_review` → `investigate`

Reason tags:

- `high_fp_score`
- `mid_fp_score`
- `low_fp_score`
- `score_unavailable` (defensive fallback)

---

## Progress Logging

The function supports `log_progress=True` (default).

It logs each major stage with an index like `[step/9]`, including:

- input row count,
- hourly bucket count,
- daily group count,
- leakage case count,
- generated plot count,
- flagged trade export count,
- completion notification.

This makes long runs auditable and easier to monitor in terminal output.

---

## Key Outputs

In the selected output folder:

- `Full_Leakage_Report_Continuous.csv`
- `Leakage_Flagged_Trades.csv`
- `Suppressed_Leakage_Candidates.csv`
- `Audit_Report.txt`
- `Leakage_*.png` (when plotting enabled)

---

## Main Config Knobs (`fp_config`)

- `contamination` (default `0.10`)
- `p_suppress` (default `0.75`)
- `p_review` (default `0.40`)
- `model_random_state` (default `42`)
- `min_training_samples` (default `8`)
- `epsilon` (default `1e-9`)

Validation rules:

- `0 <= p_review < p_suppress <= 1`
- `0 < contamination <= 0.5`

---

## Practical Governance Workflow

1. Run monthly on rolling history.
2. Track outcomes for reviewed/investigated cases.
3. Revisit `contamination`, `p_review`, `p_suppress` quarterly.
4. Keep model random state fixed for reproducibility unless intentionally changed.

This keeps triage simple, data-driven, and auditable.
