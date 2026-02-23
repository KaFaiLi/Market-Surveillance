# Intraday Risk Leakage — v1 Logic (Auditor View)

This document explains the detection logic implemented in `riskleakage_v1.py` from an auditor's perspective. It covers data normalization, the detection algorithm, outputs, assumptions and limitations, and how to investigate flagged cases.

## Purpose

Detect cases where intraday position peaks are materially larger than the end-of-day position — i.e., the firm was exposed to higher intraday risk than what EOD snapshots record ("leakage"). The goal is to provide auditable metrics and trade-level detail to support investigation.

---

## 1) Timestamp normalisation (critical for auditability)

- Input: execution timestamps like `2024-02-09T17:00:29.719+01:00[Europe/Paris]`.
- Parsing steps:
  1. Drop optional bracketed IANA suffix (if present) and keep the ISO timestamp + numeric offset.
  2. Parse to a UTC-aware datetime.
  3. Resolve the instrument's market timezone using the instrument currency (e.g., CHF → `Europe/Zurich`) from `MARKET_CLOSE_TIMES`.
  4. Convert the UTC datetime to the market-local timezone.
  5. Truncate to the hour to produce an `hour_bucket` (e.g., 09:00 represents events from 09:00:00 to 09:59:59 local time).
  6. IMPORTANT: the timezone is stripped from `hour_bucket` (stored as naive local time) before aggregation to avoid cross-timezone key collisions in pandas groupby.

Audit implications:
- All intraday times are expressed and compared in the instrument's market local time, making detection consistent with exchange hours and easier for human review.
- The naive-hour approach prevents pandas from collapsing identical UTC instants from different markets into one group.

---

## 2) Signed quantity

- Trades are converted to signed flow:
  - `BUY` -> `+quantity`
  - `SELL` -> `-quantity`

This preserves direction and lets cumulative sums reconstruct intraday position paths.

---

## 3) Hourly net flow aggregation

- Aggregation key: `(portfolioId, underlyingId, maturity, underlyingCurrency, hour_bucket)`.
- For each hour bucket per position key, net signed quantity is summed.

Why hourly buckets?
- Hourly granularity balances resolution and noise; it reconstructs the intraday position path without being overly sensitive to tick-level noise.

---

## 4) Rebuild intraday position path

- The cumulative intraday position is reconstructed by applying a running sum (`cumsum`) of hourly net flows within each position key.
- This yields a time series of the book at hour boundaries for each (portfolio × instrument × maturity) group.

Auditor note: This is computed solely from raw executions and hence is independently verifiable from the trade tape.

---

## 5) Leakage metrics per daily group

For each daily group `(execDate, portfolioId, underlyingId, maturity, underlyingCurrency)` the following are calculated:

- `Prior_EOD_Position`: Position carried into the day (computed as SOD − first hour net flow).
- `SOD_Position`: Position after the first hour bucket.
- `EOD_Position`: Position after the final hour bucket.
- `Max_Intraday_Position`: Maximum absolute cumulative position across all buckets.

Derived metrics:
- `Leakage_Gap = Max_Intraday_Position - EOD_Position`
- `Max_to_EOD_Ratio = Max_Intraday_Position / (EOD_Position + 1e-9)`
- `Max_to_Prior_EOD_Ratio = Max_Intraday_Position / (Prior_EOD_Position + 1e-9)`
- `Max_to_Baseline_EOD_Ratio` where `Baseline_EOD = max(Prior_EOD_Position, EOD_Position)`

Flagging rule (all must hold):
1. At least 3 hourly buckets in the day (`bin_count > 2`).
2. Not a mixed-zero SOD/EOD edge case (filters out ambiguous single-sided zeros).
3. `Max_Intraday_Position > EOD_Position`.
4. `Max_Intraday_Position > Prior_EOD_Position`.

Rationale:
- This combination isolates intraday peaks that are larger than both the prior carry and the close, indicating the position expanded intraday and then reduced again — the classic leakage pattern.

---

## 6) Charting and selection

- Flagged groups are ranked by a chosen metric (default: `Leakage_Gap`).
- The top `plot_top_pct`% (e.g., 5%) are plotted (capped by `max_plots`).
- Each plot shows:
  - Hourly cumulative position bars.
  - A dashed red line connecting SOD to EOD (visual baseline).
  - Title with peak, EOD, and ranking metric.
  - X-axis labelled with local market timezone name for clarity.

Auditor tip: charts are intended for quick triage and provide immediate visual evidence of the intraday peak vs EOD.

---

## 7) Trade-level mapping for investigation

- All original trade rows belonging to flagged groups are extracted and enriched with the group's leakage metrics.
- This allows an investigator to drill down from the flagged aggregate to the specific trades, times, and quantities that created the pattern.

Suggested audit steps:
1. Open the flagged group's chart to verify the intraday peak visually.
2. Load the corresponding rows in `Leakage_Flagged_Trades.csv` and examine sequence, participants (if available), brokers, and any clustering in time.
3. Cross-check with trader logs, order-management-system records, or external confirmations to determine intentionality.

---

## 8) Outputs

- `Full_Leakage_Report_Continuous.csv` — row per (date × position key) with metrics and `Leakage_Detected` flag.
- `Leakage_Flagged_Trades.csv` — individual trade rows for flagged groups with metrics.
- `Audit_Report.txt` — human-readable summary and ranking tables for auditors.
- `Leakage_*.png` — plots for top-ranked flagged cases.

---

## Limitations & audit cautions

- No cross-portfolio netting: positions are evaluated per `portfolioId`. Firm-level exposure may require consolidation outside this tool.
- No absolute materiality filter: tiny flows with large ratios will be flagged; auditors should apply materiality thresholds when triaging.
- `Prior_EOD_Position` is inferred from the first-hour flow, not from an external position snapshot. If prior-day trades/snaps are missing, this estimate can be inaccurate.
- The logic detects patterns, not intent. A legitimate intraday hedge or temporary market-making exposure will be indistinguishable from an intentional obfuscation of risk without further investigation.

---

## How auditors should use the outputs (recommended workflow)

1. Run the analysis or use pre-generated outputs found in `hourly_risk_analysis_continuous/`.
2. Review `Audit_Report.txt` for summary counts and highest-risk groups.
3. Open the plotted PNG for top-ranked cases to confirm visual evidence.
4. For confirmed-suspicious cases, open the matching rows in `Leakage_Flagged_Trades.csv` and reconstruct the trade sequence.
5. Escalate to trade owners, compliance, and operations with the chart + supporting trade list + recommended next steps (e.g., replay in OMS, check trader notes).

---

## Contact / Next steps
- If you want, I can:
  - Add a materiality filter (min absolute peak value) to reduce noise.
  - Produce consolidated firm-level leakage by netting across portfolios.
  - Add CSV columns that show the exact timestamps of SOD peak and EOD unwind for faster matching against logs.


*File saved: `riskleakage_v1_audit_logic.md`*
