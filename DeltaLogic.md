# Delta Exposure Logic — Risk Leakage Analysis

## Overview

This document describes the delta exposure calculation methodology applied in `riskleakage_delta.py`. The system extends the existing position-based leakage detection by computing EUR-denominated delta exposure across the portfolio intraday, enabling a more economically meaningful leakage signal.

Delta exposure replaces raw position as the **primary leakage flag driver**. Raw position metrics are retained as informational columns.

---

## 1. Instrument Types

Two instrument types are supported, identified by the `dealType` field in the trading data.

| dealType | Instrument | Description |
|----------|------------|-------------|
| `FUT` | Futures | Exchange-traded futures contracts with a point value multiplier |
| `SHA` | Shares (Equities) | Cash equities, delta-adjusted by beta |

---

## 2. Per-Trade Delta Notional

Delta notional is computed at the individual trade level immediately after FX rate merging.

### 2.1 Futures (FUT)

```
Delta Notional = signed_qty × price × futurePointValue × FX_rate_to_EUR
```

Where:
- `signed_qty` = `quantity × side`, with `side = +1` (Buy) or `side = −1` (Sell)
- `price` = last traded price (used as a proxy for market price)
- `futurePointValue` = contract multiplier (e.g. 50 for ES, 10 for Stoxx)
- `FX_rate_to_EUR` = conversion rate from instrument currency to EUR, sourced from `currency_rates.xlsx`

### 2.2 Shares (SHA)

```
Delta Notional = β × price × signed_qty × FX_rate_to_EUR
```

Where:
- `β` (beta) = sensitivity of the stock to the index. **Currently hardcoded to 1.0** as a placeholder pending integration of a beta dataset.
- `price` = last traded price (premium field), used as a proxy for spot price
- `signed_qty` = shares held (signed for buy/sell direction)
- `FX_rate_to_EUR` = reporting-currency FX rate from `currency_rates.xlsx`

> **Note on Beta:** Beta = 1.0 is a conservative default. Once a beta dataset is integrated, `beta_i` should be merged on instrument identifier and execution date to reflect time-varying sensitivity.

---

## 3. FX Rate Mapping

FX rates are sourced from `currency_rates.xlsx`. Each row represents a currency pair and its rate to EUR for a given date.

- Rates are merged onto the trade dataset on `execDate` and instrument `currency`.
- If a rate is missing for a currency/date combination, the fallback is the most recent available rate for that currency.
- EUR-denominated instruments have an FX rate of 1.0.

---

## 4. Latest Traded Price (Mark-to-Market Proxy)

Market prices are not available. The **latest traded price** within a position group is used as the price proxy for mark-to-market valuation.

- Trades are sorted by `execTime_parsed` ascending within each position group.
- The last traded `premium` within each hour bucket is taken as the representative price for that hour.
- Prices are then **forward-filled** across subsequent hour buckets where no trades occurred, so that a position always carries a valuation even in quiet hours.
- `futurePointValue` is forward-filled in the same manner.

---

## 5. Hourly Aggregation

Trades are bucketed into hourly intervals (`hour_bucket`) per position group. The position group key is:

```
[execDate, portfolioId, accountId, dealType, underlying]
```

For each position group × hour bucket, the following are computed:

| Column | Aggregation |
|--------|-------------|
| `signed_qty` | Sum (net flow for the hour) |
| `delta_notional` | Sum (net delta flow for the hour) |
| `latest_price` | Last traded price in the hour, then forward-filled |
| `latest_fpv` | Last futurePointValue in the hour, then forward-filled |
| `rate_to_eur` | First (constant within a day/currency) |
| `dealType` | First (consistent within a position group) |

---

## 6. Cumulative Delta Exposure (Snapshot Method)

The **snapshot approach** is used to compute cumulative delta exposure — i.e., a mark-to-market valuation of the net position at each hour — rather than a pure cumulative sum of trade-level delta flows.

```
cumulative_pos(t) = Σ signed_qty up to hour t

cumulative_delta_exposure(t) = cumulative_pos(t) × latest_price(t) × multiplier(t) × FX_rate_to_EUR
```

Where `multiplier(t)` is:
- `futurePointValue` for FUT
- `beta` (1.0) for SHA

This ensures the exposure reflects the **current economic value** of the net position at each point in time, incorporating price movements even when no new trades occur.

---

## 7. Portfolio-Level Aggregation

Portfolio-level delta exposure aggregates across all underlyings within a portfolio for each hour.

**Steps:**

1. For each underlying, generate a **full hourly grid** spanning the earliest to latest traded hour on each `execDate`.
2. Forward-fill each underlying's `cumulative_delta_exposure` into hours where it did not trade (position persists, price carried forward).
3. Sum `cumulative_delta_exposure` across all underlyings per `[execDate, portfolioId, hour_bucket]` to obtain `portfolio_delta_exposure`.

This means an underlying's exposure does not drop to zero simply because it was inactive in a given hour — it persists until it is unwound.

---

## 8. Leakage Detection — Delta Exposure Basis

Leakage detection operates on both the underlying level and the portfolio level using the same logic.

### 8.1 Key Metrics

| Metric | Description |
|--------|-------------|
| `SOD_Delta_Exposure` | Delta exposure at start of day (first hour bucket) |
| `EOD_Delta_Exposure` | Delta exposure at end of day (last hour bucket) |
| `Max_Intraday_Delta_Exposure` | Peak absolute delta exposure during the day |
| `Prior_EOD_Delta_Exposure` | EOD delta exposure from the previous trading day |
| `Delta_Leakage_Gap` | `Max_Intraday_Delta_Exposure − EOD_Delta_Exposure` |
| `Delta_Max_to_EOD_Ratio` | `Max / EOD` ratio |
| `Delta_Max_to_Prior_EOD_Ratio` | `Max / Prior EOD` ratio |
| `Delta_Max_to_Baseline_EOD_Ratio` | `Max / max(EOD, Prior EOD)` ratio |

### 8.2 Leakage Flag Conditions

A leakage flag (`Leakage_Detected = True`) is raised when **all** of the following are true:

1. `Max_Intraday_Delta_Exposure > EOD_Delta_Exposure` — the peak exposure was not maintained to end of day
2. `Max_Intraday_Delta_Exposure > Prior_EOD_Delta_Exposure` — the peak exceeded the prior close (not merely unwinding an inherited position)
3. `bin_count > 2` — at least 3 active hour buckets, filtering out single-trade events
4. `NOT mixed_zero_sod_eod` — filters cases where SOD and EOD are both zero (round-trip noise)

> Position-based metrics (`SOD_Position`, `EOD_Position`, `Leakage_Gap`, etc.) are computed and retained as informational columns but **do not drive the leakage flag**.

---

## 9. Output Files

| File | Description |
|------|-------------|
| `Full_Leakage_Report_Continuous.csv` | Per-underlying leakage report with all delta exposure metrics |
| `Portfolio_Delta_Exposure_Report.csv` | Portfolio-level aggregated delta exposure, one row per portfolio × date |
| `Leakage_Flagged_Trades.csv` | Individual trades belonging to flagged position groups, enriched with `delta_notional` and leakage metrics |
| `Audit_Report.txt` | Statistical summary of delta exposure metrics, flagged case detail, and portfolio-level summary |
| `*.png` plots | Per-underlying dual-axis plots (position bars + delta exposure line) and portfolio-level delta exposure bar charts |

---

## 10. Assumptions and Limitations

| Area | Assumption / Limitation |
|------|------------------------|
| **Beta** | Hardcoded to 1.0 for all SHA instruments. Future integration with a beta dataset will allow instrument- and date-specific values. |
| **Market Price** | No real-time market data. Latest traded price (`premium`) is used as a proxy. This underestimates exposure for positions acquired early in the day if prices moved significantly. |
| **FX Rates** | Daily FX rates are used. Intraday FX movements are not modelled. |
| **futurePointValue** | Sourced from trade data and forward-filled. Assumed constant within a trading day per contract. |
| **Netting** | Positions are netted within the position group key (`portfolioId × accountId × dealType × underlying`). Cross-instrument netting is captured at the portfolio level only. |
| **EOD Definition** | End of day is the last traded hour bucket for each position group on each `execDate`. It does not correspond to a fixed market close time. |
| **Short Positions** | Negative `signed_qty` produces negative delta exposure, correctly representing a short position. The `Max_Intraday_Delta_Exposure` uses absolute value to detect peak exposure regardless of direction. |

---

## 11. Worked Example

### FUT Trade

| Field | Value |
|-------|-------|
| dealType | FUT |
| quantity | 10 |
| way | Sell |
| premium (price) | 4,500.00 |
| futurePointValue | 50 |
| currency | USD |
| FX_rate_to_EUR | 0.92 |

```
signed_qty       = 10 × (−1) = −10
delta_notional   = −10 × 4,500 × 50 × 0.92 = −2,070,000 EUR
```

### SHA Trade

| Field | Value |
|-------|-------|
| dealType | SHA |
| quantity | 5,000 |
| way | Buy |
| premium (price) | 150.00 |
| beta | 1.0 |
| currency | GBP |
| FX_rate_to_EUR | 1.17 |

```
signed_qty       = 5,000 × (+1) = 5,000
delta_notional   = 1.0 × 150 × 5,000 × 1.17 = 877,500 EUR
```

---

*Last updated: February 2026*