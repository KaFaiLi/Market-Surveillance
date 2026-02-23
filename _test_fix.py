"""Quick test to verify underlyingCurrency grouping fix."""
import sys
import pandas as pd
from riskleakage_v1 import analyze_intraday_leakage_continuous
from riskleakage_v1 import _currency_to_timezone, _parse_exec_time_to_utc, _to_local_exec_time

out = open("_test_fix_output.txt", "w")
sys.stdout = out

df = pd.read_csv("output/synthetic_trading_data.csv")
print(f"Input rows: {len(df)}")
print(f"Unique currencies: {sorted(df['underlyingCurrency'].dropna().unique())}")

results, flagged = analyze_intraday_leakage_continuous(
    df,
    output_folder="hourly_risk_analysis_continuous",
    plot_top_pct=0,
    plot_metric="Max_to_EOD_Ratio",
    max_plots=0,
    debug_sorting=False,
)

print(f"\nResults shape: {results.shape}")
print(f"Columns: {list(results.columns)}")
print(f"\nLeakage cases: {results['Leakage_Detected'].sum()}")
leakage = results[results["Leakage_Detected"]]
print("\nSample leakage rows:")
print(leakage[["ExecDate","Portfolio","Underlying","Maturity","Currency","Bin_Count"]].head(10).to_string(index=False))

# Verify no mixed tz per group after fix
df2 = df.copy()
df2["market_timezone"] = df2["underlyingCurrency"].apply(_currency_to_timezone)
df2["execTime_utc"] = df2["execTime"].apply(_parse_exec_time_to_utc)
df2["execTime_local"] = df2.apply(lambda r: _to_local_exec_time(r["execTime_utc"], r["market_timezone"]), axis=1)
df2["hour_bucket"] = df2["execTime_local"].apply(lambda ts: ts.replace(minute=0, second=0, microsecond=0) if ts is not None else None)

keys = ["portfolioId", "underlyingId", "maturity", "underlyingCurrency"]
mixed = df2.groupby(keys)["market_timezone"].nunique(dropna=False)
print(f"\nGroups with mixed market_timezone: {int((mixed > 1).sum())}  (expected: 0)")

bad = 0
for gkeys, grp in df2.groupby(keys):
    tzs = grp["hour_bucket"].dropna().apply(lambda x: str(getattr(x, "tzinfo", None))).nunique()
    if tzs > 1:
        bad += 1
print(f"Groups with mixed tzinfo in hour_bucket: {bad}  (expected: 0)")

out.close()
sys.stdout = sys.__stdout__
print("Test complete. See _test_fix_output.txt")
