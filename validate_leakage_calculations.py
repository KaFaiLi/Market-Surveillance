"""
Validation script to verify leakage calculations are correct.

Reads Leakage_Flagged_Trades.csv and Full_Leakage_Report_Continuous.csv,
then spot-checks the arithmetic for a sample of flagged groups.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def validate_leakage_calculations(
    flagged_trades_path: str,
    full_report_path: str,
    sample_size: int = 5,
):
    """Validate leakage calculations by recomputing from trade-level data.

    Parameters
    ----------
    flagged_trades_path : str
        Path to Leakage_Flagged_Trades.csv
    full_report_path : str
        Path to Full_Leakage_Report_Continuous.csv
    sample_size : int
        Number of random groups to validate (or all if fewer exist)
    """
    print("=" * 80)
    print("LEAKAGE CALCULATION VALIDATION")
    print("=" * 80)

    # Load data
    flagged_df = pd.read_csv(flagged_trades_path)
    report_df = pd.read_csv(full_report_path)

    print(f"\nLoaded {len(flagged_df):,} flagged trades")
    print(f"Loaded {len(report_df):,} daily position groups")

    # Filter to leakage-detected groups only
    leakage_report = report_df[report_df["Leakage_Detected"]].copy()
    print(f"Found {len(leakage_report):,} leakage-detected groups\n")

    if leakage_report.empty:
        print("No leakage detected - nothing to validate.")
        return

    # Sample groups to check
    n_sample = min(sample_size, len(leakage_report))
    sample_groups = leakage_report.sample(n=n_sample, random_state=42)

    print(f"Validating {n_sample} randomly sampled groups...\n")
    print("=" * 80)

    all_passed = True
    for idx, group_row in sample_groups.iterrows():
        exec_date = group_row["ExecDate"]
        portfolio = group_row["Portfolio"]
        underlying = group_row["Underlying"]
        maturity = group_row["Maturity"]

        print(f"\n[Group {idx + 1}] {exec_date} | {portfolio} | {underlying}")
        print("-" * 80)

        # Extract trades for this group
        group_trades = flagged_df[
            (flagged_df["ExecDate"] == exec_date)
            & (flagged_df["Portfolio"] == portfolio)
            & (flagged_df["Underlying"] == underlying)
            & (flagged_df["Maturity"] == maturity)
        ].copy()

        if group_trades.empty:
            print("  ⚠️  No trades found for this group (possible data mismatch)")
            all_passed = False
            continue

        # Parse hour buckets if not present (recreate from execTime_parsed or execTime)
        if "hour_bucket" not in group_trades.columns:
            if "execTime_parsed" in group_trades.columns:
                group_trades["hour_bucket"] = pd.to_datetime(
                    group_trades["execTime_parsed"]
                ).dt.floor("H")
            elif "execTime" in group_trades.columns:
                # Parse execTime and floor to hour
                group_trades["execTime_parsed"] = pd.to_datetime(
                    group_trades["execTime"].str.split("[").str[0],
                    errors="coerce",
                    utc=True,
                )
                group_trades["hour_bucket"] = group_trades["execTime_parsed"].dt.floor("H")
            else:
                print("  ⚠️  Cannot determine hour buckets - missing time columns")
                all_passed = False
                continue

        # Aggregate by hour bucket (matching the analysis logic)
        hourly_agg = (
            group_trades.groupby("hour_bucket")["signed_qty"]
            .sum()
            .reset_index()
            .sort_values("hour_bucket")
        )

        # Expected values from the report
        expected_prior_eod = group_row["Prior_EOD_Position"]
        expected_sod = group_row["SOD_Position"]
        expected_eod = group_row["EOD_Position"]
        expected_max_intraday = group_row["Max_Intraday_Position"]
        expected_leakage_gap = group_row["Leakage_Gap"]
        expected_max_to_eod_ratio = group_row["Max_to_EOD_Ratio"]

        # Compute actual values from hourly aggregated trades
        net_flow = hourly_agg["signed_qty"].sum()
        
        # Check 1: Prior_EOD + net_flow = EOD
        computed_eod = expected_prior_eod + net_flow
        eod_match = np.isclose(computed_eod, expected_eod, rtol=1e-9, atol=1e-6)

        print(f"  Trade count           : {len(group_trades):>15,}")
        print(f"  Hour buckets          : {len(hourly_agg):>15,}")
        print(f"  Prior_EOD_Position    : {expected_prior_eod:>15,.2f}")
        print(f"  Net Flow (Σ signed_qty): {net_flow:>15,.2f}")
        print(f"  Prior_EOD + Net Flow  : {computed_eod:>15,.2f}")
        print(f"  Expected EOD_Position : {expected_eod:>15,.2f}")
        print(f"  ✓ EOD Match           : {'PASS' if eod_match else 'FAIL ❌'}")

        if not eod_match:
            all_passed = False
            print(f"    Difference          : {abs(computed_eod - expected_eod):,.6f}")

        # Check 2: Rebuild cumulative position path from HOURLY buckets
        hourly_agg["cumulative_position"] = expected_prior_eod + hourly_agg["signed_qty"].cumsum()
        
        computed_sod = hourly_agg["cumulative_position"].iloc[0]
        computed_eod_final = hourly_agg["cumulative_position"].iloc[-1]
        computed_max_intraday = hourly_agg["cumulative_position"].abs().max()

        # Verify: first hour flow should match SOD - Prior_EOD
        first_hour_flow = hourly_agg["signed_qty"].iloc[0]
        expected_first_hour = expected_sod - expected_prior_eod
        first_hour_match = np.isclose(first_hour_flow, expected_first_hour, rtol=1e-9, atol=1e-6)

        sod_match = np.isclose(computed_sod, expected_sod, rtol=1e-9, atol=1e-6)
        eod_final_match = np.isclose(computed_eod_final, expected_eod, rtol=1e-9, atol=1e-6)
        max_match = np.isclose(computed_max_intraday, expected_max_intraday, rtol=1e-9, atol=1e-6)

        print(f"\n  First Hour Flow Check:")
        print(f"    First hour net flow : {first_hour_flow:>15,.2f}")
        print(f"    Expected (SOD-Prior): {expected_first_hour:>15,.2f}")
        print(f"    ✓ First Hour Match  : {'PASS' if first_hour_match else 'FAIL ❌'}")

        print(f"\n  SOD_Position (after 1st hour):")
        print(f"    Computed            : {computed_sod:>15,.2f}")
        print(f"    Expected            : {expected_sod:>15,.2f}")
        print(f"    ✓ SOD Match         : {'PASS' if sod_match else 'FAIL ❌'}")

        print(f"\n  Max_Intraday_Position:")
        print(f"    Computed            : {computed_max_intraday:>15,.2f}")
        print(f"    Expected            : {expected_max_intraday:>15,.2f}")
        print(f"    ✓ Max Match         : {'PASS' if max_match else 'FAIL ❌'}")

        if not first_hour_match:
            all_passed = False
        if not sod_match:
            all_passed = False
        if not max_match:
            all_passed = False

        # Check 3: Leakage gap calculation
        computed_leakage_gap = computed_max_intraday - abs(expected_eod)
        gap_match = np.isclose(computed_leakage_gap, expected_leakage_gap, rtol=1e-9, atol=1e-6)

        print(f"\n  Leakage_Gap:")
        print(f"    Computed (Max - |EOD|): {computed_leakage_gap:>15,.2f}")
        print(f"    Expected              : {expected_leakage_gap:>15,.2f}")
        print(f"    ✓ Gap Match           : {'PASS' if gap_match else 'FAIL ❌'}")

        if not gap_match:
            all_passed = False

        # Check 4: Max-to-EOD ratio
        computed_ratio = computed_max_intraday / (abs(expected_eod) + 1e-9)
        ratio_match = np.isclose(computed_ratio, expected_max_to_eod_ratio, rtol=1e-6, atol=1e-6)

        print(f"\n  Max_to_EOD_Ratio:")
        print(f"    Computed              : {computed_ratio:>15,.4f}")
        print(f"    Expected              : {expected_max_to_eod_ratio:>15,.4f}")
        print(f"    ✓ Ratio Match         : {'PASS' if ratio_match else 'FAIL ❌'}")

        if not ratio_match:
            all_passed = False

        # Summary for this group
        group_pass = eod_match and first_hour_match and sod_match and max_match and gap_match and ratio_match
        print(f"\n  {'✅ ALL CHECKS PASSED' if group_pass else '❌ SOME CHECKS FAILED'}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ VALIDATION COMPLETE - ALL SAMPLED GROUPS PASSED")
    else:
        print("❌ VALIDATION COMPLETE - SOME GROUPS FAILED (see details above)")
    print("=" * 80 + "\n")

    return all_passed


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Default paths - adjust as needed
    base_folder = Path("hourly_risk_analysis_continuous")
    
    if not base_folder.exists():
        print(f"Folder not found: {base_folder}")
        print("Looking for recent output folders...")
        output_dir = Path("output")
        if output_dir.exists():
            # Find most recent _smoke_test or other folder
            candidates = [
                d for d in output_dir.iterdir() 
                if d.is_dir() and (d / "Leakage_Flagged_Trades.csv").exists()
            ]
            if candidates:
                base_folder = sorted(candidates, key=lambda x: x.stat().st_mtime)[-1]
                print(f"Using most recent folder: {base_folder}")
            else:
                print("No valid output folders found.")
                sys.exit(1)
        else:
            print("No output directory found. Run the analysis first.")
            sys.exit(1)

    flagged_csv = base_folder / "Leakage_Flagged_Trades.csv"
    full_report_csv = base_folder / "Full_Leakage_Report_Continuous.csv"

    if not flagged_csv.exists() or not full_report_csv.exists():
        print(f"Missing required files in {base_folder}")
        print(f"  Flagged trades: {flagged_csv.exists()}")
        print(f"  Full report: {full_report_csv.exists()}")
        sys.exit(1)

    # Run validation
    validate_leakage_calculations(
        str(flagged_csv),
        str(full_report_csv),
        sample_size=5,
    )
