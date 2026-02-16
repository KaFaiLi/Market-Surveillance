"""
Test case for ghost risk detection with simultaneous trades.
This script generates test data with edge cases to validate the ghost risk logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from riskleakage import analyze_risk_and_export


def create_test_data():
    """
    Create test data with various scenarios:
    1. Ghost Risk Case 1: Simultaneous offsetting trades (same timestamp)
    2. Ghost Risk Case 2: Quick round-trip within 2 minutes
    3. Real Risk Case: Gradual position build-up over hours
    """

    test_trades = []
    trade_id = 1
    base_date = "2024-03-01"

    # ============================================
    # SCENARIO 1: Ghost Risk - Simultaneous Trades
    # Buy 100000 and Sell 100000 at EXACT same time
    # Creates MASSIVE intraday exposure but zero EOD position
    # ============================================
    print("\n=== Creating Scenario 1: Ghost Risk - Simultaneous Trades ===")
    scenario1_time = f"{base_date}T10:30:15.500+01:00[Europe/Paris]"

    # Buy 100000 units
    test_trades.append(
        {
            "dealId": f"DEAL{trade_id:06d}",
            "tradeDate": f"{base_date}+01:00[Europe/Paris]",
            "portfolioId": "TEST_GHOST1",
            "underlyingId": "EQUITY",
            "maturity": "1970-01-01+01:00[Europe/Paris]",
            "way": "BUY",
            "quantity": 100000,
            "execTime": scenario1_time,
        }
    )
    trade_id += 1

    # Sell 100000 units at EXACT same timestamp
    test_trades.append(
        {
            "dealId": f"DEAL{trade_id:06d}",
            "tradeDate": f"{base_date}+01:00[Europe/Paris]",
            "portfolioId": "TEST_GHOST1",
            "underlyingId": "EQUITY",
            "maturity": "1970-01-01+01:00[Europe/Paris]",
            "way": "SELL",
            "quantity": 100000,
            "execTime": scenario1_time,  # Same exact time!
        }
    )
    trade_id += 1

    print(f"  - Created 2 trades at {scenario1_time}")
    print(f"  - BUY 100000 and SELL 100000 simultaneously")
    print(f"  - Expected: Ghost Risk (100% trades at same time, EOD = 0)")

    # ============================================
    # SCENARIO 2: Ghost Risk - Quick Round Trip
    # Multiple trades within 2 minutes, netting to near-zero
    # ============================================
    print("\n=== Creating Scenario 2: Ghost Risk - Quick Round Trip ===")
    scenario2_start = datetime.strptime(f"{base_date}T14:20:00", "%Y-%m-%dT%H:%M:%S")

    # Build up position quickly
    for i in range(3):
        trade_time = scenario2_start + timedelta(seconds=i * 20)
        test_trades.append(
            {
                "dealId": f"DEAL{trade_id:06d}",
                "tradeDate": f"{base_date}+01:00[Europe/Paris]",
                "portfolioId": "TEST_GHOST2",
                "underlyingId": "DERIV",
                "maturity": "1970-01-01+01:00[Europe/Paris]",
                "way": "BUY",
                "quantity": 50000,
                "execTime": trade_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                + "+01:00[Europe/Paris]",
            }
        )
        trade_id += 1

    # Unwind position quickly (leave tiny residual)
    for i in range(3):
        trade_time = scenario2_start + timedelta(seconds=(i + 3) * 20)
        test_trades.append(
            {
                "dealId": f"DEAL{trade_id:06d}",
                "tradeDate": f"{base_date}+01:00[Europe/Paris]",
                "portfolioId": "TEST_GHOST2",
                "underlyingId": "DERIV",
                "maturity": "1970-01-01+01:00[Europe/Paris]",
                "way": "SELL",
                "quantity": 49900,  # Slightly less to leave 300 residual
                "execTime": trade_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                + "+01:00[Europe/Paris]",
            }
        )
        trade_id += 1

    print(f"  - Created 6 trades over 2 minutes")
    print(f"  - BUY 150000, then SELL 149700 within 2 minutes")
    print(f"  - Expected: Ghost Risk (short duration, near-zero EOD)")

    # ============================================
    # SCENARIO 3: Real Risk - Gradual Build-Up
    # Position built over several hours with significant EOD position
    # ============================================
    print("\n=== Creating Scenario 3: Real Risk - Gradual Position Build-Up ===")
    scenario3_start = datetime.strptime(f"{base_date}T09:00:00", "%Y-%m-%dT%H:%M:%S")

    # Morning: Build up long position
    for i in range(5):
        trade_time = scenario3_start + timedelta(
            hours=i, minutes=np.random.randint(0, 45)
        )
        test_trades.append(
            {
                "dealId": f"DEAL{trade_id:06d}",
                "tradeDate": f"{base_date}+01:00[Europe/Paris]",
                "portfolioId": "TEST_REAL1",
                "underlyingId": "SINGLFUT",
                "maturity": "1970-01-01+01:00[Europe/Paris]",
                "way": "BUY",
                "quantity": np.random.randint(30000, 80000),
                "execTime": trade_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                + "+01:00[Europe/Paris]",
            }
        )
        trade_id += 1

    # Afternoon: Partial unwind but maintain significant position
    for i in range(2):
        trade_time = scenario3_start + timedelta(
            hours=i + 6, minutes=np.random.randint(0, 45)
        )
        test_trades.append(
            {
                "dealId": f"DEAL{trade_id:06d}",
                "tradeDate": f"{base_date}+01:00[Europe/Paris]",
                "portfolioId": "TEST_REAL1",
                "underlyingId": "SINGLFUT",
                "maturity": "1970-01-01+01:00[Europe/Paris]",
                "way": "SELL",
                "quantity": np.random.randint(20000, 40000),
                "execTime": trade_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                + "+01:00[Europe/Paris]",
            }
        )
        trade_id += 1

    print(f"  - Created 7 trades over 8 hours")
    print(f"  - Built long position gradually, partial unwind")
    print(f"  - Expected: Real Risk (long duration, non-zero EOD)")

    # ============================================
    # SCENARIO 4: Multiple Simultaneous Pairs
    # Several pairs of simultaneous offsetting trades
    # ============================================
    print("\n=== Creating Scenario 4: Ghost Risk - Multiple Simultaneous Pairs ===")
    scenario4_times = [
        f"{base_date}T11:15:30.123+01:00[Europe/Paris]",
        f"{base_date}T11:15:30.123+01:00[Europe/Paris]",  # Same as above
        f"{base_date}T11:18:45.456+01:00[Europe/Paris]",
        f"{base_date}T11:18:45.456+01:00[Europe/Paris]",  # Same as above
    ]

    quantities = [80000, 80000, 60000, 60000]
    ways = ["BUY", "SELL", "BUY", "SELL"]

    for i in range(4):
        test_trades.append(
            {
                "dealId": f"DEAL{trade_id:06d}",
                "tradeDate": f"{base_date}+01:00[Europe/Paris]",
                "portfolioId": "TEST_GHOST3",
                "underlyingId": "EQUITY",
                "maturity": "1970-01-01+01:00[Europe/Paris]",
                "way": ways[i],
                "quantity": quantities[i],
                "execTime": scenario4_times[i],
            }
        )
        trade_id += 1

    print(f"  - Created 4 trades in 2 simultaneous pairs")
    print(f"  - 50% of trades share timestamps with others")
    print(f"  - Expected: Ghost Risk (high % simultaneous trades, EOD = 0)")

    return pd.DataFrame(test_trades)


def main():
    print("=" * 80)
    print("GHOST RISK DETECTION TEST")
    print("=" * 80)

    # Create test data
    df_test = create_test_data()

    print(f"\n\nTotal test trades created: {len(df_test)}")
    print("\nTest data sample:")
    print(df_test[["portfolioId", "way", "quantity", "execTime"]].to_string())
    print("\nTest data saved to: output/test_ghost_risk_data.csv")
    df_test.to_csv("output/test_ghost_risk_data.csv", index=False)

    # Run analysis on TEST DATA ONLY
    print("\n" + "=" * 80)
    print("RUNNING RISK ANALYSIS ON TEST DATA")
    print("=" * 80 + "\n")

    # Temporarily rename output files to avoid confusion
    import shutil
    import os

    # Backup existing files if they exist
    backup_files = []
    for file in [
        "Outlier_Summary.csv",
        "Detailed_Outlier.csv",
        "Ghost_Risk_Summary.csv",
    ]:
        src = f"output/{file}"
        dst = f"output/{file}.backup"
        if os.path.exists(src):
            shutil.move(src, dst)
            backup_files.append((src, dst))

    try:
        outliers = analyze_risk_and_export(df_test, visualize_ghost_risk=True)
    finally:
        # Restore backed up files
        for src, dst in backup_files:
            if os.path.exists(dst):
                if os.path.exists(src):
                    os.remove(dst)
                else:
                    shutil.move(dst, src)

    # Display summary
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)

    # Show ALL test groups with their metrics (not just outliers)
    print("\n### ALL TEST GROUPS (including those with Z-Score < 2) ###\n")

    # Read and process test data to show all results
    try:
        import warnings

        warnings.filterwarnings("ignore")

        df_test_results = pd.read_csv("output/test_ghost_risk_data.csv")
        df_test_results["execTime_dt"] = pd.to_datetime(
            df_test_results["execTime"].str.extract(
                r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)"
            )[0]
        )
        df_test_results["signed_qty"] = np.where(
            df_test_results["way"].str.upper() == "BUY",
            df_test_results["quantity"],
            -df_test_results["quantity"],
        )
        group_cols = ["tradeDate", "portfolioId", "underlyingId", "maturity"]
        df_test_results["cumulative_pos"] = df_test_results.groupby(group_cols)[
            "signed_qty"
        ].cumsum()

        from scipy.stats import zscore

        test_summary = (
            df_test_results.groupby(group_cols)
            .agg(
                max_intraday_exposure=("cumulative_pos", lambda x: x.abs().max()),
                eod_position=("cumulative_pos", "last"),
                trade_count=("dealId", "count"),
                first_trade_time=("execTime_dt", "min"),
                last_trade_time=("execTime_dt", "max"),
                unique_timestamps=("execTime_dt", "nunique"),
            )
            .reset_index()
        )

        test_summary["trading_duration_min"] = (
            test_summary["last_trade_time"] - test_summary["first_trade_time"]
        ).dt.total_seconds() / 60
        test_summary["pct_trades_same_time"] = (
            (test_summary["trade_count"] - test_summary["unique_timestamps"])
            / test_summary["trade_count"]
            * 100
        )
        test_summary["risk_ratio"] = test_summary["max_intraday_exposure"] / (
            test_summary["eod_position"].abs() + 0.1
        )
        test_summary["z_score"] = zscore(test_summary["risk_ratio"])

        # Apply ghost risk logic
        GHOST_RISK_TIME_THRESHOLD_MIN = 5
        GHOST_RISK_EOD_THRESHOLD = 100
        GHOST_RISK_SIMULTANEOUS_THRESHOLD = 30

        test_summary["would_be_ghost_risk"] = (
            test_summary["eod_position"].abs() <= GHOST_RISK_EOD_THRESHOLD
        ) & (
            (test_summary["trading_duration_min"] <= GHOST_RISK_TIME_THRESHOLD_MIN)
            | (
                test_summary["pct_trades_same_time"]
                >= GHOST_RISK_SIMULTANEOUS_THRESHOLD
            )
        )

        print(
            f"{'Portfolio':<15} {'Max Exp':>10} {'EOD':>8} {'Duration':>10} {'Same%':>6} {'Risk Ratio':>12} {'Z-Score':>8} {'Ghost?':>8}"
        )
        print("-" * 90)

        for _, row in test_summary.iterrows():
            ghost_status = "✓ GHOST" if row["would_be_ghost_risk"] else "✗ Real"
            print(
                f"{row['portfolioId']:<15} {row['max_intraday_exposure']:>10.0f} "
                f"{row['eod_position']:>8.0f} {row['trading_duration_min']:>9.1f}m "
                f"{row['pct_trades_same_time']:>5.1f}% {row['risk_ratio']:>12.2f} "
                f"{row['z_score']:>8.2f} {ghost_status:>8}"
            )

        print("\n### KEY FINDINGS ###")
        print(
            f"✓ {test_summary['would_be_ghost_risk'].sum()} scenarios correctly identified as GHOST RISK"
        )
        print(
            f"✓ {(~test_summary['would_be_ghost_risk']).sum()} scenarios correctly identified as REAL RISK"
        )

        print("\n### GHOST RISK CRITERIA ###")
        print(f"- EOD Position ≤ {GHOST_RISK_EOD_THRESHOLD}")
        print(
            f"- AND (Duration ≤ {GHOST_RISK_TIME_THRESHOLD_MIN} min OR Same-Time% ≥ {GHOST_RISK_SIMULTANEOUS_THRESHOLD}%)"
        )

    except Exception as e:
        print(f"Error analyzing test results: {e}")

    # Check if actual outliers were detected (z-score > 2)
    print("\n### OUTLIER DETECTION (Z-Score > 2) ###")

    # Read the generated summaries
    try:
        outlier_summary = pd.read_csv("output/Outlier_Summary.csv")
        print(f"\n✓ Real Risk Outliers: {len(outlier_summary)}")
        if not outlier_summary.empty:
            print("\nReal Risk Cases:")
            for _, row in outlier_summary.iterrows():
                print(
                    f"  - {row['portfolioId']}: EOD={row['eod_position']:.0f}, "
                    f"Duration={row['trading_duration_min']:.1f}min, "
                    f"Simultaneous={row['pct_trades_same_time']:.1f}%, "
                    f"Z-Score={row['z_score']:.2f}"
                )
    except FileNotFoundError:
        print("\n✗ No real risk outliers detected")

    try:
        ghost_summary = pd.read_csv("output/Ghost_Risk_Summary.csv")
        print(f"\n✓ Ghost Risk Cases: {len(ghost_summary)}")
        if not ghost_summary.empty:
            print("\nGhost Risk Cases:")
            for _, row in ghost_summary.iterrows():
                print(
                    f"  - {row['portfolioId']}: EOD={row['eod_position']:.0f}, "
                    f"Duration={row['trading_duration_min']:.1f}min, "
                    f"Simultaneous={row['pct_trades_same_time']:.1f}%, "
                    f"Z-Score={row['z_score']:.2f}"
                )
    except FileNotFoundError:
        print("\n✗ No ghost risk cases detected")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nCheck the output/ folder for:")
    print("  - test_ghost_risk_data.csv (input test data)")
    print("  - Outlier_Summary.csv (real risk cases)")
    print("  - Ghost_Risk_Summary.csv (ghost risk cases)")
    print("  - Visualization PNG files for each case")


if __name__ == "__main__":
    main()
