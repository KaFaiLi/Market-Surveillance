"""
Marking the Close Analysis
===========================
This script analyzes equity booking data to identify potential "marking the close" issues.
It identifies trades executed in the last 15 minutes before market close and ranks them by quantity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
import re
import os
from zoneinfo import ZoneInfo  # Python 3.9+

# =============================================================================
# MARKET CLOSE TIME MAPPING
# =============================================================================
# Market close times mapped by currency (in local market time)
# Note: These are the regular trading session close times

MARKET_CLOSE_TIMES = {
    # Europe
    "GBP": {
        "market": "London Stock Exchange (LSE)",
        "close_time": time(16, 30),
        "timezone": "Europe/London",
    },
    "EUR": {
        "market": "Euronext (Paris/Amsterdam/Brussels)",
        "close_time": time(17, 30),
        "timezone": "Europe/Paris",
    },
    "CHF": {
        "market": "SIX Swiss Exchange",
        "close_time": time(17, 30),
        "timezone": "Europe/Zurich",
    },
    "SEK": {
        "market": "Nasdaq Stockholm",
        "close_time": time(17, 30),
        "timezone": "Europe/Stockholm",
    },
    "NOK": {
        "market": "Oslo Børs",
        "close_time": time(16, 20),
        "timezone": "Europe/Oslo",
    },
    "DKK": {
        "market": "Nasdaq Copenhagen",
        "close_time": time(17, 0),
        "timezone": "Europe/Copenhagen",
    },
    "PLN": {
        "market": "Warsaw Stock Exchange",
        "close_time": time(17, 0),
        "timezone": "Europe/Warsaw",
    },
    "CZK": {
        "market": "Prague Stock Exchange",
        "close_time": time(17, 0),
        "timezone": "Europe/Prague",
    },
    "HUF": {
        "market": "Budapest Stock Exchange",
        "close_time": time(17, 0),
        "timezone": "Europe/Budapest",
    },
    "RUB": {
        "market": "Moscow Exchange",
        "close_time": time(18, 50),
        "timezone": "Europe/Moscow",
    },
    # Asia Pacific
    "JPY": {
        "market": "Tokyo Stock Exchange",
        "close_time": time(15, 0),
        "timezone": "Asia/Tokyo",
    },
    "HKD": {
        "market": "Hong Kong Stock Exchange",
        "close_time": time(16, 0),
        "timezone": "Asia/Hong_Kong",
    },
    "CNY": {
        "market": "Shanghai/Shenzhen Stock Exchange",
        "close_time": time(15, 0),
        "timezone": "Asia/Shanghai",
    },
    "CNH": {
        "market": "Shanghai/Shenzhen Stock Exchange",
        "close_time": time(15, 0),
        "timezone": "Asia/Shanghai",
    },
    "SGD": {
        "market": "Singapore Exchange",
        "close_time": time(17, 0),
        "timezone": "Asia/Singapore",
    },
    "KRW": {
        "market": "Korea Exchange",
        "close_time": time(15, 30),
        "timezone": "Asia/Seoul",
    },
    "TWD": {
        "market": "Taiwan Stock Exchange",
        "close_time": time(13, 30),
        "timezone": "Asia/Taipei",
    },
    "INR": {
        "market": "National Stock Exchange of India",
        "close_time": time(15, 30),
        "timezone": "Asia/Kolkata",
    },
    "AUD": {
        "market": "Australian Securities Exchange",
        "close_time": time(16, 0),
        "timezone": "Australia/Sydney",
    },
    "NZD": {
        "market": "New Zealand Exchange",
        "close_time": time(17, 0),
        "timezone": "Pacific/Auckland",
    },
    "MYR": {
        "market": "Bursa Malaysia",
        "close_time": time(17, 0),
        "timezone": "Asia/Kuala_Lumpur",
    },
    "THB": {
        "market": "Stock Exchange of Thailand",
        "close_time": time(16, 30),
        "timezone": "Asia/Bangkok",
    },
    "IDR": {
        "market": "Indonesia Stock Exchange",
        "close_time": time(15, 0),
        "timezone": "Asia/Jakarta",
    },
    "PHP": {
        "market": "Philippine Stock Exchange",
        "close_time": time(15, 30),
        "timezone": "Asia/Manila",
    },
    "VND": {
        "market": "Ho Chi Minh Stock Exchange",
        "close_time": time(15, 0),
        "timezone": "Asia/Ho_Chi_Minh",
    },
    # North America
    "USD": {
        "market": "NYSE/NASDAQ",
        "close_time": time(16, 0),
        "timezone": "America/New_York",
    },
    "CAD": {
        "market": "Toronto Stock Exchange",
        "close_time": time(16, 0),
        "timezone": "America/Toronto",
    },
    "MXN": {
        "market": "Mexican Stock Exchange",
        "close_time": time(15, 0),
        "timezone": "America/Mexico_City",
    },
}

# Toggle for volume comparison vs rest-of-day average
ENABLE_VOLUME_COMPARISON = True

# Output folder for all generated files
OUTPUT_DIR = "output"


def ensure_output_dir(output_dir=OUTPUT_DIR):
    """
    Ensure the output directory exists.
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def parse_timestamp(ts_string):
    """
    Parse timestamp string in format '2024-01-02T17:32:38.333+01:00[Europe/Paris]'
    Returns a timezone-aware datetime object in Paris time.

    Note: All execTime values are stored in Paris time regardless of the actual market.
    """
    if pd.isna(ts_string) or ts_string == "":
        return None

    try:
        # Extract the datetime part before the timezone bracket
        match = re.match(
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)([\+\-]\d{2}:\d{2})?\[?([^\]]+)?\]?",
            str(ts_string),
        )
        if match:
            dt_str = match.group(1)
            # Parse without timezone first
            if "." in dt_str:
                dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f")
            else:
                dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
            # Attach Paris timezone since all times are in Paris time
            paris_tz = ZoneInfo("Europe/Paris")
            dt = dt.replace(tzinfo=paris_tz)
            return dt
        return None
    except Exception as e:
        print(f"Error parsing timestamp: {ts_string} - {e}")
        return None


def convert_paris_to_market_time(paris_dt, market_timezone):
    """
    Convert a Paris timezone datetime to the local market timezone.

    Parameters:
    -----------
    paris_dt : datetime
        Datetime object in Paris timezone
    market_timezone : str
        Target timezone string (e.g., 'America/New_York')

    Returns:
    --------
    datetime
        Datetime converted to the market's local timezone
    """
    if paris_dt is None or market_timezone is None:
        return None

    try:
        market_tz = ZoneInfo(market_timezone)
        return paris_dt.astimezone(market_tz)
    except Exception as e:
        print(f"Error converting timezone: {e}")
        return paris_dt


def get_market_info(currency_id):
    """
    Get market information based on currency ID.
    Returns market name, close time, and timezone.
    """
    if currency_id in MARKET_CLOSE_TIMES:
        return MARKET_CLOSE_TIMES[currency_id]
    else:
        return {"market": "Unknown", "close_time": None, "timezone": None}


def is_near_market_close(
    exec_time_paris, close_time, market_timezone, minutes_before=15
):
    """
    Check if execution time is within specified minutes before market close.

    The exec_time is in Paris timezone and needs to be converted to the market's
    local timezone for comparison with the market close time.

    Parameters:
    -----------
    exec_time_paris : datetime
        Execution time in Paris timezone
    close_time : time
        Market close time in local market time
    market_timezone : str
        Market's timezone string
    minutes_before : int
        Window before market close (default: 15 minutes)

    Returns:
    --------
    bool
        True if the trade is within the specified window before market close
    """
    if exec_time_paris is None or close_time is None or market_timezone is None:
        return False

    # Convert Paris time to market local time
    exec_time_local = convert_paris_to_market_time(exec_time_paris, market_timezone)
    if exec_time_local is None:
        return False

    exec_time_only = exec_time_local.time()

    # Calculate the start of the "near close" window
    close_datetime = datetime.combine(datetime.today(), close_time)
    window_start_datetime = close_datetime - timedelta(minutes=minutes_before)
    window_start = window_start_datetime.time()

    # Check if execution time is within the window
    return window_start <= exec_time_only <= close_time


def analyze_marking_the_close(df, minutes_before_close=15, return_analysis=False):
    """
    Analyze trades for potential marking the close behavior.

    Parameters:
    -----------
    df : pd.DataFrame
        The booking data with columns as specified in the schema
    minutes_before_close : int
        Number of minutes before market close to consider (default: 15)

    Returns:
    --------
    pd.DataFrame
        Filtered and ranked DataFrame with trades near market close
    """
    # Create a copy to avoid modifying original
    df_analysis = df.copy()

    # Parse execution time
    df_analysis["execTime_parsed"] = df_analysis["execTime"].apply(parse_timestamp)

    # Add market information based on currency
    df_analysis["market_info"] = df_analysis["currencyId"].apply(get_market_info)
    df_analysis["market_name"] = df_analysis["market_info"].apply(lambda x: x["market"])
    df_analysis["market_close_time"] = df_analysis["market_info"].apply(
        lambda x: x["close_time"]
    )
    df_analysis["market_timezone"] = df_analysis["market_info"].apply(
        lambda x: x["timezone"]
    )

    # Convert Paris time to local market time for each trade
    df_analysis["execTime_local"] = df_analysis.apply(
        lambda row: convert_paris_to_market_time(
            row["execTime_parsed"], row["market_timezone"]
        ),
        axis=1,
    )

    # Check if trade is near market close
    df_analysis["is_near_close"] = df_analysis.apply(
        lambda row: is_near_market_close(
            row["execTime_parsed"],
            row["market_close_time"],
            row["market_timezone"],
            minutes_before_close,
        ),
        axis=1,
    )

    # Compute minutes to close for all rows (local market time)
    def minutes_to_close_all(row):
        if row["execTime_local"] is None or row["market_close_time"] is None:
            return None
        exec_time = row["execTime_local"].time()
        close_time = row["market_close_time"]
        exec_minutes = exec_time.hour * 60 + exec_time.minute + exec_time.second / 60
        close_minutes = close_time.hour * 60 + close_time.minute
        return close_minutes - exec_minutes

    df_analysis["minutes_to_close_all"] = df_analysis.apply(
        minutes_to_close_all, axis=1
    )
    df_analysis["trade_date_local"] = df_analysis["execTime_local"].apply(
        lambda x: x.date() if x is not None else None
    )

    # Filter trades near market close
    df_near_close = df_analysis[df_analysis["is_near_close"]].copy()

    # Add signed quantity (positive for Buy, negative for Sell)
    df_near_close["signed_quantity"] = df_near_close.apply(
        lambda row: row["quantity"] if row["way"] == "Buy" else -row["quantity"], axis=1
    )

    # Calculate minutes before close (using local market time)
    def minutes_to_close(row):
        if row["execTime_local"] is None or row["market_close_time"] is None:
            return None
        exec_time = row["execTime_local"].time()
        close_time = row["market_close_time"]
        exec_minutes = exec_time.hour * 60 + exec_time.minute + exec_time.second / 60
        close_minutes = close_time.hour * 60 + close_time.minute
        return close_minutes - exec_minutes

    df_near_close["minutes_to_close"] = df_near_close.apply(minutes_to_close, axis=1)

    # Add formatted local execution time for display
    df_near_close["execTime_local_str"] = df_near_close["execTime_local"].apply(
        lambda x: x.strftime("%H:%M:%S") if x is not None else ""
    )

    # Add flag reason for export
    df_near_close["flag_reason"] = (
        f"Exec within last {minutes_before_close} minutes before market close"
    )

    # Rank by quantity (absolute value, largest first)
    df_near_close["quantity_rank"] = (
        df_near_close["quantity"].abs().rank(ascending=False, method="dense")
    )

    # Sort by quantity descending
    df_near_close = df_near_close.sort_values("quantity", ascending=False)

    if return_analysis:
        return df_near_close, df_analysis
    return df_near_close


def create_summary_by_asset(df_near_close):
    """
    Create a summary of trades near market close grouped by asset.
    """
    summary = (
        df_near_close.groupby(["assetName", "currencyId", "market_name"])
        .agg(
            {
                "quantity": "sum",
                "signed_quantity": "sum",
                "dealId": "count",
                "premium": "sum",
            }
        )
        .reset_index()
    )

    summary.columns = [
        "Asset",
        "Currency",
        "Market",
        "Total_Quantity",
        "Net_Quantity",
        "Trade_Count",
        "Total_Premium",
    ]
    summary = summary.sort_values("Total_Quantity", ascending=False)

    return summary


def create_summary_by_trader(df_near_close):
    """
    Create a summary of trades near market close grouped by trader.
    """
    summary = (
        df_near_close.groupby(["traderPopsId", "assetName"])
        .agg(
            {
                "quantity": "sum",
                "signed_quantity": "sum",
                "dealId": "count",
                "premium": "sum",
            }
        )
        .reset_index()
    )

    summary.columns = [
        "Trader_ID",
        "Asset",
        "Total_Quantity",
        "Net_Quantity",
        "Trade_Count",
        "Total_Premium",
    ]
    summary = summary.sort_values("Total_Quantity", ascending=False)

    return summary


def _slugify(text):
    """
    Create a safe filename slug from text.
    """
    if text is None:
        return "unknown"
    text = re.sub(r"[^a-zA-Z0-9\-\s]", "", str(text))
    text = re.sub(r"\s+", "_", text.strip())
    return text.lower() if text else "unknown"


def create_market_severity_summary(df_near_close, minutes_before_close=15):
    """
    Create a market-level summary with a severity score for potential marking the close.
    """
    if df_near_close.empty:
        return pd.DataFrame()

    df_temp = df_near_close.copy()
    df_temp["buy_qty"] = np.where(df_temp["way"] == "Buy", df_temp["quantity"], 0)
    df_temp["sell_qty"] = np.where(df_temp["way"] == "Sell", df_temp["quantity"], 0)

    market_summary = (
        df_temp.groupby(["market_name", "currencyId"])
        .agg(
            Total_Quantity=("quantity", "sum"),
            Net_Quantity=("signed_quantity", "sum"),
            Trade_Count=("dealId", "count"),
            Unique_Traders=("traderPopsId", "nunique"),
            Unique_Assets=("assetName", "nunique"),
            Avg_Minutes_To_Close=("minutes_to_close", "mean"),
            Median_Minutes_To_Close=("minutes_to_close", "median"),
            Buy_Quantity=("buy_qty", "sum"),
            Sell_Quantity=("sell_qty", "sum"),
        )
        .reset_index()
    )

    # Imbalance between buy and sell
    market_summary["Buy_Sell_Imbalance"] = (
        market_summary["Buy_Quantity"] - market_summary["Sell_Quantity"]
    ).abs() / (market_summary["Total_Quantity"] + 1e-9)

    # Normalize components for severity score
    max_total_qty = (
        market_summary["Total_Quantity"].max() if not market_summary.empty else 1
    )
    max_trade_count = (
        market_summary["Trade_Count"].max() if not market_summary.empty else 1
    )

    market_summary["Total_Quantity_Norm"] = (
        market_summary["Total_Quantity"] / max_total_qty
    )
    market_summary["Trade_Count_Norm"] = market_summary["Trade_Count"] / max_trade_count
    market_summary["Proximity_Norm"] = 1 - (
        market_summary["Avg_Minutes_To_Close"] / minutes_before_close
    )
    market_summary["Proximity_Norm"] = market_summary["Proximity_Norm"].clip(
        lower=0, upper=1
    )

    # Severity score: higher quantity, more trades, stronger imbalance, closer to close = higher risk
    market_summary["Severity_Score"] = (
        0.40 * market_summary["Total_Quantity_Norm"]
        + 0.25 * market_summary["Trade_Count_Norm"]
        + 0.25 * market_summary["Buy_Sell_Imbalance"]
        + 0.10 * market_summary["Proximity_Norm"]
    )

    # Severity tiering
    market_summary["Severity_Level"] = pd.cut(
        market_summary["Severity_Score"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"],
    )

    market_summary = market_summary.sort_values("Severity_Score", ascending=False)
    return market_summary


def add_volume_comparison(df_analysis, df_near_close, minutes_before_close=15):
    """
    Add volume comparison metrics to flagged deals.
    Compares close-window volume against average 15-min volume in the rest of the day.
    """
    if df_near_close.empty:
        return df_near_close

    valid = df_analysis["minutes_to_close_all"].notna() & (
        df_analysis["minutes_to_close_all"] >= 0
    )
    df_valid = df_analysis[valid].copy()

    group_cols = ["market_name", "assetName", "trade_date_local"]

    def _compute_group(g):
        close_qty = g.loc[g["is_near_close"], "quantity"].sum()
        rest_qty = g.loc[~g["is_near_close"], "quantity"].sum()
        max_minutes = g["minutes_to_close_all"].max()
        rest_duration = max(max_minutes - minutes_before_close, 0)
        avg_rest_15 = (
            (rest_qty / rest_duration * minutes_before_close)
            if rest_duration > 0
            else np.nan
        )
        pct = (
            (close_qty / avg_rest_15 * 100)
            if avg_rest_15 and avg_rest_15 > 0
            else np.nan
        )
        return pd.Series(
            {
                "close_window_qty": close_qty,
                "rest_day_qty": rest_qty,
                "rest_day_avg_15m_qty": avg_rest_15,
                "close_vs_rest_avg_pct": pct,
            }
        )

    comparison = df_valid.groupby(group_cols).apply(_compute_group).reset_index()

    df_near_close = df_near_close.merge(comparison, on=group_cols, how="left")
    return df_near_close


def plot_market_severity(market_summary, top_n=15, output_dir=OUTPUT_DIR):
    """
    Plot market severity scores for comparison across markets.
    """
    if market_summary.empty:
        print("No market summary available for severity plot.")
        return

    plot_df = market_summary.head(top_n)
    colors = plot_df["Severity_Level"].map(
        {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}
    )

    plt.figure(figsize=(14, 6))
    bars = plt.bar(
        range(len(plot_df)),
        plot_df["Severity_Score"],
        color=colors,
        edgecolor="black",
        alpha=0.8,
    )
    plt.xticks(range(len(plot_df)), plot_df["market_name"], rotation=45, ha="right")
    plt.ylabel("Severity Score")
    plt.title("Market Severity Comparison (Potential Marking the Close)")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, plot_df["Severity_Score"]):
        plt.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ensure_output_dir(output_dir)
    output_path = os.path.join(output_dir, "market_severity_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nChart saved as '{output_path}'")


def plot_marking_the_close_by_market(
    df_near_close, top_n=12, top_markets=6, output_dir=OUTPUT_DIR
):
    """
    Create per-market visualizations for marking the close analysis.
    """
    if df_near_close.empty:
        print("No trades found near market close. Cannot create market visualizations.")
        return

    # Use top markets by total quantity
    market_order = (
        df_near_close.groupby("market_name")["quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(top_markets)
        .index.tolist()
    )

    for market in market_order:
        df_market = df_near_close[df_near_close["market_name"] == market].copy()

        buy_sell_summary = (
            df_market.groupby(["assetName", "way"])
            .agg({"quantity": "sum"})
            .reset_index()
        )
        buy_sell_pivot = buy_sell_summary.pivot(
            index="assetName", columns="way", values="quantity"
        ).fillna(0)

        if "Buy" not in buy_sell_pivot.columns:
            buy_sell_pivot["Buy"] = 0
        if "Sell" not in buy_sell_pivot.columns:
            buy_sell_pivot["Sell"] = 0

        buy_sell_pivot["Net_Quantity"] = buy_sell_pivot["Buy"] - buy_sell_pivot["Sell"]
        buy_sell_pivot["Total_Quantity"] = (
            buy_sell_pivot["Buy"] + buy_sell_pivot["Sell"]
        )
        buy_sell_pivot = buy_sell_pivot.sort_values(
            "Total_Quantity", ascending=False
        ).head(top_n)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(
            f"{market} - Last 15 Minutes Before Close", fontsize=14, fontweight="bold"
        )

        # Net quantity
        ax1 = axes[0]
        colors = ["green" if x >= 0 else "red" for x in buy_sell_pivot["Net_Quantity"]]
        ax1.bar(
            range(len(buy_sell_pivot)),
            buy_sell_pivot["Net_Quantity"],
            color=colors,
            edgecolor="black",
            alpha=0.7,
        )
        ax1.set_xlabel("Asset Name")
        ax1.set_ylabel("Net Quantity")
        ax1.set_xticks(range(len(buy_sell_pivot)))
        ax1.set_xticklabels(buy_sell_pivot.index, rotation=45, ha="right")
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax1.grid(axis="y", alpha=0.3)

        # Buy/Sell
        ax2 = axes[1]
        x = np.arange(len(buy_sell_pivot))
        width = 0.8
        ax2.bar(
            x,
            buy_sell_pivot["Buy"],
            width,
            label="Buy",
            color="green",
            edgecolor="darkgreen",
            alpha=0.7,
        )
        ax2.bar(
            x,
            -buy_sell_pivot["Sell"],
            width,
            label="Sell",
            color="red",
            edgecolor="darkred",
            alpha=0.7,
        )
        ax2.set_xlabel("Asset Name")
        ax2.set_ylabel("Quantity (Buy: +, Sell: -)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(buy_sell_pivot.index, rotation=45, ha="right")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax2.legend(loc="upper right")
        ax2.grid(axis="y", alpha=0.3)

        ensure_output_dir(output_dir)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"marking_the_close_{_slugify(market)}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"\nChart saved as '{output_path}'")


def plot_timeline_distribution_by_market(
    df_near_close, top_markets=6, output_dir=OUTPUT_DIR
):
    """
    Create per-market timeline distributions for trades near market close.
    """
    if df_near_close.empty:
        print(
            "No trades found near market close. Cannot create timeline visualizations."
        )
        return

    market_order = (
        df_near_close.groupby("market_name")["quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(top_markets)
        .index.tolist()
    )

    for market in market_order:
        df_market = df_near_close[df_near_close["market_name"] == market].copy()

        fig, ax = plt.subplots(figsize=(12, 5))
        df_market["minutes_bucket"] = df_market["minutes_to_close"].round(0)
        timeline = (
            df_market.groupby("minutes_bucket")
            .agg({"quantity": "sum", "dealId": "count"})
            .reset_index()
        )

        ax1 = ax
        ax2 = ax1.twinx()

        ax1.bar(
            timeline["minutes_bucket"],
            timeline["quantity"],
            color="steelblue",
            alpha=0.7,
            label="Total Quantity",
        )
        ax2.plot(
            timeline["minutes_bucket"],
            timeline["dealId"],
            color="red",
            marker="o",
            linewidth=2,
            label="Number of Trades",
        )

        ax1.set_xlabel("Minutes Before Market Close")
        ax1.set_ylabel("Total Quantity", color="steelblue")
        ax2.set_ylabel("Number of Trades", color="red")
        ax1.set_title(
            f"{market} - Trade Distribution in Last 15 Minutes", fontweight="bold"
        )

        ax1.invert_xaxis()
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax2.tick_params(axis="y", labelcolor="red")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax1.grid(axis="y", alpha=0.3)

        ensure_output_dir(output_dir)
        plt.tight_layout()
        filename = f"marking_the_close_timeline_{_slugify(market)}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"\nTimeline chart saved as '{output_path}'")


def export_flagged_deals_to_excel(
    df_near_close, market_summary, output_path=None, output_dir=OUTPUT_DIR
):
    """
    Export flagged deals and market summary to an Excel file.
    """
    if df_near_close.empty:
        print("No flagged deals to export.")
        return

    export_cols = [
        "dealId",
        "dealNature",
        "dealType",
        "portfolioId",
        "assetName",
        "currencyId",
        "market_name",
        "way",
        "quantity",
        "signed_quantity",
        "premium",
        "execTime",
        "execTime_local_str",
        "market_close_time",
        "minutes_to_close",
        "traderPopsId",
        "tradeDate",
        "valueDate",
        "inputDate",
        "flag_reason",
        "close_window_qty",
        "rest_day_qty",
        "rest_day_avg_15m_qty",
        "close_vs_rest_avg_pct",
    ]
    export_cols = [c for c in export_cols if c in df_near_close.columns]
    ensure_output_dir(output_dir)
    if output_path is None:
        output_path = os.path.join(output_dir, "marking_the_close_flagged_deals.xlsx")

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_near_close[export_cols].sort_values("minutes_to_close").to_excel(
                writer, sheet_name="flagged_deals", index=False
            )
            if market_summary is not None and not market_summary.empty:
                market_summary.to_excel(
                    writer, sheet_name="market_severity", index=False
                )
        print(f"\nFlagged deals exported to '{output_path}'")
    except ImportError:
        fallback_path = output_path.replace(".xlsx", ".csv")
        df_near_close[export_cols].to_csv(fallback_path, index=False)
        print(
            f"\nopenpyxl not available. Saved flagged deals to '{fallback_path}' instead."
        )


def plot_marking_the_close_analysis(df_near_close, top_n=20, output_dir=OUTPUT_DIR):
    """
    Create visualizations for marking the close analysis.

    Parameters:
    -----------
    df_near_close : pd.DataFrame
        Filtered DataFrame with trades near market close
    top_n : int
        Number of top assets to display in the charts
    """
    if df_near_close.empty:
        print("No trades found near market close. Cannot create visualizations.")
        return

    # Prepare data for plotting
    # Group by asset and way for buy/sell analysis
    buy_sell_summary = (
        df_near_close.groupby(["assetName", "way"])
        .agg({"quantity": "sum"})
        .reset_index()
    )

    # Pivot for buy/sell bars
    buy_sell_pivot = buy_sell_summary.pivot(
        index="assetName", columns="way", values="quantity"
    ).fillna(0)

    # Ensure both Buy and Sell columns exist
    if "Buy" not in buy_sell_pivot.columns:
        buy_sell_pivot["Buy"] = 0
    if "Sell" not in buy_sell_pivot.columns:
        buy_sell_pivot["Sell"] = 0

    # Calculate net quantity
    buy_sell_pivot["Net_Quantity"] = buy_sell_pivot["Buy"] - buy_sell_pivot["Sell"]

    # Sort by total quantity and get top N
    buy_sell_pivot["Total_Quantity"] = buy_sell_pivot["Buy"] + buy_sell_pivot["Sell"]
    buy_sell_pivot = buy_sell_pivot.sort_values("Total_Quantity", ascending=False).head(
        top_n
    )

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # ==========================================================================
    # Plot 1: Net Quantity Bar Chart
    # ==========================================================================
    ax1 = axes[0]
    colors = ["green" if x >= 0 else "red" for x in buy_sell_pivot["Net_Quantity"]]
    bars1 = ax1.bar(
        range(len(buy_sell_pivot)),
        buy_sell_pivot["Net_Quantity"],
        color=colors,
        edgecolor="black",
        alpha=0.7,
    )

    ax1.set_xlabel("Asset Name", fontsize=12)
    ax1.set_ylabel("Net Quantity", fontsize=12)
    ax1.set_title(
        f"Net Quantity in Last 15 Minutes Before Market Close\n(Top {top_n} by Total Quantity)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(range(len(buy_sell_pivot)))
    ax1.set_xticklabels(buy_sell_pivot.index, rotation=45, ha="right", fontsize=10)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars1, buy_sell_pivot["Net_Quantity"]):
        height = bar.get_height()
        ax1.annotate(
            f"{val:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -10),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=9,
        )

    # ==========================================================================
    # Plot 2: Buy (Positive) and Sell (Negative) Bar Chart
    # ==========================================================================
    ax2 = axes[1]
    x = np.arange(len(buy_sell_pivot))
    width = 0.8

    # Buy bars (positive)
    bars_buy = ax2.bar(
        x,
        buy_sell_pivot["Buy"],
        width,
        label="Buy",
        color="green",
        edgecolor="darkgreen",
        alpha=0.7,
    )

    # Sell bars (negative)
    bars_sell = ax2.bar(
        x,
        -buy_sell_pivot["Sell"],
        width,
        label="Sell",
        color="red",
        edgecolor="darkred",
        alpha=0.7,
    )

    ax2.set_xlabel("Asset Name", fontsize=12)
    ax2.set_ylabel("Quantity (Buy: +, Sell: -)", fontsize=12)
    ax2.set_title(
        f"Buy vs Sell Quantity in Last 15 Minutes Before Market Close\n(Top {top_n} by Total Quantity)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(buy_sell_pivot.index, rotation=45, ha="right", fontsize=10)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars_buy, buy_sell_pivot["Buy"]):
        if val > 0:
            ax2.annotate(
                f"{val:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="darkgreen",
            )

    for bar, val in zip(bars_sell, buy_sell_pivot["Sell"]):
        if val > 0:
            ax2.annotate(
                f"-{val:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                color="darkred",
            )

    ensure_output_dir(output_dir)
    output_path = os.path.join(output_dir, "marking_the_close_analysis.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nChart saved as '{output_path}'")


def plot_timeline_distribution(df_near_close, output_dir=OUTPUT_DIR):
    """
    Create a timeline distribution showing when trades occurred relative to market close.
    """
    if df_near_close.empty:
        print(
            "No trades found near market close. Cannot create timeline visualization."
        )
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by minutes to close
    df_near_close["minutes_bucket"] = df_near_close["minutes_to_close"].round(0)
    timeline = (
        df_near_close.groupby("minutes_bucket")
        .agg({"quantity": "sum", "dealId": "count"})
        .reset_index()
    )

    # Create dual axis plot
    ax1 = ax
    ax2 = ax1.twinx()

    bars = ax1.bar(
        timeline["minutes_bucket"],
        timeline["quantity"],
        color="steelblue",
        alpha=0.7,
        label="Total Quantity",
    )
    line = ax2.plot(
        timeline["minutes_bucket"],
        timeline["dealId"],
        color="red",
        marker="o",
        linewidth=2,
        label="Number of Trades",
    )

    ax1.set_xlabel("Minutes Before Market Close", fontsize=12)
    ax1.set_ylabel("Total Quantity", fontsize=12, color="steelblue")
    ax2.set_ylabel("Number of Trades", fontsize=12, color="red")
    ax1.set_title(
        "Trade Distribution in Last 15 Minutes Before Market Close",
        fontsize=14,
        fontweight="bold",
    )

    ax1.invert_xaxis()  # 15 minutes on left, 0 on right
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="red")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.grid(axis="y", alpha=0.3)

    ensure_output_dir(output_dir)
    output_path = os.path.join(output_dir, "marking_the_close_timeline.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nTimeline chart saved as '{output_path}'")


def print_market_close_times():
    """
    Print a formatted table of all market close times.
    """
    print("\n" + "=" * 80)
    print("MARKET CLOSE TIMES REFERENCE")
    print("=" * 80)
    print(f"{'Currency':<10} {'Market':<45} {'Close Time':<12} {'Timezone'}")
    print("-" * 80)

    for currency, info in sorted(MARKET_CLOSE_TIMES.items()):
        close_str = (
            info["close_time"].strftime("%H:%M") if info["close_time"] else "N/A"
        )
        print(f"{currency:<10} {info['market']:<45} {close_str:<12} {info['timezone']}")

    print("=" * 80 + "\n")


def main(data_path=None, df=None, enable_volume_comparison=True):
    """
    Main function to run the marking the close analysis.

    Parameters:
    -----------
    data_path : str, optional
        Path to the CSV/Excel file containing the booking data
    df : pd.DataFrame, optional
        DataFrame containing the booking data (if already loaded)

    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # Print market close times reference
    print_market_close_times()

    # Ensure output directory exists
    ensure_output_dir(OUTPUT_DIR)

    # Load data
    if df is None and data_path is not None:
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
    elif df is None:
        raise ValueError("Either data_path or df must be provided.")

    print(f"Loaded {len(df)} total trades")
    print(f"Date range: {df['tradeDate'].min()} to {df['tradeDate'].max()}")
    print(f"Unique currencies: {df['currencyId'].nunique()}")
    print(f"Unique assets: {df['assetName'].nunique()}")

    # Run analysis
    print("\n" + "=" * 80)
    print("ANALYZING TRADES NEAR MARKET CLOSE (Last 15 minutes)")
    print("=" * 80)

    df_near_close, df_analysis = analyze_marking_the_close(
        df, minutes_before_close=15, return_analysis=True
    )

    if enable_volume_comparison and not df_near_close.empty:
        df_near_close = add_volume_comparison(
            df_analysis=df_analysis,
            df_near_close=df_near_close,
            minutes_before_close=15,
        )

    print(
        f"\nFound {len(df_near_close)} trades executed in the last 15 minutes before market close"
    )

    if not df_near_close.empty:
        # Create summaries
        summary_by_asset = create_summary_by_asset(df_near_close)
        summary_by_trader = create_summary_by_trader(df_near_close)
        market_severity = create_market_severity_summary(
            df_near_close, minutes_before_close=15
        )

        print("\n" + "-" * 60)
        print("TOP 10 ASSETS BY TOTAL QUANTITY NEAR MARKET CLOSE:")
        print("-" * 60)
        print(summary_by_asset.head(10).to_string(index=False))

        print("\n" + "-" * 60)
        print("TOP 10 TRADERS BY TOTAL QUANTITY NEAR MARKET CLOSE:")
        print("-" * 60)
        print(summary_by_trader.head(10).to_string(index=False))

        print("\n" + "-" * 60)
        print("MARKET SEVERITY RANKING (Potential Marking the Close):")
        print("-" * 60)
        if not market_severity.empty:
            print(
                market_severity[
                    [
                        "market_name",
                        "currencyId",
                        "Severity_Score",
                        "Severity_Level",
                        "Total_Quantity",
                        "Trade_Count",
                        "Buy_Sell_Imbalance",
                        "Avg_Minutes_To_Close",
                    ]
                ]
                .head(10)
                .to_string(index=False)
            )

        # Create visualizations
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        # Overall summary plots
        plot_market_severity(market_severity, top_n=15)

        # Market-specific plots (top markets by quantity)
        plot_marking_the_close_by_market(df_near_close, top_n=12, top_markets=6)
        plot_timeline_distribution_by_market(df_near_close, top_markets=6)

        # Detailed trade list
        print("\n" + "-" * 60)
        print("DETAILED TRADES RANKED BY QUANTITY (Top 20):")
        print("-" * 60)
        print(
            "Note: execTime (Paris) shows the original timestamp, execTime_local shows the converted local market time"
        )
        display_cols = [
            "dealId",
            "assetName",
            "currencyId",
            "market_name",
            "way",
            "quantity",
            "execTime_local_str",
            "market_close_time",
            "minutes_to_close",
            "traderPopsId",
        ]
        print(df_near_close[display_cols].head(20).to_string(index=False))

        # Export flagged deals to Excel
        export_flagged_deals_to_excel(df_near_close, market_severity)

        results = {
            "near_close_trades": df_near_close,
            "summary_by_asset": summary_by_asset,
            "summary_by_trader": summary_by_trader,
            "market_severity": market_severity,
        }
    else:
        results = {
            "near_close_trades": df_near_close,
            "summary_by_asset": pd.DataFrame(),
            "summary_by_trader": pd.DataFrame(),
            "market_severity": pd.DataFrame(),
        }

    return results


# =============================================================================
# EXAMPLE USAGE WITH SAMPLE DATA
# =============================================================================
if __name__ == "__main__":
    # Create sample data for demonstration
    sample_data = {
        "dealNature": ["REGULAR_MARKET"] * 20,
        "dealId": list(range(1598671033, 1598671053)),
        "dealType": ["FUT"] * 10 + ["OPT"] * 10,
        "inputDate": ["2024-01-02T17:32:38.333+01:00[Europe/Paris]"] * 20,
        "tradeDate": ["2024-01-02+01:00[Europe/Paris]"] * 20,
        "valueDate": ["2024-01-03+01:00[Europe/Paris]"] * 20,
        "portfolioId": ["AH-FTSE", "AH-DAX", "AH-CAC"] * 6 + ["AH-SP500", "AH-NIKKEI"],
        "way": ["Buy", "Sell"] * 10,
        "quantity": [
            100,
            50,
            200,
            75,
            150,
            300,
            25,
            80,
            120,
            90,
            45,
            180,
            60,
            95,
            110,
            140,
            55,
            85,
            130,
            70,
        ],
        "remainQuantity": [0.0] * 20,
        "assetName": [
            "FTSE1",
            "DAX1",
            "CAC1",
            "FTSE1",
            "DAX1",
            "SP500",
            "NIKKEI",
            "HSI",
            "KOSPI",
            "ASX200",
        ]
        * 2,
        "currencyId": [
            "GBP",
            "EUR",
            "EUR",
            "GBP",
            "EUR",
            "USD",
            "JPY",
            "HKD",
            "KRW",
            "AUD",
        ]
        * 2,
        "maturity": ["2024-03-15+01:00[Europe/Paris]"] * 20,
        "premium": [7728.0, 15000.0, 8500.0, 12000.0, 9800.0] * 4,
        "underlyingId": ["6", "7", "8", "6", "7"] * 4,
        "productId": [0] * 20,
        "associatedId": [0] * 20,
        "optionType": [""] * 20,
        "strike": [0.0] * 20,
        "market": ["LIFFE", "EUREX", "EURONEXT", "LIFFE", "EUREX"] * 4,
        # Execution times near market close - ALL IN PARIS TIME
        # Paris is UTC+1 in winter, so we need to calculate what Paris time corresponds to near market close
        # GBP: London closes 16:30 local (UTC+0), so Paris time = 17:30
        # EUR: Paris closes 17:30 local (UTC+1)
        # USD: NY closes 16:00 local (UTC-5), so Paris time = 22:00
        # JPY: Tokyo closes 15:00 local (UTC+9), so Paris time = 07:00
        # HKD: HK closes 16:00 local (UTC+8), so Paris time = 09:00
        # KRW: Seoul closes 15:30 local (UTC+9), so Paris time = 07:30
        # AUD: Sydney closes 16:00 local (UTC+11), so Paris time = 06:00
        "execTime": [
            "2024-01-02T17:20:31.96+01:00[Europe/Paris]",  # GBP - London 16:20 (10 min before close)
            "2024-01-02T17:18:45.12+01:00[Europe/Paris]",  # EUR - 12 min before close
            "2024-01-02T17:25:00.00+01:00[Europe/Paris]",  # EUR - 5 min before close
            "2024-01-02T17:28:15.33+01:00[Europe/Paris]",  # GBP - London 16:28 (2 min before close)
            "2024-01-02T17:22:30.00+01:00[Europe/Paris]",  # EUR - 8 min before close
            "2024-01-02T21:50:00.00+01:00[Europe/Paris]",  # USD - NY 15:50 (10 min before close)
            "2024-01-02T06:55:00.00+01:00[Europe/Paris]",  # JPY - Tokyo 14:55 (5 min before close)
            "2024-01-02T08:52:00.00+01:00[Europe/Paris]",  # HKD - HK 15:52 (8 min before close)
            "2024-01-02T07:20:00.00+01:00[Europe/Paris]",  # KRW - Seoul 15:20 (10 min before close)
            "2024-01-02T05:48:00.00+01:00[Europe/Paris]",  # AUD - Sydney 15:48 (12 min before close)
            "2024-01-02T17:25:31.96+01:00[Europe/Paris]",  # GBP - London 16:25
            "2024-01-02T17:20:45.12+01:00[Europe/Paris]",  # EUR
            "2024-01-02T17:28:00.00+01:00[Europe/Paris]",  # EUR
            "2024-01-02T17:29:15.33+01:00[Europe/Paris]",  # GBP - London 16:29
            "2024-01-02T17:24:30.00+01:00[Europe/Paris]",  # EUR
            "2024-01-02T21:55:00.00+01:00[Europe/Paris]",  # USD - NY 15:55
            "2024-01-02T06:58:00.00+01:00[Europe/Paris]",  # JPY - Tokyo 14:58
            "2024-01-02T08:55:00.00+01:00[Europe/Paris]",  # HKD - HK 15:55
            "2024-01-02T07:25:00.00+01:00[Europe/Paris]",  # KRW - Seoul 15:25
            "2024-01-02T05:50:00.00+01:00[Europe/Paris]",  # AUD - Sydney 15:50
        ],
        "shortLong": ["L", "S"] * 10,
        "validDeal": [""] * 20,
        "withdrawal": [False] * 20,
        "strategyId": [0] * 20,
        "localBroker": [""] * 20,
        "underlyingCurrency": [
            "GBP",
            "EUR",
            "EUR",
            "GBP",
            "EUR",
            "USD",
            "JPY",
            "HKD",
            "KRW",
            "AUD",
        ]
        * 2,
        "underlyingType": ["Index"] * 20,
        "underlyingName": [
            "FTSE",
            "DAX",
            "CAC",
            "FTSE",
            "DAX",
            "S&P500",
            "NIKKEI",
            "HSI",
            "KOSPI",
            "ASX",
        ]
        * 2,
        "traderPopsId": ["10000072600", "10000072601", "10000072602"] * 6
        + ["10000072600", "10000072601"],
        "swapswireId": [""] * 20,
        "confirm": [""] * 20,
        "mifidClassification": [""] * 20,
        "buyingAccountingCenter": ["AH"] * 20,
        "sellingAccountingCenter": [""] * 20,
        "pointValue": [0.0] * 20,
        "fileCode": [""] * 20,
        "category": [""] * 20,
    }

    # Create sample DataFrame
    df_sample = pd.DataFrame(sample_data)

    print("=" * 80)
    print("MARKING THE CLOSE ANALYSIS")
    print("Identifying trades executed in the last 15 minutes before market close")
    print("=" * 80)

    # Run analysis with sample data
    results = main(df=df_sample, enable_volume_comparison=ENABLE_VOLUME_COMPARISON)

    # To use with your actual data, uncomment and modify:
    # results = main(data_path='path/to/your/booking_data.csv')
    # OR
    # df = pd.read_csv('your_data.csv')  # or load from database
    # results = main(df=df)
