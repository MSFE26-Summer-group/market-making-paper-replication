"""Inspect snapshot and metadata of the monthly LOB parquet file."""

from __future__ import annotations

import pathlib

import polars as pl

PARQUET_PATH = pathlib.Path(
    "/home/ares/R/wfbryan/data/btc_usdt_20221010_20221110_lob.parquet"
)


def main() -> None:
    if not PARQUET_PATH.exists():
        print(f"Error: File not found at {PARQUET_PATH}")
        return

    print(f"Loading {PARQUET_PATH} ...")
    df = pl.read_parquet(PARQUET_PATH)

    print("\n=== Dataset Overview ===")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\n=== Columns ===")
    print(df.columns)

    print("\n=== Schema ===")
    for col, dtype in df.schema.items():
        if col.startswith("BTCUSDT.ask_price_3") or col == "mid_price":
            print("...")
            break
        print(f"  {col}: {dtype}")

    print("\n=== First 3 Rows (Key Columns) ===")
    key_cols = [
        col
        for col in [
            "timestamp",
            "mid_price",
            "BTCUSDT.ask_price_1",
            "BTCUSDT.ask_size_1",
            "BTCUSDT.bid_price_1",
            "BTCUSDT.bid_size_1",
            "net_trade_sign",
            "trade_count",
            "ema_ofi",
        ]
        if col in df.columns
    ]
    print(df.select(key_cols).head(3))

    print("\n=== Time Bounds ===")
    if "timestamp" in df.columns:
        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        print(f"Min timestamp: {ts_min}")
        print(f"Max timestamp: {ts_max}")

    print("\n=== Null Value Counts ===")
    null_counts = df.null_count()
    total_nulls = sum(null_counts.row(0))
    print(f"Total null cells across all columns: {total_nulls}")


if __name__ == "__main__":
    main()
