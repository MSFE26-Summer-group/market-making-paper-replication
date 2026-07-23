"""Dynamic market state: Realized Volatility, RSI, and Order Strength Index (OSI).

Paper reference: III-A2. RV (Eq. 3) and RSI (Eq. 4) are computed directly
from `mid_price` and transfer over cleanly. OSI (Eq. 2) is defined over
three event categories -- new market orders, new limit orders, and
cancellations -- but our data only lets us reconstruct the market-order
(trade) category:

- `order_strength_index_proxy` uses only columns already in the LOB
  snapshot parquet (cheap, always available, default in FeatureConfig).
- `order_strength_index_from_ticks` recomputes the same trade-category OSI
  from the raw per-trade side+volume in ticks.parquet (faithful, but a full
  pass over ~69M rows per window -- not wired into the default pipeline).

Neither path can recover the limit-order-submission or cancellation
categories: this dataset has no order-book event log, only trade prints and
periodic LOB snapshots.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd


def _time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(df["timestamp"].to_numpy(), unit="s"))


def realized_volatility(
    df: pd.DataFrame, windows_seconds: tuple[int, ...]
) -> pd.DataFrame:
    """Eq. 3: sqrt(sum of squared log-returns) over trailing time windows."""
    idx = _time_index(df)
    log_ret = pd.Series(np.log(df["mid_price"].to_numpy()), index=idx).diff()
    sq = log_ret**2
    out = {f"rv_{w}s": np.sqrt(sq.rolling(f"{w}s").sum()) for w in windows_seconds}
    return pd.DataFrame(out).reset_index(drop=True)


def relative_strength_index(
    df: pd.DataFrame, windows_seconds: tuple[int, ...]
) -> pd.DataFrame:
    """Eq. 4, as literally written in the paper: Gain / (Gain + Loss), bounded [0, 1].

    Note this is not the textbook Wilder RSI (which rescales RS to 0-100);
    we replicate the paper's own formula.
    """
    idx = _time_index(df)
    price = pd.Series(df["mid_price"].to_numpy(), index=idx)
    diff = price.diff()
    gain = diff.clip(lower=0)
    loss = (-diff).clip(lower=0)
    out = {}
    for w in windows_seconds:
        gain_sum = gain.rolling(f"{w}s").sum()
        loss_sum = loss.rolling(f"{w}s").sum()
        denom = (gain_sum + loss_sum).replace(0, np.nan)
        out[f"rsi_{w}s"] = gain_sum / denom
    return pd.DataFrame(out).reset_index(drop=True)


def order_strength_index_proxy(
    df: pd.DataFrame, windows_seconds: tuple[int, ...]
) -> pd.DataFrame:
    """OSI proxy built only from columns already in the LOB snapshot parquet.

    - Count-based `osi_n`: a genuine replication using `net_trade_sign` /
      `trade_count` (signed and total trade counts since the previous
      snapshot), rolled up over each window. This is exactly the paper's
      "number of orders" OSI, restricted to the trade category.
    - Volume-based `osi_v`: the snapshot file only has *unsigned*
      `trade_volume`, so a true signed volume imbalance can't be derived
      here. We substitute the pipeline's existing `ema_ofi` (order-flow
      imbalance) column as a stand-in signal, rolled up the same way.

    For a faithful volume+count OSI restricted to trades, see
    `order_strength_index_from_ticks`.
    """
    idx = _time_index(df)
    sign = pd.Series(df["net_trade_sign"].to_numpy(), index=idx)
    count = pd.Series(df["trade_count"].to_numpy(), index=idx)
    ofi = pd.Series(df["ema_ofi"].to_numpy(), index=idx)
    out = {}
    for w in windows_seconds:
        count_sum = count.rolling(f"{w}s").sum().replace(0, np.nan)
        out[f"osi_n_{w}s"] = sign.rolling(f"{w}s").sum() / count_sum
        out[f"osi_v_{w}s"] = ofi.rolling(f"{w}s").mean()
    return pd.DataFrame(out).reset_index(drop=True)


def order_strength_index_from_ticks(
    df: pd.DataFrame, ticks_parquet_path: str, windows_seconds: tuple[int, ...]
) -> pd.DataFrame:
    """Faithful trade-category OSI computed from raw per-trade side+volume.

    For each LOB snapshot timestamp, aggregates `ticks_parquet_path` trades
    in the trailing window via a DuckDB range join.

    Not called by `build_feature_dataset` by default: `ticks.parquet` is
    ~69M rows, and this issues one range-join aggregation per window. Enable
    explicitly via `FeatureConfig(use_real_osi=True)` once you're ready to
    pay the compute cost; the naive range join here may need optimizing
    (e.g. an as-of/cumulative-sum approach) before running it at full scale.
    """
    con = duckdb.connect()
    snapshots = pd.DataFrame(
        {"snapshot_id": np.arange(len(df)), "ts": df["timestamp"].to_numpy()}
    )
    con.register("snapshots", snapshots)

    out = pd.DataFrame({"snapshot_id": snapshots["snapshot_id"]})
    for w in windows_seconds:
        query = f"""
            SELECT s.snapshot_id AS snapshot_id,
                   SUM(CASE WHEN t.side = 1 THEN t.amount ELSE 0 END) AS buy_vol,
                   SUM(CASE WHEN t.side = -1 THEN t.amount ELSE 0 END) AS sell_vol,
                   SUM(CASE WHEN t.side = 1 THEN 1 ELSE 0 END) AS buy_n,
                   SUM(CASE WHEN t.side = -1 THEN 1 ELSE 0 END) AS sell_n
            FROM snapshots s
            LEFT JOIN read_parquet('{ticks_parquet_path}') t
              ON t.timestamp > s.ts - {w} AND t.timestamp <= s.ts
            GROUP BY s.snapshot_id
        """
        agg = (
            con.execute(query)
            .df()
            .set_index("snapshot_id")
            .reindex(snapshots["snapshot_id"])
        )
        vol_denom = (agg["buy_vol"] + agg["sell_vol"]).replace(0, np.nan)
        n_denom = (agg["buy_n"] + agg["sell_n"]).replace(0, np.nan)
        out[f"osi_v_{w}s"] = ((agg["buy_vol"] - agg["sell_vol"]) / vol_denom).to_numpy()
        out[f"osi_n_{w}s"] = ((agg["buy_n"] - agg["sell_n"]) / n_denom).to_numpy()
    con.close()
    return out.drop(columns="snapshot_id").reset_index(drop=True)
