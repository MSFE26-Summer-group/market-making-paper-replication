# Feature Engineering Pipeline (Attn-LOB Pretraining)

This page documents `paper_replication.features` — the code that turns the raw
BTC/USDT limit order book (LOB) snapshots into a labeled dataset ready to
train the Attn-LOB model from the paper (Section III-A / III-B). See
`notebooks/02_feature_engineering.ipynb` (listed on the
[Notebooks page](notebooks/index.md)) for a runnable walkthrough on the real
data, and the [API Reference](api/index.md) for the full auto-generated
docstrings.

## In plain English

The paper's model doesn't look at the order book once — it looks at a short
*movie* of the order book (the last 50 snapshots) and tries to guess whether
the price is about to go up, down, or stay flat. Before any of that can
happen, the raw data needs to be cut into overlapping clips, put on a
comparable scale (a $19{,}000 price and a 0.001 BTC size can't sit in the
same neural network input un-adjusted), paired with some slower-moving
"mood of the market" signals (how volatile is it right now, is the market
predominantly buying or selling), and given an answer key (what actually
happened to the price afterwards).

That's the whole pipeline in one sentence: **slice the order book into
overlapping windows, put every window on the same scale, compute a handful
of market-mood indicators, and label each window with what the price did
next.**

## Why our version differs from the paper

The paper's data is one row per *event* (a new order, a cancellation, a
trade — anything that changes the book), arriving roughly every 60-150ms.
Ours (`data/btc_usdt_20221019_20221030_lob.parquet`) is one row every ~10
seconds instead — a fixed-interval snapshot, not an event log. That single
difference cascades through the whole pipeline:

| Paper | This replication | Why |
|---|---|---|
| `T=50` events of context (~a few seconds) | `T=50` rows of context (~8 minutes) | Row-count kept the same as the paper; wall-clock time is ~100x longer because our rows are ~100x further apart |
| `k=10` events ahead for the label | `k=10` rows ahead (~100 seconds) | Same idea — same row count, very different real time |
| `alpha=1e-5` (fixed) | `alpha` auto-calibrated per run | The paper's threshold was tuned to a different asset's tick-level noise and doesn't transfer numerically to 10s BTC bars |
| OSI over 3 event categories (new market orders, new limit orders, cancellations) | OSI restricted to the trade category only | This dataset has no order-book event log — only executed trades and periodic snapshots, so the other two categories can't be reconstructed at all |

None of this is hidden — every module below says explicitly, in its
docstrings and in the notes here, exactly where it diverges from the paper
and why.

## The pipeline, step by step

```text
LOB parquet (88,727 rows, 10 levels, ~10s apart)
        │
        ▼
1. load_lob_snapshot()        sort by time, pull the columns we need
        │
        ▼
2. lob_state_matrix()          the 40 raw price/size numbers per row
   + rolling_windows()         cut into overlapping 50-row clips
   + normalize_lob_state()     rescale each clip so the network can use it
        │                                           │
        ▼                                           ▼
3. realized_volatility()         4. compute_l_t()
   relative_strength_index()        + calibrate_alpha()
   order_strength_index_proxy()     + direction_labels()
   ("what mood is the market in")   ("what did price do next")
        │                                           │
        └───────────────┬───────────────────────────┘
                         ▼
              5. build_feature_dataset()
         aligns windows + mood features + labels
         on the same row, drops incomplete edges
                         │
                         ▼
              6. chronological_split()
         train / val / test, in time order, no shuffling
                         │
                         ▼
              AttnLOBDataset  (ready for the model)
```

Every stage below is one module in `src/paper_replication/features/`.

---

## `config.py` — the dials

**In simple terms:** one settings object, `FeatureConfig`, that holds every
number the rest of the pipeline needs: how many order book levels to use,
how long a "clip" is, how far ahead to check the price, and so on. Change a
number here instead of hunting through the code.

**Technical details:**

- `n_levels = 10` — number of bid/ask levels to use (this dataset has 10 available).
- `window_T = 50` — rows per LOB-state window (paper Eq. 1's rolling window length `T`).
- `horizon_k = 10` — rows of look-back/look-ahead used to build the direction label (paper Eq. 5-7's `k`).
- `label_alpha: float | None = None` — the label's dead-zone threshold. `None` means "calibrate it from the data" (see `labels.py` below); set a float to pin it manually.
- `label_target_balance = 1/3` — used only when `label_alpha` is `None`; the calibration aims for this fraction of samples in each of the up/down classes.
- `osi_windows_seconds`, `rv_windows_seconds`, `rsi_windows_seconds` — trailing windows (in real seconds) for the three dynamic features, matching the paper's 10s/60s/300s (OSI) and 5min/10min/30min (RV, RSI).
- `use_real_osi = False` — stays proxy-mode by default; see `dynamic_state.py` for what flipping it does.

---

## `lob_state.py` — the raw order book "movie"

**In simple terms:** loads the order book columns from the parquet file, cuts
the long time series into overlapping short clips (like a sliding window
over a video), and rescales each clip so prices and sizes are on comparable
footing — otherwise a model would mostly just learn "BTC costs about
$19,000" instead of anything useful about the book's shape.

**Technical details:**

- `level_column_names(symbol, n_levels)` — builds the column name list
  `[ask_price_1, ask_size_1, bid_price_1, bid_size_1, ask_price_2, ...]`,
  matching the paper's LOB state vector ordering (Eq. 1).
- `load_lob_snapshot(parquet_path, symbol, n_levels)` — pulls those columns
  plus `timestamp`, `mid_price`, `net_trade_sign`, `trade_count`, `ema_ofi`
  (the last three feed `dynamic_state.py`) via DuckDB, sorted and
  de-duplicated by timestamp.
- `lob_state_matrix(df, symbol, n_levels)` — returns the raw
  `(n_rows, 4 * n_levels)` matrix, i.e. `(88727, 40)` for this dataset.
- `rolling_windows(matrix, window_T)` — turns `(n_rows, F)` into
  `(n_rows - window_T + 1, window_T, F)` overlapping windows via
  `numpy.lib.stride_tricks.sliding_window_view` (no copying until the
  normalization step). Window `i`'s last row is source row `i + window_T - 1`
  — this offset is what `dataset.py` uses to line windows back up with
  labels and dynamic features.
- `normalize_lob_state(windows, n_levels)` — **per window**, z-normalizes
  the price columns (`(x - mean) / std`) and max-normalizes the volume
  columns (`x / max`), matching the paper's description (Section III-B2):
  *"we perform z-norm to the stationary price sequence and max-norm for
  the volume sequence."* Normalizing per-window rather than globally means
  the model only ever sees relative price movement within a clip, not the
  absolute BTC price level — which is what makes clips from October 19th
  and October 30th comparable even though BTC's price drifted between them.

---

## `labels.py` — the answer key

**In simple terms:** for every point in time, look a little bit into the
future and a little bit into the past, and decide: did the price mostly go
up, mostly go down, or stay flat? That decision becomes the label the model
is trained to predict.

**Technical details:**

- `compute_l_t(mid_price, k)` computes, for each row `t`,
  `l_t = (future_mean(t) - past_mean(t)) / past_mean(t)`, where
  `past_mean(t)` is the average of the `k` rows ending at `t` and
  `future_mean(t)` is the average of the `k` rows immediately after `t`
  (paper Eq. 6-7). We follow the standard DeepLOB formulation the paper
  cites as its source ([23], Zhang & Zohren) rather than the paper's own
  ambiguous `t∓i` notation, since the paper explicitly borrows the labeling
  method from there.
- `direction_labels(l_t, alpha)` turns `l_t` into `{-1, 0, +1}` (down /
  stationary / up) via a dead-zone threshold `alpha` (paper Eq. 5):
  `l_t > alpha → up`, `l_t < -alpha → down`, otherwise `stationary`.
- `calibrate_alpha(l_t, target_balance)` picks `alpha` empirically so each
  of the up/down classes gets roughly `target_balance` of the samples,
  instead of reusing the paper's `alpha=1e-5` (which was tuned to a
  different asset's tick-level noise and doesn't transfer to 10s BTC bars).
  It does this by taking the `(1 - 2 * target_balance)` quantile of `|l_t|`.
- Both functions return/accept plain `numpy` arrays with `NaN` at the
  series' edges, where there isn't enough history or lookahead to compute
  a label — `dataset.py` drops those rows.

---

## `dynamic_state.py` — the market's "mood"

**In simple terms:** the order book snapshot alone doesn't say much about
*momentum* — whether the market has been trending, how choppy it's been
recently, or whether buyers or sellers have had the upper hand. This module
adds three such signals, each computed over a few different time windows
(fast/medium/slow), the same way you'd glance at a 1-minute, 1-hour, and
1-day chart before making a call.

**Technical details** — three signals, three functions each:

- `realized_volatility(df, windows_seconds)` — **RV**, paper Eq. 3:
  `sqrt(sum of squared log-returns)` of `mid_price` over each trailing
  time window (default 5/10/30 minutes). Time-based (`pandas` `.rolling("Ns")`
  on a `DatetimeIndex`), so it's correct even though rows aren't evenly
  spaced (10s vs 20s gaps in this dataset).
- `relative_strength_index(df, windows_seconds)` — **RSI**, paper Eq. 4,
  implemented exactly as the paper writes it: `Gain / (Gain + Loss)`,
  bounded `[0, 1]`. Note this is *not* the textbook Wilder RSI (which
  rescales to 0-100) — we replicate the paper's own (nonstandard) formula.
- `order_strength_index_proxy(df, windows_seconds)` — **OSI**, paper Eq. 2,
  restricted to what this dataset can actually support. The paper defines
  OSI over three event categories (new market orders, new limit orders,
  cancellations); this dataset only records trade prints and periodic book
  snapshots, so only the trade category is reconstructable at all —
  wired into `osi_n_*` (count-based, from `net_trade_sign` / `trade_count`,
  a genuine replication of the paper's "number of orders" OSI restricted to
  trades) and `osi_v_*` (volume-based; since the snapshot file has no
  buy/sell split for `trade_volume`, this substitutes the pipeline's
  existing `ema_ofi` order-flow-imbalance column as a stand-in).
- `order_strength_index_from_ticks(df, ticks_parquet_path, windows_seconds)`
  — the faithful version of OSI, computed directly from raw per-trade
  side + volume in `ticks.parquet` (~69M rows) via a DuckDB range join.
  **Not called by the default pipeline** — set
  `FeatureConfig(use_real_osi=True)` and pass `ticks_parquet_path` to
  `build_feature_dataset` to enable it. It's unit-tested and correct, just
  expensive (naive range join, one per window), so it's opt-in rather than
  the default.

---

## `dataset.py` — putting it all together

**In simple terms:** this is the conductor. It calls every piece above,
lines everything up so that "clip ending at row 500", "market mood at row
500", and "what happened after row 500" all refer to the exact same moment,
throws away the handful of rows at the very start/end where a full clip or
a full label isn't available, and hands back one tidy bundle. It also
splits that bundle into train/validation/test periods in time order — never
shuffled, so the model is never accidentally trained on the future and
tested on the past.

**Technical details:**

- `AttnLOBDataset` — the output dataclass:
    - `lob_state`: `(n_samples, window_T, 4 * n_levels)`, normalized.
    - `dynamic_state`: `(n_samples, n_dynamic_features)` — RV + RSI + OSI concatenated.
    - `dynamic_feature_names`: column names for the array above (e.g. `rv_300s`, `osi_n_10s`, ...).
    - `labels`: `(n_samples,)` `int8` in `{-1, 0, 1}`.
    - `timestamps`: `(n_samples,)` — the snapshot timestamp of each window's *last* row.
    - `alpha_used`: the (possibly auto-calibrated) label threshold actually used.
- `build_feature_dataset(lob_parquet_path, config, symbol, ticks_parquet_path=None)`:
    1. Loads the snapshot file and builds the raw/normalized LOB-state windows.
    2. Computes RV, RSI, and OSI (proxy by default, real-from-ticks if `config.use_real_osi=True`).
    3. Computes `l_t`, resolves `alpha` (calibrated or fixed), and derives labels.
    4. Aligns everything on the row index of each window's *last* row, restricted to rows where a full window, a full label, and finite dynamic features all exist — `[max(window_T - 1, horizon_k - 1), n - 1 - (horizon_k - 1)]`. Any remaining `NaN` rows (e.g. from the earliest dynamic-feature windows, before there's enough time history) are dropped.
- `chronological_split(dataset, test_frac=0.5, val_frac_of_train=0.2)` —
  slices the (already time-ordered) dataset into train/val/test by index,
  no shuffling. Mirrors the paper's split (first half of the month for
  train+val, second half for test), scaled to however much history the
  dataset covers.

---

## Known limitations (read before trusting the numbers)

- **Wall-clock mismatch**: `window_T` and `horizon_k` are event-count
  parameters borrowed unchanged from the paper, but our "events" are ~10s
  snapshots instead of ~100ms order-book events. A window that was a few
  seconds of context in the paper is ~8 minutes here.
- **`alpha` isn't the paper's `alpha`**: it's recalibrated per run from the
  data (see `labels.py`), so exact numeric comparisons to the paper's
  reported threshold aren't meaningful — only the resulting class balance is.
- **OSI is trade-only**: two of the paper's three OSI event categories
  (new limit orders, cancellations) cannot be reconstructed from this
  dataset at all, in either the proxy or real-from-ticks path.
- **Per-split class balance drifts**: `alpha` is calibrated globally, but
  11 days of BTC/USDT isn't stationary — the train split (mid-October) came
  out more "stationary"-heavy, the test split (late October) more trending.
  This is a real property of the data, not a bug, and is worth remembering
  when interpreting pretraining accuracy on the test split.
- **Single asset**: the paper validates across three stocks in different
  sectors; this replication currently has one instrument (BTC/USDT), so
  cross-sectional generalization can't be tested the same way.

## Where things live

| What | Where |
|---|---|
| Feature engineering code | `src/paper_replication/features/` |
| Unit tests | `tests/test_features_*.py` |
| Runnable walkthrough on real data | `notebooks/02_feature_engineering.ipynb` |
| Cached output of a full run | `data/processed/attn_lob_pretrain.npz` (git-ignored — regenerate via the notebook) |
| Auto-generated API docs | [API Reference](api/index.md) |
