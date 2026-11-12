"""
td_stack.py - Trader Dale order flow book, Trading Setup #4: STACKED IMBALANCES (book pages 57-62).

Book: Imbalance = one side 300%+ of the other (Ask >= 300% of Bid = buying imbalance, shown at the
Ask; Bid >= 300% of Ask = selling imbalance, shown at the Bid). 3+ imbalances of the same kind
on top of each other = Stacked Imbalance zone (green support / red resistance). Price must first
leave by 1-2 whole 30m candles, then wait for the pullback (from above = long, from below = short),
first touch only. SL/TP/fees = same as td_book.py (setup 1).

Needs td_book.py, td_footprint.py, td_node.py, td_big.py (for the verdict) and the built 5m data.
NO new download.

STEP 1  pick the volume floor by COUNTING only (target ~5 stacked zones/day, never by P&L):
    python td_stack.py count
STEP 2  zero-fee screen, then the real run:
    python td_stack.py run --minvol auto --feemaker 0 --feetaker 0 --slip 0
    python td_stack.py run --minvol auto --tpclose next

OUR CHOICES (book gives no number): imbalance compares DIAGONALLY (Ask at a row vs Bid one row below,
Bid at a row vs Ask one row above - the standard footprint method; --same compares inside the row),
ratio 3.0, stack size 3 rows, edge rows of a candle are not rated, a stack needs the dominant volume
>= minvol BTC (book default is 0, which on BTC creates noise from 0.001 BTC cells), a new stack is
dropped if a same-side stack overlapping it was created in the previous 100 candles.
Pass criteria (fixed in advance): >=150 trades, avg R>0 after costs, t>=2.5, positive in both years.
"""
import argparse

import numpy as np
import pandas as pd

import td_book as tb
import td_node as tn
import td_big as bg

DATA = tb.DATA
MINVOL_GRID = [0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12]
TARGET_PER_DAY = 5.0
CELLS = None
MINVOL = 0.0
SIZE = 3
MAXAGE = 100


def imbalance_cells(fp30, row, ratio=3.0, diagonal=True):
    """Every (candle, row) with an imbalance. Columns: candle, k, side(+1 buy,-1 sell), vol."""
    d = fp30[["candle", "level", "buy", "sell"]].copy()
    d["k"] = np.round(d["level"] / row).astype("int64")
    d = d.sort_values(["candle", "k"]).reset_index(drop=True)
    kmin = d.groupby("candle")["k"].transform("min")
    kmax = d.groupby("candle")["k"].transform("max")
    key = d[["candle", "k", "buy", "sell"]]
    below = key.assign(k=key["k"] + 1)[["candle", "k", "sell"]].rename(columns={"sell": "sell_below"})
    above = key.assign(k=key["k"] - 1)[["candle", "k", "buy"]].rename(columns={"buy": "buy_above"})
    d = d.merge(below, on=["candle", "k"], how="left").merge(above, on=["candle", "k"], how="left")
    if diagonal:
        opp_buy = d["sell_below"].fillna(0.0).to_numpy()      # bid one row below the ask
        opp_sell = d["buy_above"].fillna(0.0).to_numpy()      # ask one row above the bid
        ok_buy = (d["k"] > kmin).to_numpy()                   # lowest row has no row below
        ok_sell = (d["k"] < kmax).to_numpy()                  # highest row has no row above
    else:
        opp_buy, opp_sell = d["sell"].to_numpy(), d["buy"].to_numpy()
        ok_buy = ok_sell = np.ones(len(d), bool)
    b, s = d["buy"].to_numpy(), d["sell"].to_numpy()
    buy_imb = ok_buy & (b > 0) & (b >= ratio * opp_buy)
    sell_imb = ok_sell & (s > 0) & (s >= ratio * opp_sell)
    out = [pd.DataFrame({"candle": d["candle"][buy_imb], "k": d["k"][buy_imb], "side": 1, "vol": b[buy_imb]}),
           pd.DataFrame({"candle": d["candle"][sell_imb], "k": d["k"][sell_imb], "side": -1, "vol": s[sell_imb]})]
    return pd.concat(out, ignore_index=True)


def stacks_from_cells(cells, minvol, size, row, maxage, dedupe=True):
    c = cells[cells["vol"] >= minvol].sort_values(["side", "candle", "k"]).reset_index(drop=True)
    if c.empty:
        return pd.DataFrame(columns=["candle", "zone_low", "zone_high", "side"])
    brk = ((c["candle"] != c["candle"].shift()) | (c["side"] != c["side"].shift())
           | (c["k"] - c["k"].shift() != 1))
    c["run"] = brk.cumsum()
    g = c.groupby("run").agg(candle=("candle", "first"), side=("side", "first"),
                             kmin=("k", "min"), kmax=("k", "max"), n=("k", "count"))
    g = g[g["n"] >= size].sort_values("candle")
    z = pd.DataFrame({"candle": g["candle"].to_numpy(), "zone_low": g["kmin"].to_numpy() * row,
                      "zone_high": (g["kmax"].to_numpy() + 1) * row, "side": g["side"].to_numpy()})
    if dedupe and len(z):
        keep, live = [], []                                  # live = (side, low, high, candle)
        span = maxage * tb.MS30
        for r in z.itertuples():
            live = [x for x in live if r.candle - x[3] <= span]
            dup = any(x[0] == r.side and x[1] < r.zone_high and r.zone_low < x[2] for x in live)
            keep.append(not dup)
            if not dup:
                live.append((r.side, r.zone_low, r.zone_high, r.candle))
        z = z[np.array(keep)]
    return z.reset_index(drop=True)


def load_fp30(symbol):
    fp5 = pd.read_csv(DATA / f"{symbol}_5m_footprint.csv")
    cd5 = pd.read_csv(DATA / f"{symbol}_5m_candles.csv").sort_values("candle").reset_index(drop=True)
    fp30, cd30 = tb.to_30m(fp5, cd5)
    return fp30, cd30


def count(P):
    fp30, cd30 = load_fp30(P.symbol)
    cells = imbalance_cells(fp30, P.row, P.ratio, not P.same)
    days = max((cd30["candle"].max() - cd30["candle"].min()) / 86_400_000, 1)
    print(f"{len(cells)} imbalance cells over {days:.0f} days (ratio {P.ratio}, "
          f"{'same-row' if P.same else 'diagonal'}, stack size {P.size})")
    print("  minvol(BTC)  stacks  per day")
    res = []
    for mv in MINVOL_GRID:
        n = len(stacks_from_cells(cells, mv, P.size, P.row, P.maxage))
        res.append((mv, n / days))
        print(f"  {mv:>10}  {n:>6}  {n / days:7.2f}")
    best = min(res, key=lambda r: abs(r[1] - TARGET_PER_DAY))
    print(f"\nclosest to {TARGET_PER_DAY:g}/day: minvol = {best[0]}  (counting only, no P&L used)")
    return best[0], cells


def find_stacks(fp30, cd30, row, mult, pocLen, pad, tf_min=30, merge=6):
    z = stacks_from_cells(CELLS, MINVOL, SIZE, row, MAXAGE)
    return z[["candle", "zone_low", "zone_high"]]


def main():
    global CELLS, MINVOL, SIZE, MAXAGE
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["count", "run"])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--row", type=float, default=10.0)
    ap.add_argument("--ratio", type=float, default=3.0)
    ap.add_argument("--size", type=int, default=3)
    ap.add_argument("--minvol", default="auto")
    ap.add_argument("--same", action="store_true", help="compare Bid and Ask inside the same row")
    bg.add_run_args(ap)
    P = ap.parse_args()

    if P.cmd == "count":
        count(P)
        return
    if P.minvol == "auto":
        MINVOL, CELLS = count(P)
        print()
    else:
        MINVOL = float(P.minvol)
        fp30, _ = load_fp30(P.symbol)
        CELLS = imbalance_cells(fp30, P.row, P.ratio, not P.same)
    SIZE, MAXAGE = P.size, P.maxage
    print(f"Using minvol = {MINVOL} BTC, ratio {P.ratio}, stack {P.size}+")
    tb.find_clusters = find_stacks
    tb.zone_windows = tn.node_windows
    tr, skips = tb.run(P)
    if not tr.empty:
        if P.date_from:
            tr = tr[tr["entry_time"] >= pd.Timestamp(P.date_from)]
        if P.date_to:
            tr = tr[tr["entry_time"] < pd.Timestamp(P.date_to)]
    print("Skipped zones:", dict(skips))
    tb.summarize(tr)
    if not tr.empty:
        bg.verdict(tr, P.split)
        tr.to_csv(DATA / f"{P.symbol}_stack_trades.csv", index=False)


if __name__ == "__main__":
    main()
