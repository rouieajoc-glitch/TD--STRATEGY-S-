"""
td_big.py - Trader Dale order flow book, Trading Setup #3: TRADES FILTER (book pages 53-56).

Book idea: show only SINGLE trades of at least X lots. Those cells are zones. Price must first
leave by at least 1-2 whole 30m candles, then we wait for the pullback (from above = long,
from below = short, first touch only). SL/TP/fees/fills = same as td_book.py (setup 1).

Needs td_book.py, td_footprint.py, td_node.py and the 5m data (td_footprint.py build --tf 5 ...).

STEP 1  download + keep only big trades (about 40 min per year, run once):
    python td_big.py build --start 2024-11-01 --end 2026-11-05 --floor 1.0
STEP 2  choose X by COUNTING ONLY (book: 5-10 signals/day). Never by P&L:
    python td_big.py count
STEP 3  zero-fee screen, then the real run:
    python td_big.py run --x auto --feemaker 0 --feetaker 0 --slip 0
    python td_big.py run --x auto --tpclose next

X is in BTC. Binance aggTrades = one taker order at one price (fits "ONE order"; a sweep over
several levels shows as several smaller trades, so big sweeps are undercounted).
Pass criteria (fixed in advance): >=150 trades, avg R>0 after costs, t>=2.5, positive in both years.
"""
import argparse
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

import td_footprint as tf
import td_book as tb
import td_node as tn

DATA = tb.DATA
BIG_DIR = DATA / "bigtrades"
MS30 = tb.MS30
X_GRID = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]
TARGET_PER_DAY = 7.0
BIG_X = 5.0          # set from --x
BIG_ROW = 10.0


def big_path(symbol):
    return DATA / f"{symbol}_bigtrades.csv"


# ------------------------------------------------------------------ build
def lean_load(zip_path, floor):
    """Memory-light read: chunks, only the columns we need. Returns (big trades, total rows)."""
    import zipfile
    keep, total = [], 0
    with zipfile.ZipFile(zip_path) as z:
        with z.open(z.namelist()[0]) as f:
            for ch in pd.read_csv(f, header=None, usecols=[1, 2, 5, 6], names=None,
                                  chunksize=500_000, dtype=str):
                ch.columns = ["price", "qty", "time", "is_buyer_maker"]
                ch = ch[pd.to_numeric(ch["price"], errors="coerce").notna()]
                total += len(ch)
                q = ch["qty"].astype(float)
                ch = ch[q >= floor].copy()
                ch["qty"] = q[q >= floor]
                ch["price"] = ch["price"].astype(float)
                ch["time"] = ch["time"].astype("int64")
                ch["is_buyer_maker"] = ch["is_buyer_maker"].str.lower().eq("true")
                keep.append(ch)
    big = pd.concat(keep, ignore_index=True)[["time", "price", "qty", "is_buyer_maker"]]
    if len(big) and big["time"].iloc[0] > 1e14:
        big["time"] //= 1000
    return big, total


def build(symbol, start, end, floor):
    BIG_DIR.mkdir(parents=True, exist_ok=True)
    d = start
    while d <= end:
        out = BIG_DIR / f"{symbol}_{d.isoformat()}_big.csv"
        if not out.exists():
            try:
                z = tf.download_day(symbol, d)
            except Exception as e:
                print(f"{d} skipped: {e}")
                d += timedelta(days=1)
                continue
            big, ntr = lean_load(z, floor)
            big.to_csv(out, index=False)
            z.unlink(missing_ok=True)
            print(f"{d} trades {ntr:>9} kept {len(big):>6}", flush=True)
        d += timedelta(days=1)
    files = sorted(BIG_DIR.glob(f"{symbol}_*_big.csv"))
    allb = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    allb.to_csv(big_path(symbol), index=False)
    print(f"merged {len(allb)} big trades from {len(files)} days -> {big_path(symbol)}")


# ------------------------------------------------------------------ zones
def sig_path(symbol, row):
    return DATA / f"{symbol}_bigsig_{row:g}.csv"


def load_sig(symbol, row):
    """One row per (30m candle, price row) = biggest single trade there. Built in chunks, cached."""
    p = sig_path(symbol, row)
    if p.exists():
        return pd.read_csv(p)
    parts = []
    for ch in pd.read_csv(big_path(symbol), usecols=["time", "price", "qty"], chunksize=2_000_000):
        ch["candle"] = (ch["time"] // MS30) * MS30
        ch["level"] = (np.floor(ch["price"] / row) * row).round(8)
        parts.append(ch.groupby(["candle", "level"], as_index=False)["qty"].max())
        del ch
    sig = pd.concat(parts).groupby(["candle", "level"], as_index=False)["qty"].max()
    sig = sig.sort_values("candle").reset_index(drop=True)
    sig.to_csv(p, index=False)
    return sig


def find_bigs(fp30, cd30, row, mult, pocLen, pad, tf_min=30, merge=6):
    s = BIG_DF[BIG_DF["qty"] >= BIG_X]
    s = s[s["candle"].isin(set(cd30["candle"]))]
    return pd.DataFrame({"candle": s["candle"].to_numpy(),
                         "zone_low": s["level"].to_numpy(),
                         "zone_high": s["level"].to_numpy() + row})


def count(symbol, row):
    sig = load_sig(symbol, row)
    days = max((sig["candle"].max() - sig["candle"].min()) / 86_400_000, 1)
    print(f"{len(sig)} (candle,row) cells with a trade >= floor over {days:.0f} days")
    print("   X(BTC)  signals  per day")
    res = []
    for x in X_GRID:
        n = int((sig["qty"] >= x).sum())
        res.append((x, n / days))
        print(f"  {x:>6}  {n:>7}  {n / days:7.2f}")
    best = min(res, key=lambda r: abs(r[1] - TARGET_PER_DAY))
    print(f"\nclosest to {TARGET_PER_DAY:g}/day: X = {best[0]}  (counting only, no P&L used)")
    return best[0]


# ------------------------------------------------------------------ run
def verdict(tr, split):
    a = tr[tr["entry_time"] < pd.Timestamp(split)]["R"]
    b = tr[tr["entry_time"] >= pd.Timestamp(split)]["R"]
    t = tn.tstat
    print(f"\nYear 1 (before {split}): {len(a)} trades, avg R {a.mean():.3f}, t {t(a):.2f}")
    print(f"Year 2 (from {split}):   {len(b)} trades, avg R {b.mean():.3f}, t {t(b):.2f}")
    r = tr["R"]
    checks = [("at least 150 trades", len(r) >= 150),
              ("positive average R after costs", r.mean() > 0),
              ("t-stat of at least 2.5", t(r) >= 2.5),
              ("positive average R in BOTH years", len(a) > 0 and len(b) > 0 and a.mean() > 0 and b.mean() > 0)]
    print("\nPass criteria (fixed in advance):")
    for n, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print("RESULT:", "edge candidate" if all(ok for _, ok in checks) else "no edge shown")


def add_run_args(p):
    p.add_argument("--x", default="auto", help="min single-trade size in BTC, or auto")
    p.add_argument("--from", dest="date_from", default=None)
    p.add_argument("--to", dest="date_to", default=None)
    p.add_argument("--split", default="2025-11-01")
    p.add_argument("--depart", type=int, default=2)
    p.add_argument("--maxage", type=int, default=100)
    p.add_argument("--mult", type=float, default=2.0)
    p.add_argument("--pocLen", type=int, default=50)
    p.add_argument("--pad", type=int, default=2)
    p.add_argument("--merge", type=int, default=6)
    p.add_argument("--fillthru", type=float, default=0.0)
    p.add_argument("--confirm", default="none", choices=["none", "limit", "absorb", "aggr", "any"])
    p.add_argument("--bigmult", type=float, default=4.0)
    p.add_argument("--absmult", type=float, default=3.0)
    p.add_argument("--avgn", type=int, default=288)
    p.add_argument("--slmethod", default="zone", choices=["zone", "lowvol", "fixed"])
    p.add_argument("--slbuf", type=float, default=1.0)
    p.add_argument("--slrange", default="clamp", choices=["clamp", "skip"])
    p.add_argument("--adrdays", type=int, default=14)
    p.add_argument("--adrlo", type=float, default=0.10)
    p.add_argument("--adrhi", type=float, default=0.20)
    p.add_argument("--tpbuf", type=float, default=2.0)
    p.add_argument("--minrr", type=float, default=1.0)
    p.add_argument("--tpclose", default="skip", choices=["skip", "next"])
    p.add_argument("--notarget", default="skip", choices=["skip", "rr"])
    p.add_argument("--rr", type=float, default=2.0)
    p.add_argument("--tplook", type=int, default=336)
    p.add_argument("--feemaker", type=float, default=0.0002)
    p.add_argument("--feetaker", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--maxhold", type=int, default=576)


def main():
    global BIG_DF, BIG_X
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "count", "run"])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--row", type=float, default=10.0)
    ap.add_argument("--start", default="2024-11-01")
    ap.add_argument("--end", default="2026-11-05")
    ap.add_argument("--floor", type=float, default=1.0)
    add_run_args(ap)
    P = ap.parse_args()

    if P.cmd == "build":
        build(P.symbol, date.fromisoformat(P.start), date.fromisoformat(P.end), P.floor)
        return
    if P.cmd == "count":
        count(P.symbol, P.row)
        return

    BIG_DF = load_sig(P.symbol, P.row)
    BIG_X = count(P.symbol, P.row) if P.x == "auto" else float(P.x)
    print(f"Using X = {BIG_X} BTC")
    tb.find_clusters = find_bigs
    tb.zone_windows = tn.node_windows
    P.symbol = P.symbol
    tr, skips = tb.run(P)
    if not tr.empty:
        if P.date_from:
            tr = tr[tr["entry_time"] >= pd.Timestamp(P.date_from)]
        if P.date_to:
            tr = tr[tr["entry_time"] < pd.Timestamp(P.date_to)]
    print("Skipped zones:", dict(skips))
    tb.summarize(tr)
    if not tr.empty:
        verdict(tr, P.split)
        tr.to_csv(DATA / f"{P.symbol}_big_trades.csv", index=False)


if __name__ == "__main__":
    main()
