"""
td_htf.py - run setups 1-4 on a HIGHER timeframe (default 4H) from the same 5m data. No download.

A 4H footprint is the exact sum of its 5m footprints (same trades, same rows), exactly like the 30m
ones. This wrapper only changes the candle length (td_book/td_node/td_big/td_stack all read it from
MS30) and a few defaults, then runs the normal script, so zones, entries, SL/TP logic, fees and the
verdict are identical to the 30m runs.

    python td_htf.py book  --feemaker 0 --feetaker 0 --slip 0          setup 1 (Volume Clusters)
    python td_htf.py node  ...                                          setup 2 (Multiple Nodes)
    python td_htf.py big   count                                        setup 3: pick X by counting only
    python td_htf.py big   run --x auto ...
    python td_htf.py stack count                                        setup 4: pick min volume by counting
    python td_htf.py stack run --minvol auto ...
Always run the zero-fee screen first, then the real run with --tpclose next.

4H DEFAULTS (fixed in advance, change nothing after seeing results):
  --tf 240            candle length in minutes
  --row 50            price rows re-bucketed from 10 to 50 USD (a 4H candle spans ~20-40 rows)
  --depart 2          2 whole candles away (book)
  --maxage 30         zone lives 30 candles = 5 days
  --tplook 42         heavy areas for take profit looked up over 42 candles = 7 days
  --adrlo/--adrhi 0.30/0.60   stop 30-60% of the daily average range (book: 10-20% for 30m)
  --maxhold 2016      5m candles = 7 days
  counting target     1 zone per day (setup 3 and 4), by counting only, never by P&L
Pass criteria (fixed in advance): >=150 trades (fewer = inconclusive, not a fail), avg R>0 after costs,
t>=2.5, positive in BOTH years. Zero-fee screen: gross avg R below ~+0.05R cannot survive costs.
Outputs are saved with a _240m suffix so the 30m files are not overwritten.
"""
import sys

import numpy as np
import pandas as pd

import td_book as tb
import td_node as tn
import td_big as bg
import td_stack as ts

DEFAULTS = {"--row": "50", "--maxage": "30", "--tplook": "42",
            "--adrlo": "0.30", "--adrhi": "0.60", "--maxhold": "2016"}


def pop_opt(args, name, default):
    if name in args:
        i = args.index(name)
        val = args[i + 1]
        del args[i:i + 2]
        return val
    return default


def make_to_tf(tf_ms, row_new, row_old=10.0):
    def to_tf(fp5, cd5):
        fp = fp5.copy()
        fp["candle"] = (fp["candle"] // tf_ms) * tf_ms
        if row_new != row_old:
            fp["level"] = (np.floor(fp["level"] / row_new + 1e-9) * row_new).round(8)
        out = fp.groupby(["candle", "level"], as_index=False)[["buy", "sell"]].sum()
        out["total"] = out["buy"] + out["sell"]
        cd = cd5.copy()
        cd["ck"] = (cd["candle"] // tf_ms) * tf_ms
        c = (cd.groupby("ck").agg(open=("open", "first"), high=("high", "max"),
                                  low=("low", "min"), close=("close", "last"))
               .reset_index().rename(columns={"ck": "candle"}))
        return out, c
    return to_tf


def main():
    args = sys.argv[1:]
    if not args or args[0] not in ("book", "node", "big", "stack"):
        print(__doc__)
        return
    setup = args.pop(0)
    tf = int(pop_opt(args, "--tf", "240"))
    tf_ms = tf * 60_000
    for flag, val in DEFAULTS.items():
        if flag not in args:
            args += [flag, val]
    row = float(args[args.index("--row") + 1])
    symbol = args[args.index("--symbol") + 1] if "--symbol" in args else "BTCUSDT"

    for m in (tb, tn, bg, ts):
        m.MS30 = tf_ms
    tb.to_30m = make_to_tf(tf_ms, row)
    orig_fc = tb.find_clusters
    tb.find_clusters = lambda *a, **k: orig_fc(*a, **{**k, "tf_min": tf})
    bg.TARGET_PER_DAY = ts.TARGET_PER_DAY = 1.0
    bg.X_GRID = bg.X_GRID + [300, 400, 500, 750, 1000, 1500]
    ts.MINVOL_GRID = [0, 1, 2, 5, 10, 20, 30, 50, 80, 120, 200, 300, 500]
    bg.sig_path = lambda s, r: bg.DATA / f"{s}_bigsig_{r:g}_{tf}m.csv"

    D = tb.DATA
    names = {"book": ["book_trades.csv", "book_equity.png"], "node": ["node_trades.csv", "node_equity.png"],
             "big": ["big_trades.csv"], "stack": ["stack_trades.csv"]}[setup]
    files = [D / f"{symbol}_{n}" for n in names]
    backup = {f: f.read_bytes() for f in files if f.exists()}
    sys.argv = [setup] + args
    try:
        {"book": tb.main, "node": tn.main, "big": bg.main, "stack": ts.main}[setup]()
    finally:
        for f in files:
            if f.exists():
                new = f.with_name(f.stem + f"_{tf}m" + f.suffix)
                new.write_bytes(f.read_bytes())
                print(f"saved {new.name}")
            if f in backup:
                f.write_bytes(backup[f])
            elif f.exists():
                f.unlink()


if __name__ == "__main__":
    main()
