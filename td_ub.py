"""
td_ub.py - Trader Dale order flow book, Trading Setup #5: UNFINISHED BUSINESS (book pages 63-71).

Book: a properly formed HIGH has 0 contracts traded at the Bid in its top row; a properly formed LOW
has 0 at the Ask in its bottom row. If not (Bid AND Ask both traded) = Unfinished Business (UB).
The author says plainly: "I don't have a trading setup based solely on Unfinished Business; I use it
primarily to support other trade setups" (p.63, p.65, p.71). So it is NOT backtested as a stand-alone
entry. The book makes two claims we can test with data:

 TEST A  "MAGNET" (p.63-64): price is drawn back to UB. We compare, on 30m candles, how often an
         extreme with UB is re-touched within 48 candles vs a properly FINISHED extreme, at the SAME
         distance from the candle close (5 distance buckets, so a far-away level is not compared with
         a near one).
 TEST B  "WARNING" (p.66-67): do not go long with UB below entry / short with UB above entry.
         Applied to the trades already produced by setups 1-4 (data/BTCUSDT_*_trades.csv).
         Flagged = an untested UB lies on the adverse side of entry within 2 R of it.

Needs td_book.py, td_footprint.py, td_stack.py, the built 5m data. NO download.
    python td_ub.py magnet
    python td_ub.py filter
Pass criteria (fixed in advance, 4 files tested in TEST B so the bar is t >= 2.5):
  A: UB re-touch rate higher than finished, z >= 2.58 pooled, and positive in both years.
  B: flagged trades have lower avg R than unflagged by Welch t <= -2.5, difference negative in both
     years, AND unflagged avg R > 0 (otherwise the filter cannot make the setup tradable).
UB 'tested' = a later candle trades into the extreme row. A UB lives 100 30m candles (test B).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import td_book as tb
import td_stack as ts

DATA = tb.DATA
MS30 = tb.MS30
HORIZON = 48          # 30m candles in which the re-touch is counted (test A)
UB_LIFE = 100         # 30m candles a UB stays 'live' (test B)
FILES = ["book", "node", "big", "stack"]


def extremes(fp30, cd30, row):
    """One row per 30m candle and side (+1 high, -1 low): ub flag, level (row low edge)."""
    g = fp30.groupby("candle")["level"]
    top = fp30.loc[g.idxmax()].set_index("candle")
    bot = fp30.loc[g.idxmin()].set_index("candle")
    c = cd30.set_index("candle")
    c = c.loc[c.index.intersection(top.index)]
    rng = (c["high"] - c["low"]).rolling(48, min_periods=24).mean().shift(1)
    hi = pd.DataFrame({"candle": c.index, "side": 1, "ub": (top.loc[c.index, "sell"] > 0).to_numpy(),
                       "level": top.loc[c.index, "level"].to_numpy(), "extreme": c["high"].to_numpy(),
                       "dist": ((c["high"] - c["close"]) / rng).to_numpy()})
    lo = pd.DataFrame({"candle": c.index, "side": -1, "ub": (bot.loc[c.index, "buy"] > 0).to_numpy(),
                       "level": bot.loc[c.index, "level"].to_numpy(), "extreme": c["low"].to_numpy(),
                       "dist": ((c["close"] - c["low"]) / rng).to_numpy()})
    return pd.concat([hi, lo], ignore_index=True), c


def retouch(ex, c, row, n=HORIZON):
    H = pd.Series(c["high"].to_numpy(), index=c.index)
    L = pd.Series(c["low"].to_numpy(), index=c.index)
    fmax = H[::-1].rolling(n, min_periods=n).max()[::-1].shift(-1)
    fmin = L[::-1].rolling(n, min_periods=n).min()[::-1].shift(-1)
    fm = ex["candle"].map(fmax).to_numpy()
    fl = ex["candle"].map(fmin).to_numpy()
    hit = np.where(ex["side"] == 1, fm >= ex["level"].to_numpy(), fl <= ex["level"].to_numpy() + row)
    ok = ~np.where(ex["side"] == 1, np.isnan(fm), np.isnan(fl))
    out = ex.assign(hit=hit.astype(float))[ok & ex["dist"].notna().to_numpy()]
    return out


def strat_diff(d):
    """UB minus finished re-touch rate, inverse-variance pooled over 5 distance strata."""
    if d.empty:
        return np.nan, np.nan, np.nan, 0
    d = d.copy()
    d["bin"] = pd.qcut(d["dist"], 5, labels=False, duplicates="drop")
    num = den = 0.0
    n_used = 0
    for _, s in d.groupby("bin"):
        a, b = s[s["ub"]]["hit"], s[~s["ub"]]["hit"]
        if len(a) < 30 or len(b) < 30:
            continue
        pa, pb = a.mean(), b.mean()
        var = max(pa * (1 - pa) / len(a) + pb * (1 - pb) / len(b), 1e-12)
        num += (pa - pb) / var
        den += 1 / var
        n_used += len(s)
    if den == 0:
        return np.nan, np.nan, np.nan, 0
    diff, se = num / den, 1 / np.sqrt(den)
    return diff, se, diff / se, n_used


def magnet(P):
    fp30, cd30 = ts.load_fp30(P.symbol)
    ex, c = extremes(fp30, cd30, P.row)
    d = retouch(ex, c, P.row)
    d["year2"] = pd.to_datetime(d["candle"], unit="ms") >= pd.Timestamp(P.split)
    print(f"{len(d)} candle extremes; UB share {d['ub'].mean():.1%}; re-touch within {HORIZON} candles\n")
    print("  group                 n      UB rate  finished rate   UB-fin diff    z")
    res = {}
    for name, sub in [("highs", d[d["side"] == 1]), ("lows", d[d["side"] == -1]), ("pooled", d),
                      ("year 1", d[~d["year2"]]), ("year 2", d[d["year2"]])]:
        diff, se, z, n = strat_diff(sub)
        pa, pb = sub[sub["ub"]]["hit"].mean(), sub[~sub["ub"]]["hit"].mean()
        res[name] = (diff, z)
        print(f"  {name:<12} {len(sub):>9}  {pa:9.3f}  {pb:11.3f}   {diff:+10.4f}  {z:7.2f}")
    ok = [("pooled z >= 2.58 and diff > 0", res["pooled"][1] >= 2.58 and res["pooled"][0] > 0),
          ("diff positive in BOTH years", res["year 1"][0] > 0 and res["year 2"][0] > 0)]
    print("\nPass criteria (fixed in advance):")
    for n, o in ok:
        print(f"  {'PASS' if o else 'FAIL'}  {n}")
    print("RESULT:", "magnet effect shown" if all(o for _, o in ok) else "no magnet effect shown")


def ub_levels(fp30, cd30, cd5, row):
    """Every UB with creation time and the first 5m candle that traded into its row."""
    ex, _ = extremes(fp30, cd30, row)
    u = ex[ex["ub"]].sort_values("candle").reset_index(drop=True)
    T5 = cd5["candle"].to_numpy()
    H5, L5 = cd5["high"].to_numpy(), cd5["low"].to_numpy()
    made = u["candle"].to_numpy() + MS30
    start = np.searchsorted(T5, made)
    horizon5 = UB_LIFE * 6
    touch = np.full(len(u), np.iinfo("int64").max, dtype="int64")
    lv, side = u["level"].to_numpy(), u["side"].to_numpy()
    for i in range(len(u)):
        s, e = start[i], min(start[i] + horizon5, len(T5))
        if s >= e:
            continue
        h = (H5[s:e] >= lv[i]) if side[i] == 1 else (L5[s:e] <= lv[i] + row)
        j = h.argmax()
        if h[j]:
            touch[i] = T5[s + j]
    return pd.DataFrame({"made": made, "price": lv + row / 2, "touch": touch,
                         "dead": made + UB_LIFE * MS30})


def flag_trades(tr, ub):
    made, price, touch, dead = (ub[k].to_numpy() for k in ["made", "price", "touch", "dead"])
    et = ((tr["entry_time"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)).to_numpy()
    flags = np.zeros(len(tr), bool)
    for i, t in enumerate(tr.itertuples()):
        live = (made <= et[i]) & (dead > et[i]) & (touch >= et[i])        # touched in earlier 5m = tested
        risk = abs(t.entry - t.stop)
        if t.dir == 1:
            near = (price < t.entry) & (t.entry - price <= 2 * risk)
        else:
            near = (price > t.entry) & (price - t.entry <= 2 * risk)
        flags[i] = bool((live & near).any())
    return flags


def welch(a, b):
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def filt(P):
    fp30, cd30 = ts.load_fp30(P.symbol)
    cd5 = pd.read_csv(DATA / f"{P.symbol}_5m_candles.csv").sort_values("candle").reset_index(drop=True)
    ub = ub_levels(fp30, cd30, cd5, P.row)
    print(f"{len(ub)} unfinished-business levels built\n")
    for name in FILES:
        p = DATA / f"{P.symbol}_{name}_trades.csv"
        if not p.exists():
            print(f"[{name}] {p.name} not found, skipped\n")
            continue
        tr = pd.read_csv(p, parse_dates=["entry_time"])
        tr["flag"] = flag_trades(tr, ub)
        y2 = tr["entry_time"] >= pd.Timestamp(P.split)
        f, u = tr[tr["flag"]]["R"], tr[~tr["flag"]]["R"]
        t = welch(f, u)
        d1 = tr[tr["flag"] & ~y2]["R"].mean() - tr[~tr["flag"] & ~y2]["R"].mean()
        d2 = tr[tr["flag"] & y2]["R"].mean() - tr[~tr["flag"] & y2]["R"].mean()
        print(f"[{name}] {len(tr)} trades | flagged {len(f)} ({len(f) / len(tr):.0%}) avg R {f.mean():+.3f}"
              f" | unflagged {len(u)} avg R {u.mean():+.3f}")
        print(f"        diff (flagged - unflagged) {f.mean() - u.mean():+.3f}, Welch t {t:.2f};"
              f" year1 diff {d1:+.3f}, year2 diff {d2:+.3f}")
        ok = [("flagged worse, t <= -2.5", t <= -2.5),
              ("flagged worse in BOTH years", d1 < 0 and d2 < 0),
              ("unflagged avg R > 0 after the file's costs", u.mean() > 0)]
        for n, o in ok:
            print(f"        {'PASS' if o else 'FAIL'}  {n}")
        print(f"        RESULT: {'warning filter helps' if all(o for _, o in ok) else 'not shown'}\n")
    print("Note: node/book CSVs hold whichever run you saved last (zero-fee or real costs); the flagged"
          " vs unflagged comparison inside one file is unaffected, the 'avg R > 0' check is.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["magnet", "filter"])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--row", type=float, default=10.0)
    ap.add_argument("--split", default="2025-11-01")
    P = ap.parse_args()
    magnet(P) if P.cmd == "magnet" else filt(P)


if __name__ == "__main__":
    main()
