"""
td_node.py - Trader Dale order flow book, Trading Setup #2: MULTIPLE NODES (book pages 47-51).

Reuses td_book.py for EVERYTHING except finding the zones: same 5m data, same stop loss
and take profit rules, same confirmations, same fees and fill simulation. So the numbers
are directly comparable with the Volume Cluster results.

Needs td_book.py and td_footprint.py in the same folder, and the 5m data already built
(python td_footprint.py build --tf 5 --start 2024-11-01 --end 2026-11-05).

Run:       python td_node.py
Examples:  python td_node.py --confirm any
           python td_node.py --tpclose next
           python td_node.py --from 2025-11-01

BOOK vs OURS (see bottom of the file).
"""
import argparse
from collections import namedtuple

import numpy as np
import pandas as pd

import td_book as tb

MS30 = tb.MS30
DATA = tb.DATA
Node = namedtuple("Node", "candle dir zone_low zone_high kind")
NODE_TOL = 1.0          # set from --tol


# ------------------------------------------------------------------ nodes
def find_nodes(fp30, cd30, row, mult, pocLen, pad, tf_min=30, merge=6):
    """Multiple Node = 2+ CONSECUTIVE 30m candles whose HVN (the row with the most total
    volume inside the candle) is at the same price (within NODE_TOL rows).
    One node per run, created when the SECOND candle closes (no look-ahead).
    Direction is not known yet; it is decided later by where price leaves to."""
    idx = fp30.groupby("candle")["total"].idxmax()
    hvn = fp30.loc[idx].set_index("candle")["level"]
    c = cd30["candle"].to_numpy()
    lv = hvn.reindex(c).to_numpy(dtype=float)
    tol = NODE_TOL * row + 1e-9
    n = len(c)
    pair = np.zeros(n, bool)
    for i in range(1, n):
        pair[i] = (c[i] - c[i - 1] == MS30 and not np.isnan(lv[i]) and not np.isnan(lv[i - 1])
                   and abs(lv[i] - lv[i - 1]) <= tol)
    rows = []
    for i in range(1, n):
        if pair[i] and not pair[i - 1]:                  # start of a run
            lo = min(lv[i], lv[i - 1])
            hi = max(lv[i], lv[i - 1]) + row
            rows.append((c[i], lo, hi))
    return pd.DataFrame(rows, columns=["candle", "zone_low", "zone_high"])


def node_windows(c30, nodes, depart, max_age):
    """Book steps 3-5. After the node: at least `depart` WHOLE 30m candles completely above
    (-> pullback from above = LONG) or completely below (-> SHORT). Then it is armed and we
    wait for the first touch. Dies if a 30m candle closes through it, or after max_age candles.
    Returns [(node_with_direction, t_arm, t_end)], never_armed."""
    idx = {t: i for i, t in enumerate(c30["candle"])}
    T = c30["candle"].to_numpy()
    H, L, C = (c30[k].to_numpy() for k in ["high", "low", "close"])
    n = len(c30)
    out, never = [], 0
    for z in nodes.itertuples():
        i0 = idx.get(z.candle)
        if i0 is None:
            continue
        lo, hi = z.zone_low, z.zone_high
        up = dn = 0
        d, t_arm, end_i = 0, None, i0
        for j in range(i0 + 1, min(i0 + 1 + max_age, n)):
            end_i = j
            if d == 0:
                up = up + 1 if L[j] > hi else 0
                dn = dn + 1 if H[j] < lo else 0
                if up >= depart:
                    d, t_arm = 1, T[j] + MS30
                elif dn >= depart:
                    d, t_arm = -1, T[j] + MS30
            elif (d == 1 and C[j] < lo) or (d == -1 and C[j] > hi):
                break
        if t_arm is None:
            never += 1
        else:
            out.append((Node(z.candle, d, lo, hi, "node"), t_arm, T[end_i] + MS30))
    return out, never


# ------------------------------------------------------------------ report
def tstat(r):
    return r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 and r.std() > 0 else float("nan")


def verdict(tr, split):
    """Pass criteria fixed BEFORE seeing any result."""
    a = tr[tr["entry_time"] < pd.Timestamp(split)]["R"]
    b = tr[tr["entry_time"] >= pd.Timestamp(split)]["R"]
    print(f"\nYear 1 (before {split}): {len(a)} trades, avg R {a.mean():.3f}, t {tstat(a):.2f}")
    print(f"Year 2 (from {split}):   {len(b)} trades, avg R {b.mean():.3f}, t {tstat(b):.2f}")
    r = tr["R"]
    checks = [("at least 150 trades", len(r) >= 150),
              ("positive average R after costs", r.mean() > 0),
              ("t-stat of at least 2", tstat(r) >= 2),
              ("positive average R in BOTH years", len(a) > 0 and len(b) > 0 and a.mean() > 0 and b.mean() > 0)]
    print("\nPass criteria (fixed in advance):")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("RESULT:", "edge candidate" if all(ok for _, ok in checks) else "no edge shown")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--row", type=float, default=10.0)
    p.add_argument("--from", dest="date_from", default=None)
    p.add_argument("--to", dest="date_to", default=None)
    p.add_argument("--split", default="2025-11-01", help="boundary between year 1 and year 2")
    # node
    p.add_argument("--tol", type=float, default=1.0, help="HVNs count as the same price within this many rows")
    p.add_argument("--depart", type=int, default=2, help="whole 30m candles away from the node (book: 2)")
    p.add_argument("--maxage", type=int, default=100, help="node expires after this many 30m candles")
    # same meaning as td_book (heavy areas for take profit, fills)
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
    P = p.parse_args()

    global NODE_TOL
    NODE_TOL = P.tol
    tb.find_clusters = find_nodes          # swap only the zone finder and the zone lifecycle
    tb.zone_windows = node_windows

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
        tr.to_csv(DATA / f"{P.symbol}_node_trades.csv", index=False)
        tb.plot_equity(tr, DATA / f"{P.symbol}_node_equity.png")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# BOOK vs OURS
# FROM THE BOOK (pages 47-51)
#  - HVN = heaviest-volume row inside one footprint; 30m chart, volume only (no bid x ask)
#  - Multiple Node = 2+ consecutive footprints with their HVN at the same price
#  - After it forms, at least 2 whole footprints completely above or below it
#  - Wait for a pullback; from above = long, from below = short; first touch only
#  - Stop loss, take profit, 10-20% range rule, confirmations: same pages as setup 1 (td_book.py)
# OUR CHOICES (the book gives no number)
#  - "same price" = within 1 row (default row = 10 USD); node zone = the HVN row(s) of the pair
#  - node is created when the 2nd candle closes; longer runs do not create extra nodes
#  - node expires after 100 30m candles or a 30m close through it
#  - NO trend / before-trend / rejection filter: the book gives no measurable rule, so every
#    node is traded. The book's author picks by eye, so this probably trades more nodes than he would.
#  - entry: limit at the near edge of the node (like setup 1); with --confirm: market entry
# ---------------------------------------------------------------------------
