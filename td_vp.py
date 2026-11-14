"""
td_vp.py - Volume Profile book, Setup #1 "Volume accumulation" (book pp.64-69, SL/PT p.81).

Book rules:
  1. Find a sideways rotation area followed by a strong move (up = buyers accumulated -> LONG later,
     down = sellers accumulated -> SHORT later).
  2. Volume profile of the rotation only; mark the POC (price with the heaviest volume).
  3. Wait for price to return to the POC. Enter at the level, NO confirmation.
  4. Intraday SL and PT = 10-20% of the average daily ATR (ATR period 200).

Our mechanical choices (fixed BEFORE results; the book leaves them open):
  rotation  = N consecutive 30m candles whose total range <= K * ADR
  strong    = within M candles after the window, a 30m CLOSE is >= S * ADR beyond the window's
              high (up) or low (down). The level only becomes tradable AFTER that candle closes
              (no look-ahead).
  windows do not overlap. POC from 5m footprint cells re-binned to --vprow USD.
  level life = --maxage 30m candles (default 240 = 5 days), FIRST touch only, limit order at the POC.
  SL = TP = --sl / --tp (default 0.15) x ADR, ADR = rolling mean of daily range (200 days, min 30).
  Costs/fills/one position at a time: same engine as td_book.py.

Commands:
  python td_vp.py count            rotations per day for a grid of K (counting only, no P&L)
  python td_vp.py run              backtest (K auto-picked by count unless --k given)
  python td_vp.py run --feemaker 0 --feetaker 0 --slip 0     zero-fee screen
"""
import argparse

import numpy as np
import pandas as pd

import td_book as tb

DATA = tb.DATA
MS30, DAY = tb.MS30, tb.DAY
K_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
TARGET_PER_DAY = 0.5
VP = dict(n=12, k=0.5, s=0.5, m=8, vprow=10.0, sl=0.15, tp=0.15, adrdays=200)


def adr_daily(cd5, n=None):
    """Average daily range (ATR on 24/7 data), 200 days, min 30, known the day before."""
    n = n or VP["adrdays"]
    d = cd5.assign(day=cd5["candle"] // DAY).groupby("day").agg(hi=("high", "max"), lo=("low", "min"))
    return (d["hi"] - d["lo"]).rolling(n, min_periods=30).mean().shift(1).to_dict()


def rotations(fp30, cd30, adr, n, k, s, m, vprow):
    T = cd30["candle"].to_numpy()
    H, L, C = (cd30[c].to_numpy() for c in ["high", "low", "close"])
    f = fp30.sort_values("candle")
    fc, fl, ft = f["candle"].to_numpy(), f["level"].to_numpy(), f["total"].to_numpy()
    out, i, N = [], n - 1, len(T)
    while i < N - 1:
        a = adr.get(T[i] // DAY)
        if a is None or np.isnan(a):
            i += 1
            continue
        hw, lw = H[i - n + 1:i + 1].max(), L[i - n + 1:i + 1].min()
        if hw - lw > k * a:
            i += 1
            continue
        d, jj = 0, None
        for j in range(i + 1, min(i + 1 + m, N)):
            if C[j] >= hw + s * a:
                d, jj = 1, j
                break
            if C[j] <= lw - s * a:
                d, jj = -1, j
                break
        if d == 0:
            i += 1
            continue
        lo_i, hi_i = np.searchsorted(fc, T[i - n + 1], "left"), np.searchsorted(fc, T[i], "right")
        lv = np.floor(fl[lo_i:hi_i] / vprow) * vprow
        if len(lv) == 0:
            i += 1
            continue
        u, inv = np.unique(lv, return_inverse=True)
        poc = u[np.bincount(inv, weights=ft[lo_i:hi_i]).argmax()]
        out.append(dict(candle=T[i], dir=d, zone_low=poc, zone_high=poc + vprow,
                        kind="vp_accum", ready=T[jj] + MS30))
        i += n                                    # windows do not overlap
    return pd.DataFrame(out, columns=["candle", "dir", "zone_low", "zone_high", "kind", "ready"])


CTX = {}


def find_rot(fp30, cd30, row, mult, poc_len, pad, tf_min=30, merge=6):
    return rotations(fp30, cd30, CTX["adr"], VP["n"], VP["k"], VP["s"], VP["m"], VP["vprow"])


def windows(c30, zones, depart, max_age):
    """Engine lifecycle (price leaves by `depart` candles, dies if a close goes through), but a level
    is armed no earlier than the candle that proved the move was strong."""
    base, never = tb.zone_windows_orig(c30, zones, depart, max_age)
    out = []
    for z, t_arm, t_end in base:
        t_arm = max(t_arm, z.ready)
        if t_arm < t_end:
            out.append((z, t_arm, t_end))
        else:
            never += 1
    return out, never


def make_levels(P, ctx, z, d, entry, t_now):
    adr = ctx["adr"].get(t_now // DAY)
    if adr is None or np.isnan(adr):
        return None, None, "noadr"
    return entry - d * VP["sl"] * adr, entry + d * VP["tp"] * adr, None


def count(symbol):
    cd5 = pd.read_csv(DATA / f"{symbol}_5m_candles.csv")
    fp5 = pd.read_csv(DATA / f"{symbol}_5m_footprint.csv")
    fp30, cd30 = tb.to_30m(fp5, cd5)
    cd30 = cd30.sort_values("candle").reset_index(drop=True)
    adr = adr_daily(cd5)
    days = max((cd30["candle"].max() - cd30["candle"].min()) / DAY, 1)
    print(f"N={VP['n']} candles, S={VP['s']}, M={VP['m']}; {days:.0f} days")
    print("     K  rotations  per day   up  down")
    res = []
    for k in K_GRID:
        r = rotations(fp30, cd30, adr, VP["n"], k, VP["s"], VP["m"], VP["vprow"])
        res.append((k, len(r) / days))
        print(f"  {k:4.1f}  {len(r):9}  {len(r) / days:7.2f}  {(r.dir == 1).sum():4}  {(r.dir == -1).sum():4}")
    best = min(res, key=lambda x: abs(x[1] - TARGET_PER_DAY))
    print(f"\nclosest to {TARGET_PER_DAY:g}/day: K = {best[0]}  (counting only, no P&L used)")
    return best[0]


def verdict(tr, split):
    import td_big
    td_big.verdict(tr, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["count", "run"])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--row", type=float, default=10.0)
    ap.add_argument("--k", default="auto")
    ap.add_argument("--n", type=int, default=VP["n"])
    ap.add_argument("--s", type=float, default=VP["s"])
    ap.add_argument("--m", type=int, default=VP["m"])
    ap.add_argument("--vprow", type=float, default=VP["vprow"])
    ap.add_argument("--sl", type=float, default=VP["sl"], help="stop as fraction of ADR (book: 0.10-0.20)")
    ap.add_argument("--tp", type=float, default=VP["tp"], help="target as fraction of ADR (book: 0.10-0.20)")
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--split", default="2025-11-01")
    ap.add_argument("--depart", type=int, default=2)
    ap.add_argument("--maxage", type=int, default=240)
    ap.add_argument("--adrdays", type=int, default=200)
    ap.add_argument("--fillthru", type=float, default=0.0)
    ap.add_argument("--confirm", default="none")
    ap.add_argument("--feemaker", type=float, default=0.0002)
    ap.add_argument("--feetaker", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--maxhold", type=int, default=576)
    ap.add_argument("--out", default="vp1")
    P = ap.parse_args()
    VP.update(n=P.n, s=P.s, m=P.m, vprow=P.vprow, sl=P.sl, tp=P.tp, adrdays=P.adrdays)
    if P.cmd == "count":
        count(P.symbol)
        return
    VP["k"] = count(P.symbol) if P.k == "auto" else float(P.k)
    print(f"Using K = {VP['k']}, SL = {P.sl} x ADR, TP = {P.tp} x ADR")
    cd5 = pd.read_csv(DATA / f"{P.symbol}_5m_candles.csv")
    CTX["adr"] = adr_daily(cd5)
    tb.zone_windows_orig = tb.zone_windows
    tb.find_clusters, tb.zone_windows, tb.make_levels = find_rot, windows, make_levels
    tb.adr_by_day = lambda cd, n: CTX["adr"]
    for k_, v_ in dict(mult=2.0, pocLen=50, pad=2, merge=6, avgn=288, bigmult=4.0, absmult=3.0,
                       slmethod="zone", slbuf=1.0, slrange="clamp", adrlo=0.10, adrhi=0.20,
                       tpbuf=2.0, minrr=1.0, tpclose="skip", notarget="skip", rr=2.0, tplook=336).items():
        setattr(P, k_, v_)
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
        tr.to_csv(DATA / f"{P.symbol}_{P.out}_trades.csv", index=False)


if __name__ == "__main__":
    main()
