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

Setup #2 "Trend" (book pp.70-72; PT/SL pp.116-121): the same level logic, but the rotation/cluster must
sit INSIDE a strong trend (price already moved >= --tmove x ADR in the same direction over the
--lb candles before the cluster) and the trend must continue after it. Enter in the trend direction
at the first touch of the cluster's POC, no confirmation.
  --setup trend   (defaults n=8 candles, lb=24, tmove=0.7)
  --tpmode volume book's volume-based PT: just before the nearest heavy 30m volume area (skip trade if the
                  PT is closer than 10% of ADR); SL stays fixed 15% of ADR. Default is fixed 1:1.
Setup #3 "Rejection" (book pp.75-79): a very strong rejection of lower (-> LONG) or higher (-> SHORT) prices.
Volume profile over the rejection area only; the level is the POC of that area; wait for price to come back,
enter at the first touch, no confirmation.
  --setup reject   swing low/high (+-6 candles) after a >= A x ADR move into it within 8 candles, then a
                   close >= A x ADR back the other way within 8 candles. Area = swing candle +-2 candles.
                   A is picked by count (target 0.5/day), never by P&L. Level armed only after the reversal
                   candle has closed (no look-ahead).
  --reversal       book p.78 "reversal trades": when a level fails (trade stopped out = price shot past it), wait
                   for price to come back to the level and enter the OPPOSITE direction, first touch. Works
                   with any setup in fixed 1:1 mode. Reported separately and combined (one position at a time).
Alternative SL (close beyond SL, 150% catastrophic) is NOT tested: Dale does not use it intraday.

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
VP = dict(n=12, k=0.5, s=0.5, m=8, vprow=10.0, sl=0.15, tp=0.15, adrdays=200, lb=0, tmove=0.0,
          setup='accum', a=0.5, w=6, win=2, p=8, q=8, b=0.1, life=240)
A_GRID = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]


def adr_daily(cd5, n=None):
    """Average daily range (ATR on 24/7 data), 200 days, min 30, known the day before."""
    n = n or VP["adrdays"]
    d = cd5.assign(day=cd5["candle"] // DAY).groupby("day").agg(hi=("high", "max"), lo=("low", "min"))
    return (d["hi"] - d["lo"]).rolling(n, min_periods=30).mean().shift(1).to_dict()


def rotations(fp30, cd30, adr, n, k, s, m, vprow, lb=0, tmove=0.0):
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
        if lb:                                    # trend setup: cluster must sit inside a trend
            i0 = i - n + 1 - lb
            if i0 < 0 or d * (C[i - n] - C[i0]) < tmove * a:
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
                        kind="vp_trend" if lb else "vp_accum", ready=T[jj] + MS30))
        i += n                                    # windows do not overlap
    return pd.DataFrame(out, columns=["candle", "dir", "zone_low", "zone_high", "kind", "ready"])


def rejections(fp30, cd30, adr, w, win, p, q, a_mult, vprow):
    T = cd30["candle"].to_numpy()
    H, L, C = (cd30[c].to_numpy() for c in ["high", "low", "close"])
    f = fp30.sort_values("candle")
    fc, fl, ft = f["candle"].to_numpy(), f["level"].to_numpy(), f["total"].to_numpy()
    out, i, N = [], max(w, p), len(T)
    while i < N - w:
        a = adr.get(T[i] // DAY)
        if a is None or np.isnan(a):
            i += 1
            continue
        side, jj = 0, None
        if L[i] == L[i - w:i + w + 1].min() and H[i - p:i].max() - L[i] >= a_mult * a:
            for j in range(i + 1, min(i + 1 + q, N)):
                if C[j] >= L[i] + a_mult * a:
                    side, jj = 1, j
                    break
        elif H[i] == H[i - w:i + w + 1].max() and H[i] - L[i - p:i].min() >= a_mult * a:
            for j in range(i + 1, min(i + 1 + q, N)):
                if C[j] <= H[i] - a_mult * a:
                    side, jj = -1, j
                    break
        if side == 0:
            i += 1
            continue
        lo_i = np.searchsorted(fc, T[i - win], "left")
        hi_i = np.searchsorted(fc, T[i + win], "right")
        lv = np.floor(fl[lo_i:hi_i] / vprow) * vprow
        if len(lv) == 0:
            i += 1
            continue
        u, inv = np.unique(lv, return_inverse=True)
        poc = u[np.bincount(inv, weights=ft[lo_i:hi_i]).argmax()]
        out.append(dict(candle=T[i + win], dir=side, zone_low=poc, zone_high=poc + vprow, kind="vp_reject",
                        ready=max(T[jj], T[i + w]) + MS30))
        i += w
    return pd.DataFrame(out, columns=["candle", "dir", "zone_low", "zone_high", "kind", "ready"])


def flips(cd30, adr, w, p, q, a_mult, b_mult, life, vprow):
    """Price action setup (book p.23-25): a strong reaction away from a level (swing low/high, same test as
    the rejection setup but price only), then a CLOSE through the level by >= B x ADR within `life`
    candles. Support breached -> new resistance -> short at the level; resistance breached -> long."""
    T = cd30["candle"].to_numpy()
    H, L, C = (cd30[c].to_numpy() for c in ["high", "low", "close"])
    out, i, N = [], max(w, p), len(T)
    while i < N - w:
        a = adr.get(T[i] // DAY)
        if a is None or np.isnan(a):
            i += 1
            continue
        side, jj = 0, None
        if L[i] == L[i - w:i + w + 1].min() and H[i - p:i].max() - L[i] >= a_mult * a:
            for j in range(i + 1, min(i + 1 + q, N)):
                if C[j] >= L[i] + a_mult * a:
                    side, jj = 1, j          # reaction up from a support
                    break
        elif H[i] == H[i - w:i + w + 1].max() and H[i] - L[i - p:i].min() >= a_mult * a:
            for j in range(i + 1, min(i + 1 + q, N)):
                if C[j] <= H[i] - a_mult * a:
                    side, jj = -1, j         # reaction down from a resistance
                    break
        if side == 0:
            i += 1
            continue
        level = L[i] if side == 1 else H[i]
        k0 = max(jj, i + w) + 1
        for k in range(k0, min(k0 + life, N)):
            ak = adr.get(T[k] // DAY)
            if ak is None or np.isnan(ak):
                continue
            if (side == 1 and C[k] < level - b_mult * ak) or (side == -1 and C[k] > level + b_mult * ak):
                d = -side                    # support became resistance -> short
                lo, hi = (level, level + vprow) if d == -1 else (level - vprow, level)
                out.append(dict(candle=T[k], dir=d, zone_low=lo, zone_high=hi, kind="pa_flip",
                                ready=T[k] + MS30))
                break
        i += w
    return pd.DataFrame(out, columns=["candle", "dir", "zone_low", "zone_high", "kind", "ready"])


CTX = {}


def find_rot(fp30, cd30, row, mult, poc_len, pad, tf_min=30, merge=6):
    if VP["setup"] == "flip":
        return flips(cd30, CTX["adr"], VP["w"], VP["p"], VP["q"], VP["a"], VP["b"], VP["life"], VP["vprow"])
    if VP["setup"] == "reject":
        return rejections(fp30, cd30, CTX["adr"], VP["w"], VP["win"], VP["p"], VP["q"], VP["a"], VP["vprow"])
    return rotations(fp30, cd30, CTX["adr"], VP["n"], VP["k"], VP["s"], VP["m"], VP["vprow"],
                     VP["lb"], VP["tmove"])


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
    rej = VP["setup"] in ("reject", "flip")
    if VP["setup"] == "flip":
        print(f"swing +-{VP['w']} candles, lookback {VP['p']}, reaction within {VP['q']}, breach close >= {VP['b']} x ADR within {VP['life']} candles; {days:.0f} days")
        print("     A  flip setups  per day  long  short")
    elif rej:
        print(f"swing +-{VP['w']} candles, area +-{VP['win']}, lookback {VP['p']}, reversal within {VP['q']}; {days:.0f} days")
        print("     A  rejections  per day   long  short")
    else:
        print(f"N={VP['n']} candles, S={VP['s']}, M={VP['m']}, trend filter lb={VP['lb']} tmove={VP['tmove']}; {days:.0f} days")
        print("     K  rotations  per day   up  down")
    res = []
    for g in (A_GRID if rej else K_GRID):
        if VP["setup"] == "flip":
            r = flips(cd30, adr, VP["w"], VP["p"], VP["q"], g, VP["b"], VP["life"], VP["vprow"])
        elif rej:
            r = rejections(fp30, cd30, adr, VP["w"], VP["win"], VP["p"], VP["q"], g, VP["vprow"])
        else:
            r = rotations(fp30, cd30, adr, VP["n"], g, VP["s"], VP["m"], VP["vprow"], VP["lb"], VP["tmove"])
        res.append((g, len(r) / days))
        print(f"  {g:4.1f}  {len(r):9}  {len(r) / days:7.2f}  {(r.dir == 1).sum():4}  {(r.dir == -1).sum():4}")
    best = min(res, key=lambda x: abs(x[1] - TARGET_PER_DAY))
    print(f"\nclosest to {TARGET_PER_DAY:g}/day: {'A' if rej else 'K'} = {best[0]}  (counting only, no P&L used)")
    return best[0]


def reversals(tr, cd5, P):
    """Book p.78: a level that failed (stopped out) is re-entered the OTHER way at its first return."""
    ms = lambda t: int((t - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1))
    T5 = cd5["candle"].to_numpy()
    O5, H5, L5, C5 = (cd5[c].to_numpy() for c in ["open", "high", "low", "close"])
    n5, life, out = len(T5), P.maxage * 6, []
    for r in tr[tr["reason"] == "stop"].itertuples():
        level = (r.stop + r.target) / 2                       # fixed 1:1 mode: level sits midway
        d = -r.dir
        k_exit = int(np.searchsorted(T5, ms(r.exit_time), "left"))
        adr = CTX["adr"].get(T5[min(k_exit, n5 - 1)] // DAY)
        if adr is None or np.isnan(adr):
            continue
        sd = VP["sl"] * adr
        stop, tp = level - d * sd, level + d * VP["tp"] * adr
        for k0 in range(k_exit + 1, min(k_exit + 1 + life, n5)):
            if (H5[k0] >= level) if d == -1 else (L5[k0] <= level):
                break
        else:
            continue
        fill = max(O5[k0], level) if d == -1 else min(O5[k0], level)
        last = min(k0 + P.maxhold, n5 - 1)
        exit_px, reason, k_x = C5[last], "timeout", last
        for k in range(k0, last + 1):
            if (L5[k] <= stop) if d == 1 else (H5[k] >= stop):
                exit_px = min(O5[k], stop) if d == 1 else max(O5[k], stop)
                reason, k_x = "stop", k
                break
            if (H5[k] >= tp) if d == 1 else (L5[k] <= tp):
                exit_px, reason, k_x = tp, "target", k
                break
        x_eff = exit_px if reason == "target" else exit_px * (1 - d * P.slip)
        fee_out = P.feemaker if reason == "target" else P.feetaker
        rr = (d * (x_eff - fill) - P.feemaker * fill - fee_out * x_eff) / sd
        out.append(dict(entry_time=pd.to_datetime(T5[k0], unit="ms"), exit_time=pd.to_datetime(T5[k_x], unit="ms"),
                        dir=d, kind="reversal", mode="limit", entry=fill, stop=stop, target=tp, exit=exit_px,
                        reason=reason, risk_pct=sd / level * 100, hours=(k_x - k0 + 1) * 5 / 60, R=rr))
    return pd.DataFrame(out)


def one_at_a_time(a, b):
    both = pd.concat([a, b]).sort_values("entry_time")
    keep, free = [], None
    for r in both.itertuples():
        if free is None or r.entry_time > free:
            keep.append(r.Index if False else r)
            free = r.exit_time
    return pd.DataFrame([k._asdict() for k in keep]).drop(columns="Index")


def verdict(tr, split):
    import td_big
    td_big.verdict(tr, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["count", "run"])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--row", type=float, default=10.0)
    ap.add_argument("--k", default="auto")
    ap.add_argument("--setup", default="accum", choices=["accum", "trend", "reject", "flip"])
    ap.add_argument("--tpmode", default="fixed", choices=["fixed", "volume"])
    ap.add_argument("--reversal", action="store_true")
    ap.add_argument("--a", default="auto")
    ap.add_argument("--b", type=float, default=0.1, help="flip: breach close beyond level, x ADR")
    ap.add_argument("--lb", type=int, default=None)
    ap.add_argument("--tmove", type=float, default=None)
    ap.add_argument("--n", type=int, default=None)
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
    ap.add_argument("--out", default=None)
    P = ap.parse_args()
    trend = P.setup == "trend"
    P.n = P.n if P.n is not None else (8 if trend else 12)
    P.lb = P.lb if P.lb is not None else (24 if trend else 0)
    P.tmove = P.tmove if P.tmove is not None else (0.7 if trend else 0.0)
    P.out = P.out or {"accum": "vp1", "trend": "vp2", "reject": "vp3", "flip": "vp4"}[P.setup]
    VP.update(n=P.n, s=P.s, m=P.m, vprow=P.vprow, sl=P.sl, tp=P.tp, adrdays=P.adrdays, lb=P.lb, tmove=P.tmove, setup=P.setup, b=P.b)
    if P.cmd == "count":
        count(P.symbol)
        return
    if P.setup in ("reject", "flip"):
        VP["a"] = count(P.symbol) if P.a == "auto" else float(P.a)
    else:
        VP["k"] = count(P.symbol) if P.k == "auto" else float(P.k)
    if P.reversal and P.tpmode != "fixed":
        raise SystemExit("--reversal needs --tpmode fixed")
    print(f"Setup {P.setup}, {'A' if P.setup in ('reject', 'flip') else 'K'} = {VP['a'] if P.setup in ('reject', 'flip') else VP['k']}, SL = {P.sl} x ADR, TP = " +
          (f"{P.tp} x ADR" if P.tpmode == "fixed" else "volume-based"))
    cd5 = pd.read_csv(DATA / f"{P.symbol}_5m_candles.csv")
    CTX["adr"] = adr_daily(cd5)
    tb.zone_windows_orig = tb.zone_windows
    tb.make_levels_orig = tb.make_levels
    tb.find_clusters, tb.zone_windows = find_rot, windows
    tb.make_levels = make_levels if P.tpmode == "fixed" else tb.make_levels_orig
    tb.adr_by_day = lambda cd, n: CTX["adr"]
    for k_, v_ in dict(mult=2.0, pocLen=50, pad=2, merge=6, avgn=288, bigmult=4.0, absmult=3.0,
                       slmethod="zone", slbuf=1.0, slrange="clamp", adrlo=0.10, adrhi=0.20,
                       tpbuf=2.0, minrr=1.0, tpclose="skip", notarget="skip", rr=2.0, tplook=336).items():
        setattr(P, k_, v_)
    if P.tpmode == "volume":                  # book p.116: skip if PT closer than 10% of ADR (SL = 15% of ADR)
        P.slmethod, P.minrr = "fixed", 0.10 / 0.15
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
    if P.reversal and not tr.empty:
        rev = reversals(tr, cd5, P)
        print("\n===== REVERSAL TRADES (re-entry of stopped-out levels, opposite direction) =====")
        if rev.empty:
            print("No reversal trades.")
            return
        tb.summarize(rev)
        verdict(rev, P.split)
        rev.to_csv(DATA / f"{P.symbol}_{P.out}_rev_trades.csv", index=False)
        print("\n===== COMBINED (original + reversal, one position at a time) =====")
        comb = one_at_a_time(tr, rev)
        tb.summarize(comb)
        verdict(comb, P.split)


if __name__ == "__main__":
    main()
