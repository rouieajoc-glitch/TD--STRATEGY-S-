"""
td_book.py - Volume Cluster strategy using the book's own rules for stop loss,
take profit and (optional) order-flow confirmation.

Needs the 5-minute data:   python td_footprint.py build --tf 5 --start ... --end ...
Run:                       python td_book.py
Examples:                  python td_book.py --confirm any
                           python td_book.py --slmethod lowvol
                           python td_book.py --from 2026-05-01        (test on the last part only)

What comes from the book and what is our own choice is listed in the README block
at the bottom of this file (search for "BOOK vs OURS").
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from td_footprint import find_clusters

DATA = Path("data")
MS5, MS30, DAY = 5 * 60_000, 30 * 60_000, 86_400_000


# ------------------------------------------------------------------ data
class Cells:
    """Fast lookup of footprint cells (level, buy, sell) for one candle."""

    def __init__(self, fp):
        f = fp.sort_values(["candle", "level"])
        self.c = f["candle"].to_numpy()
        self.lv = f["level"].to_numpy()
        self.b = f["buy"].to_numpy()
        self.s = f["sell"].to_numpy()

    def get(self, t):
        a = np.searchsorted(self.c, t, "left")
        b = np.searchsorted(self.c, t, "right")
        return self.lv[a:b], self.b[a:b], self.s[a:b]


def to_30m(fp5, cd5):
    fp = fp5.copy()
    fp["candle"] = (fp["candle"] // MS30) * MS30
    fp30 = fp.groupby(["candle", "level"], as_index=False)[["buy", "sell"]].sum()
    fp30["total"] = fp30["buy"] + fp30["sell"]
    cd = cd5.copy()
    cd["c30"] = (cd["candle"] // MS30) * MS30
    cd30 = (cd.groupby("c30")
              .agg(open=("open", "first"), high=("high", "max"),
                   low=("low", "min"), close=("close", "last"))
              .reset_index().rename(columns={"c30": "candle"}))
    return fp30, cd30


def heavy_zones(fp30, cd30, mult, poc_len, pad, row):
    """Every 30m candle with an unusually heavy POC (any direction) = take-profit areas."""
    idx = fp30.groupby("candle")["total"].idxmax()
    poc = fp30.loc[idx, ["candle", "level", "total"]].rename(
        columns={"level": "poc_level", "total": "poc_vol"})
    c = cd30.merge(poc, on="candle", how="left")
    c["avg"] = c["poc_vol"].rolling(poc_len).mean().shift(1)
    h = c[c["poc_vol"] >= mult * c["avg"]]
    return (h["candle"].to_numpy() + MS30,                       # ready time
            (h["poc_level"] - pad * row).to_numpy(),             # low
            (h["poc_level"] + row + pad * row).to_numpy())       # high


def adr_by_day(cd5, n):
    d = cd5.assign(day=cd5["candle"] // DAY).groupby("day").agg(hi=("high", "max"), lo=("low", "min"))
    return (d["hi"] - d["lo"]).rolling(n).mean().shift(1).to_dict()


# --------------------------------------------------------- zone lifecycle (30m)
def zone_windows(c30, zones, depart, max_age):
    """For each zone: when it becomes valid (price left it `depart` full candles)
    and when it dies (30m close through it, or max_age). Returns list of (zone, t_arm, t_end)."""
    idx = {t: i for i, t in enumerate(c30["candle"])}
    T = c30["candle"].to_numpy()
    H, L, C = (c30[k].to_numpy() for k in ["high", "low", "close"])
    n = len(c30)
    out, never = [], 0
    for z in zones.itertuples():
        i0 = idx.get(z.candle)
        if i0 is None:
            continue
        d, lo, hi = z.dir, z.zone_low, z.zone_high
        dep, t_arm, end_i = 0, None, i0
        for j in range(i0 + 1, min(i0 + 1 + max_age, n)):
            end_i = j
            if (d == 1 and C[j] < lo) or (d == -1 and C[j] > hi):
                break
            if t_arm is None:
                away = (L[j] > hi) if d == 1 else (H[j] < lo)
                dep = dep + 1 if away else 0
                if dep >= depart:
                    t_arm = T[j] + MS30
        if t_arm is None:
            never += 1
        else:
            out.append((z, t_arm, T[end_i] + MS30))
    return out, never


# ---------------------------------------------------------- stop loss / take profit
def make_levels(P, ctx, z, d, entry, t_now):
    """Returns (stop, tp, skip_reason). Uses only information available at t_now."""
    row, lo, hi = P.row, z.zone_low, z.zone_high

    # 1) stop location (book: methods 1-3)
    if P.slmethod == "zone":              # method 2: beyond the high/low of the S/R area
        stop = lo - P.slbuf * row if d == 1 else hi + P.slbuf * row
    elif P.slmethod == "lowvol":          # method 3: in the low-volume area behind the heavy area
        lv, b, s = ctx["cells30"].get(z.candle)
        tot = b + s
        stop = lo - P.slbuf * row if d == 1 else hi + P.slbuf * row
        if len(lv):
            mean_row = tot.mean()
            if d == 1:
                for k in np.argsort(-lv):
                    if lv[k] < lo and tot[k] < mean_row:
                        stop = lv[k] - P.slbuf * row
                        break
            else:
                for k in np.argsort(lv):
                    if lv[k] >= hi and tot[k] < mean_row:
                        stop = lv[k] + row + P.slbuf * row
                        break
    else:                                 # method 1: fixed (15% of average daily range)
        adr0 = ctx["adr"].get(t_now // DAY)
        if adr0 is None or np.isnan(adr0):
            return None, None, "noadr"
        stop = entry - d * 0.15 * adr0

    # 2) book rule: stop distance should be 10-20% of the average daily range
    adr = ctx["adr"].get(t_now // DAY)
    if adr is None or np.isnan(adr):
        return None, None, "noadr"
    dist = d * (entry - stop)
    if dist <= 0:
        return None, None, "bad_stop"
    lo_b, hi_b = P.adrlo * adr, P.adrhi * adr
    if dist < lo_b or dist > hi_b:
        if P.slrange == "skip":
            return None, None, "sl_range"
        dist = min(max(dist, lo_b), hi_b)          # use a fixed SL inside the 10-20% range
        stop = entry - d * dist
    risk = dist

    # 3) take profit: just before the nearest heavy-volume area (30m footprints)
    rdy, hl, hh = ctx["heavy"]
    ok = (rdy <= t_now) & (rdy >= t_now - P.tplook * MS30)
    if d == 1:
        ok &= hl > entry
        order = np.argsort(np.where(ok, hl, np.inf))
        n_ok = ok.sum()
    else:
        ok &= hh < entry
        order = np.argsort(np.where(ok, -hh, np.inf))
        n_ok = ok.sum()
    tp = None
    for k in order[:n_ok]:
        cand = (hl[k] - P.tpbuf * row) if d == 1 else (hh[k] + P.tpbuf * row)
        reward = d * (cand - entry)
        if reward > 0 and reward / risk >= P.minrr:
            tp = cand
            break
        if P.tpclose == "skip":
            return None, None, "tp_close"          # book tip #2, option 1: skip the trade
    if tp is None:
        if P.notarget == "rr":
            tp = entry + d * P.rr * risk
        else:
            return None, None, "no_target"
    return stop, tp, None


# ------------------------------------------------------------- confirmations (5m)
def is_confirmed(mode, d, lv, b, s, lo, hi, row, avg, P):
    if mode == "none":
        return True
    if np.isnan(avg) or len(lv) == 0:
        return False
    inz = (lv + row > lo) & (lv <= hi)
    res = []
    if mode in ("limit", "any"):                    # big limit order: Bid (sell vol) for long, Ask (buy vol) for short
        side = s if d == 1 else b
        res.append(inz.any() and side[inz].max() >= P.bigmult * avg)
    if mode in ("absorb", "any"):                   # heavy volume on Bid AND Ask in the same cell
        res.append(inz.any() and bool(((b[inz] >= P.absmult * avg) & (s[inz] >= P.absmult * avg)).any()))
    if mode in ("aggr", "any"):                     # aggressive orders + delta
        delta = b.sum() - s.sum()
        if d == 1:
            res.append(delta > 0 and (b > s).mean() > 0.5)
        else:
            res.append(delta < 0 and (s > b).mean() > 0.5)
    return any(res)


# ------------------------------------------------------------------ backtest
def run(P):
    fp5 = pd.read_csv(DATA / f"{P.symbol}_5m_footprint.csv")
    cd5 = pd.read_csv(DATA / f"{P.symbol}_5m_candles.csv").sort_values("candle").reset_index(drop=True)
    fp30, cd30 = to_30m(fp5, cd5)
    cd30 = cd30.sort_values("candle").reset_index(drop=True)

    zones = find_clusters(fp30, cd30, P.row, P.mult, P.pocLen, P.pad, tf_min=30, merge=P.merge)
    wins, never = zone_windows(cd30, zones, P.depart, P.maxage)
    print(f"{len(zones)} zones, {len(wins)} became valid (price left them), {never} never did")

    ctx = {"adr": adr_by_day(cd5, P.adrdays),
           "heavy": heavy_zones(fp30, cd30, P.mult, P.pocLen, P.pad, P.row),
           "cells30": Cells(fp30)}
    cells5 = Cells(fp5) if P.confirm != "none" else None

    T5 = cd5["candle"].to_numpy()
    O5, H5, L5, C5 = (cd5[k].to_numpy() for k in ["open", "high", "low", "close"])
    n5 = len(cd5)

    g = fp5.groupby("candle").agg(vol=("total", "sum"), n=("level", "count")).reindex(T5)
    avg_side = (g["vol"].rolling(P.avgn, min_periods=P.avgn // 2).sum().shift(1)
                / (2 * g["n"].rolling(P.avgn, min_periods=P.avgn // 2).sum().shift(1))).to_numpy()

    skips, sigs = Counter(), []
    skips["never_valid"] = never
    for z, t_arm, t_end in wins:
        d, lo, hi = z.dir, z.zone_low, z.zone_high
        i_s = np.searchsorted(T5, t_arm, "left")
        i_e = np.searchsorted(T5, t_end, "left")
        entered = False
        if P.confirm == "none":                          # limit order at the near edge of the zone
            entry_nom = hi if d == 1 else lo
            stop, tp, why = make_levels(P, ctx, z, d, entry_nom, t_arm)
            if why:
                skips[why] += 1
                continue
            for k in range(i_s, i_e):
                touched = (L5[k] <= hi - P.fillthru) if d == 1 else (H5[k] >= lo + P.fillthru)
                if touched:
                    fill = min(O5[k], entry_nom) if d == 1 else max(O5[k], entry_nom)
                    sigs.append(dict(k0=k, fill=fill, entry=entry_nom, stop=stop, tp=tp, d=d,
                                     kind=z.kind, mode="limit"))
                    entered = True
                    break
        else:                                            # market entry when the confirmation shows up
            for k in range(i_s, i_e):
                if (d == 1 and C5[k] < lo) or (d == -1 and C5[k] > hi):
                    break                                # zone failed
                touching = (L5[k] <= hi and C5[k] >= lo) if d == 1 else (H5[k] >= lo and C5[k] <= hi)
                if not touching:
                    continue
                lv, b, s = cells5.get(T5[k])
                if is_confirmed(P.confirm, d, lv, b, s, lo, hi, P.row, avg_side[k], P):
                    entry = C5[k]
                    stop, tp, why = make_levels(P, ctx, z, d, entry, T5[k] + MS5)
                    if why:
                        skips[why] += 1
                        entered = True                   # counted as a skip, stop looking
                        break
                    sigs.append(dict(k0=k + 1, fill=entry, entry=entry, stop=stop, tp=tp, d=d,
                                     kind=z.kind, mode="market"))
                    entered = True
                    break
        if not entered:
            skips["no_entry"] += 1

    # one position at a time, in time order
    sigs.sort(key=lambda s: s["k0"])
    trades, free = [], -1
    for s in sigs:
        k0, d, stop, tp = s["k0"], s["d"], s["stop"], s["tp"]
        if k0 <= free or k0 >= n5:
            skips["busy"] += 1
            continue
        last = min(k0 + P.maxhold, n5 - 1)
        exit_px, reason, k_exit = C5[last], "timeout", last
        for k in range(k0, last + 1):
            hit_stop = (L5[k] <= stop) if d == 1 else (H5[k] >= stop)
            hit_tp = (H5[k] >= tp) if d == 1 else (L5[k] <= tp)
            if hit_stop:                                  # stop first if both in one candle
                exit_px = min(O5[k], stop) if d == 1 else max(O5[k], stop)
                reason, k_exit = "stop", k
                break
            if hit_tp:
                exit_px, reason, k_exit = tp, "target", k
                break
        risk = d * (s["entry"] - stop)
        market_in = s["mode"] == "market"
        e_eff = s["fill"] * (1 + d * P.slip) if market_in else s["fill"]
        x_eff = exit_px if reason == "target" else exit_px * (1 - d * P.slip)
        fee_in = P.feetaker if market_in else P.feemaker
        fee_out = P.feemaker if reason == "target" else P.feetaker
        r = (d * (x_eff - e_eff) - fee_in * e_eff - fee_out * x_eff) / risk
        trades.append(dict(entry_time=pd.to_datetime(T5[k0], unit="ms"),
                           exit_time=pd.to_datetime(T5[k_exit], unit="ms"),
                           dir=d, kind=s["kind"], mode=s["mode"], entry=s["fill"], stop=stop,
                           target=tp, exit=exit_px, reason=reason, risk_pct=risk / s["entry"] * 100,
                           hours=(k_exit - k0 + 1) * 5 / 60, R=r))
        free = k_exit
    tr = pd.DataFrame(trades)
    return tr, skips


# ------------------------------------------------------------------- report
def summarize(tr):
    if tr.empty:
        print("No trades.")
        return
    r = tr["R"]
    wins, losses = r[r > 0].sum(), r[r <= 0].sum()
    eq = r.cumsum()
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 and r.std() > 0 else float("nan")
    print(f"Trades: {len(r)}   Win rate: {(r > 0).mean():.1%}   Avg R: {r.mean():.3f}   Total R: {r.sum():.1f}")
    print(f"Profit factor: {wins / abs(losses) if losses else float('inf'):.2f}   "
          f"Max drawdown: {(eq.cummax() - eq).max():.1f} R   t-stat: {t:.2f}")
    print(f"Avg stop distance: {tr['risk_pct'].mean():.2f}% of price   Avg hold: {tr['hours'].mean():.1f} h")
    for col in ["dir", "kind", "reason"]:
        x = tr.groupby(col)["R"].agg(["count", "mean", "sum"]).round(3)
        if col == "dir":
            x = x.rename(index={1: "long", -1: "short"})
        print(x.to_string())


def plot_equity(tr, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tr["exit_time"], tr["R"].cumsum(), drawstyle="steps-post")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title("Book mode - cumulative R")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("Saved", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--row", type=float, default=10.0)
    p.add_argument("--from", dest="date_from", default=None)
    p.add_argument("--to", dest="date_to", default=None)
    # clusters (same defaults as the baseline backtest)
    p.add_argument("--mult", type=float, default=2.0)
    p.add_argument("--pocLen", type=int, default=50)
    p.add_argument("--pad", type=int, default=2)
    p.add_argument("--merge", type=int, default=6)
    p.add_argument("--depart", type=int, default=2)
    p.add_argument("--maxage", type=int, default=40)
    p.add_argument("--fillthru", type=float, default=0.0, help="USD price must trade THROUGH the level")
    # confirmation
    p.add_argument("--confirm", default="none", choices=["none", "limit", "absorb", "aggr", "any"])
    p.add_argument("--bigmult", type=float, default=4.0)
    p.add_argument("--absmult", type=float, default=3.0)
    p.add_argument("--avgn", type=int, default=288)
    # stop loss
    p.add_argument("--slmethod", default="zone", choices=["zone", "lowvol", "fixed"])
    p.add_argument("--slbuf", type=float, default=1.0, help="rows beyond the level")
    p.add_argument("--slrange", default="clamp", choices=["clamp", "skip"])
    p.add_argument("--adrdays", type=int, default=14)
    p.add_argument("--adrlo", type=float, default=0.10)
    p.add_argument("--adrhi", type=float, default=0.20)
    # take profit
    p.add_argument("--tpbuf", type=float, default=2.0, help="rows before the heavy area")
    p.add_argument("--minrr", type=float, default=1.0)
    p.add_argument("--tpclose", default="skip", choices=["skip", "next"])
    p.add_argument("--notarget", default="skip", choices=["skip", "rr"])
    p.add_argument("--rr", type=float, default=2.0)
    p.add_argument("--tplook", type=int, default=336, help="30m candles to look back for heavy areas")
    # costs
    p.add_argument("--feemaker", type=float, default=0.0002)
    p.add_argument("--feetaker", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--maxhold", type=int, default=576, help="5m candles")
    P = p.parse_args()

    tr, skips = run(P)
    if not tr.empty:
        if P.date_from:
            tr = tr[tr["entry_time"] >= pd.Timestamp(P.date_from)]
        if P.date_to:
            tr = tr[tr["entry_time"] < pd.Timestamp(P.date_to)]
    print("Skipped zones:", dict(skips))
    summarize(tr)
    if not tr.empty:
        tr.to_csv(DATA / f"{P.symbol}_book_trades.csv", index=False)
        plot_equity(tr, DATA / f"{P.symbol}_book_equity.png")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# BOOK vs OURS
# FROM THE BOOK
#  - Zones: heavy-volume cluster inside a trend or a rejection (30m, volume only)
#  - Price must leave the zone, then pull back; enter at the start of the cluster
#  - Direction: uptrend / rejection of lows = long, downtrend / rejection of highs = short
#  - Stop: beyond the high/low of the S/R area, or in the low-volume area behind it (30m)
#  - Stop distance should be 10-20% of the average daily range; else skip or use a fixed SL in range
#  - Take profit: in a heavy-volume area, a little BEFORE it (30m). If it is too close: skip the
#    trade, or aim for the next heavy area
#  - Confirmations (optional, 5m): big limit order, absorption, aggressive orders + delta
#  - Order-flow imbalance trailing exists in the book; NOT coded yet.
#  - Cumulative-delta divergence confirmation (needs 1m data); NOT coded yet.
# OUR CHOICES (the book gives no number)
#  - "heavy" = POC volume >= 2x average, "trend" = EMA50 slope, "rejection" = 60% wick
#  - zone = POC row +/- 2 rows; zones merged if same direction, overlapping, within 6 candles
#  - ADR = average of the last 14 daily ranges; stop buffer 1 row; TP buffer 2 rows
#  - Minimum reward:risk 1.0 for tip #2; look back 7 days for heavy areas
#  - "unusually large" = 4x (limit order) / 3x (absorption) the recent average cell volume
#  - fees: maker 0.02%, taker 0.05%; slippage 0.02% on market fills; max hold 2 days
# ---------------------------------------------------------------------------
