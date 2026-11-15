"""
CRT (Candle Range Theory) backtest, BTCUSDT 5m candles only (OHLC, no footprint needed).

Rules fixed BEFORE looking at results (source: the CRT write-up + "Model 1 entry" post):
  Candle 1 = a closed HTF candle -> CRH (high), CRL (low).
  Candle 2 = the next HTF candle. It must take ONE side of candle 1 (sweep) and come back inside.
  Candle 3 = expansion toward the unswept side. We hold at most until the end of candle 3.
  Pairs (--pair): 4h/15m (primary), 1h/5m, 1d/1h.  Bins are UTC-aligned (--hoffset shifts them).

  --entry mss   (primary, "Model 1"): inside candle 2, after the first sweep, enter at the close of the
                first LTF candle that closes back through the reference swing level while still inside
                candle 1's range (bullish: close above the last confirmed LTF swing high, and <= CRH;
                bearish mirror). Swing = 2 candles each side, known 2 candles later (no look-ahead).
                Stop = extreme made since the sweep started. Candle 2's own close is NOT used (unknown then).
                If the other side of candle 1 gets taken before the entry, no trade.
  --entry close (simple): enter at the close of candle 2 if it swept exactly one side and closed back
                inside. Stop = candle 2's extreme.
  --target opposite (primary) = far side of candle 1;  --target mid = 50% of candle 1.
  --minrr 1.0: skip if reward < 1R (the write-up says deep sweeps with bad RR are avoided).
  Market entry (taker + slippage), target = limit (maker), stop/timeout = taker + slippage.
  Stop first if stop and target are both inside one 5m candle. One position at a time.

Pass criteria (fixed in advance, stricter than the first nine tests because this is test #10+):
  >= 150 trades, avg R > 0 after costs, t >= 3.0, positive in BOTH years (split 2025-11-01).
Zero-fee screen first. Gross edge below about +0.2R cannot survive costs.

Usage:
  python td_crt.py run --feemaker 0 --feetaker 0 --slip 0     # zero-fee screen
  python td_crt.py run                                          # real costs
  python td_crt.py run --entry close | --target mid | --pair 1h/5m | --pair 1d/1h
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def resample(cd5, ms, off):
    c = cd5.copy()
    c["b"] = ((c["candle"] - off) // ms) * ms + off
    g = (c.groupby("b").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                            close=("close", "last"), n=("open", "size")).reset_index()
          .rename(columns={"b": "candle"}))
    full = ms // MS["5m"]
    return g[g["n"] == full].drop(columns="n").reset_index(drop=True)


def swing_levels(H, L):
    """Last confirmed 2/2 fractal high and low as of each candle (confirmed 2 candles after the extreme)."""
    n = len(H)
    sh, sl = np.full(n, np.nan), np.full(n, np.nan)
    for j in range(2, n - 2):
        if H[j] > max(H[j - 2], H[j - 1], H[j + 1], H[j + 2]):
            sh[j + 2] = H[j]
        if L[j] < min(L[j - 2], L[j - 1], L[j + 1], L[j + 2]):
            sl[j + 2] = L[j]
    return pd.Series(sh).ffill().to_numpy(), pd.Series(sl).ffill().to_numpy()


def simulate(d, fill, stop, target, k0, t_end_idx, O5, H5, L5, C5, P):
    """Walk 5m candles from k0 to t_end_idx. Returns (R, reason, k_exit, exit_px)."""
    sd = d * (fill - stop)
    exit_px, reason, k_x = C5[t_end_idx], "timeout", t_end_idx
    for k in range(k0, t_end_idx + 1):
        if (L5[k] <= stop) if d == 1 else (H5[k] >= stop):
            exit_px = min(O5[k], stop) if d == 1 else max(O5[k], stop)
            reason, k_x = "stop", k
            break
        if (H5[k] >= target) if d == 1 else (L5[k] <= target):
            exit_px, reason, k_x = target, "target", k
            break
    x_eff = exit_px if reason == "target" else exit_px * (1 - d * P.slip)
    fee_out = P.feemaker if reason == "target" else P.feetaker
    r = (d * (x_eff - fill) - P.feetaker * fill - fee_out * x_eff) / sd
    return r, reason, k_x, exit_px


def run(P):
    cd5 = pd.read_csv(DATA / f"{P.symbol}_5m_candles.csv").sort_values("candle").reset_index(drop=True)
    htf_s, ltf_s = P.pair.split("/")
    hms, lms, off = MS[htf_s], MS[ltf_s], int(P.hoffset * 3_600_000)
    htf = resample(cd5, hms, off)
    ltf = resample(cd5, lms, 0) if lms > MS["5m"] else cd5[["candle", "open", "high", "low", "close"]]
    T5 = cd5["candle"].to_numpy()
    O5, H5, L5, C5 = (cd5[c].to_numpy() for c in ["open", "high", "low", "close"])
    TL = ltf["candle"].to_numpy()
    HL, LL, CL = (ltf[c].to_numpy() for c in ["high", "low", "close"])
    SH, SL = swing_levels(HL, LL)
    TH = htf["candle"].to_numpy()
    skips = dict(not_consecutive=0, no_sweep=0, both_sides=0, no_trigger=0, rr_below_min=0, busy=0, no_swing=0)
    out, free_at, cands = [], -1, 0
    for i in range(len(htf) - 2):
        t2 = TH[i + 1]
        if TH[i + 1] - TH[i] != hms or TH[i + 2] - TH[i + 1] != hms:
            skips["not_consecutive"] += 1
            continue
        crh, crl = htf["high"].iloc[i], htf["low"].iloc[i]
        h2, l2, c2 = htf["high"].iloc[i + 1], htf["low"].iloc[i + 1], htf["close"].iloc[i + 1]
        t_end = t2 + 2 * hms                                   # end of candle 3
        t_end_idx = int(np.searchsorted(T5, t_end, "left")) - 1
        up, dn = h2 > crh, l2 < crl
        if not (up or dn):
            skips["no_sweep"] += 1
            continue
        if up and dn:
            skips["both_sides"] += 1
            continue
        d = 1 if dn else -1                                    # low swept -> long; high swept -> short
        if P.entry == "close":
            if not (crl <= c2 <= crh):
                skips["no_sweep"] += 1                         # broke out and stayed out = not CRT
                continue
            cands += 1
            t_in = t2 + hms
            k0 = int(np.searchsorted(T5, t_in, "left"))
            if k0 > t_end_idx:
                continue
            raw = C5[k0 - 1]
            stop = l2 if d == 1 else h2
        else:
            a = int(np.searchsorted(TL, t2, "left"))
            b = int(np.searchsorted(TL, t2 + hms, "left"))
            s = None
            for m in range(a, b):
                if (LL[m] < crl) if d == 1 else (HL[m] > crh):
                    s = m
                    break
            if s is None:
                continue
            cands += 1
            ref = SH[s] if d == 1 else SL[s]
            if np.isnan(ref):
                skips["no_swing"] += 1
                continue
            ext, trig = (LL[s] if d == 1 else HL[s]), None
            for m in range(s, b):
                ext = min(ext, LL[m]) if d == 1 else max(ext, HL[m])
                if (HL[m] > crh) if d == 1 else (LL[m] < crl):
                    break                                       # other side taken first: invalid
                inside = (CL[m] <= crh) if d == 1 else (CL[m] >= crl)
                if ((CL[m] > ref) if d == 1 else (CL[m] < ref)) and inside:
                    trig = m
                    break
            if trig is None:
                skips["no_trigger"] += 1
                continue
            t_in = TL[trig] + lms
            k0 = int(np.searchsorted(T5, t_in, "left"))
            if k0 > t_end_idx:
                continue
            raw, stop = CL[trig], ext
        fill = raw * (1 + d * P.slip)
        target = (crh if d == 1 else crl) if P.target == "opposite" else (crh + crl) / 2
        sd = d * (fill - stop)
        if sd <= 0 or d * (target - fill) <= 0 or d * (target - fill) / sd < P.minrr:
            skips["rr_below_min"] += 1
            continue
        if T5[k0] <= free_at:
            skips["busy"] += 1
            continue
        r, reason, k_x, xp = simulate(d, fill, stop, target, k0, t_end_idx, O5, H5, L5, C5, P)
        free_at = T5[k_x] + MS["5m"] - 1
        out.append(dict(entry_time=pd.to_datetime(T5[k0], unit="ms"), exit_time=pd.to_datetime(T5[k_x], unit="ms"),
                        dir=d, entry=fill, stop=stop, target=target, exit=xp, reason=reason,
                        rr_plan=d * (target - fill) / sd, risk_pct=sd / fill * 100, hours=(k_x - k0 + 1) * 5 / 60, R=r))
    return pd.DataFrame(out), skips, cands, len(htf)


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def summarize(tr):
    R = tr["R"].to_numpy()
    eq = np.cumsum(R)
    dd = (np.maximum.accumulate(eq) - eq).max()
    pf = R[R > 0].sum() / abs(R[R < 0].sum()) if (R < 0).any() else float("inf")
    print(f"Trades: {len(R)}   Win rate: {(R > 0).mean() * 100:.1f}%   Avg R: {R.mean():.3f}   Total R: {R.sum():.1f}")
    print(f"Profit factor: {pf:.2f}   Max drawdown: {dd:.1f} R   t-stat: {tstat(R):.2f}")
    print(f"Avg stop distance: {tr['risk_pct'].mean():.2f}% of price   Avg planned RR: {tr['rr_plan'].mean():.2f}   "
          f"Avg hold: {tr['hours'].mean():.1f} h")
    print(tr.groupby(tr["dir"].map({1: "long", -1: "short"}))["R"].agg(["count", "mean", "sum"]).round(3))
    print(tr.groupby("reason")["R"].agg(["count", "mean", "sum"]).round(3))


def verdict(tr, split):
    R = tr["R"].to_numpy()
    y1 = tr[tr["entry_time"] < pd.Timestamp(split)]["R"]
    y2 = tr[tr["entry_time"] >= pd.Timestamp(split)]["R"]
    print(f"\nYear 1 (before {split}): {len(y1)} trades, avg R {y1.mean():.3f}, t {tstat(y1):.2f}")
    print(f"Year 2 (from {split}):   {len(y2)} trades, avg R {y2.mean():.3f}, t {tstat(y2):.2f}")
    ok = [("at least 150 trades", len(R) >= 150), ("positive average R after costs", R.mean() > 0),
          ("t-stat of at least 3.0", tstat(R) >= 3.0),
          ("positive average R in BOTH years", len(y1) > 0 and len(y2) > 0 and y1.mean() > 0 and y2.mean() > 0)]
    print("\nPass criteria (fixed in advance):")
    for name, v in ok:
        print(f"  {'PASS' if v else 'FAIL'}  {name}")
    print("RESULT:", "EDGE CANDIDATE (needs out-of-sample confirmation)" if all(v for _, v in ok) else "no edge shown")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--pair", default="4h/15m", choices=["4h/15m", "1h/5m", "1d/1h"])
    ap.add_argument("--entry", default="mss", choices=["mss", "close"])
    ap.add_argument("--target", default="opposite", choices=["opposite", "mid"])
    ap.add_argument("--minrr", type=float, default=1.0)
    ap.add_argument("--hoffset", type=float, default=0.0, help="shift HTF bins by hours (UTC)")
    ap.add_argument("--split", default="2025-11-01")
    ap.add_argument("--feemaker", type=float, default=0.0002)
    ap.add_argument("--feetaker", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--out", default=None)
    P = ap.parse_args()
    tr, skips, cands, nh = run(P)
    print(f"CRT {P.pair}, entry {P.entry}, target {P.target}, minrr {P.minrr}; {nh} HTF candles, {cands} CRT candidates")
    print("Skipped:", skips)
    if tr.empty:
        print("No trades.")
        return
    summarize(tr)
    verdict(tr, P.split)
    name = P.out or f"crt_{P.pair.replace('/', '_')}_{P.entry}_{P.target}"
    tr.to_csv(DATA / f"{P.symbol}_{name}_trades.csv", index=False)


if __name__ == "__main__":
    main()
