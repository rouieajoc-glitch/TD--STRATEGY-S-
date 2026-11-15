"""
CRT "video version" (Smart Risk, Candle Range Theory trading model): trend-following CRT with a
lower-timeframe fair value gap (FVG) entry. BTCUSDT 5m candles only. Needs td_crt.py in the same folder.

Rules fixed BEFORE looking at results (the video leaves some things discretionary; these are my choices):
  Trend (1H, --ema 50, --slope 10): uptrend = EMA50 of the range candle is above its value 10 candles
        earlier AND the range candle closes above the EMA. Downtrend = the mirror. Trade only with the trend.
  Range candle = a pullback candle against the trend: BEARISH 1H candle in an uptrend (long setup),
        BULLISH 1H candle in a downtrend (short setup). Its high/low = CRH/CRL.
  Sweep candle = the very next 1H candle. Long: it trades below CRL, does not exceed CRH, and CLOSES back
        inside the range. Short: mirror. If it closes outside the range the setup is skipped (video).
  Entry: after the sweep candle has CLOSED, look at the next 1H candle's worth of 5m candles for the first
        FVG in the trade direction (bullish: low of candle k > high of candle k-2; bearish: mirror), all three
        candles formed after the sweep closed. Place a limit order at the edge of the gap nearest to price
        (top of a bullish gap / bottom of a bearish gap). The order lives --valid 12 five-minute candles
        and is cancelled if price reaches +1R before filling.
  Stop (--stop gap, primary): beyond the far edge of the gap (video: "below the gap").
        --stop sweep (secondary, declared up front): beyond the sweep candle's extreme (video: "or the range").
  Exit: half the position at +1R; the stop of the rest moves to break-even; the rest targets the far side of
        the range candle (my stand-in for "next strong level"). Anything still open is closed after --hold 12
        hours. Skip if the far side is closer than 1R. Stop first if stop and target share one 5m candle.
  Costs: entry limit = maker, 1R and far-side targets = maker, stop / break-even / timeout = taker + slippage.
  One position at a time.
  Fill candle: its high may have come BEFORE the touch, so a target hit in the fill candle itself is not counted
  (a stop hit is: price must pass through the entry on its way to a low below the stop). This is slightly
  pessimistic for the gap-stop version, whose stop can be smaller than one 5m candle.

Pass criteria (fixed in advance, as for the other CRT tests):
  >= 150 trades, avg R > 0 after costs, t >= 3.0, positive in BOTH years (split 2025-11-01).
Zero-fee screen first: a gross edge below about +0.2R cannot survive costs.

Usage:
  python td_crt_video.py run --feemaker 0 --feetaker 0 --slip 0
  python td_crt_video.py run
  python td_crt_video.py run --stop sweep --feemaker 0 --feetaker 0 --slip 0
"""
import argparse

import numpy as np
import pandas as pd

import td_crt as tc

MS, DATA = tc.MS, tc.DATA


def simulate(d, fill, stop, t1, t2, k0, k_end, O5, H5, L5, C5, P):
    """Half at +1R (t1), rest to t2 with break-even stop. Returns (R, reason, k_exit, exit_px)."""
    sd = d * (fill - stop)
    half = 0.5
    r1 = None
    cur_stop, k_x, reason, exit_px = stop, k_end, "timeout", C5[k_end]
    phase2_from = k0
    for k in range(k0, k_end + 1):
        if r1 is None:
            hit_stop = (L5[k] <= stop) if d == 1 else (H5[k] >= stop)
            if hit_stop:
                px = min(O5[k], stop) if d == 1 else max(O5[k], stop)
                x = px * (1 - d * P.slip)
                r = (d * (x - fill) - P.feemaker * fill - P.feetaker * x) / sd
                return r, "stop", k, px
            if k > k0 and ((H5[k] >= t1) if d == 1 else (L5[k] <= t1)):
                r1 = (d * (t1 - fill) - P.feemaker * fill - P.feemaker * t1) / sd
                phase2_from = k + 1
                break
    if r1 is None:
        x = C5[k_end] * (1 - d * P.slip)
        r = (d * (x - fill) - P.feemaker * fill - P.feetaker * x) / sd
        return r, "timeout", k_end, C5[k_end]
    cur_stop = fill                                           # break-even for the rest
    for k in range(phase2_from, k_end + 1):
        if (L5[k] <= cur_stop) if d == 1 else (H5[k] >= cur_stop):
            px = min(O5[k], cur_stop) if d == 1 else max(O5[k], cur_stop)
            x = px * (1 - d * P.slip)
            r2 = (d * (x - fill) - P.feemaker * fill - P.feetaker * x) / sd
            return half * r1 + half * r2, "be", k, px
        if (H5[k] >= t2) if d == 1 else (L5[k] <= t2):
            r2 = (d * (t2 - fill) - P.feemaker * fill - P.feemaker * t2) / sd
            return half * r1 + half * r2, "target", k, t2
    x = C5[k_end] * (1 - d * P.slip)
    r2 = (d * (x - fill) - P.feemaker * fill - P.feetaker * x) / sd
    return half * r1 + half * r2, "timeout", k_end, C5[k_end]


def run(P):
    cd5 = pd.read_csv(DATA / f"{P.symbol}_5m_candles.csv").sort_values("candle").reset_index(drop=True)
    hms = MS[P.htf]
    htf = tc.resample(cd5, hms, 0)
    TH = htf["candle"].to_numpy()
    OH, HH, LH, CH = (htf[c].to_numpy() for c in ["open", "high", "low", "close"])
    ema = htf["close"].ewm(span=P.ema, adjust=False).mean().to_numpy()
    T5 = cd5["candle"].to_numpy()
    n5 = len(T5)
    O5, H5, L5, C5 = (cd5[c].to_numpy() for c in ["open", "high", "low", "close"])
    per = hms // MS["5m"]
    skips = dict(gaps=0, no_trend=0, not_pullback=0, no_sweep=0, no_fvg=0, no_fill=0, rr_below_min=0, busy=0)
    out, free_at, cands = [], -1, 0
    warm = max(P.ema, P.slope) + 5
    for i in range(warm, len(htf) - 2):
        if TH[i + 1] - TH[i] != hms:
            skips["gaps"] += 1
            continue
        up = ema[i] > ema[i - P.slope] and CH[i] > ema[i]
        dn = ema[i] < ema[i - P.slope] and CH[i] < ema[i]
        if up and CH[i] < OH[i]:
            d = 1
        elif dn and CH[i] > OH[i]:
            d = -1
        else:
            skips["no_trend" if not (up or dn) else "not_pullback"] += 1
            continue
        crh, crl = HH[i], LH[i]
        h2, l2, c2 = HH[i + 1], LH[i + 1], CH[i + 1]
        if d == 1:
            ok = l2 < crl and h2 <= crh and crl < c2 <= crh
        else:
            ok = h2 > crh and l2 >= crl and crl <= c2 < crh
        if not ok:
            skips["no_sweep"] += 1
            continue
        cands += 1
        t2 = TH[i + 1]
        k_s = int(np.searchsorted(T5, t2 + hms, "left"))          # first 5m candle after the sweep closed
        kf = None
        for k in range(k_s + 2, min(k_s + per, n5)):
            if (L5[k] > H5[k - 2]) if d == 1 else (H5[k] < L5[k - 2]):
                kf = k
                break
        if kf is None:
            skips["no_fvg"] += 1
            continue
        if d == 1:
            entry, gap_far = L5[kf], H5[kf - 2]
        else:
            entry, gap_far = H5[kf], L5[kf - 2]
        stop = gap_far if P.stop == "gap" else (l2 if d == 1 else h2)
        sd0 = d * (entry - stop)
        t2px = crh if d == 1 else crl
        if sd0 <= 0 or d * (t2px - entry) / sd0 < P.minrr:
            skips["rr_below_min"] += 1
            continue
        t1px = entry + d * sd0
        k_end = min(int(np.searchsorted(T5, t2 + (1 + P.hold) * hms, "left")) - 1, n5 - 1)
        k0 = None
        for k in range(kf + 1, min(kf + 1 + P.valid, n5)):
            touch = (L5[k] <= entry) if d == 1 else (H5[k] >= entry)
            ran = (H5[k] >= t1px) if d == 1 else (L5[k] <= t1px)
            if touch:
                k0 = k
                break
            if ran:
                break
        if k0 is None or k0 > k_end:
            skips["no_fill"] += 1
            continue
        fill = min(O5[k0], entry) if d == 1 else max(O5[k0], entry)
        sd = d * (fill - stop)
        if sd <= 0:
            skips["no_fill"] += 1
            continue
        if T5[k0] <= free_at:
            skips["busy"] += 1
            continue
        t1x = fill + d * sd
        r, reason, k_x, xp = simulate(d, fill, stop, t1x, t2px, k0, k_end, O5, H5, L5, C5, P)
        free_at = T5[k_x] + MS["5m"] - 1
        out.append(dict(entry_time=pd.to_datetime(T5[k0], unit="ms"), exit_time=pd.to_datetime(T5[k_x], unit="ms"),
                        dir=d, entry=fill, stop=stop, target=t2px, exit=xp, reason=reason,
                        rr_plan=d * (t2px - fill) / sd, risk_pct=sd / fill * 100, hours=(k_x - k0 + 1) * 5 / 60, R=r))
    return pd.DataFrame(out), skips, cands, len(htf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--htf", default="1h", choices=["1h", "4h"])
    ap.add_argument("--stop", default="gap", choices=["gap", "sweep"])
    ap.add_argument("--ema", type=int, default=50)
    ap.add_argument("--slope", type=int, default=10)
    ap.add_argument("--valid", type=int, default=12, help="5m candles the FVG limit order stays live")
    ap.add_argument("--hold", type=int, default=12, help="HTF candles to hold after the sweep candle")
    ap.add_argument("--minrr", type=float, default=1.0)
    ap.add_argument("--split", default="2025-11-01")
    ap.add_argument("--feemaker", type=float, default=0.0002)
    ap.add_argument("--feetaker", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    P = ap.parse_args()
    tr, skips, cands, nh = run(P)
    print(f"CRT video version, {P.htf} range candles, stop {P.stop}; {nh} HTF candles, {cands} valid sweeps with trend")
    print("Skipped:", skips)
    if tr.empty:
        print("No trades.")
        return
    tc.summarize(tr)
    tc.verdict(tr, P.split)
    tr.to_csv(DATA / f"{P.symbol}_crtvideo_{P.htf}_{P.stop}_trades.csv", index=False)


if __name__ == "__main__":
    main()
