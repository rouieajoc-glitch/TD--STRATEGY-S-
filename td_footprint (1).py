"""
TraderDale-style Volume Cluster tools for Binance USDT-M futures (free data).

Steps:
  1) python td_footprint.py build    --symbol BTCUSDT --start 2026-10-01 --end 2026-10-14
  2) python td_footprint.py clusters --symbol BTCUSDT
  3) python td_footprint.py plot     --symbol BTCUSDT --candles 40
  4) python td_footprint.py backtest --symbol BTCUSDT

Needs: pip install pandas numpy requests matplotlib
"""
import argparse
import io
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
DATA = Path("data")
COLS = ["agg_id", "price", "qty", "first_id", "last_id", "time", "is_buyer_maker"]


# ---------------------------------------------------------------- download
def download_day(symbol: str, day: date) -> Path:
    DATA.mkdir(exist_ok=True)
    name = f"{symbol}-aggTrades-{day.isoformat()}.zip"
    path = DATA / name
    if path.exists():
        return path
    url = f"{BASE}/{symbol}/{name}"
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"{url} -> HTTP {r.status_code}")
    path.write_bytes(r.content)
    return path


def load_trades(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COLS, low_memory=False)
    # Some files have a header row: drop any row where price is not numeric.
    df = df[pd.to_numeric(df["price"], errors="coerce").notna()].copy()
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    df["time"] = df["time"].astype("int64")
    if df["time"].iloc[0] > 1e14:  # microseconds -> milliseconds
        df["time"] //= 1000
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().eq("true")
    return df


# --------------------------------------------------------------- footprint
def trades_to_footprint(df: pd.DataFrame, tf_min: int, row_size: float):
    """Returns (footprint, candles).
    footprint columns: candle, level, buy, sell, total
      buy  = volume of aggressive buyers (hit the ask)  -> is_buyer_maker == False
      sell = volume of aggressive sellers (hit the bid) -> is_buyer_maker == True
    """
    tf_ms = tf_min * 60_000
    df = df.sort_values(["time", "agg_id"], kind="stable")
    df["candle"] = (df["time"] // tf_ms) * tf_ms
    df["level"] = (np.floor(df["price"] / row_size) * row_size).round(8)
    df["sell"] = np.where(df["is_buyer_maker"], df["qty"], 0.0)
    df["buy"] = np.where(df["is_buyer_maker"], 0.0, df["qty"])

    fp = df.groupby(["candle", "level"], as_index=False)[["buy", "sell"]].sum()
    fp["total"] = fp["buy"] + fp["sell"]

    g = df.groupby("candle")["price"]
    candles = pd.DataFrame(
        {"open": g.first(), "high": g.max(), "low": g.min(), "close": g.last()}
    ).reset_index()
    return fp, candles


def build(symbol, start, end, tf_min, row_size, keep_zips=False):
    days_dir = DATA / "days"
    days_dir.mkdir(parents=True, exist_ok=True)
    fps, cds = [], []
    day = start
    while day <= end:
        tag = f"{symbol}_{tf_min}m_{row_size:g}_{day}"
        fpf, cdf = days_dir / f"{tag}_fp.csv", days_dir / f"{tag}_cd.csv"
        if not (fpf.exists() and cdf.exists()):
            try:
                z = download_day(symbol, day)
                fp, cd = trades_to_footprint(load_trades(z), tf_min, row_size)
                fp.to_csv(fpf, index=False)
                cd.to_csv(cdf, index=False)
                if not keep_zips:
                    z.unlink()
                print(f"{day} ok ({len(fp)} cells)")
            except Exception as e:  # missing day, network error, ...
                print(f"{day} skipped: {e}")
                day += timedelta(days=1)
                continue
        else:
            print(f"{day} cached")
        fps.append(pd.read_csv(fpf))
        cds.append(pd.read_csv(cdf))
        day += timedelta(days=1)
    if not fps:
        raise SystemExit("No data built.")
    pd.concat(fps).to_csv(DATA / f"{symbol}_{tf_min}m_footprint.csv", index=False)
    pd.concat(cds).to_csv(DATA / f"{symbol}_{tf_min}m_candles.csv", index=False)
    print(f"Saved footprint + candles for {len(fps)} days in ./data")


# ---------------------------------------------------------------- clusters
def merge_zones(hit, row_size, tf_min, window=6):
    """Merge same-direction zones that overlap and formed within `window` candles.
    The merged zone is ready when its LAST cluster candle has closed."""
    tf_ms = tf_min * 60_000
    out = []
    for r in hit.sort_values("candle").to_dict("records"):
        r["n"] = 1
        for o in reversed(out):
            if r["candle"] - o["candle"] > window * tf_ms:
                break
            if (o["dir"] == r["dir"]
                    and r["zone_low"] <= o["zone_high"] + 2 * row_size
                    and r["zone_high"] >= o["zone_low"] - 2 * row_size):
                o["zone_low"] = min(o["zone_low"], r["zone_low"])
                o["zone_high"] = max(o["zone_high"], r["zone_high"])
                o["candle"], o["time"] = r["candle"], r["time"]
                o["ratio"] = max(o["ratio"], r["ratio"])
                o["poc_vol"] = max(o["poc_vol"], r["poc_vol"])
                o["n"] += 1
                break
        else:
            out.append(r)
    return pd.DataFrame(out)


def find_clusters(fp, candles, row_size, mult=2.0, poc_len=50, pad_rows=2,
                  ema_len=50, wick_pct=0.6, tf_min=30, merge=6):
    c = candles.sort_values("candle").reset_index(drop=True)
    poc_idx = fp.groupby("candle")["total"].idxmax()
    poc = fp.loc[poc_idx, ["candle", "level", "total"]].rename(
        columns={"level": "poc_level", "total": "poc_vol"}
    )
    c = c.merge(poc, on="candle", how="left")
    c["poc_avg"] = c["poc_vol"].rolling(poc_len).mean().shift(1)
    c["ratio"] = c["poc_vol"] / c["poc_avg"]

    c["ema"] = c["close"].ewm(span=ema_len, adjust=False).mean()
    up = (c["close"] > c["ema"]) & (c["ema"] > c["ema"].shift(5))
    down = (c["close"] < c["ema"]) & (c["ema"] < c["ema"].shift(5))
    rng = (c["high"] - c["low"]).clip(lower=row_size)
    body_lo = c[["open", "close"]].min(axis=1)
    body_hi = c[["open", "close"]].max(axis=1)
    lower_wick = (body_lo - c["low"]) / rng
    upper_wick = (c["high"] - body_hi) / rng

    rej_low = (lower_wick >= wick_pct) & (c["poc_level"] + row_size <= body_lo)
    rej_high = (upper_wick >= wick_pct) & (c["poc_level"] >= body_hi)

    c["kind"] = np.select(
        [rej_low, rej_high, up, down], ["rejection", "rejection", "trend", "trend"], ""
    )
    c["dir"] = np.select([rej_low, rej_high, up, down], [1, -1, 1, -1], 0)

    hit = c[(c["ratio"] >= mult) & (c["dir"] != 0)].copy()
    hit["zone_low"] = hit["poc_level"] - pad_rows * row_size
    hit["zone_high"] = hit["poc_level"] + row_size + pad_rows * row_size
    hit["time"] = pd.to_datetime(hit["candle"], unit="ms")
    hit = hit[["time", "candle", "dir", "kind", "poc_level", "poc_vol", "ratio",
               "zone_low", "zone_high"]]
    if merge > 0 and len(hit):
        hit = merge_zones(hit, row_size, tf_min, merge)
    else:
        hit["n"] = 1
    return hit


# -------------------------------------------------------------------- plot
def plot_window(fp, candles, clusters, row_size, n, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    c = candles.sort_values("candle").tail(n).reset_index(drop=True)
    sub = fp[fp["candle"].isin(c["candle"])]
    vmax = sub["total"].max()
    pos = {t: i for i, t in enumerate(c["candle"])}

    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), 8))
    for r in sub.itertuples():
        x = pos[r.candle]
        shade = 1 - 0.9 * (r.total / vmax)  # dark = heavy volume
        ax.add_patch(Rectangle((x - 0.4, r.level), 0.8, row_size,
                               facecolor=(shade, shade, shade), edgecolor="none"))
    for i, r in c.iterrows():
        ax.plot([i, i], [r.low, r.high], color="black", lw=0.6)

    for z in clusters[clusters["candle"].isin(c["candle"])].itertuples():
        x = pos[z.candle]
        col = "green" if z.dir == 1 else "red"
        ax.add_patch(Rectangle((x - 0.45, z.zone_low), n - x, z.zone_high - z.zone_low,
                               facecolor=col, alpha=0.3, edgecolor=col, lw=1))

    ax.set_xlim(-1, n)
    ax.set_ylim(c["low"].min() - row_size * 3, c["high"].max() + row_size * 3)
    ax.set_xticks(range(0, n, max(1, n // 12)))
    ax.set_xticklabels(
        [pd.to_datetime(t, unit="ms").strftime("%m-%d %H:%M")
         for t in c["candle"].iloc[:: max(1, n // 12)]],
        rotation=45, ha="right", fontsize=7)
    ax.set_title("Volume footprint (dark = heavy) with detected clusters "
                 "(green = long zone, red = short zone)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("Saved", out)


# ---------------------------------------------------------------- backtest
def backtest(candles, zones, depart=2, max_age=40, rr=2.0, stop_atr=0.5,
             fee=0.0005, slip=0.0002, max_hold=48, atr_len=14):
    """Candle-based backtest of the cluster pullback setup.
    - A zone is valid after `depart` consecutive candles fully leave it.
    - Entry: limit at the near edge when price returns (fills at open if gapped).
    - Stop: beyond the far edge by stop_atr*ATR. Target: rr * risk.
    - If stop and target are both inside one candle, the stop is assumed first.
    - fee is charged per side, slip applies to entry, stop and timeout fills.
    """
    c = candles.sort_values("candle").reset_index(drop=True)
    prev = c["close"].shift()
    tr = pd.concat([c["high"] - c["low"], (c["high"] - prev).abs(),
                    (c["low"] - prev).abs()], axis=1).max(axis=1)
    c["atr"] = tr.rolling(atr_len).mean()
    idx = {t: i for i, t in enumerate(c["candle"])}
    O, H, L, C, A = (c[k].to_numpy() for k in ["open", "high", "low", "close", "atr"])
    n = len(c)

    sigs = []
    for z in zones.itertuples():
        i0 = idx.get(z.candle)
        if i0 is None or np.isnan(A[i0]):
            continue
        d, lo, hi = z.dir, z.zone_low, z.zone_high
        dep, armed = 0, False
        for j in range(i0 + 1, min(i0 + 1 + max_age, n)):
            if d == 1:
                if C[j] < lo:
                    break
                if not armed:
                    dep = dep + 1 if L[j] > hi else 0
                    armed = dep >= depart
                elif L[j] <= hi:
                    sigs.append((j, min(O[j], hi), z, A[i0]))
                    break
            else:
                if C[j] > hi:
                    break
                if not armed:
                    dep = dep + 1 if H[j] < lo else 0
                    armed = dep >= depart
                elif H[j] >= lo:
                    sigs.append((j, max(O[j], lo), z, A[i0]))
                    break

    sigs.sort(key=lambda s: s[0])
    trades, free_at = [], -1
    for j, entry, z, atr in sigs:
        if j <= free_at:
            continue
        d = z.dir
        stop = z.zone_low - stop_atr * atr if d == 1 else z.zone_high + stop_atr * atr
        risk = (entry - stop) if d == 1 else (stop - entry)
        if risk <= 0:
            continue
        tgt = entry + d * rr * risk
        last = min(j + max_hold, n - 1)
        exit_px, reason, k_exit = C[last], "timeout", last
        for k in range(j, last + 1):
            if d == 1:
                hit_stop, hit_tgt = L[k] <= stop, H[k] >= tgt
                gap_stop = min(O[k], stop) if k > j else stop
            else:
                hit_stop, hit_tgt = H[k] >= stop, L[k] <= tgt
                gap_stop = max(O[k], stop) if k > j else stop
            if hit_stop:
                exit_px, reason, k_exit = gap_stop, "stop", k
                break
            if hit_tgt:
                exit_px, reason, k_exit = tgt, "target", k
                break
        e_eff = entry * (1 + d * slip)
        x_eff = exit_px if reason == "target" else exit_px * (1 - d * slip)
        r_mult = d * (x_eff - e_eff) / risk - 2 * fee * entry / risk
        trades.append({
            "entry_time": pd.to_datetime(c["candle"][j], unit="ms"),
            "exit_time": pd.to_datetime(c["candle"][k_exit], unit="ms"),
            "dir": d, "kind": z.kind, "entry": entry, "stop": stop, "target": tgt,
            "exit": exit_px, "reason": reason, "R": r_mult,
        })
        free_at = k_exit
    return pd.DataFrame(trades)


def summarize(tr):
    if tr.empty:
        print("No trades.")
        return
    wins, losses = tr[tr["R"] > 0]["R"].sum(), tr[tr["R"] <= 0]["R"].sum()
    eq = tr["R"].cumsum()
    print(f"Trades: {len(tr)}   Win rate: {(tr['R'] > 0).mean():.1%}")
    print(f"Avg R: {tr['R'].mean():.3f}   Total R: {tr['R'].sum():.1f}   "
          f"Profit factor: {wins / abs(losses) if losses else float('inf'):.2f}")
    print(f"Max drawdown: {(eq.cummax() - eq).max():.1f} R")
    print(tr.groupby("dir")["R"].agg(["count", "mean", "sum"]).rename(
        index={1: "long", -1: "short"}).round(3).to_string())
    print(tr.groupby("kind")["R"].agg(["count", "mean", "sum"]).round(3).to_string())
    print(tr["reason"].value_counts().to_string())


def plot_equity(tr, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tr["exit_time"], tr["R"].cumsum(), drawstyle="steps-post")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title("Cumulative R (after fees and slippage)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("Saved", out)


# -------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "clusters", "plot", "backtest"])
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default=(date.today() - timedelta(days=14)).isoformat())
    p.add_argument("--end", default=(date.today() - timedelta(days=2)).isoformat())
    p.add_argument("--tf", type=int, default=30, help="candle minutes")
    p.add_argument("--row", type=float, default=10.0, help="price row size in USDT")
    p.add_argument("--mult", type=float, default=2.0)
    p.add_argument("--pocLen", type=int, default=50)
    p.add_argument("--pad", type=int, default=2)
    p.add_argument("--candles", type=int, default=40)
    p.add_argument("--merge", type=int, default=6, help="merge window in candles (0=off)")
    p.add_argument("--keepzips", action="store_true", help="do not delete downloaded zips")
    p.add_argument("--depart", type=int, default=2)
    p.add_argument("--maxage", type=int, default=40)
    p.add_argument("--rr", type=float, default=2.0)
    p.add_argument("--stopatr", type=float, default=0.5)
    p.add_argument("--fee", type=float, default=0.0005, help="per side, 0.0005 = 0.05%%")
    p.add_argument("--slip", type=float, default=0.0002)
    a = p.parse_args()

    fpf = DATA / f"{a.symbol}_{a.tf}m_footprint.csv"
    cdf = DATA / f"{a.symbol}_{a.tf}m_candles.csv"

    if a.cmd == "build":
        build(a.symbol, date.fromisoformat(a.start), date.fromisoformat(a.end), a.tf, a.row, a.keepzips)
        return
    fp, cd = pd.read_csv(fpf), pd.read_csv(cdf)
    cl = find_clusters(fp, cd, a.row, a.mult, a.pocLen, a.pad,
                       tf_min=a.tf, merge=a.merge)
    if a.cmd == "clusters":
        cl.to_csv(DATA / f"{a.symbol}_{a.tf}m_clusters.csv", index=False)
        print(cl[["time", "dir", "kind", "poc_level", "ratio", "n"]].tail(20).to_string(index=False))
        print(f"{len(cl)} clusters found in {len(cd)} candles")
    elif a.cmd == "backtest":
        tr = backtest(cd, cl, a.depart, a.maxage, a.rr, a.stopatr, a.fee, a.slip)
        tr.to_csv(DATA / f"{a.symbol}_{a.tf}m_trades.csv", index=False)
        summarize(tr)
        if not tr.empty:
            plot_equity(tr, DATA / f"{a.symbol}_equity.png")
    else:
        plot_window(fp, cd, cl, a.row, a.candles, DATA / f"{a.symbol}_clusters.png")


if __name__ == "__main__":
    main()
