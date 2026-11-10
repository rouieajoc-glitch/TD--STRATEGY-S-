"""
TraderDale-style Volume Cluster tools for Binance USDT-M futures (free data).

Steps:
  1) python td_footprint.py build    --symbol BTCUSDT --start 2026-10-01 --end 2026-10-14
  2) python td_footprint.py clusters --symbol BTCUSDT
  3) python td_footprint.py plot     --symbol BTCUSDT --candles 40

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


def build(symbol, start, end, tf_min, row_size):
    fps, cds = [], []
    day = start
    while day <= end:
        try:
            z = download_day(symbol, day)
            fp, cd = trades_to_footprint(load_trades(z), tf_min, row_size)
            fps.append(fp)
            cds.append(cd)
            print(f"{day} ok ({len(fp)} cells)")
        except Exception as e:  # missing day, network error, ...
            print(f"{day} skipped: {e}")
        day += timedelta(days=1)
    if not fps:
        raise SystemExit("No data built.")
    pd.concat(fps).to_csv(DATA / f"{symbol}_{tf_min}m_footprint.csv", index=False)
    pd.concat(cds).to_csv(DATA / f"{symbol}_{tf_min}m_candles.csv", index=False)
    print("Saved footprint + candles in ./data")


# ---------------------------------------------------------------- clusters
def find_clusters(fp, candles, row_size, mult=2.0, poc_len=50, pad_rows=2,
                  ema_len=50, wick_pct=0.6):
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
    return hit[["time", "candle", "dir", "kind", "poc_level", "poc_vol", "ratio",
                "zone_low", "zone_high"]]


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
                               facecolor=col, alpha=0.12, edgecolor=col, lw=1))

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


# -------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "clusters", "plot"])
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default=(date.today() - timedelta(days=14)).isoformat())
    p.add_argument("--end", default=(date.today() - timedelta(days=2)).isoformat())
    p.add_argument("--tf", type=int, default=30, help="candle minutes")
    p.add_argument("--row", type=float, default=10.0, help="price row size in USDT")
    p.add_argument("--mult", type=float, default=2.0)
    p.add_argument("--pocLen", type=int, default=50)
    p.add_argument("--pad", type=int, default=2)
    p.add_argument("--candles", type=int, default=40)
    a = p.parse_args()

    fpf = DATA / f"{a.symbol}_{a.tf}m_footprint.csv"
    cdf = DATA / f"{a.symbol}_{a.tf}m_candles.csv"

    if a.cmd == "build":
        build(a.symbol, date.fromisoformat(a.start), date.fromisoformat(a.end), a.tf, a.row)
        return
    fp, cd = pd.read_csv(fpf), pd.read_csv(cdf)
    cl = find_clusters(fp, cd, a.row, a.mult, a.pocLen, a.pad)
    if a.cmd == "clusters":
        cl.to_csv(DATA / f"{a.symbol}_{a.tf}m_clusters.csv", index=False)
        print(cl.tail(20).to_string(index=False))
        print(f"{len(cl)} clusters found in {len(cd)} candles")
    else:
        plot_window(fp, cd, cl, a.row, a.candles, DATA / f"{a.symbol}_clusters.png")


if __name__ == "__main__":
    main()
