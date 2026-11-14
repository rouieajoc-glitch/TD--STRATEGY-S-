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
Alternative SL (close beyond SL, 150% catastrophic) is NOT tested: Dale does not use it intraday.

Commands:
  python td_vp.py count            rotations per day for a grid of K (counting only, no P&L)
  python td_vp.py run              backtest (K auto-picked by count unless --k given)
  python td_vp.py run --feemaker 0 --feetaker 0 --slip 0     zero-fee screen
"""
import argparse

import numpy as np
import pandas as pd
