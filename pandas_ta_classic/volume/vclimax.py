# Volume Climax (VCLIMAX)
from typing import Any

import numpy as np
from pandas import DataFrame, Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_float, _pos_int, degenerate_div, nan_on_short_input


@nan_on_short_input
def vclimax(
    high: Series,
    low: Series,
    close: Series,
    volume: Series,
    length: int | None = None,
    lookback: int | None = None,
    mult: float | None = None,
    run: int | None = None,
    drop: float | None = None,
    window: int | None = None,
    close_pct: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: Volume Climax (VCLIMAX)"""
    # Validate arguments
    length = _pos_int(length, 50, "length")
    lookback = _pos_int(lookback, 20, "lookback")
    mult = _pos_float(mult, 2.0, "mult")
    run = _pos_int(run, 3, "run")
    drop = _pos_float(drop, 8.0, "drop")
    window = _pos_int(window, 5, "window")
    # Above 0.5, so a bar cannot close both near its high and near its low:
    # a selling and a buying climax exclude each other.
    close_pct = _pos_float(close_pct, 0.6, "close_pct", gt=0.5, lt=1)
    # The first bar with every input of the signal: `length` volumes and
    # `lookback` lows/highs before it, `run` closing changes and the close
    # `window` bars before the previous one.
    _length = max(length, lookback, run + 1, window + 1) + 1
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    volume = verify_series(volume, _length)
    offset = get_offset(offset)

    if high is None or low is None or close is None or volume is None:
        return None

    # Calculate Result
    # Relative volume against the `length` bars before this one. No volume on a
    # window with no volume is 0/0 and reads 0.0 (tests/test_degenerate_input.py).
    rvol = degenerate_div(volume, volume.rolling(length).mean().shift(1))

    # Close position in the bar's range: 1 on the high, 0 on the low. A bar
    # with high == low reads 0.5, the middle, as ad() and cmf() read it.
    high_low_range = high - low
    cpos = ((close - low) / high_low_range.where(high_low_range != 0)).mask((high_low_range == 0) & close.notna(), 0.5)

    floor = low.rolling(lookback).min().shift(1)
    ceiling = high.rolling(lookback).max().shift(1)

    # Down (up) closes among the `run` bars before this one; NaN where a close is missing.
    change = close.diff()
    known = change.notna()
    downs = (change < 0).astype(float).where(known).rolling(run).sum().shift(1)
    ups = (change > 0).astype(float).where(known).rolling(run).sum().shift(1)
    move = (close.shift(1) / close.shift(1 + window) - 1) * 100

    # A comparison with NaN is False and would read as "no climax": the signal
    # is NaN wherever one of its inputs is.
    valid = rvol.notna() & cpos.notna() & floor.notna() & ceiling.notna() & downs.notna() & move.notna()
    heavy = rvol >= mult
    selling = heavy & (low <= floor) & (cpos >= close_pct) & ((downs >= run) | (move <= -drop))
    buying = heavy & (high >= ceiling) & (cpos <= 1 - close_pct) & ((ups >= run) | (move >= drop))
    signal = Series(np.where(selling, 1.0, np.where(buying, -1.0, 0.0)), index=close.index).where(valid)

    # Follow-through: the bar after a climax moves away from it.
    prior = signal.shift(1)
    higher = (low > low.shift(1)) & (close > close.shift(1))
    lower = (high < high.shift(1)) & (close < close.shift(1))
    follow = Series(np.where((prior == 1) & higher, 1.0, np.where((prior == -1) & lower, -1.0, 0.0)), index=close.index)
    follow = follow.where(prior.notna() & cpos.notna())

    # Offset
    rvol, cpos, signal, follow = apply_offset([rvol, cpos, signal, follow], offset)

    rvol, cpos, signal, follow = apply_fill([rvol, cpos, signal, follow], **kwargs)

    # Name and Categorize it
    _props = f"_{length}_{lookback}_{mult}_{run}_{drop}_{window}_{close_pct}"
    rvol.name = f"VCLIMAXr_{length}"
    cpos.name = "VCLIMAXp"
    signal.name = f"VCLIMAX{_props}"
    follow.name = f"VCLIMAXft{_props}"
    rvol.category = cpos.category = signal.category = follow.category = "volume"

    # Prepare DataFrame to return
    df = DataFrame({rvol.name: rvol, cpos.name: cpos, signal.name: signal, follow.name: follow})
    df.name = f"VCLIMAX{_props}"
    df.category = "volume"

    return df


vclimax.__doc__ = """Volume Climax (VCLIMAX)

Labels Wyckoff-style climax bars. A selling climax (capitulation) is very heavy
volume on a new low after a run of selling, with the bar closing in the upper
part of its range: supply was met by demand. A buying climax (blow-off) is the
mirror image on a new high, closing in the lower part of the range. In
Wyckoff and Volume Spread Analysis terms the bar marks effort without result:
the volume did not carry the price through.

The indicator labels events; it does not call bottoms or tops. A climax can
repeat several times before a trend turns, and often does not turn it at all.
The follow-through column marks the bar after a climax that moved away from
it (a higher low and a higher close after a selling climax, a lower high and a
lower close after a buying climax), which is the usual confirmation.

Every value at bar t uses bars t and earlier only.

Sources:
    Richard D. Wyckoff's method: selling and buying climaxes
    Anna Coulling, "A Complete Guide to Volume Price Analysis", 2013
    https://chartschool.stockcharts.com/table-of-contents/market-analysis/wyckoff-analysis-articles/the-wyckoff-method-a-tutorial

Calculation:
    Default Inputs:
        length=50, lookback=20, mult=2.0, run=3, drop=8.0, window=5, close_pct=0.6

    RVOL = volume / SMA(volume, length).shift(1)  # the `length` bars before this one
    CPOS = (close - low) / (high - low)  # 0.5 on a bar with high == low

    new_low = low <= MIN(low, lookback).shift(1)
    new_high = high >= MAX(high, lookback).shift(1)
    down_run = every one of the `run` closes up to close[t-1] below the one before it
    up_run = every one of the `run` closes up to close[t-1] above the one before it
    MOVE = 100 * (close[t-1] / close[t-1-window] - 1)

    SIGNAL = +1 if RVOL >= mult and new_low and CPOS >= close_pct
                  and (down_run or MOVE <= -drop)
             -1 if RVOL >= mult and new_high and CPOS <= 1 - close_pct
                  and (up_run or MOVE >= drop)
              0 otherwise

    FOLLOW = +1 if SIGNAL[t-1] == +1 and low > low[t-1] and close > close[t-1]
             -1 if SIGNAL[t-1] == -1 and high < high[t-1] and close < close[t-1]
              0 otherwise

    SIGNAL is NaN until max(length, lookback, run + 1, window + 1) bars of
    history exist, and wherever one of its inputs is missing; FOLLOW is NaN
    where the previous SIGNAL or this bar's high, low or close is.

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    length (int): Bars in the volume average, before the current bar. Default: 50
    lookback (int): Bars before the current one that a new low or high must reach. Default: 20
    mult (float): Relative volume that counts as heavy. Default: 2.0
    run (int): Consecutive down (up) closes into the bar. Default: 3
    drop (float): Percent move over `window` bars into the bar, the alternative to `run`. Default: 8.0
    window (int): Bars the `drop` move is measured over. Default: 5
    close_pct (float): Close position a selling climax must reach; a buying climax must
        stay at or below 1 - close_pct. Must be above 0.5 and below 1. Default: 0.6
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: VCLIMAXr (relative volume), VCLIMAXp (close position),
    VCLIMAX (signal) and VCLIMAXft (follow-through) columns.
"""
