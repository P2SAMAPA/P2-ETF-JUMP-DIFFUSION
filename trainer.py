"""
Main training script – Daily, Global, and Shrinking modes.

FIXES:
  Bug 3 — prepare_returns_series received a positionally-sliced df whose
           RangeIndex didn't match the DatetimeIndex used for the macro
           slice. All slicing now done via date comparisons on the
           Date column BEFORE passing to prepare_returns_series, and
           the slice is passed as the full filtered frame (not via .loc
           on a mismatched boolean mask).

  Bug 4 — date_mask / aligned_mask mismatch: macro_df uses DatetimeIndex
           while df_master uses RangeIndex. Replaced with date-range
           filtering on both DataFrames using the same start/end dates
           so the two slices are always aligned.

  Bug 5 — Shrinking windows used future data: the window end was set
           to start + 2*DAILY_LOOKBACK which for recent start_years
           extends beyond the hold-out period. Fixed by capping the
           window end at TODAY - DAILY_LOOKBACK days so every window
           has an out-of-sample period to validate against. Windows
           are also now weighted by hold-out realised return (IC proxy)
           rather than in-sample expected_return.
"""

import os
import pandas as pd
import numpy as np

import config
import data_manager
from jump_diffusion_model import MertonJumpDiffusion
import push_results


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_vix_series(macro_df: pd.DataFrame, date_index: pd.DatetimeIndex):
    """Return VIX series aligned to date_index, forward-filled."""
    if 'VIX' not in macro_df.columns:
        return None
    return macro_df['VIX'].reindex(date_index, method='ffill')


def fit_and_forecast(ret_series, macro_series=None, lambda_cap=10.0):
    """Fit Merton model and return forecast dict."""
    vix_avg = (float(macro_series.mean())
               if macro_series is not None and len(macro_series) > 0
               else 20.0)
    model = MertonJumpDiffusion(
        jump_threshold_std=config.JUMP_THRESHOLD_STD,
        lambda_cap=lambda_cap,
        macro_conditioning=config.USE_MACRO_CONDITIONING,
        vix_avg=vix_avg,
    )
    ret_values   = ret_series.values if isinstance(ret_series, pd.Series) else np.asarray(ret_series)
    macro_values = (macro_series.values
                    if macro_series is not None and isinstance(macro_series, pd.Series)
                    else macro_series)
    success = model.fit(ret_values, macro_values)
    if success:
        return model.forecast()
    mean_ret = float(np.mean(ret_values) * 252)
    return {
        'expected_return': mean_ret, 'diffusion_drift': mean_ret,
        'jump_intensity': 0.0, 'jump_mean': 0.0, 'jump_adjustment': 0.0,
    }


def compute_universe_results(tickers, df_master, macro_df,
                              start_date=None, end_date=None,
                              lambda_cap=config.LAMBDA_CAP_DAILY):
    """
    FIX Bug 3 & 4: filter both df_master and macro_df by start_date/end_date
    using consistent date comparisons — no boolean mask re-indexing.

    prepare_returns_series is given the full (filtered) df_master so it can
    call set_index('Date') cleanly, and macro_slice shares the same date range.
    """
    # Filter df_master by date range
    mask = pd.Series([True] * len(df_master), index=df_master.index)
    if start_date is not None:
        mask &= df_master['Date'] >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= df_master['Date'] <= pd.Timestamp(end_date)
    df_slice = df_master.loc[mask].copy()

    # Filter macro_df by same date range using its DatetimeIndex
    macro_mask = pd.Series([True] * len(macro_df), index=macro_df.index)
    if start_date is not None:
        macro_mask &= macro_df.index >= pd.Timestamp(start_date)
    if end_date is not None:
        macro_mask &= macro_df.index <= pd.Timestamp(end_date)
    macro_slice = macro_df.loc[macro_mask]

    results = {}
    for ticker in tickers:
        ret_series = data_manager.prepare_returns_series(df_slice, ticker)
        if len(ret_series) < config.MIN_OBSERVATIONS:
            continue
        vix_series = _get_vix_series(macro_slice, ret_series.index)
        fc = fit_and_forecast(ret_series, vix_series, lambda_cap)
        results[ticker] = {
            'ticker':          ticker,
            'expected_return': fc['expected_return'],
            'diffusion_drift': fc['diffusion_drift'],
            'jump_intensity':  fc['jump_intensity'],
            'jump_mean':       fc['jump_mean'],
            'jump_adjustment': fc['jump_adjustment'],
        }

    sorted_items = sorted(results.items(),
                          key=lambda x: x[1]['expected_return'], reverse=True)
    top3 = [{"ticker": t, **d} for t, d in sorted_items[:3]]
    return results, top3


def run_shrinking_windows(df_master, tickers, macro_df):
    """
    FIX Bug 5: each window now ends at most DAILY_LOOKBACK days before the
    latest date in the dataset, so every window has genuine out-of-sample
    data to validate against.

    Votes are weighted by the hold-out realised mean return of the chosen
    ticker (rank-IC proxy) rather than the in-sample expected_return.
    This prevents windows that happened to have high in-sample drift from
    dominating when that drift didn't persist.
    """
    latest_date  = df_master['Date'].max()
    holdout_days = config.DAILY_LOOKBACK        # reserve this many days as OOS
    max_end_date = latest_date - pd.Timedelta(days=holdout_days)

    windows = []
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        train_start = pd.Timestamp(f"{start_year}-01-01")
        # Cap window end so there is always out-of-sample data after it
        train_end   = min(
            train_start + pd.Timedelta(days=config.DAILY_LOOKBACK * 2),
            max_end_date
        )
        if train_end <= train_start + pd.Timedelta(days=config.MIN_OBSERVATIONS):
            continue   # window too short

        oos_start = train_end + pd.Timedelta(days=1)
        oos_end   = min(oos_start + pd.Timedelta(days=holdout_days), latest_date)

        # Fit on training window
        best_ticker = None
        best_ret    = -np.inf
        for ticker in tickers:
            ret_series = data_manager.prepare_returns_series(
                df_master.loc[
                    (df_master['Date'] >= train_start) &
                    (df_master['Date'] <= train_end)
                ], ticker)
            if len(ret_series) < config.MIN_OBSERVATIONS:
                continue
            macro_mask = (macro_df.index >= train_start) & (macro_df.index <= train_end)
            vix_series = _get_vix_series(macro_df.loc[macro_mask], ret_series.index)
            fc = fit_and_forecast(ret_series, vix_series, config.LAMBDA_CAP_DAILY)
            if fc['expected_return'] > best_ret:
                best_ret    = fc['expected_return']
                best_ticker = ticker

        if best_ticker is None:
            continue

        # FIX Bug 5: weight vote by hold-out realised return (not in-sample)
        oos_ret = data_manager.prepare_returns_series(
            df_master.loc[
                (df_master['Date'] >= oos_start) &
                (df_master['Date'] <= oos_end)
            ], best_ticker)
        oos_weight = float(oos_ret.mean() * 252) if len(oos_ret) > 5 else 0.0

        windows.append({
            'window_start':    start_year,
            'window_end':      train_end.year,
            'ticker':          best_ticker,
            'expected_return': best_ret,
            'oos_weight':      oos_weight,
        })

    if not windows:
        return None

    # Weighted vote: each window contributes oos_weight (clamped to ≥0)
    vote_score = {}
    vote_count = {}
    for w in windows:
        t   = w['ticker']
        wgt = max(w['oos_weight'], 0.0)   # only reward positive OOS performance
        vote_score[t] = vote_score.get(t, 0.0) + wgt
        vote_count[t] = vote_count.get(t, 0) + 1

    # Primary sort: weighted score; secondary: raw count
    pick       = max(vote_score, key=lambda t: (vote_score[t], vote_count[t]))
    conviction = vote_count[pick] / len(windows) * 100

    return {
        'ticker':      pick,
        'conviction':  conviction,
        'num_windows': len(windows),
        'windows':     windows,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set")
        return

    df_master = data_manager.load_master_data()
    macro_df  = data_manager.prepare_macro_features(df_master)

    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== {universe_name} ===")
        universe_out = {}

        # ── Daily (504d) ──────────────────────────────────────────
        daily_start = df_master['Date'].iloc[-config.DAILY_LOOKBACK]
        daily_results, daily_top3 = compute_universe_results(
            tickers, df_master, macro_df,
            start_date=daily_start,
            lambda_cap=config.LAMBDA_CAP_DAILY,
        )
        universe_out['daily'] = {'top_picks': daily_top3, 'universes': daily_results}
        print(f"  Daily top: {daily_top3[0]['ticker']}" if daily_top3 else "  Daily: no data")

        # ── Global (2008-present) ─────────────────────────────────
        global_results, global_top3 = compute_universe_results(
            tickers, df_master, macro_df,
            start_date=config.GLOBAL_TRAIN_START,
            lambda_cap=config.LAMBDA_CAP_GLOBAL,
        )
        universe_out['global'] = {'top_picks': global_top3, 'universes': global_results}
        print(f"  Global top: {global_top3[0]['ticker']}" if global_top3 else "  Global: no data")

        # ── Shrinking windows ─────────────────────────────────────
        shrinking = run_shrinking_windows(df_master, tickers, macro_df)
        if shrinking:
            universe_out['shrinking'] = shrinking
            print(f"  Shrinking consensus: {shrinking['ticker']} "
                  f"({shrinking['conviction']:.0f}% conviction)")

        all_results[universe_name] = universe_out

    push_results.push_daily_result({"run_date": config.TODAY, "universes": all_results})
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    main()
