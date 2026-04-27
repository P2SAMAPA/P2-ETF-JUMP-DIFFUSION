"""
Main training script – Daily, Global, and Shrinking modes.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from jump_diffusion_model import MertonJumpDiffusion
import push_results


def fit_and_forecast(ret_series, macro_series=None, lambda_cap=10.0):
    """Fit Merton model and return forecast dict."""
    model = MertonJumpDiffusion(
        jump_threshold_std=config.JUMP_THRESHOLD_STD,
        lambda_cap=lambda_cap,
        macro_conditioning=config.USE_MACRO_CONDITIONING,
        vix_avg=macro_series.mean() if macro_series is not None and len(macro_series) > 0 else 20.0
    )
    ret_values = ret_series.values if isinstance(ret_series, pd.Series) else ret_series
    macro_values = macro_series.values if macro_series is not None and isinstance(macro_series, pd.Series) else macro_series
    success = model.fit(ret_values, macro_values)
    if success:
        return model.forecast()
    else:
        mean_ret = np.mean(ret_values) * 252
        return {
            'expected_return': mean_ret,
            'diffusion_drift': mean_ret,
            'jump_intensity': 0.0, 'jump_mean': 0.0, 'jump_adjustment': 0.0
        }


def compute_universe_results(tickers, df_master, macro_df, date_mask=None, lambda_cap=config.LAMBDA_CAP_DAILY):
    """Fit jump‑diffusion for each ticker on a specific data slice (or full history)."""
    if date_mask is not None:
        df = df_master.loc[date_mask]          # already has Date column
        macro_slice = macro_df.loc[date_mask]
    else:
        df = df_master
        macro_slice = macro_df

    results = {}
    for ticker in tickers:
        ret_series = data_manager.prepare_returns_series(df, ticker)
        if len(ret_series) < config.MIN_OBSERVATIONS:
            continue
        # Fetch VIX series aligned to returns
        vix_series = macro_slice['VIX'].reindex(ret_series.index).ffill() if 'VIX' in macro_slice.columns else None
        fc = fit_and_forecast(ret_series, vix_series, lambda_cap)
        results[ticker] = {
            'ticker': ticker,
            'expected_return': fc['expected_return'],
            'diffusion_drift': fc['diffusion_drift'],
            'jump_intensity': fc['jump_intensity'],
            'jump_mean': fc['jump_mean'],
            'jump_adjustment': fc['jump_adjustment']
        }

    sorted_items = sorted(results.items(), key=lambda x: x[1]['expected_return'], reverse=True)
    top3 = [{"ticker": t, **d} for t, d in sorted_items[:3]]
    return results, top3


def run_shrinking_windows(df_master, tickers, macro_df):
    """Shrinking windows consensus."""
    windows = []
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        sd = pd.Timestamp(f"{start_year}-01-01")
        ed = sd + pd.Timedelta(days=config.DAILY_LOOKBACK * 2)
        mask = (df_master['Date'] >= sd) & (df_master['Date'] <= ed)
        best_ticker = None
        best_ret = -np.inf
        for ticker in tickers:
            ret_series = data_manager.prepare_returns_series(df_master[mask], ticker)
            if len(ret_series) < config.MIN_OBSERVATIONS:
                continue
            vix_series = macro_df.loc[ret_series.index, 'VIX'] if 'VIX' in macro_df.columns else None
            fc = fit_and_forecast(ret_series, vix_series, config.LAMBDA_CAP_DAILY)
            if fc['expected_return'] > best_ret:
                best_ret = fc['expected_return']
                best_ticker = ticker
        if best_ticker:
            windows.append({
                'window_start': start_year, 'window_end': start_year + 2,
                'ticker': best_ticker, 'expected_return': best_ret
            })

    if not windows:
        return None
    vote = {}
    for w in windows:
        vote[w['ticker']] = vote.get(w['ticker'], 0) + 1
    pick = max(vote, key=vote.get)
    conviction = vote[pick] / len(windows) * 100
    return {'ticker': pick, 'conviction': conviction, 'num_windows': len(windows), 'windows': windows}


def main():
    import os
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set")
        return

    df_master = data_manager.load_master_data()               # wide DataFrame with 'Date' column
    macro_df = data_manager.prepare_macro_features(df_master)  # indexed by Date

    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== {universe_name} ===")
        universe_out = {}

        # Daily (504d)
        daily_mask = (df_master['Date'] >= df_master['Date'].iloc[-config.DAILY_LOOKBACK])
        daily_results, daily_top3 = compute_universe_results(
            tickers, df_master, macro_df, daily_mask, config.LAMBDA_CAP_DAILY)
        universe_out['daily'] = {'top_picks': daily_top3, 'universes': daily_results}
        print(f"  Daily top: {daily_top3[0]['ticker']}" if daily_top3 else "  Daily: no data")

        # Global (2008‑YTD)
        global_mask = df_master['Date'] >= config.GLOBAL_TRAIN_START
        global_results, global_top3 = compute_universe_results(
            tickers, df_master, macro_df, global_mask, config.LAMBDA_CAP_GLOBAL)
        universe_out['global'] = {'top_picks': global_top3, 'universes': global_results}
        print(f"  Global top: {global_top3[0]['ticker']}" if global_top3 else "  Global: no data")

        # Shrinking windows
        shrinking = run_shrinking_windows(df_master, tickers, macro_df)
        if shrinking:
            universe_out['shrinking'] = shrinking
            print(f"  Shrinking consensus: {shrinking['ticker']} ({shrinking['conviction']:.0f}%)")

        all_results[universe_name] = universe_out

    push_results.push_daily_result({"run_date": config.TODAY, "universes": all_results})
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    main()
