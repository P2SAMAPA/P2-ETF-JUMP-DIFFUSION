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


def fit_and_forecast(returns_series, macro_series=None, lambda_cap=10.0):
    """Fit Merton model and return forecast dict."""
    model = MertonJumpDiffusion(
        jump_threshold_std=config.JUMP_THRESHOLD_STD,
        lambda_cap=lambda_cap,
        macro_conditioning=config.USE_MACRO_CONDITIONING,
        vix_avg=macro_series.mean() if macro_series is not None and len(macro_series) > 0 else 20.0
    )
    success = model.fit(returns_series.values, macro_series.values if macro_series is not None else None)
    if success:
        return model.forecast()
    else:
        return {'expected_return': np.mean(returns_series) * 252,
                'diffusion_drift': np.mean(returns_series) * 252,
                'jump_intensity': 0, 'jump_mean': 0, 'jump_adjustment': 0}


def compute_universe_results(tickers, returns, macro, lambda_cap):
    """Fit jump‑diffusion for each ticker and compute top 3."""
    results = {}
    for ticker in tickers:
        ret_series = data_manager.prepare_returns_series(returns, ticker)
        if len(ret_series) < config.MIN_OBSERVATIONS:
            continue
        macro_series = macro['VIX'] if 'VIX' in macro.columns else None
        fc = fit_and_forecast(ret_series, macro_series, lambda_cap)
        results[ticker] = {
            'ticker': ticker,
            'expected_return': fc['expected_return'],
            'diffusion_drift': fc['diffusion_drift'],
            'jump_intensity': fc['jump_intensity'],
            'jump_mean': fc['jump_mean'],
            'jump_adjustment': fc['jump_adjustment']
        }

    sorted_tickers = sorted(results.items(), key=lambda x: x[1]['expected_return'], reverse=True)
    top3 = [{k: v for k, v in d.items() if k != 'ticker'} | {'ticker': t}
            for t, d in sorted_tickers[:3]]
    return results, top3


def run_shrinking_windows(df_master, tickers, macro):
    """Shrinking windows consensus."""
    windows = []
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        sd = pd.Timestamp(f"{start_year}-01-01")
        ed = pd.Timestamp(f"{start_year}-01-01") + pd.Timedelta(days=config.DAILY_LOOKBACK * 2)
        mask = (df_master['Date'] >= sd) & (df_master['Date'] <= ed)
        window_df = df_master[mask]
        r = data_manager.prepare_returns_series(window_df, tickers[0])
        if len(r) < config.MIN_OBSERVATIONS:
            continue

        best_ticker = None
        best_ret = -np.inf
        for ticker in tickers:
            ret_series = data_manager.prepare_returns_series(window_df, ticker)
            if len(ret_series) < config.MIN_OBSERVATIONS:
                continue
            fc = fit_and_forecast(ret_series, macro['VIX'] if 'VIX' in macro.columns else None,
                                  config.LAMBDA_CAP_DAILY)
            if fc['expected_return'] > best_ret:
                best_ret = fc['expected_return']
                best_ticker = ticker
        if best_ticker:
            windows.append({'window_start': start_year, 'window_end': start_year + 2,
                            'ticker': best_ticker, 'expected_return': best_ret})

    if not windows:
        return None
    vote = {}
    for w in windows:
        vote[w['ticker']] = vote.get(w['ticker'], 0) + 1
    pick = max(vote, key=vote.get)
    conviction = vote[pick] / len(windows) * 100
    return {'ticker': pick, 'conviction': conviction, 'windows': windows}


def main():
    import os
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set")
        return

    df_master = data_manager.load_master_data()
    macro = data_manager.prepare_macro_features(df_master)

    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== {universe_name} ===")
        returns = data_manager.prepare_returns_series(df_master.set_index('Date'), tickers[0])
        returns_all = df_master.set_index('Date')
        macro_uni = macro.loc[returns_all.index].dropna()

        universe_out = {}

        # Daily (504d) – original
        daily_ret = returns_all.iloc[-config.DAILY_LOOKBACK:]
        daily_macro = macro_uni.loc[daily_ret.index] if len(macro_uni) > 0 else pd.DataFrame()
        daily_results, daily_top3 = compute_universe_results(
            tickers, df_master, daily_macro, config.LAMBDA_CAP_DAILY)
        universe_out['daily'] = {'top_picks': daily_top3, 'universes': daily_results}
        print(f"  Daily top: {daily_top3[0]['ticker']}" if daily_top3 else "  Daily: no data")

        # Global (full history) – new
        global_ret = returns_all[returns_all.index >= config.GLOBAL_TRAIN_START]
        global_macro = macro_uni.loc[global_ret.index] if len(macro_uni) > 0 else pd.DataFrame()
        global_results, global_top3 = compute_universe_results(
            tickers, df_master, global_macro, config.LAMBDA_CAP_GLOBAL)
        universe_out['global'] = {'top_picks': global_top3, 'universes': global_results}
        print(f"  Global top: {global_top3[0]['ticker']}" if global_top3 else "  Global: no data")

        # Shrinking windows
        shrinking = run_shrinking_windows(df_master, tickers, macro_uni)
        if shrinking:
            universe_out['shrinking'] = shrinking
            print(f"  Shrinking consensus: {shrinking['ticker']} ({shrinking['conviction']:.0f}%)")

        all_results[universe_name] = universe_out

    push_results.push_daily_result({"run_date": config.TODAY, "universes": all_results})
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    main()
