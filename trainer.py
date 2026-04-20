"""
Main training script for Jump-Diffusion engine.
Fits Merton model to each ETF and ranks by jump-adjusted expected return.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from jump_diffusion_model import MertonJumpDiffusion
import push_results

def run_jump_diffusion():
    print(f"=== P2-ETF-JUMP-DIFFUSION Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    
    all_results = {}
    top_picks = {}
    
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        universe_results = {}
        
        for ticker in tickers:
            print(f"  Fitting {ticker}...")
            returns = data_manager.prepare_returns_series(df_master, ticker)
            if len(returns) < config.MIN_OBSERVATIONS:
                continue
            
            recent = returns.iloc[-config.LOOKBACK_WINDOW:].values
            model = MertonJumpDiffusion(jump_threshold_std=config.JUMP_THRESHOLD_STD)
            success = model.fit(recent)
            
            if success:
                forecast = model.forecast()
                universe_results[ticker] = {
                    'ticker': ticker,
                    'expected_return': forecast['expected_return'],
                    'diffusion_drift': forecast['diffusion_drift'],
                    'jump_intensity': forecast['jump_intensity'],
                    'jump_mean': forecast['jump_mean'],
                    'jump_adjustment': forecast['jump_adjustment']
                }
            else:
                universe_results[ticker] = {
                    'ticker': ticker,
                    'expected_return': np.mean(recent) * 252,
                    'diffusion_drift': np.mean(recent) * 252,
                    'jump_intensity': 0,
                    'jump_mean': 0,
                    'jump_adjustment': 0
                }
        
        all_results[universe_name] = universe_results
        
        # Top 3 by expected return
        sorted_tickers = sorted(universe_results.items(), 
                                key=lambda x: x[1]['expected_return'], reverse=True)
        top_picks[universe_name] = [
            {'ticker': t, **d} for t, d in sorted_tickers[:3]
        ]
    
    # Shrinking windows
    shrinking_results = {}
    df_master = df_master.sort_values('Date')
    
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        window_label = f"{start_year}-{start_year + 2}"
        print(f"\n--- Shrinking Window: {window_label} ---")
        
        mask = df_master['Date'] >= start_date
        df_window = df_master[mask].copy()
        if len(df_window) < config.MIN_OBSERVATIONS:
            continue
        
        window_top = {}
        for universe_name, tickers in config.UNIVERSES.items():
            best_ticker = None
            best_ret = -np.inf
            for ticker in tickers:
                returns = data_manager.prepare_returns_series(df_window, ticker)
                if len(returns) < config.MIN_OBSERVATIONS:
                    continue
                recent = returns.iloc[:config.LOOKBACK_WINDOW].values
                if len(recent) < config.MIN_OBSERVATIONS:
                    continue
                model = MertonJumpDiffusion(jump_threshold_std=config.JUMP_THRESHOLD_STD)
                if model.fit(recent):
                    fc = model.forecast()['expected_return']
                else:
                    fc = np.mean(recent) * 252
                if fc > best_ret:
                    best_ret = fc
                    best_ticker = ticker
            if best_ticker:
                window_top[universe_name] = {
                    'ticker': best_ticker,
                    'expected_return': best_ret
                }
        shrinking_results[window_label] = {
            'start_year': start_year,
            'top_picks': window_top
        }
    
    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "jump_threshold_std": config.JUMP_THRESHOLD_STD
        },
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        },
        "shrinking_windows": shrinking_results
    }
    
    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_jump_diffusion()
