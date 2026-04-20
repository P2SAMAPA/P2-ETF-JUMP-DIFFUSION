# P2-ETF-JUMP-DIFFUSION

**Merton Jump-Diffusion Model for ETF Return Forecasting**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-JUMP-DIFFUSION/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-JUMP-DIFFUSION/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--jump--diffusion--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-jump-diffusion-results)

## Overview

`P2-ETF-JUMP-DIFFUSION` fits a **Merton jump‑diffusion model** to each ETF's daily returns, separating continuous diffusion from discrete jumps. The engine outputs a **jump‑adjusted expected return** that accounts for both drift and the average effect of jumps, then ranks ETFs for next‑day trading.

## Methodology

1. **Jump Detection**: Identify returns exceeding a threshold (default: 2.5σ).
2. **Maximum Likelihood Estimation**: Fit the five Merton parameters (μ, σ, λ, μⱼ, σⱼ).
3. **Forecast**: Expected return = μ + λ·μⱼ (annualized).
4. **Ranking**: ETFs sorted by jump‑adjusted expected return.

## Universe
FI/Commodities, Equity Sectors, Combined (23 ETFs)

## Usage
```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
