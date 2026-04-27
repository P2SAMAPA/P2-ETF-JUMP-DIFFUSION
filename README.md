# P2-ETF-JUMP-DIFFUSION

**Merton Jump‑Diffusion Model – Macro‑Conditioned Jump Intensity for ETF Return Forecasting**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-JUMP-DIFFUSION/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-JUMP-DIFFUSION/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--jump--diffusion--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-jump-diffusion-results)

## Overview

`P2-ETF-JUMP-DIFFUSION` fits a **Merton jump‑diffusion model** to each ETF's daily returns, separating continuous diffusion from discrete jumps. The model now includes **macro‑conditioned jump intensity**: λ is scaled by the current VIX level relative to its historical average, capturing the tendency for jumps to cluster during high‑volatility periods. ETFs are ranked by jump‑adjusted expected return.

## Methodology

1. **Merton Model**: Five parameters (μ, σ, λ, μⱼ, σⱼ) estimated via maximum likelihood.
2. **Macro Conditioning**: Jump intensity λ is scaled by `VIX_current / VIX_avg`, making the model regime‑aware.
3. **Three Training Modes**:
   - **Daily (504d)** – Most recent 2 years, λ cap = 10/yr.
   - **Global (2008‑YTD)** – Full history, λ cap = 25/yr (to accommodate crisis years).
   - **Shrinking Windows Consensus** – Most frequently selected ETF across rolling windows.

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
Dashboard
Three sub‑tabs per universe: Daily, Global, Shrinking Consensus.

Hero card with jump‑adjusted expected return, jump adjustment, and λ.

Top 3 & All ETFs tables.
