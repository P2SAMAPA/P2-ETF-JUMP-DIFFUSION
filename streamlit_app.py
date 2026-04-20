"""
Streamlit Dashboard for Jump-Diffusion Engine.
Displays jump-adjusted expected returns and top ETF picks.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Jump-Diffusion", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; color: white; }
    .hero-return { font-size: 2rem; font-weight: 600; color: white; }
    .return-positive { color: #28a745; font-weight: 600; }
    .return-negative { color: #dc3545; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def return_badge(ret):
    if ret >= 0:
        return f'<span class="return-positive">+{ret*100:.2f}%</span>'
    return f'<span class="return-negative">{ret*100:.2f}%</span>'

def display_hero_card(ticker: str, exp_ret: float, jump_adj: float, jump_intensity: float):
    ret_str = f"{exp_ret*100:+.2f}%"
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">📈 TOP PICK (Jump‑Adjusted Expected Return)</div>
        <div class="hero-ticker">{ticker}</div>
        <div class="hero-return">Exp Return: {ret_str}</div>
        <div style="margin-top: 1rem; color: rgba(255,255,255,0.9);">
            Jump Adjustment: {jump_adj*100:+.2f}% | Jump Intensity: {jump_intensity:.2f}/year
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_forecast_table(universe_data: dict):
    rows = []
    for ticker, m in universe_data.items():
        rows.append({
            'Ticker': ticker,
            'Exp Return': f"{m['expected_return']*100:.2f}%",
            'Diffusion Drift': f"{m['diffusion_drift']*100:.2f}%",
            'Jump Adjustment': f"{m['jump_adjustment']*100:.2f}%",
            'Jump Intensity': f"{m['jump_intensity']:.2f}"
        })
    df = pd.DataFrame(rows).sort_values('Exp Return', ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")
st.sidebar.divider()
st.sidebar.markdown("### 📈 Jump-Diffusion Parameters")
st.sidebar.markdown(f"- Lookback: **{config.LOOKBACK_WINDOW} days**")
st.sidebar.markdown(f"- Jump Threshold: **{config.JUMP_THRESHOLD_STD} σ**")

st.markdown('<div class="main-header">📈 P2Quant Jump-Diffusion</div>', unsafe_allow_html=True)
st.markdown('<div>Merton Jump‑Diffusion Model – Separating Drift from Discontinuous Jumps</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_trading']
shrinking = data.get('shrinking_windows', {})

tab1, tab2 = st.tabs(["📋 Daily Trading", "📆 Shrinking Windows"])

with tab1:
    top_picks = daily['top_picks']
    universes_data = daily['universes']
    subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]
    
    for subtab, key in zip(subtabs, universe_keys):
        with subtab:
            if key in universes_data:
                picks = top_picks.get(key, [])
                if picks:
                    top = picks[0]
                    st.markdown("### 🏆 Top Pick for Tomorrow")
                    display_hero_card(top['ticker'], top['expected_return'],
                                      top['jump_adjustment'], top['jump_intensity'])
                st.markdown("### 📋 All ETFs (Ranked by Expected Return)")
                display_forecast_table(universes_data[key])

with tab2:
    if not shrinking:
        st.warning("No shrinking windows data yet.")
        st.stop()
    st.markdown("### Top Picks Across Historical Windows")
    subtabs_sw = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    for subtab, key in zip(subtabs_sw, universe_keys):
        with subtab:
            rows = []
            for label, winfo in sorted(shrinking.items(), key=lambda x: x[1]['start_year'], reverse=True):
                top = winfo['top_picks'].get(key, {})
                if top:
                    rows.append({
                        'Window': label,
                        'Top Pick': top['ticker'],
                        'Exp Return': f"{top['expected_return']*100:.2f}%"
                    })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
