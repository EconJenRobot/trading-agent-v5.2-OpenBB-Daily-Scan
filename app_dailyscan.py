import os
# ==========================================
# 【終極大絕招】強制關閉 OpenBB 啟動時的自動構建與鎖定檔機制
# ==========================================
os.environ["OPENBB_AUTO_BUILD"] = "False"
os.environ["OPENBB_BUILD"] = "False"
os.environ["OPENBB_USER_SETTINGS_DIRECTORY"] = "/tmp/openbb/settings"
os.environ["OPENBB_DATA_DIRECTORY"] = "/tmp/openbb/data"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from openbb import obb  # 榮耀回歸

# ==========================================
# 0. 網頁基本設定 (必須在最上層)
# ==========================================
st.set_page_config(
    page_title="Trading Agent V5.2 - OpenBB 雲端核心版",
    page_icon="🦅",
    layout="wide"
)

# ==========================================
# 1. 數據獲取與核心計算邏輯 (OpenBB v4 雲端標準與備援雙引擎版)
# ==========================================
@st.cache_data(ttl=3600)  # 快取 1 小時
def fetch_and_process_data(tickers, benchmark_ticker, bond_10y_ticker):
    try:
        # 【嘗試引擎 A】：OpenBB v4 標準最新語法
        # 註：v4 支援用逗號字串（例如 "NVDA,TSLA"）一次抓取多檔
        data_raw = obb.equity.price.historical(
            symbol=",".join(tickers),
            provider="yfinance",
            start_date=(pd.Timestamp.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
        )
        data_df = data_raw.to_dataframe()
        
        bench_raw = obb.equity.price.historical(symbol=benchmark_ticker, provider="yfinance", start_date=(pd.Timestamp.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d'))
        bench_df = bench_raw.to_dataframe()['close'].squeeze()
        
        bond_raw = obb.equity.price.historical(symbol=bond_10y_ticker, provider="yfinance", start_date=(pd.Timestamp.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d'))
        bond_df = bond_raw.to_dataframe()['close'].squeeze()
        
        # 解析與標準化對齊數據
        if 'symbol' in data_df.columns:
            price_df = data_df.pivot(columns='symbol', values='close').ffill().fillna(0)
            volume_df = data_df.pivot(columns='symbol', values='volume').fillna(0)
        elif isinstance(data_df.index, pd.MultiIndex):
            price_df = data_df['close'].unstack(level='symbol').ffill().fillna(0)
            volume_df = data_df['volume'].unstack(level='symbol').fillna(0)
        else:
            if len(tickers) == 1:
                price_df = pd.DataFrame({tickers[0]: data_df['close']})
                volume_df = pd.DataFrame({tickers[0]: data_df['volume']})
            else:
                price_df = data_df.pivot(columns='symbol', values='close').ffill().fillna(0)
                volume_df = data_df.pivot(columns='symbol', values='volume').fillna(0)
                
        return price_df, volume_df, bench_df, bond_df

    except Exception as e:
        # 【啟動防線 B】：萬一雲端 OpenBB 缺少 Extension 罷工，立刻無縫切換至純 yfinance 備援
        try:
            import yfinance as yf
            data_raw = yf.download(tickers, period="3mo", group_by='ticker')
            bench_raw = yf.download(benchmark_ticker, period="3mo")
            bond_raw = yf.download(bond_10y_ticker, period="3mo")
            
            bench_df = bench_raw['Close'].squeeze()
            bond_df = bond_raw['Close'].squeeze()
            
            if len(tickers) == 1:
                price_df = pd.DataFrame({tickers[0]: data_raw['Close']})
                volume_df = pd.DataFrame({tickers[0]: data_raw['Volume']})
            else:
                price_df = pd.DataFrame({t: data_raw[t]['Close'] for t in tickers if t in data_raw.columns.levels[0]}).ffill().fillna(0)
                volume_df = pd.DataFrame({t: data_raw[t]['Volume'] for t in tickers if t in data_raw.columns.levels[0]}).fillna(0)
                
            return price_df, volume_df, bench_df, bond_df
        except Exception as inner_e:
            st.sidebar.error(f"所有數據引擎均獲取失敗: {str(inner_e)}")
            return None, None, None, None

# ==========================================
# 2. 側邊欄配置 (Sidebar)
# ==========================================
st.sidebar.header("⚙️ 偵測雷達參數配置")
ticker_input = st.sidebar.text_area(
    "監控標的群組 (用逗號隔開)", 
    "NVDA, TSLA, ARM, AMD, DELL, INTC, NOK, MRVL, MU, ANET, BB, AMZN, META, MSFT, AVGO"
)
tickers = [t.strip().upper() for t in ticker_input.split(",")]
benchmark_ticker = st.sidebar.text_input("大盤對照基準", "QQQ")
bond_10y_ticker = st.sidebar.text_input("美債殖利率錨定", "^TNX")

if st.sidebar.button("🔄 立即重新整理數據"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 3. 主畫面與數據載入
# ==========================================
st.title("🦅 Trading Agent V5.2 控制台")
st.subheader("OpenBB 數據引擎雲端版 — 成功突破 Linux 權限鎖，重返機構級數據軌道")
st.markdown("---")

with st.spinner("🕵️ Agent 正在調取量化交易數據歷史..."):
    price_df, volume_df, bench_df, bond_df = fetch_and_process_data(tickers, benchmark_ticker, bond_10y_ticker)

if price_df is None or price_df.empty:
    st.error("⚠️ 獲取數據時發生異常，請檢查
