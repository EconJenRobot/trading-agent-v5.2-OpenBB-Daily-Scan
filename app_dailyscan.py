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

# 【終極突破】注入自訂 CSS，徹底榨乾螢幕寬度，將 Streamlit 預設的左右大白邊完全去除
st.markdown(
    """
    <style>
        .block-container {
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 100% !important;
        }
    </style>
    """,
    unsafe_html=True
)

# ==========================================
# 1. 數據獲取與核心計算邏輯 (OpenBB v4 雲端標準與備援雙引擎版)
# ==========================================
@st.cache_data(ttl=3600)  # 快取 1 小時
def fetch_and_process_data(tickers, benchmark_ticker, bond_10y_ticker):
    try:
        # 【嘗試引擎 A】：OpenBB v4 標準最新語法
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
    st.error("⚠️ 獲取數據時發生異常，請檢查代號、網路或 Secrets 配置後重試。")
    st.stop()

# ==========================================
# 4. 量化因子計算
# ==========================================
current_bond_yield = float(bond_df.iloc[-1])
bond_ma20 = float(bond_df.rolling(20).mean().iloc[-1])
macro_risk_trigger = current_bond_yield > bond_ma20

current_price = price_df.iloc[-1]
ma20 = price_df.rolling(20).mean().iloc[-1]
ma50 = price_df.rolling(50).mean().iloc[-1]
std20 = price_df.rolling(20).std().iloc[-1]
bollinger_upper = ma20 + (2 * std20)
bollinger_lower = ma20 - (2 * std20)

factor_momentum = (current_price - ma50) / ma50.replace(0, np.nan)
v_ma5 = volume_df.rolling(5).mean().iloc[-1]
v_ma20 = volume_df.rolling(20).mean().iloc[-1]
factor_volume = v_ma5 / v_ma20.replace(0, np.nan)

daily_ret = price_df.pct_change()
obv = pd.DataFrame(0.0, index=price_df.index, columns=price_df.columns)
for t in tickers:
    if t in price_df.columns:
        direction = np.sign(daily_ret[t]).fillna(0)
        obv[t] = (direction * volume_df[t]).cumsum()
obv_slope = (obv.iloc[-1] - obv.iloc[-10]) / obv.iloc[-10].replace(0, np.nan)
price_change_10d = (price_df.iloc[-1] - price_df.iloc[-10]) / price_df.iloc[-10].replace(0, np.nan)
factor_accumulation = obv_slope - price_change_10d

z_a = (factor_momentum - factor_momentum.mean()) / (factor_momentum.std() if factor_momentum.std() != 0 else 1)
z_b = (factor_volume - factor_volume.mean()) / (factor_volume.std() if factor_volume.std() != 0 else 1)
z_c = (factor_accumulation - factor_accumulation.mean()) / (factor_accumulation.std() if factor_accumulation.std() != 0 else 1)
total_scores = z_a * 0.40 + z_b * 0.30 + z_c * 0.30

system_tags = {}
stop_loss_prices = {}
take_profit_prices = {}

# ==========================================
# 5. 決策樹核心過濾系統 (字串適度縮短精簡，避免文字過長撐寬欄位)
# ==========================================
for t in tickers:
    if t not in current_price.index: continue
    p = current_price[t]
    m20 = ma20[t]
    b_up = bollinger_upper[t]
    b_low = bollinger_lower[t]
    inst_strength = factor_accumulation[t]
    vol_change = factor_volume[t]
    
    support_line = m20 if p > m20 else b_low
    
    if p > m20:
        if inst_strength > 1.0:
            system_tags[t] = "🔥 多頭-機構高度鎖倉"
            stop_loss_prices[t] = f"${support_line * 0.997:.2f}"
            take_profit_prices[t] = f"${b_up:.2f}"
            total_scores[t] += 1.5 
        elif inst_strength < 0.1 and vol_change > 1.2:
            system_tags[t] = "⚠️ 散戶泡沫-缺乏機構"
            stop_loss_prices[t] = "逢高分批平倉"
            take_profit_prices[t] = "不建議追高"
            total_scores[t] -= 2.0 
        elif inst_strength > 0 and vol_change > 1.0:
            system_tags[t] = "🦅 多頭-標準量價流入"
            stop_loss_prices[t] = f"${support_line * 0.997:.2f}"
            take_profit_prices[t] = f"${b_up:.2f}"
        else:
            system_tags[t] = "⚠️ 多頭-量價背離觀察"
            stop_loss_prices[t] = "不建議入場"
            take_profit_prices[t] = "不建議入場"
            total_scores[t] -= 0.5
            
    elif p <= b_low or (p < m20 and inst_strength > 2.0):
        if inst_strength > 0:
            system_tags[t] = "⚡ 震盪-低位主力吸籌"
            stop_loss_prices[t] = f"${p * 0.997:.2f}"
            take_profit_prices[t] = f"${m20:.2f}"
            total_scores[t] += 1.0
        else:
            system_tags[t] = "🛑 空頭-左側摸底危險"
            stop_loss_prices[t] = "禁買"
            take_profit_prices[t] = "禁買"
            total_scores[t] -= 1.0
    else:
        system_tags[t] = "💀 空頭-建議平倉/不碰"
        stop_loss_prices[t] = "禁買"
        take_profit_prices[t] = "禁買"
        total_scores[t] -= 1.5

# ==========================================
# 6. 構建資料表與動態配資
# ==========================================
dashboard = pd.DataFrame({
    '當前價': current_price,
    '20MA位置': ma20,
    '機構吸籌': factor_accumulation,
    '5日量能': factor_volume,
    '決策樹系統分類': pd.Series(system_tags),
    '圖表安全止損位': pd.Series(stop_loss_prices),
    '圖表預期止盈位': pd.Series(take_profit_prices),
    '決策樹修正總分': total_scores
}).dropna(subset=['決策樹系統分類']).sort_values(by='決策樹修正總分', ascending=False)

total_allocation = 0.25 if macro_risk_trigger else 0.60
positive_stocks = dashboard[dashboard['決策樹修正總分'] > 0]

dashboard['動態配資比率'] = 0.0
dashboard['動態配資權重_顯示'] = "0.0%"

if not positive_stocks.empty:
    scores_pos = positive_stocks['決策樹修正總分']
    shifted_scores = scores_pos - scores_pos.min() + 1
    weights = (shifted_scores / shifted_scores.sum()) * total_allocation
    for t in positive_stocks.index:
        dashboard.loc[t, '動態配資比率'] = float(weights[t])
        dashboard.loc[t, '動態配資權重_顯示'] = f"{weights[t]*100:.1f}%"

# ==========================================
# 7. 網頁 UI 渲染區 (全螢幕終極解鎖版)
# ==========================================
col_macro1, col_macro2, col_macro3 = st.columns(3)
with col_macro1:
    if macro_risk_trigger:
        st.error("⚠️ 🛑 總經警報：美債殖利率飆升")
    else:
        st.success("🟢 總經安全：美債環境穩定")
with col_macro2:
    st.metric(label="10年美債當前殖利率", value=f"{current_bond_yield:.2f}%", delta=f"{current_bond_yield - bond_ma20:+.2f}% vs 20MA")
with col_macro3:
    st.metric(label="核心風險防禦系統 — 建議總權限位上限", value=f"{total_allocation*100:.0f}%")

st.markdown("---")

# ==========================================
# 上層：大表格獨佔 100% 寬度（配合 CSS 邊距歸零，達到真正全寬）
# ==========================================
st.subheader("📊 策略雷達掃描矩陣")

display_df = dashboard.copy()
display_df['當前價'] = display_df['當前價'].map(lambda x: f"${x:.2f}")
display_df['20MA位置'] = display_df['20MA位置'].map(lambda x: f"${x:.2f}")
display_df['機構吸籌'] = display_df['機構吸籌'].map(lambda x: f"{x:+.2f}")
display_df['5日量能'] = display_df['5日量能'].map(lambda x: f"{x:.2f}x")
display_df['決策樹修正總分'] = display_df['決策樹修正總分'].map(lambda x: f"{x:.2f}")

display_df = display_df[['決策樹系統分類', '當前價', '20MA位置', '機構吸籌', '5日量能', '圖表安全止損位', '圖表預期止盈位', '動態配資權重_顯示', '決策樹修正總分']]
display_df.columns = ['系統分類', '當前價', '20MA位置', '機構吸籌因子', '5日量比', '建議止損位', '預期止盈位', '配資權重', '策略總分']

# 啟用全寬、加高顯示 (600)，給予所有欄位完美的伸展視野
st.dataframe(
    display_df, 
    use_container_width=True, 
    height=600
)

st.markdown("---")

# ==========================================
# 下層：圓餅圖獨立水平置中
# ==========================================
allocated_sum = dashboard['動態配資比率'].sum()
cash_ratio = 1.0 - allocated_sum

pie_data = dashboard[dashboard['動態配資比率'] > 0].copy()
cash_row = pd.DataFrame({'動態配資比率': [cash_ratio]}, index=['💵 現金防守水位 (Cash)'])
pie_df = pd.concat([pie_data[['動態配資比率']], cash_row])
pie_df['標的'] = pie_df.index

fig = px.pie(
    pie_df, 
    values='動態配資比率', 
    names='標的', 
    hole=0.4,
    title="🎯 動態配資權重分佈",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(
    title_font=dict(size=18),
    title_x=0.44,  # 精準水平置中標題
    showlegend=True, 
    legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5), # 下方橫排標籤
    margin=dict(t=60, b=60, l=10, r=10), 
    height=550
)

# 使用三欄式比例控位，將圓餅圖完美鎖定在中央
col_space1, col_pie_main, col_space2 = st.columns([1, 2, 1])
with col_pie_main:
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(f"數據最後更新日期：{price_df.index[-1].strftime('%Y-%m-%d')} | 驅動核心：OpenBB Core v4 Standard / yfinance Intelligent Engine")
