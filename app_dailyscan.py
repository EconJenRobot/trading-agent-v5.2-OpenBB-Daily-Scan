import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from openbb import obb

# ==========================================
# 0. 網頁基本設定 (必須在最上層)
# ==========================================
st.set_page_config(
    page_title="Trading Agent V5.2 - OpenBB 數據核心版",
    page_icon="🦅",
    layout="wide"
)

# ==========================================
# 1. 數據獲取與核心計算邏輯 (加入 Streamlit 快取機制)
# ==========================================
@st.cache_data(ttl=3600)  # 快取 1 小時，避免重複頻繁請求數據
def fetch_and_process_data(tickers, benchmark_ticker, bond_10y_ticker):
    try:
        # 抓取個股數據
        data_raw = obb.equity.price.historical(symbol=tickers, provider="yfinance", period="3mo")
        data_df = data_raw.to_dataframe()
        
        # 抓取大盤與美債殖利率
        bench_raw = obb.equity.price.historical(symbol=benchmark_ticker, provider="yfinance", period="3mo")
        bench_df = bench_raw.to_dataframe()['close'].squeeze()
        
        bond_raw = obb.equity.price.historical(symbol=bond_10y_ticker, provider="yfinance", period="3mo")
        bond_df = bond_raw.to_dataframe()['close'].squeeze()
        
        # 標準化對齊
        if isinstance(data_df.columns, pd.MultiIndex):
            price_df = data_df['close'].ffill().fillna(0)
            volume_df = data_df['volume'].fillna(0)
        else:
            price_df = data_df.pivot(columns='symbol', values='close').ffill().fillna(0)
            volume_df = data_df.pivot(columns='symbol', values='volume').fillna(0)
            
        return price_df, volume_df, bench_df, bond_df
    except Exception as e:
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

# 重新整理按鈕
if st.sidebar.button("🔄 立即重新整理數據"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 3. 主畫面與數據載入
# ==========================================
st.title("🦅 Trading Agent V5.2 控制台")
st.subheader("OpenBB 數據引擎升級版 — 修正指標邊界漏洞，完美融合機構鎖倉邏輯")
st.markdown("---")

with st.spinner("🕵️ Agent 正在從 OpenBB 調取機構級數據歷史..."):
    price_df, volume_df, bench_df, bond_df = fetch_and_process_data(tickers, benchmark_ticker, bond_10y_ticker)

if price_df is None:
    st.error("⚠️ OpenBB 獲取數據時發生異常，請檢查代號是否正確或稍後再試。")
    st.stop()

# ==========================================
# 4. 量化因子計算
# ==========================================
current_bond_yield = bond_df.iloc[-1]
bond_ma20 = bond_df.rolling(20).mean().iloc[-1]
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

# 核心基礎分計算
z_a = (factor_momentum - factor_momentum.mean()) / (factor_momentum.std() if factor_momentum.std() != 0 else 1)
z_b = (factor_volume - factor_volume.mean()) / (factor_volume.std() if factor_volume.std() != 0 else 1)
z_c = (factor_accumulation - factor_accumulation.mean()) / (factor_accumulation.std() if factor_accumulation.std() != 0 else 1)
total_scores = z_a * 0.40 + z_b * 0.30 + z_c * 0.30

system_tags = {}
stop_loss_prices = {}
take_profit_prices = {}

# -------------------------------------------------------------------------
# 5. 決策樹核心過濾系統
# -------------------------------------------------------------------------
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
            system_tags[t] = "🔥 順勢多頭-機構高度鎖倉 (真核心)"
            stop_loss_prices[t] = f"${support_line * 0.997:.2f}"
            take_profit_prices[t] = f"${b_up:.2f}"
            total_scores[t] += 1.5 
        elif inst_strength < 0.1 and vol_change > 1.2:
            system_tags[t] = "⚠️ 散戶情緒泡沫-無機構支持"
            stop_loss_prices[t] = "建議逢高分批平倉"
            take_profit_prices[t] = "不建議追高"
            total_scores[t] -= 2.0 
        elif inst_strength > 0 and vol_change > 1.0:
            system_tags[t] = "🦅 順勢多頭-標準量價流入"
            stop_loss_prices[t] = f"${support_line * 0.997:.2f}"
            take_profit_prices[t] = f"${b_up:.2f}"
        else:
            system_tags[t] = "⚠️ 順勢多頭-量價背離觀察"
            stop_loss_prices[t] = "不建議入場"
            take_profit_prices[t] = "不建議入場"
            total_scores[t] -= 0.5
            
    elif p <= b_low or (p < m20 and inst_strength > 2.0):
        if inst_strength > 0:
            system_tags[t] = "⚡ 震盪系統-低位主力吸籌"
            stop_loss_prices[t] = f"${p * 0.997:.2f}"
            take_profit_prices[t] = f"${m20:.2f}"
            total_scores[t] += 1.0
        else:
            system_tags[t] = "🛑 順勢空頭-左側摸底危險"
            stop_loss_prices[t] = "禁買"
            take_profit_prices[t] = "禁買"
            total_scores[t] -= 1.0
    else:
        system_tags[t] = "💀 空頭系統-建議平倉/不碰"
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
# 7. 網頁 UI 渲染區 (已修正冒號漏洞)
# ==========================================

# 頂部總經區塊
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

# 視覺化看板：左邊表格，右邊配資圖
col_left, col_right = st.columns([5, 3])

with col_left:
    st.subheader("📊 策略雷達掃描矩陣")
    
    # 格式化輸出表格
    display_df = dashboard.copy()
    display_df['當前價'] = display_df['當前價'].map(lambda x: f"${x:.2f}")
    display_df['20MA位置'] = display_df['20MA位置'].map(lambda x: f"${x:.2f}")
    display_df['機構吸籌'] = display_df['機構吸籌'].map(lambda x: f"{x:+.2f}")
    display_df['5日量能'] = display_df['5日量能'].map(lambda x: f"{x:.2f}x")
    display_df['決策樹修正總分'] = display_df['決策樹修正總分'].map(lambda x: f"{x:.2f}")
    
    # 重新排列欄位順序更直覺
    display_df = display_df[['決策樹系統分類', '當前價', '20MA位置', '機構吸籌', '5日量能', '圖表安全止損位', '圖表預期止盈位', '動態配資權重_顯示', '決策樹修正總分']]
    display_df.columns = ['系統分類', '當前價', '20MA位置', '機構吸籌因子', '5日量比', '建議止損位', '預期止盈位', '動態配資權重', '策略修正總分']
    
    # 使用最新 width="stretch" 語法
    st.dataframe(display_df, width="stretch", height=500)

with col_right:
    st.subheader("🎯 動態配資權重圓餅圖")
    
    # 製作圓餅圖數據，包含未配置的現金水位
    allocated_sum = dashboard['動態配資比率'].sum()
    cash_ratio = 1.0 - allocated_sum
    
    pie_data = dashboard[dashboard['動態配資比率'] > 0].copy()
    
    # 新增現金列
    cash_row = pd.DataFrame({
        '動態配資比率': [cash_ratio]
    }, index=['💵 現金防守水位 (Cash)'])
    
    pie_df = pd.concat([pie_data[['動態配資比率']], cash_row])
    pie_df['標的'] = pie_df.index
    
    fig = px.pie(
        pie_df, 
        values='動態配資比率', 
        names='標的', 
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=450)
    
    # 使用最新 width="stretch" 語法
    st.plotly_chart(fig, width="stretch")

# 底部數據日期腳註
st.caption(f"數據最後更新日期：{price_df.index[-1].strftime('%Y-%m-%d')} | 驅動核心：OpenBB Core Engine V5.2")