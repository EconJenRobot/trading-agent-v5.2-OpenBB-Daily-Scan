import os
# ==========================================
# 【終極防線】強制關閉 OpenBB 啟動時的自動構建與鎖定檔機制
# ==========================================
os.environ["OPENBB_AUTO_BUILD"] = "False"
os.environ["OPENBB_BUILD"] = "False"
os.environ["OPENBB_USER_SETTINGS_DIRECTORY"] = "/tmp/openbb/settings"
os.environ["OPENBB_DATA_DIRECTORY"] = "/tmp/openbb/data"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from openbb import obb

# ==========================================
# 0. 網頁基本設定 (必須在最上層)
# ==========================================
st.set_page_config(
    page_title="Trading Agent V5.2 - OpenBB 雲端核心版",
    page_icon="🦅",
    layout="wide"
)

# ==========================================
# 1. 數據獲取與核心計算邏輯 (自適應單/多股引擎)
# ==========================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_and_process_data(tickers_tuple, benchmark_ticker, bond_10y_ticker):
    tickers = list(tickers_tuple)
    try:
        # 【嘗試引擎 A】：OpenBB v4 標準語法
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
        
        # 強制標準化轉換：確保無論一檔還是多檔，輸出都是 DataFrame 格式
        if 'symbol' in data_df.columns:
            price_df = data_df.pivot(columns='symbol', values='close').ffill()
            volume_df = data_df.pivot(columns='symbol', values='volume').fillna(0)
        elif isinstance(data_df.index, pd.MultiIndex):
            price_df = data_df['close'].unstack(level='symbol').ffill()
            volume_df = data_df['volume'].unstack(level='symbol').fillna(0)
        else:
            # 只有一檔股票的特殊邊界狀況防禦
            active_ticker = tickers[0]
            price_df = pd.DataFrame({active_ticker: data_df['close']}).ffill()
            volume_df = pd.DataFrame({active_ticker: data_df['volume']}).fillna(0)
                
        return price_df, volume_df, bench_df, bond_df

    except Exception as e:
        # 【防線 B】： yfinance 備援引擎 (同樣加入單/多股標準化)
        try:
            import yfinance as yf
            data_raw = yf.download(tickers, period="3mo", group_by='ticker', progress=False)
            bench_raw = yf.download(benchmark_ticker, period="3mo", progress=False)
            bond_raw = yf.download(bond_10y_ticker, period="3mo", progress=False)
            
            bench_df = bench_raw['Close'].squeeze()
            bond_df = bond_raw['Close'].squeeze()
            
            price_df = pd.DataFrame(index=bench_df.index)
            volume_df = pd.DataFrame(index=bench_df.index)
            
            if len(tickers) == 1:
                t = tickers[0]
                if not data_raw.empty:
                    target_col = 'Close' if 'Close' in data_raw.columns else ('Adj Close' if 'Adj Close' in data_raw.columns else None)
                    if target_col:
                        price_df[t] = data_raw[target_col]
                        volume_df[t] = data_raw['Volume']
            else:
                for t in tickers:
                    if t in data_raw.columns.levels[0]:
                        price_df[t] = data_raw[t]['Close']
                        volume_df[t] = data_raw[t]['Volume']
            
            price_df = price_df.ffill()
            volume_df = volume_df.fillna(0)
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
# 自動過濾掉空格與因手殘打錯產生的奇形怪狀字元
tickers = [t.strip().upper() for t in ticker_input.replace("，", ",").split(",") if t.strip() and t.strip().isalpha()]
benchmark_ticker = st.sidebar.text_input("大盤對照基準", "QQQ")
bond_10y_ticker = st.sidebar.text_input("美債殖利率錨定", "^TNX")

if st.sidebar.button("🔄 立即重新整理數據"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 3. 主畫面與數據載入
# ==========================================
st.title("🦅 Trading Agent V5.2 控制台")
st.subheader("OpenBB 數據引擎雲端版")
st.markdown("---")

if not tickers:
    st.warning("⚠️ 請在左側輸入至少一檔有效的股票代號（例如: NVDA）。")
    st.stop()

with st.spinner("🕵️ Agent 正在調取量化交易數據歷史..."):
    price_df, volume_df, bench_df, bond_df = fetch_and_process_data(tuple(tickers), benchmark_ticker, bond_10y_ticker)

if price_df is None or price_df.empty:
    st.error("⚠️ 獲取數據時發生異常，請檢查代號是否正確、網路或 Secrets 配置後重試。")
    st.stop()

# 有些股票可能因打錯找不到，重新校正最終有拿到數據的 tickers 清單
valid_tickers = [t for t in tickers if t in price_df.columns]

if not valid_tickers:
    st.error("⚠️ 輸入的標的皆無法取得有效的歷史價格，請檢查美股代號。")
    st.stop()

# ==========================================
# 4. 自適應量化因子計算 (防禦單股變異)
# ==========================================
current_bond_yield = float(bond_df.iloc[-1])
bond_ma20 = float(bond_df.rolling(20).mean().iloc[-1])
macro_risk_trigger = current_bond_yield > bond_ma20

current_price = price_df[valid_tickers].iloc[-1]
ma20 = price_df[valid_tickers].rolling(20).mean().iloc[-1]
ma50 = price_df[valid_tickers].rolling(50).mean().iloc[-1]
std20 = price_df[valid_tickers].rolling(20).std().iloc[-1]
bollinger_upper = ma20 + (2 * std20)
bollinger_lower = ma20 - (2 * std20)

factor_momentum = (current_price - ma50) / ma50.replace(0, np.nan)
v_ma5 = volume_df[valid_tickers].rolling(5).mean().iloc[-1]
v_ma20 = volume_df[valid_tickers].rolling(20).mean().iloc[-1]
factor_volume = v_ma5 / v_ma20.replace(0, np.nan)

daily_ret = price_df[valid_tickers].pct_change()
obv = pd.DataFrame(0.0, index=price_df.index, columns=valid_tickers)
for t in valid_tickers:
    direction = np.sign(daily_ret[t]).fillna(0)
    obv[t] = (direction * volume_df[t]).cumsum()
obv_slope = (obv.iloc[-1] - obv.iloc[-10]) / obv.iloc[-10].replace(0, np.nan)
price_change_10d = (price_df[valid_tickers].iloc[-1] - price_df[valid_tickers].iloc[-10]) / price_df[valid_tickers].iloc[-10].replace(0, np.nan)
factor_accumulation = obv_slope - price_change_10d

# 安全統計轉化：當只有一檔或兩檔股票時，Series 的 std() 為 0 或 NaN，此處做安全填補
momentum_std = factor_momentum.std() if len(valid_tickers) > 1 and factor_momentum.std() != 0 else 1
volume_std = factor_volume.std() if len(valid_tickers) > 1 and factor_volume.std() != 0 else 1
accum_std = factor_accumulation.std() if len(valid_tickers) > 1 and factor_accumulation.std() != 0 else 1

z_a = (factor_momentum - factor_momentum.mean()) / momentum_std
z_b = (factor_volume - factor_volume.mean()) / volume_std
z_c = (factor_accumulation - factor_accumulation.mean()) / accum_std

# 轉化回 Series 結構防止單股計算失效
z_a = pd.Series(z_a, index=valid_tickers).fillna(0)
z_b = pd.Series(z_b, index=valid_tickers).fillna(0)
z_c = pd.Series(z_c, index=valid_tickers).fillna(0)
total_scores = z_a * 0.40 + z_b * 0.30 + z_c * 0.30

system_tags = {}
stop_loss_prices = {}
take_profit_prices = {}

# ==========================================
# 5. 決策樹邏輯
# ==========================================
for t in valid_tickers:
    p = current_price[t]
    m20 = ma20[t]
    b_up = bollinger_upper[t]
    b_low = bollinger_lower[t]
    inst_strength = factor_accumulation[t]
    vol_change = factor_volume[t]
    
    support_line = m20 if p > m20 else b_low
    
    if p > m20:
        if inst_strength > 1.0:
            system_tags[t] = "🔥 多頭-機構高鎖倉"
            stop_loss_prices[t] = f"${support_line * 0.997:.2f}"
            take_profit_prices[t] = f"${b_up:.2f}"
            total_scores[t] += 1.5 
        elif inst_strength < 0.1 and vol_change > 1.2:
            system_tags[t] = "⚠️ 散戶情緒泡沫"
            stop_loss_prices[t] = "建議逢高平倉"
            take_profit_prices[t] = "不追高"
            total_scores[t] -= 2.0 
        elif inst_strength > 0 and vol_change > 1.0:
            system_tags[t] = "🦅 多頭-標準量價流入"
            stop_loss_prices[t] = f"${support_line * 0.997:.2f}"
            take_profit_prices[t] = f"${b_up:.2f}"
        else:
            system_tags[t] = "⚠️ 多頭-量價背離"
            stop_loss_prices[t] = "觀望"
            take_profit_prices[t] = "觀望"
            total_scores[t] -= 0.5
            
    elif p <= b_low or (p < m20 and inst_strength > 2.0):
        if inst_strength > 0:
            system_tags[t] = "⚡ 震盪-主力吸籌"
            stop_loss_prices[t] = f"${p * 0.997:.2f}"
            take_profit_prices[t] = f"${m20:.2f}"
            total_scores[t] += 1.0
        else:
            system_tags[t] = "🛑 空頭-左側危險"
            stop_loss_prices[t] = "禁買"
            take_profit_prices[t] = "禁買"
            total_scores[t] -= 1.0
    else:
        system_tags[t] = "💀 空頭-建議平倉"
        stop_loss_prices[t] = "禁買"
        take_profit_prices[t] = "禁買"
        total_scores[t] -= 1.5

# ==========================================
# 6. 資料表與權重分配
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
}, index=valid_tickers).sort_values(by='決策樹修正總分', ascending=False)

total_allocation = 0.25 if macro_risk_trigger else 0.60
positive_stocks = dashboard[dashboard['決策樹修正總分'] > 0]

dashboard['動態配資比率'] = 0.0
dashboard['動態配資權重_顯示'] = "0.0%"

if not positive_stocks.empty:
    scores_pos = positive_stocks['決策樹修正總分']
    # 修正：單股時分數分布做安全歸一
    if len(scores_pos) == 1:
        weights = pd.Series([total_allocation], index=scores_pos.index)
    else:
        shifted_scores = scores_pos - scores_pos.min() + 1
        weights = (shifted_scores / shifted_scores.sum()) * total_allocation
        
    for t in positive_stocks.index:
        dashboard.loc[t, '動態配資比率'] = float(weights[t])
        dashboard.loc[t, '動態配資權重_顯示'] = f"{weights[t]*100:.1f}%"

# ==========================================
# 7. UI 渲染
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
st.subheader("📊 策略雷達掃描矩陣")

display_df = dashboard.copy()
display_df['當前價'] = display_df['當前價'].map(lambda x: f"${x:.2f}")
display_df['20MA位置'] = display_df['20MA位置'].map(lambda x: f"${x:.2f}")
display_df['機構吸籌'] = display_df['機構吸籌'].map(lambda x: f"{x:+.2f}")
display_df['5日量能'] = display_df['5日量能'].map(lambda x: f"{x:.2f}x")
display_df['決策樹修正總分'] = display_df['決策樹修正總分'].map(lambda x: f"{x:.2f}")

display_df = display_df[['決策樹系統分類', '當前價', '20MA位置', '機構吸籌', '5日量能', '圖表安全止損位', '圖表預期止盈位', '動態配資權重_顯示', '決策樹修正總分']]
display_df.columns = ['系統分類', '當前價', '20MA位置', '機構籌碼', '5日量比', '建議止損位', '預期止盈位', '配資權重', '策略總分']

st.dataframe(
    display_df, 
    use_container_width=True, 
    height=400,
    column_config={
        "系統分類": st.column_config.TextColumn("系統分類", width="medium"),
        "當前價": st.column_config.TextColumn("當前價", width="small"),
        "20MA位置": st.column_config.TextColumn("20MA位置", width="small"),
        "機構籌碼": st.column_config.TextColumn("機構籌碼", width="small"),
        "5日量比": st.column_config.TextColumn("5日量比", width="small"),
        "建議止損位": st.column_config.TextColumn("建議止損位", width="small"),
        "預期止盈位": st.column_config.TextColumn("預期止盈位", width="small"),
        "配資權重": st.column_config.TextColumn("配資權重", width="small"),
        "策略總分": st.column_config.TextColumn("策略總分", width="small")
    }
)

st.markdown("---")

# 下層圓餅圖 (支援單股配資 100% 現金劃分)
allocated_sum = dashboard['動態配資比率'].sum()
cash_ratio = max(0.0, 1.0 - allocated_sum)

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
    title_x=0.44, 
    showlegend=True, 
    legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
    margin=dict(t=60, b=60, l=10, r=10), 
    height=550
)

col_space1, col_pie_main, col_space2 = st.columns([1, 2, 1])
with col_pie_main:
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(f"數據最後更新日期：{price_df.index[-1].strftime('%Y-%m-%d')} | 驅動核心：OpenBB Core v4 / yfinance Robust Multi-Engine")
