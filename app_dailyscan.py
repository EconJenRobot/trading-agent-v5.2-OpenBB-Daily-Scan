import os
import xml.etree.ElementTree as ET
import urllib.parse
import requests
from datetime import datetime

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
import yfinance as yf
from openbb import obb

# 引入 NLP 情緒分析庫並下載字典
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)

# ==========================================
# 0. 網頁基本設定 (必須在最上層)
# ==========================================
st.set_page_config(
    page_title="Trading Agent V5.5 - 華爾街量化雷達核心版",
    page_icon="🦅",
    layout="wide"
)

# ==========================================
# 1. 核心引擎：新聞抓取與「實時情緒打分」
# ==========================================
def fetch_safe_news_with_sentiment(ticker):
    """
    雙通道 RSS 穿透技術 + VADER 金融情緒分析
    """
    news_list = []
    titles_to_analyze = []
    sia = SentimentIntensityAnalyzer()
    
    # 升級金融情緒字典
    sia.lexicon.update({
        'surge': 2.0, 'surges': 2.0, 'soar': 2.5, 'soars': 2.5, 'rally': 1.8, 'highs': 1.5,
        'slip': -1.5, 'slips': -1.5, 'slump': -2.0, 'slumps': -2.0, 'drop': -1.5, 'falls': -1.5,
        'crash': -3.0, 'beat': 1.5, 'split': 1.0, 'upgrade': 1.8, 'downgrade': -2.0, 'cautious': -1.2,
        'bubble': -1.5, 'unreasonable': -1.0, 'absurd': -1.5, 'risk': -1.0, 'growth': 1.2
    })

    # 通道一：Yahoo RSS
    try:
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(rss_url, headers=headers, timeout=3)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for item in root.findall('.//item')[:3]:
                title = item.find('title').text if item.find('title') is not None else "No Title"
                news_list.append(f"📰 {title} (Yahoo RSS)")
                titles_to_analyze.append(title)
    except Exception: pass

    # 通道二：Google RSS
    if len(news_list) < 3:
        try:
            query = urllib.parse.quote(f"{ticker} stock")
            g_rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(g_rss_url, timeout=3)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for item in root.findall('.//item')[:3]:
                    title = item.find('title').text if item.find('title') is not None else "No Title"
                    source = item.find('source').text if item.find('source') is not None else "Google News"
                    title_clean = title.rsplit(" - ", 1)[0] if " - " in title else title
                    
                    if f"📰 {title_clean} ({source})" not in news_list:
                        news_list.append(f"📰 {title_clean} ({source})")
                        titles_to_analyze.append(title_clean)
        except Exception: pass

    while len(news_list) < 3:
        news_list.append(f"⚪ {ticker} 暫無更多即時新聞")

    if titles_to_analyze:
        scores = [sia.polarity_scores(t)['compound'] for t in titles_to_analyze]
        avg_sentiment = float(np.mean(scores))
    else:
        avg_sentiment = 0.0

    return news_list, avg_sentiment

# ==========================================
# 2. 數據獲取與混合計算邏輯 (自適應單/多股 + 全指標清洗)
# ==========================================
@st.cache_data(ttl=900, show_spinner=False)  # 新聞即時性高，快取縮短為15分鐘
def fetch_and_process_hybrid_data(tickers_tuple, benchmark_ticker, bond_10y_ticker):
    tickers = list(tickers_tuple)
    real_data = {}
    
    # 獲取美債與大盤歷史數據
    try:
        bond_raw = obb.equity.price.historical(symbol=bond_10y_ticker, provider="yfinance", period="100d")
        bond_df = bond_raw.to_dataframe()['close'].squeeze()
        bench_raw = obb.equity.price.historical(symbol=benchmark_ticker, provider="yfinance", period="100d")
        bench_df = bench_raw.to_dataframe()['close'].squeeze()
    except Exception:
        # 總經 yfinance 備援
        bond_df = yf.download(bond_10y_ticker, period="6mo", progress=False)['Close'].squeeze()
        bench_df = yf.download(benchmark_ticker, period="6mo", progress=False)['Close'].squeeze()

    for tk in tickers:
        try:
            # 指標 A: 透過 yfinance 獲取最即時「不延遲」現價
            ticker_yf = yf.Ticker(tk)
            current_price = ticker_yf.fast_info.get('last_price', None)
            if current_price is None or np.isnan(current_price):
                df_yf_live = ticker_yf.history(period="1d")
                current_price = float(df_yf_live['Close'].iloc[-1])

            # 指標 B: 調用 OpenBB 獲取脫水後的標準歷史 K 線
            try:
                df_obb = obb.equity.price.historical(tk, provider="yfinance", period="100d").to_df()
            except Exception:
                df_obb = yf.download(tk, period="6mo", progress=False)
                df_obb.columns = [c.lower() for c in df_obb.columns]
                
            df_obb = df_obb.dropna()

            # 指標 C: 50MA 乖離率
            ma_50 = float(df_obb['close'].rolling(window=50).mean().iloc[-1])
            dynamic_bias = (current_price - ma_50) / ma_50

            # 指標 D: 5日量能變動比率
            today_vol = float(df_obb['volume'].iloc[-1])
            avg_5d_vol = float(df_obb['volume'].iloc[-6:-1].mean())
            dynamic_vol_change = today_vol / avg_5d_vol if avg_5d_vol > 0 else 1.0

            # 指標 E: 精準 Chaikin Money Flow (CMF) 機構吸籌指標
            close_p = df_obb['close']
            high_p = df_obb['high']
            low_p = df_obb['low']
            vol_p = df_obb['volume']
            denom = (high_p - low_p).replace(0, 0.01)
            clv = ((close_p - low_p) - (high_p - close_p)) / denom
            cmf_5d = (clv * vol_p).rolling(window=5).sum() / vol_p.rolling(window=5).sum()
            dynamic_inst_strength = float(cmf_5d.iloc[-1]) * 10

            # 指標 F: 同步穿透 RSS 情緒打分
            news_content, news_sentiment_score = fetch_safe_news_with_sentiment(tk)

            real_data[tk] = {
                'price': current_price,
                'bias': dynamic_bias,
                'vol': dynamic_vol_change,
                'inst': dynamic_inst_strength,
                'news_score': news_sentiment_score,
                'news_raw': news_content
            }
        except Exception as e:
            real_data[tk] = {'price': 0.0, 'bias': 0.0, 'vol': 1.0, 'inst': 0.0, 'news_score': 0.0, 'news_raw': [f"❌ 數據流加載異常: {str(e)}"]}

    return real_data, bench_df, bond_df

# ==========================================
# 3. 側邊欄配置 (Sidebar)
# ==========================================
st.sidebar.header("⚙️ 偵測雷達參數配置")
ticker_input = st.sidebar.text_area(
    "監控標的群組 (用逗號隔開)", 
    "NVDA, TSLA, ARM, AMD, DELL, INTC, NOK, MRVL, MU, ANET, BB, AMZN, META, MSFT, AVGO"
)
tickers = [t.strip().upper() for t in ticker_input.replace("，", ",").split(",") if t.strip() and t.strip().isalpha()]
benchmark_ticker = st.sidebar.text_input("大盤對照基準", "QQQ")
bond_10y_ticker = st.sidebar.text_input("美債殖利率錨定", "^TNX")

if st.sidebar.button("🔄 立即重新整理數據"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 4. 主畫面控制台
# ==========================================
st.title("🦅 Trading Agent V5.5 華爾街級終極控制台")
st.subheader("混合 yfinance 實時流 + OpenBB 脫水矩陣 + NLP 金融情緒分析引擎")
st.markdown("---")

if not tickers:
    st.warning("⚠️ 請在左側輸入至少一檔有效的股票代號。")
    st.stop()

with st.spinner("🕵️ 量化特工正在執行跨資產 Z-Score 清洗並穿透 RSS 新聞網路..."):
    real_data, bench_df, bond_df = fetch_and_process_hybrid_data(tuple(tickers), benchmark_ticker, bond_10y_ticker)

# 轉換成 DataFrame
raw_list = []
for tk, d in real_data.items():
    if d['price'] > 0:  # 排除完全獲取失敗的標的
        raw_list.append({
            'ticker': tk, 'price': d['price'], 'bias': d['bias'], 
            'vol': d['vol'], 'inst': d['inst'], 'news_score': d['news_score'], 'news_raw': d['news_raw']
        })

if not raw_list:
    st.error("⚠️ 無法獲取任何標的的有效數據，請檢查代號或網路環境。")
    st.stop()

df_metrics = pd.DataFrame(raw_list)

# ==========================================
# 5. 橫截面 Z-Score 標準化與動態反轉權重計算
# ==========================================
if len(df_metrics) > 1:
    df_metrics['inst_z'] = (df_metrics['inst'] - df_metrics['inst'].mean()) / (df_metrics['inst'].std() if df_metrics['inst'].std() > 0 else 1)
    df_metrics['vol_z'] = (df_metrics['vol'] - df_metrics['vol'].mean()) / (df_metrics['vol'].std() if df_metrics['vol'].std() > 0 else 1)
else:
    df_metrics['inst_z'] = 0.0
    df_metrics['vol_z'] = 0.0

processed_list = []
system_tags = {}

for idx, row in df_metrics.iterrows():
    tk = row['ticker']
    # 動態新聞權重判定 (防止高檔利多出貨陷阱)
    if row['bias'] > 0.50:
        current_news_weight = 0.0
        news_status_tag = "⚠️ 過熱失效"
        system_tags[tk] = "🛑 乖離過熱限倉"
    else:
        current_news_weight = 0.10
        news_status_tag = "🟢 正常引入"
        
    # 華爾街多因子綜合得分公式
    comp_score = (row['inst_z'] * 0.35) + (row['vol_z'] * 0.20) + (row['news_score'] * current_news_weight) - (row['bias'] * 0.35)
    
    # 分類決策樹簡化標籤
    if system_tags.get(tk) is None:
        if comp_score > 0.2 and row['inst'] > 0:
            system_tags[tk] = "🔥 多頭-機構高鎖倉"
        elif comp_score < -0.2:
            system_tags[tk] = "💀 空頭-建議平倉"
        elif row['vol'] > 1.5 and row['inst'] < 0:
            system_tags[tk] = "⚠️ 散戶情緒泡沫"
        else:
            system_tags[tk] = "🦅 震盪標準區間"

    processed_list.append({
        'ticker': tk, 'price': row['price'], 'bias': row['bias'], 
        'vol': row['vol'], 'inst': row['inst'], 'news_score': row['news_score'],
        'news_weight_tag': news_status_tag, 'score': comp_score, 'news_raw': row['news_raw'],
        'tag': system_tags[tk]
    })

df_ranked = pd.DataFrame(processed_list).sort_values(by='score', ascending=False).reset_index(drop=True)

# 總經美債警報計算
current_bond_yield = float(bond_df.iloc[-1])
bond_ma20 = float(bond_df.rolling(20).mean().iloc[-1])
macro_risk_trigger = current_bond_yield > bond_ma20
total_allocation = 0.25 if macro_risk_trigger else 0.60

# 計算動態配資
df_ranked['weight_val'] = 0.0
df_ranked['建議配資'] = "0.0%"
positive_scores = df_ranked[df_ranked['score'] > 0.0]

if not positive_scores.empty:
    if len(positive_scores) == 1:
        df_ranked.loc[df_ranked['score'] > 0, 'weight_val'] = total_allocation
    else:
        shifted = positive_scores['score'] - positive_scores['score'].min() + 1
        weights = (shifted / shifted.sum()) * total_allocation
        for t_idx in positive_scores.index:
            df_ranked.loc[t_idx, 'weight_val'] = float(weights[t_idx])

for idx, row in df_ranked.iterrows():
    if row['bias'] > 0.50:
        df_ranked.loc[idx, '建議配資'] = "0.0% (🛑限倉)"
    elif row['weight_val'] > 0:
        # 限制單檔最大權重不超過 5.5% 呼應原底層控管
        final_w = min(row['weight_val'] * 10, 5.5)
        df_ranked.loc[idx, '建議配資'] = f"{final_w:.1f}%"

# ==========================================
# 6. UI 頂部總經面板渲染
# ==========================================
col_macro1, col_macro2, col_macro3 = st.columns(3)
with col_macro1:
    if macro_risk_trigger:
        st.error("⚠️ 🛑 總經警報：美債殖利率飆升中")
    else:
        st.success("🟢 總經安全：美債環境穩定")
with col_macro2:
    st.metric(label="10年美債當前殖利率", value=f"{current_bond_yield:.2f}%", delta=f"{current_bond_yield - bond_ma20:+.2f}% vs 20MA")
with col_macro3:
    st.metric(label="資產防衛水位上限", value=f"{(total_allocation)*100:.0f}%", delta="防守狀態" if macro_risk_trigger else "主攻狀態")

st.markdown("---")

# ==========================================
# 7. 數據主表格展示 (與底層視覺完美對齊)
# ==========================================
st.subheader("📊 策略雷達綜合矩陣 (Z-Score 交叉純化版)")

display_df = df_ranked.copy()
display_df['排名'] = display_df.index + 1
display_df['當前實時價'] = display_df['price'].map(lambda x: f"${x:.2f}")
display_df['動態50MA乖離'] = display_df['bias'].map(lambda x: f"{x*100:+.1f}%")
display_df['動態5日量能'] = display_df['vol'].map(lambda x: f"{x:.2f}x")
display_df['動態機構吸籌'] = display_df['inst'].map(lambda x: f"{x:+.2f}")
display_df['新聞得分'] = display_df['news_score'].map(lambda x: f"{x:+.2f}")
display_df['量化綜合總分'] = display_df['score'].map(lambda x: f"{x:.4f}")

# 重整欄位順序
display_table = display_df[['排名', 'ticker', 'tag', '當前實時價', '動態50MA乖離', '動態5日量能', '動態機構吸籌', '新聞得分', 'news_weight_tag', '量化綜合總分', '建議配資']]
display_table.columns = ['排名', '標的', '系統分類', '當前實時價', '50MA乖離', '5日量比', '機構吸籌', '新聞情緒', '新聞狀態', '綜合總分', '建議配資']

st.dataframe(
    display_table,
    use_container_width=True,
    height=380,
    column_config={
        "排名": st.column_config.NumberColumn("排名", width="small"),
        "標的": st.column_config.TextColumn("標的", width="small"),
        "系統分類": st.column_config.TextColumn("系統分類", width="medium"),
        "綜合總分": st.column_config.TextColumn("綜合總分", width="small"),
        "建議配資": st.column_config.TextColumn("建議配資", width="small"),
    }
)

st.markdown("---")

# ==========================================
# 8. 🔥 核心亮點功能：標的深度新聞雷達與 Gemini 分析報告區
# ==========================================
st.subheader("🕵️ 標的前瞻因子探針 & 即時新聞摘要分析")
st.info("💡 請在下方點擊您想深入探測的股票，即可查看實時多因子數據與 RSS 華爾街新聞摘要：")

# 建立分頁切換
tabs = st.tabs([f"🌟 {row['ticker']}" for idx, row in df_ranked.iterrows()])

for idx, row in df_ranked.iterrows():
    with tabs[idx]:
        tk = row['ticker']
        col_t1, col_t2 = st.columns([1, 2])
        
        with col_t1:
            st.markdown(f"### 🦅 **{tk} 量化探針**")
            st.metric("當前實時價格", f"${row['price']:.2f}")
            st.metric("量化綜合總分", f"{row['score']:.4f}")
            st.metric("建議戰略配資", row['建議配資'])
            
            # 指標儀表板小清單
            st.markdown(f"""
            **核心量化因子明細：**
            * 📈 **50MA 乖離率**： `{row['bias']*100:+.1f}%`
            * 📊 **5日量能爆發力**： `{row['vol']:.2f}x`
            * 🏦 **機構籌碼吸籌力 (CMF)**： `{row['inst']:+.2f}`
            * 💬 **新聞情緒 VADER 分數**： `{row['news_score']:+.2f}` (`{row['news_weight_tag']}`)
            """)
            
        with col_t2:
            st.markdown(f"### 📰 **華爾街即時新聞摘要 ({tk})**")
            for news in row['news_raw']:
                st.markdown(f"{news}")
                
            st.markdown("---")
            # 為 Gemini 準備的一鍵分析區塊
            st.caption("📋 專為大模型設計的前瞻性解讀文本 (點擊右上角可一鍵複製)：")
            report_text = f"【決策雷達前瞻報告】\n標的: {tk} | 排名: 第 {idx+1} 名\n" \
                          f"當前價格: ${row['price']:.2f} | 綜合總分: {row['score']:.4f} | 建議配資: {row['建議配資']}\n" \
                          f"數據明細 -> 50MA乖離: {row['bias']*100:+.1f}% | 5日量能: {row['vol']:.2f}x | 機構吸籌: {row['inst']:+.2f} | 新聞情緒: {row['news_score']:+.2f}\n" \
                          f"最新新聞焦點:\n" + "\n".join(row['news_raw'])
            st.code(report_text, language="text")

st.markdown("---")

# 下層圓餅圖
allocated_sum = df_ranked['weight_val'].sum()
cash_ratio = max(0.0, 1.0 - allocated_sum)

pie_data = df_ranked[df_ranked['weight_val'] > 0].copy()
cash_row = pd.DataFrame({'weight_val': [cash_ratio]}, index=['💵 現金防守水位 (Cash)'])
if not pie_data.empty:
    pie_data.index = pie_data['ticker']
    pie_df = pd.concat([pie_data[['weight_val']], cash_row])
else:
    pie_df = cash_row
pie_df['標的'] = pie_df.index

fig = px.pie(
    pie_df, values='weight_val', names='標的', hole=0.4,
    title="🎯 終極防禦型動態配資權重分佈",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title_x=0.44, height=450, margin=dict(t=40, b=40, l=10, r=10))

col_space1, col_pie_main, col_space2 = st.columns([1, 2, 1])
with col_pie_main:
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(f"數據即時同步時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 驅動核心：OpenBB Core v4 / VADER Sentiment Engine")
