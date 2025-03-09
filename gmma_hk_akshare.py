import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="GMMA 港股分析工具",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("顾比多重移动平均线 (GMMA) 港股图表")
st.markdown("""
此应用程序显示使用 akshare 数据的香港股票的顾比多重移动平均线 (GMMA) 图表。  
可以分析单个股票或自动扫描最近出现买入信号的股票。
""")

def has_recent_crossover_hk(ticker, days_to_check=3):
    try:
        end_date = datetime.today().strftime('%Y%m%d')
        start_date = (datetime.today() - timedelta(days=120)).strftime('%Y%m%d')
        
        stock_data = ak.stock_hk_hist(symbol=ticker, period="daily", start_date=start_date, end_date=end_date, adjust="")
        if stock_data.empty:
            return False, None
        
        stock_data.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open'}, inplace=True)
        stock_data['date'] = pd.to_datetime(stock_data['日期'])
        stock_data.set_index('date', inplace=True)
        stock_data.sort_index(inplace=True)
        
        for period in [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]:
            stock_data[f"EMA{period}"] = stock_data["收盘"].ewm(span=period, adjust=False).mean()
        
        short_terms = [3, 5, 8, 10, 12, 15]
        long_terms = [30, 35, 40, 45, 50, 60]
        stock_data['avg_short_ema'] = stock_data[[f'EMA{period}' for period in short_terms]].mean(axis=1)
        stock_data['avg_long_ema'] = stock_data[[f'EMA{period}' for period in long_terms]].mean(axis=1)
        
        stock_data['short_above_long'] = stock_data['avg_short_ema'] > stock_data['avg_long_ema']
        stock_data['crossover'] = stock_data['short_above_long'] & (~stock_data['short_above_long'].shift(1, fill_value=False))
        
        recent_data = stock_data.iloc[-days_to_check:]
        has_crossover = recent_data['crossover'].any()
        
        return has_crossover, stock_data if has_crossover else None
    except:
        return False, None

st.sidebar.title("分析模式")
analysis_mode = st.sidebar.radio("选择模式", ["自动扫描买入信号", "单一股票分析"])

if analysis_mode == "单一股票分析":
    st.sidebar.title("股票输入")
    ticker = st.sidebar.text_input("输入港股代码（例如，00593）", "00593")
    
    # ...existing code for single stock analysis...
    # Replace ak.stock_zh_a_hist with ak.stock_hk_hist
    # Adjust column names accordingly (开盘, 收盘, 最高, 最低)
    # The rest of the logic remains the same, just adapt to HK data structure

else:  # Auto scan mode
    st.sidebar.title("扫描设置")
    days_to_check = st.sidebar.slider("检查最近几天内的信号", 1, 7, 1)
    max_stocks = st.sidebar.slider("最多显示股票数量", 1, 200, 200)
    
    if st.sidebar.button("开始扫描"):
        with st.spinner("正在扫描港股买入信号，这可能需要一些时间..."):
            stock_info_df = ak.stock_hk_main_board_spot_em()
            stock_info_df = stock_info_df[['代码', '名称']]
            
            crossover_stocks = []
            progress_bar = st.progress(0)
            
            for i, row in enumerate(stock_info_df.itertuples()):
                ticker = row.代码
                name = row.名称
                
                has_crossover, stock_data = has_recent_crossover_hk(ticker, days_to_check)
                if has_crossover:
                    crossover_stocks.append((ticker, name, stock_data))
                if len(crossover_stocks) >= max_stocks:
                    break
                progress_bar.progress(min((i+1)/len(stock_info_df), 1.0))
            
            progress_bar.progress(1.0)
            
            if len(crossover_stocks) == 0:
                st.warning(f"没有找到在最近 {days_to_check} 天内出现买入信号的港股。")
            else:
                st.success(f"找到 {len(crossover_stocks)} 只在最近 {days_to_check} 天内出现买入信号的港股。")
                summary_df = pd.DataFrame(
                    [(t, n) for t, n, _ in crossover_stocks], 
                    columns=["代码", "名称"]
                )
                st.subheader("买入信号港股列表")
                st.table(summary_df)
                # Optionally, add detailed charts for each stock as in original code
