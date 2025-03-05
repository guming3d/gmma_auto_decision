import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import time
from functools import lru_cache
import os
import json

# Set page layout to wide mode
st.set_page_config(
    page_title="GMMA 股票分析工具",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("顾比多重移动平均线 (GMMA) 图表")
st.markdown("""
此应用程序显示使用 akshare 数据的中国 A 股股票的古普利多重移动平均线 (GMMA) 图表。  
可以分析单个股票或自动扫描最近出现买入信号的股票。
""")

# Function to check if a stock has a recent crossover
def has_recent_crossover(ticker, days_to_check=3):
    try:
        # Calculate date range for the past 2 months (enough data to calculate EMAs)
        end_date = datetime.today().strftime('%Y%m%d')
        start_date = (datetime.today() - timedelta(days=120)).strftime('%Y%m%d')
        
        # Fetch stock data using akshare
        stock_data = ak.stock_zh_a_hist(symbol=ticker, period="daily", 
                                         start_date=start_date, end_date=end_date, adjust="")
        if stock_data.empty:
            return False, None
            
        # Rename columns and process data
        stock_data.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open'}, inplace=True)
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data.set_index('date', inplace=True)
        stock_data.sort_index(inplace=True)
        
        # Calculate EMAs
        for period in [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]:
            stock_data[f"EMA{period}"] = stock_data["close"].ewm(span=period, adjust=False).mean()
        
        # Calculate average EMAs
        short_terms = [3, 5, 8, 10, 12, 15]
        long_terms = [30, 35, 40, 45, 50, 60]
        stock_data['avg_short_ema'] = stock_data[[f'EMA{period}' for period in short_terms]].mean(axis=1)
        stock_data['avg_long_ema'] = stock_data[[f'EMA{period}' for period in long_terms]].mean(axis=1)
        
        # Detect crossovers
        stock_data['short_above_long'] = stock_data['avg_short_ema'] > stock_data['avg_long_ema']
        stock_data['crossover'] = False
        
        # Find crossover points - FIX: Use loc[] instead of chained assignment
        for i in range(1, len(stock_data)):
            if not stock_data['short_above_long'].iloc[i-1] and stock_data['short_above_long'].iloc[i]:
                # Replace: stock_data['crossover'].iloc[i] = True
                stock_data.loc[stock_data.index[i], 'crossover'] = True
        
        # Check if there's a crossover in the last 'days_to_check' days
        recent_data = stock_data.iloc[-days_to_check:]
        has_crossover = recent_data['crossover'].any()
        
        return has_crossover, stock_data if has_crossover else None
    except Exception as e:
        print(f"Error checking {ticker}: {str(e)}")
        return False, None

# Add a caching mechanism for expensive API calls with local file support
@st.cache_data(ttl=3600)  # Cache data for 1 hour in Streamlit's cache
def fetch_industry_data():
    """Fetch and cache all industry data, using local file when possible"""
    try:
        # Define directory for cache files
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Find the most recent industry cache file
        cache_files = [f for f in os.listdir(cache_dir) if f.startswith('industry_data_') and f.endswith('.json')]
        latest_file = None
        is_cache_valid = False
        
        if cache_files:
            # Get the most recent file
            cache_files.sort(reverse=True)  # Sort by filename (which includes date)
            latest_file = os.path.join(cache_dir, cache_files[0])
            
            # Extract date from filename (industry_data_YYYY-MM-DD.json)
            try:
                file_date_str = cache_files[0].replace('industry_data_', '').replace('.json', '')
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                # Check if file is less than 2 months old
                is_cache_valid = (datetime.now() - file_date).days < 60
            except:
                is_cache_valid = False
        
        # Load from cache file if valid
        if is_cache_valid and latest_file and os.path.exists(latest_file):
            progress_text = st.empty()
            progress_text.text("从本地缓存加载行业数据...")
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            progress_text.empty()
            return cached_data
        
        # If cache is invalid or doesn't exist, fetch fresh data
        progress_text = st.empty()
        progress_text.text("正在从服务器获取行业数据...")
        
        # Get all industry names
        industry_df = ak.stock_board_industry_name_em()
        industry_list = industry_df["板块名称"].tolist()
        
        # Create dictionaries to store industry data
        industry_stocks = {}  # Industry -> List of stocks
        industry_counts = {}  # Industry -> Count of stocks
        stock_to_industry = {}  # Stock code -> Industry
        
        # Get stock data for each industry
        for i, industry in enumerate(industry_list):
            progress_text.text(f"正在获取行业数据: {i+1}/{len(industry_list)} - {industry}")
            try:
                # Fetch stocks in this industry
                industry_stocks_df = ak.stock_board_industry_cons_em(symbol=industry)
                if not industry_stocks_df.empty:
                    # Process stocks in this industry
                    stocks_list = []
                    for _, row in industry_stocks_df.iterrows():
                        stock_code = row["代码"].split('.')[0].zfill(6)
                        stock_name = row["名称"]
                        stocks_list.append((stock_code, stock_name))
                        # Map stock code to industry
                        stock_to_industry[stock_code] = industry
                    
                    # Store data
                    industry_stocks[industry] = stocks_list
                    industry_counts[industry] = len(stocks_list)
                else:
                    industry_stocks[industry] = []
                    industry_counts[industry] = 0
            except Exception as e:
                print(f"Error fetching stocks for {industry}: {str(e)}")
                industry_stocks[industry] = []
                industry_counts[industry] = 0
        
        # Prepare data structure for caching
        industry_data = {
            "industry_list": industry_list,
            "industry_stocks": industry_stocks,
            "industry_counts": industry_counts,
            "stock_to_industry": stock_to_industry,
            "fetch_date": datetime.now().strftime('%Y-%m-%d')
        }
        
        # Save to a new cache file with current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        cache_file = os.path.join(cache_dir, f'industry_data_{current_date}.json')
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(industry_data, f, ensure_ascii=False, indent=2)
        
        # Clean up old cache files (keep only the most recent 3)
        cache_files = [f for f in os.listdir(cache_dir) if f.startswith('industry_data_') and f.endswith('.json')]
        if len(cache_files) > 3:
            cache_files.sort()  # Sort by date ascending
            for old_file in cache_files[:-3]:  # Remove all but the 3 most recent
                try:
                    os.remove(os.path.join(cache_dir, old_file))
                except:
                    pass
        
        progress_text.empty()
        return industry_data
    
    except Exception as e:
        st.error(f"获取行业数据失败: {str(e)}")
        return {
            "industry_list": [],
            "industry_stocks": {},
            "industry_counts": {},
            "stock_to_industry": {},
            "fetch_date": datetime.now().strftime('%Y-%m-%d')
        }

# Sidebar options
st.sidebar.title("分析模式")
analysis_mode = st.sidebar.radio("选择模式", ["自动扫描买入信号","单一股票分析"])

if analysis_mode == "单一股票分析":
    # Single stock analysis mode - similar to the original code
    st.sidebar.title("股票输入")
    ticker = st.sidebar.text_input("输入 6 位股票代码（例如，000001）", "000001")
    
    st.sidebar.title("显示选项")
    show_short_term = st.sidebar.checkbox("显示短期 EMA", value=True)
    show_long_term = st.sidebar.checkbox("显示长期 EMA", value=True)
    
    # Calculate date range for the past 6 months
    end_date = datetime.today().strftime('%Y%m%d')
    start_date = (datetime.today() - timedelta(days=180)).strftime('%Y%m%d')
    
    # Fetch and process stock data
    with st.spinner("获取数据中..."):
        try:
            # Remove exchange suffix if present (e.g., '000001.SZ' -> '000001')
            ticker = ticker.split('.')[0]
            if not ticker.isdigit() or len(ticker) != 6:
                st.error("请输入有效的 6 位股票代码。")
            else:
                # Fetch stock data using akshare
                stock_data = ak.stock_zh_a_hist(symbol=ticker, period="daily", start_date=start_date, end_date=end_date, adjust="")
                if stock_data.empty:
                    st.error("未找到所输入股票代码的数据。请检查代码并重试。")
                else:
                    # Rename columns from Chinese to English
                    stock_data.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open'}, inplace=True)
                    # Set 'date' as index and sort by date
                    stock_data['date'] = pd.to_datetime(stock_data['date'])
                    stock_data.set_index('date', inplace=True)
                    stock_data.sort_index(inplace=True)
                    
                    # Calculate Exponential Moving Averages (EMAs)
                    for period in [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]:
                        stock_data[f"EMA{period}"] = stock_data["close"].ewm(span=period, adjust=False).mean()
                    
                    # Define short-term and long-term EMAs
                    short_terms = [3, 5, 8, 10, 12, 15]
                    long_terms = [30, 35, 40, 45, 50, 60]
                    
                    # Calculate average of short-term and long-term EMAs for each day
                    stock_data['avg_short_ema'] = stock_data[[f'EMA{period}' for period in short_terms]].mean(axis=1)
                    stock_data['avg_long_ema'] = stock_data[[f'EMA{period}' for period in long_terms]].mean(axis=1)
                    
                    # Detect crossovers (short-term crossing above long-term)
                    stock_data['short_above_long'] = stock_data['avg_short_ema'] > stock_data['avg_long_ema']
                    stock_data['crossover'] = False
                    
                    # Find the exact crossover points (when short_above_long changes from False to True)
                    for i in range(1, len(stock_data)):
                        if not stock_data['short_above_long'].iloc[i-1] and stock_data['short_above_long'].iloc[i]:
                            # Replace: stock_data['crossover'].iloc[i] = True
                            stock_data.loc[stock_data.index[i], 'crossover'] = True
                    
                    # Create Plotly figure
                    fig = go.Figure()
                    
                    # Add candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=stock_data.index,
                        open=stock_data["open"],
                        high=stock_data[["open", "close"]].max(axis=1),
                        low=stock_data[["open", "close"]].min(axis=1),
                        close=stock_data["close"],
                        increasing_line_color='green',
                        decreasing_line_color='red',
                        name="Price"
                    ))
                    
                    # Add short-term EMAs (blue)
                    if show_short_term:
                        for i, period in enumerate([3, 5, 8, 10, 12, 15]):
                            fig.add_trace(go.Scatter(
                                x=stock_data.index,
                                y=stock_data[f"EMA{period}"],
                                mode="lines",
                                name=f"EMA{period}",
                                line=dict(color="blue", width=1),
                                legendgroup="short_term",
                                showlegend=(i == 0)
                            ))
                    
                    # Add long-term EMAs (red)
                    if show_long_term:
                        for i, period in enumerate([30, 35, 40, 45, 50, 60]):
                            fig.add_trace(go.Scatter(
                                x=stock_data.index,
                                y=stock_data[f"EMA{period}"],
                                mode="lines",
                                name=f"EMA{period}",
                                line=dict(color="red", width=1),
                                legendgroup="long_term",
                                showlegend=(i == 0)
                            ))
                    
                    # Add average short-term and long-term EMAs to visualize crossover
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['avg_short_ema'],
                        mode="lines",
                        name="Avg Short-term EMAs",
                        line=dict(color="blue", width=2, dash='dot'),
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['avg_long_ema'],
                        mode="lines",
                        name="Avg Long-term EMAs",
                        line=dict(color="red", width=2, dash='dot'),
                    ))
                    
                    # Mark crossover signals on the chart
                    crossover_dates = stock_data[stock_data['crossover']].index
                    for date in crossover_dates:
                        price_at_crossover = stock_data.loc[date, 'close']
                        # Add vertical line at crossover
                        fig.add_shape(
                            type="line",
                            x0=date,
                            y0=price_at_crossover * 0.97,
                            x1=date,
                            y1=price_at_crossover * 1.03,
                            line=dict(color="green", width=3),
                        )
                        # Add annotation
                        fig.add_annotation(
                            x=date,
                            y=price_at_crossover * 1.04,
                            text="买入信号",
                            showarrow=True,
                            arrowhead=1,
                            arrowcolor="green",
                            arrowsize=1,
                            arrowwidth=2,
                            font=dict(color="green", size=12)
                        )
                    
                    # Count and display the number of signals
                    signal_count = len(crossover_dates)
                    if signal_count > 0:
                        last_signal = crossover_dates[-1].strftime('%Y-%m-%d') if signal_count > 0 else "None"
                        signal_info = f"**买入信号**: 共 {signal_count} 个, 最近信号日期: {last_signal}"
                        fig.add_annotation(
                            x=0.02,
                            y=0.98,
                            xref="paper",
                            yref="paper",
                            text=signal_info,
                            showarrow=False,
                            font=dict(size=14, color="green"),
                            bgcolor="white",
                            bordercolor="green",
                            borderwidth=1,
                            align="left"
                        )
                    
                    # Customize plot layout
                    fig.update_layout(
                        title=f"股票 {ticker} GMMA 图表 (标记: 短期EMA从下方穿过长期EMA)",
                        xaxis_title="日期",
                        yaxis_title="价格",
                        legend_title="图例",
                        hovermode="x unified",
                        template="plotly_white"
                    )
                    
                    # Display the plot in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display crossover days in a table
                    if len(crossover_dates) > 0:
                        st.subheader("买入信号日期")
                        signal_df = pd.DataFrame(crossover_dates, columns=["日期"])
                        signal_df["日期"] = signal_df["日期"].dt.strftime('%Y-%m-%d')
                        st.table(signal_df)
        except Exception as e:
            st.error(f"获取数据时出错: {str(e)}")

else:  # Auto scan mode
    st.sidebar.title("扫描设置")
    days_to_check = st.sidebar.slider("检查最近几天内的信号", 1, 7, 3)
    max_stocks = st.sidebar.slider("最多显示股票数量", 1, 200, 10)
    
    # Add industry selection option
    scan_mode = st.sidebar.radio("扫描范围", ["按行业板块","全部 A 股"])
    
    selected_industry = None
    
    # Industry board selection
    if scan_mode == "按行业板块":
        try:
            # Fetch all industry data once (cached)
            with st.spinner("获取行业板块数据..."):
                industry_data = fetch_industry_data()
                industry_list = industry_data["industry_list"]
                industry_counts = industry_data["industry_counts"]
                industry_stocks = industry_data["industry_stocks"]
                
                # Get fresh industry price change data directly from API
                fresh_industry_df = ak.stock_board_industry_name_em()
                industry_change_pct = {row["板块名称"]: row["涨跌幅"] 
                                       for _, row in fresh_industry_df.iterrows()}
            
            # Remove loading message from sidebar
            st.sidebar.text("")
            
            # Format options to include stock count and 涨跌幅: "行业名 (123股) ↑2.50%"
            industry_options = []
            for ind in industry_list:
                count = industry_counts[ind]
                # Get the latest change percentage for this industry
                change_pct = industry_change_pct.get(ind, 0)
                
                # Format percentage with arrow indicator and color
                if change_pct > 0:
                    pct_str = f"↑{change_pct:.2f}%"
                elif change_pct < 0:
                    pct_str = f"↓{abs(change_pct):.2f}%"
                else:
                    pct_str = f"0.00%"
                    
                # Create the formatted option
                option = f"{ind} ({count}股) {pct_str}"
                industry_options.append(option)
            
            # Create a mapping from formatted name back to original name
            industry_name_mapping = {option: ind for ind, option in zip(industry_list, industry_options)}
            
            # Get fresh industry order directly from the API for sorting
            try:
                ordered_industry_list = fresh_industry_df["板块名称"].tolist()
                
                # Create a mapping for sorting based on the original order
                industry_order = {ind: idx for idx, ind in enumerate(ordered_industry_list)}
                
                # Sort industry_options based on the original order
                industry_options_sorted = sorted(
                    industry_options,
                    key=lambda x: industry_order.get(industry_name_mapping[x], float('inf'))
                )
            except Exception as e:
                # Fallback to unsorted if API call fails
                st.sidebar.warning(f"无法获取行业排序，将使用默认顺序: {str(e)}")
                industry_options_sorted = industry_options
            
            # Use the sorted options in the multiselect
            default_option = industry_options_sorted[0] if industry_options_sorted else None
            selected_industry_options = st.sidebar.multiselect(
                "选择行业板块 (可多选)",
                options=industry_options_sorted,
                default=[default_option] if default_option else []
            )
            
            # Convert the selected formatted options back to original industry names
            selected_industries = [industry_name_mapping[opt] for opt in selected_industry_options]
            
            # Show the selected industry info
            if selected_industries:
                total_stocks = sum(industry_counts[ind] for ind in selected_industries)
                industries_text = ", ".join(selected_industries)
                st.sidebar.info(f"已选择: {industries_text}\n\n共计约 {total_stocks} 只股票")
        except Exception as e:
            st.sidebar.error(f"获取行业板块失败: {str(e)}")

    if st.sidebar.button("开始扫描"):
        with st.spinner("正在扫描买入信号，这可能需要一些时间..."):
            try:
                # Variable to track if we have valid stocks to scan
                have_stocks_to_scan = True
                
                # Get stock list based on scan mode
                if scan_mode == "按行业板块" and selected_industries:
                    # Create an empty DataFrame to store all industry stocks
                    all_industry_stocks_list = []
                    
                    # Use cached data instead of making new API calls
                    for industry in selected_industries:
                        if industry in industry_stocks:
                            all_industry_stocks_list.extend(industry_stocks[industry])
                    
                    if all_industry_stocks_list:
                        # Convert to DataFrame
                        stock_info_df = pd.DataFrame(all_industry_stocks_list, columns=["code", "name"])
                        # Remove duplicates
                        stock_info_df = stock_info_df.drop_duplicates(subset=["code"])
                    else:
                        st.error("未能获取所选行业的股票列表。")
                        have_stocks_to_scan = False
                else:
                    # Get all A-share stock codes and names
                    stock_info_df = ak.stock_info_a_code_name()
                
                # Only proceed if we have stocks to scan
                if have_stocks_to_scan:
                    # Show how many stocks will be scanned
                    stock_count = len(stock_info_df)
                    st.info(f"准备扫描 {stock_count} 只股票...")
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Container for results
                    results_container = st.container()
                    results_title = st.empty()
                    
                    # Counter for stocks with crossover
                    crossover_stocks = []
                    
                    # Create industry mapping dictionary for multiple industry case
                    industry_mapping = {}
                    
                    # If in industry mode, map stock codes to their industries
                    if scan_mode == "按行业板块":
                        # Create industry mapping from the cached data
                        industry_mapping = industry_data["stock_to_industry"]
                    
                    # Loop through selected stocks
                    for i, row in enumerate(stock_info_df.itertuples()):
                        # Update progress
                        progress_bar.progress(min((i+1)/stock_count, 1.0))
                        
                        ticker = row.code
                        name = row.name
                        
                        # Skip stocks with special prefixes only if scanning all stocks
                        if scan_mode == "全部 A 股" and ticker.startswith(('688', '300', '8', '4')):
                            continue
                            
                        # Check for crossover
                        has_crossover, stock_data = has_recent_crossover(ticker, days_to_check)
                        
                        if has_crossover:
                            # Get industry information for the stock
                            if scan_mode == "按行业板块":
                                # Use the mapped industry from the cached data
                                industry = industry_mapping.get(ticker, "未知")
                            else:
                                # For all A-shares mode, try to get industry info
                                try:
                                    # First check if we have it in the cache
                                    if ticker in industry_data["stock_to_industry"]:
                                        industry = industry_data["stock_to_industry"][ticker]
                                    else:
                                        # If not cached, fetch it directly
                                        stock_info = ak.stock_individual_info_em(symbol=ticker)
                                        industry = stock_info.loc[stock_info['item'] == '所属行业', 'value'].iloc[0]
                                except:
                                    industry = "未知"
                            
                            # Include industry in the crossover_stocks list
                            crossover_stocks.append((ticker, name, industry, stock_data))
                            
                            # Create tab for this stock
                            with st.expander(f"{ticker} - {name} ({industry})", expanded=True):
                                # Create GMMA chart
                                fig = go.Figure()
                                
                                # Add candlestick chart
                                fig.add_trace(go.Candlestick(
                                    x=stock_data.index,
                                    open=stock_data["open"],
                                    high=stock_data[["open", "close"]].max(axis=1),
                                    low=stock_data[["open", "close"]].min(axis=1),
                                    close=stock_data["close"],
                                    increasing_line_color='red',
                                    decreasing_line_color='green',
                                    name="Price"
                                ))
                                
                                # Add short-term EMAs (blue)
                                for i, period in enumerate([3, 5, 8, 10, 12, 15]):
                                    fig.add_trace(go.Scatter(
                                        x=stock_data.index,
                                        y=stock_data[f"EMA{period}"],
                                        mode="lines",
                                        name=f"EMA{period}",
                                        line=dict(color="blue", width=1),
                                        legendgroup="short_term",
                                        showlegend=(i == 0)
                                    ))
                                
                                # Add long-term EMAs (red)
                                for i, period in enumerate([30, 35, 40, 45, 50, 60]):
                                    fig.add_trace(go.Scatter(
                                        x=stock_data.index,
                                        y=stock_data[f"EMA{period}"],
                                        mode="lines",
                                        name=f"EMA{period}",
                                        line=dict(color="red", width=1),
                                        legendgroup="long_term",
                                        showlegend=(i == 0)
                                    ))
                                
                                # Add average EMAs
                                fig.add_trace(go.Scatter(
                                    x=stock_data.index,
                                    y=stock_data['avg_short_ema'],
                                    mode="lines",
                                    name="Avg Short-term EMAs",
                                    line=dict(color="blue", width=2, dash='dot'),
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=stock_data.index,
                                    y=stock_data['avg_long_ema'],
                                    mode="lines",
                                    name="Avg Long-term EMAs",
                                    line=dict(color="red", width=2, dash='dot'),
                                ))
                                
                                # Mark crossover signals
                                crossover_dates = stock_data[stock_data['crossover']].index
                                for date in crossover_dates:
                                    price_at_crossover = stock_data.loc[date, 'close']
                                    # fig.add_shape(
                                    #     type="line",
                                    #     x0=date,
                                    #     y0=price_at_crossover * 0.97,
                                    #     x1=date,
                                    #     y1=price_at_crossover * 1.03,
                                    #     line=dict(color="orange", width=3),
                                    # )
                                    fig.add_annotation(
                                        x=date,
                                        y=price_at_crossover * 1.04,
                                        text="买入信号",
                                        showarrow=True,
                                        arrowhead=1,
                                        arrowcolor="green",
                                        arrowsize=1,
                                        arrowwidth=2,
                                        font=dict(color="green", size=12)
                                    )
                                
                                # Layout
                                fig.update_layout(
                                    title=f"{ticker} - {name} GMMA 图表",
                                    xaxis_title="日期",
                                    yaxis_title="价格",
                                    legend_title="图例",
                                    hovermode="x unified",
                                    template="plotly_white",
                                    height=500
                                )
                                
                                # Display the plot
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Check if we found enough stocks
                        if len(crossover_stocks) >= max_stocks:
                            break
                    
                    # Final update
                    progress_bar.progress(1.0)
                    
                    # Final message
                    if len(crossover_stocks) == 0:
                        st.warning(f"没有找到在最近 {days_to_check} 天内出现买入信号的股票。")
                    else:
                        scan_scope = f"所选 {len(selected_industries)} 个行业" if scan_mode == "按行业板块" else "全部 A 股"
                        st.success(f"在{scan_scope}中找到 {len(crossover_stocks)} 只在最近 {days_to_check} 天内出现买入信号的股票。")
                        
                        # Create a summary table with industry information
                        summary_df = pd.DataFrame(
                            [(t, n, ind) for t, n, ind, _ in crossover_stocks], 
                            columns=["代码", "名称", "所属行业"]
                        )
                        st.subheader("买入信号股票列表")
                        st.table(summary_df)
                
            except Exception as e:
                st.error(f"扫描过程中出错: {str(e)}")
    else:
        if scan_mode == "按行业板块":
            if selected_industries:
                industry_count = len(selected_industries)
                total_stocks = sum(industry_counts.get(ind, 0) for ind in selected_industries)
                industries_text = f"{industry_count} 个行业 (约 {total_stocks} 只股票)" if industry_count > 1 else f"{selected_industries[0]} (约 {industry_counts.get(selected_industries[0], 0)} 只股票)"
                st.info(f"请点击'开始扫描'按钮以查找 {industries_text} 中最近出现买入信号的股票。")
            else:
                st.info("请先选择至少一个行业板块，然后点击'开始扫描'按钮。")
        else:
            st.info("请点击'开始扫描'按钮以查找最近出现买入信号的股票。")
