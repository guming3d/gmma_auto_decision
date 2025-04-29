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
import concurrent.futures
import gc

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

# Cache directory setup
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Function to load stock data with caching
@st.cache_data(ttl=300)  # Cache for 1 hour
def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data with caching to avoid repeated API calls"""
    try:
        # Fetch stock data using akshare
        stock_data = ak.stock_zh_a_hist(symbol=ticker, period="daily", 
                                     start_date=start_date, end_date=end_date, adjust="")
        if stock_data.empty:
            return None
            
        # Rename columns and process data
        stock_data.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open'}, inplace=True)
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data.set_index('date', inplace=True)
        stock_data.sort_index(inplace=True)
        
        return stock_data
    except Exception as e:
        print(f"Error fetching {ticker}: {str(e)}")
        return None

# Function to check if a stock has a recent crossover (optimized)
def has_recent_crossover(ticker, days_to_check=3, history_days=365):
    try:
        # Calculate date range for the past history_days (enough data to calculate EMAs)
        end_date = datetime.today().strftime('%Y%m%d')
        # Get more historical data - add 30% more days for proper EMA calculation
        calculation_days = int(history_days * 1.3)
        start_date = (datetime.today() - timedelta(days=calculation_days)).strftime('%Y%m%d')
        
        # Fetch stock data using cached function
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        if stock_data is None or stock_data.empty:
            return False, None
            
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
        
        # Find crossover points - Use loc[] instead of chained assignment
        for i in range(1, len(stock_data)):
            if not stock_data['short_above_long'].iloc[i-1] and stock_data['short_above_long'].iloc[i]:
                stock_data.loc[stock_data.index[i], 'crossover'] = True
        
        # Check if there's a crossover in the last 'days_to_check' days
        recent_data = stock_data.iloc[-days_to_check:]
        has_crossover = recent_data['crossover'].any()
        
        # If no crossover, return early and free memory
        if not has_crossover:
            return False, None
            
        return has_crossover, stock_data
    except Exception as e:
        print(f"Error checking {ticker}: {str(e)}")
        return False, None

# Add a caching mechanism for expensive API calls with local file support
def fetch_industry_data():
    """Fetch and cache all industry data, using local file when possible"""
    try:
        # Find the most recent industry cache file
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.startswith('industry_data_') and f.endswith('.json')]
        latest_file = None
        is_cache_valid = False
        
        if cache_files:
            # Get the most recent file
            cache_files.sort(reverse=True)  # Sort by filename (which includes date)
            latest_file = os.path.join(CACHE_DIR, cache_files[0])
            
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
        cache_file = os.path.join(CACHE_DIR, f'industry_data_{current_date}.json')
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(industry_data, f, ensure_ascii=False, indent=2)
        
        # Clean up old cache files (keep only the most recent 3)
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.startswith('industry_data_') and f.endswith('.json')]
        if len(cache_files) > 3:
            cache_files.sort()  # Sort by date ascending
            for old_file in cache_files[:-3]:  # Remove all but the 3 most recent
                try:
                    os.remove(os.path.join(CACHE_DIR, old_file))
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

# Process batch of stocks to identify crossovers
def process_stock_batch(stock_batch, days_to_check, industry_mapping=None, display_days=365):
    crossover_stocks = []
    
    for ticker, name in stock_batch:
        # Skip stocks with special prefixes 
        if ticker.startswith(('688', '300', '8', '4')):
            continue
            
        # Check for crossover
        has_crossover, stock_data = has_recent_crossover(ticker, days_to_check, history_days=display_days)
        
        if has_crossover:
            # Get industry information for the stock
            industry = industry_mapping.get(ticker, "未知") if industry_mapping else "未知"
            
            # Include industry in the crossover_stocks list
            crossover_stocks.append((ticker, name, industry, stock_data))
    
    return crossover_stocks

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
    
    # Add history length selection dropdown
    history_options = {
        "10 年": 3650,
        "5 年": 1825,
        "2 年": 730,
        "12 个月": 365,
        "6 个月": 180,
        "3 个月": 90
    }
    selected_history = st.sidebar.selectbox(
        "历史数据长度",
        options=list(history_options.keys()),
        index=3  # Default to 12 months
    )
    
    # Calculate date range based on selected period
    end_date = datetime.today().strftime('%Y%m%d')
    # Set display period to the selected period
    display_days = history_options[selected_history]
    # Add extra time for calculation (30% more) to ensure proper EMA calculation
    calculation_days = int(display_days * 1.3)
    
    display_start_date = (datetime.today() - timedelta(days=display_days)).strftime('%Y%m%d')
    calculation_start_date = (datetime.today() - timedelta(days=calculation_days)).strftime('%Y%m%d')
    
    # Fetch and process stock data
    with st.spinner("获取数据中..."):
        try:
            # Remove exchange suffix if present (e.g., '000001.SZ' -> '000001')
            ticker = ticker.split('.')[0]
            if not ticker.isdigit() or len(ticker) != 6:
                st.error("请输入有效的 6 位股票代码。")
            else:
                # Fetch stock data using akshare with extended history for proper EMA calculation
                stock_data = fetch_stock_data(ticker, calculation_start_date, end_date)
                if stock_data is None or stock_data.empty:
                    st.error("未找到所输入股票代码的数据。请检查代码并重试。")
                else:
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
                            stock_data.loc[stock_data.index[i], 'crossover'] = True
                    
                    # Only display the last 6 months in the chart
                    display_date = pd.to_datetime(display_start_date)
                    display_data = stock_data[stock_data.index >= display_date]
                    
                    # Create Plotly figure
                    fig = go.Figure()
                    
                    # Add candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=display_data.index,
                        open=display_data["open"],
                        high=display_data[["open", "close"]].max(axis=1),
                        low=display_data[["open", "close"]].min(axis=1),
                        close=display_data["close"],
                        increasing_line_color='green',
                        decreasing_line_color='red',
                        name="Price"
                    ))
                    
                    # Add short-term EMAs (blue)
                    if show_short_term:
                        for i, period in enumerate([3, 5, 8, 10, 12, 15]):
                            fig.add_trace(go.Scatter(
                                x=display_data.index,
                                y=display_data[f"EMA{period}"],
                                mode="lines",
                                name=f"EMA{period}",
                                line=dict(color="skyblue", width=1),
                                legendgroup="short_term",
                                showlegend=(i == 0)
                            ))
                    
                    # Add long-term EMAs (red)
                    if show_long_term:
                        for i, period in enumerate([30, 35, 40, 45, 50, 60]):
                            fig.add_trace(go.Scatter(
                                x=display_data.index,
                                y=display_data[f"EMA{period}"],
                                mode="lines",
                                name=f"EMA{period}",
                                line=dict(color="lightcoral", width=1),
                                legendgroup="long_term",
                                showlegend=(i == 0)
                            ))
                    
                    # Add average short-term and long-term EMAs to visualize crossover
                    fig.add_trace(go.Scatter(
                        x=display_data.index,
                        y=display_data['avg_short_ema'],
                        mode="lines",
                        name="Avg Short-term EMAs",
                        line=dict(color="blue", width=2, dash='dot'),
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=display_data.index,
                        y=display_data['avg_long_ema'],
                        mode="lines",
                        name="Avg Long-term EMAs",
                        line=dict(color="red", width=2, dash='dot'),
                    ))
                    
                    # Mark crossover signals on the chart
                    crossover_dates = display_data[display_data['crossover']].index
                    for date in crossover_dates:
                        price_at_crossover = display_data.loc[date, 'close']
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
                        title=f"股票 {ticker} GMMA 图表 ({selected_history} 历史数据, 标记: 短期EMA从下方穿过长期EMA)",
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
    days_to_check = st.sidebar.slider("检查最近几天内的信号", 1, 7, 1)
    max_stocks = st.sidebar.slider("最多显示股票数量", 1, 500, 200)
    batch_size = st.sidebar.slider("批处理大小 (较小值更省内存)", 10, 100, 50)
    
    # Add history length selection dropdown for chart display
    history_options = {
        "10 年": 3650,
        "5 年": 1825,
        "2 年": 730,
        "12 个月": 365,
        "6 个月": 180,
        "3 个月": 90
    }
    selected_history = st.sidebar.selectbox(
        "历史数据长度",
        options=list(history_options.keys()),
        index=3  # Default to 12 months
    )
    display_days = history_options[selected_history]
    
    # Add more filtering options
    st.sidebar.title("筛选选项")
    exclude_st = st.sidebar.checkbox("排除ST股票", value=False)
    exclude_new = st.sidebar.checkbox("排除上市不足3个月的股票", value=False)
    exclude_688 = st.sidebar.checkbox("排除科创板股票 (688开头)", value=False)
    exclude_300 = st.sidebar.checkbox("排除创业板股票 (300开头)", value=False)
    
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
                    # Apply filtering based on user preferences
                    if exclude_st:
                        stock_info_df = stock_info_df[~stock_info_df["name"].str.contains("ST", case=False)]
                    
                    # Apply code-based filtering
                    if exclude_688:
                        stock_info_df = stock_info_df[~stock_info_df["code"].str.startswith("688")]
                    
                    if exclude_300:
                        stock_info_df = stock_info_df[~stock_info_df["code"].str.startswith("300")]
                    
                    # Show how many stocks will be scanned
                    stock_count = len(stock_info_df)
                    st.info(f"准备扫描 {stock_count} 只股票，分批处理以优化内存使用...")
                    
                    # Create UI elements for real-time updates
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_count = st.empty()
                    
                    # Create containers for results that will be updated with each batch
                    st.subheader("买入信号股票列表")
                    table_container = st.empty()
                    
                    # Create an expander section for all charts
                    st.subheader("GMMA 图表 (实时更新)")
                    charts_container = st.container()
                    
                    # Counter for stocks with crossover
                    crossover_stocks = []
                    
                    # If in industry mode, map stock codes to their industries
                    industry_mapping = industry_data["stock_to_industry"] if scan_mode == "按行业板块" else {}
                    
                    # Convert DataFrame to list for batch processing
                    stocks_to_scan = list(zip(stock_info_df["code"], stock_info_df["name"]))
                    
                    # Calculate total number of batches
                    num_batches = (len(stocks_to_scan) + batch_size - 1) // batch_size
                    
                    # Process stocks in batches to limit memory usage
                    for batch_idx in range(num_batches):
                        # Update progress and status
                        progress_bar.progress(min(batch_idx / num_batches, 1.0))
                        batch_start = batch_idx * batch_size
                        batch_end = min((batch_idx + 1) * batch_size, len(stocks_to_scan))
                        current_batch = stocks_to_scan[batch_start:batch_end]
                        
                        status_text.text(f"处理批次 {batch_idx+1}/{num_batches}，股票 {batch_start+1}-{batch_end}/{stock_count}")
                        
                        # Process current batch
                        batch_results = process_stock_batch(current_batch, days_to_check, industry_mapping, display_days)
                        
                        # If we found new stocks with crossovers
                        if batch_results:
                            # Add new results to our master list
                            crossover_stocks.extend(batch_results)
                            
                            # Update the results counter
                            results_count.info(f"已找到 {len(crossover_stocks)} 只出现买入信号的股票")
                            
                            # Update summary table with all found stocks
                            summary_df = pd.DataFrame(
                                [(t, n, ind) for t, n, ind, _ in crossover_stocks], 
                                columns=["代码", "名称", "所属行业"]
                            )
                            table_container.dataframe(summary_df)
                            
                            # Display charts for new stocks in this batch
                            with charts_container:
                                for ticker, name, industry, stock_data in batch_results:
                                    with st.expander(f"{ticker} - {name} ({industry})", expanded=True):
                                        # Create GMMA chart
                                        fig = go.Figure()
                                        
                                        # Filter to display based on selected history length
                                        display_start_date = (datetime.today() - timedelta(days=display_days))
                                        display_data = stock_data[stock_data.index >= display_start_date]
                                        
                                        # Add candlestick chart
                                        fig.add_trace(go.Candlestick(
                                            x=display_data.index,
                                            open=display_data["open"],
                                            high=display_data[["open", "close"]].max(axis=1),
                                            low=display_data[["open", "close"]].min(axis=1),
                                            close=display_data["close"],
                                            increasing_line_color='red',
                                            decreasing_line_color='green',
                                            name="Price"
                                        ))
                                        
                                        # Add short-term EMAs (blue)
                                        for i, period in enumerate([3, 5, 8, 10, 12, 15]):
                                            fig.add_trace(go.Scatter(
                                                x=display_data.index,
                                                y=display_data[f"EMA{period}"],
                                                mode="lines",
                                                name=f"EMA{period}",
                                                line=dict(color="skyblue", width=1),
                                                legendgroup="short_term",
                                                showlegend=(i == 0)
                                            ))
                                        
                                        # Add long-term EMAs (red)
                                        for i, period in enumerate([30, 35, 40, 45, 50, 60]):
                                            fig.add_trace(go.Scatter(
                                                x=display_data.index,
                                                y=display_data[f"EMA{period}"],
                                                mode="lines",
                                                name=f"EMA{period}",
                                                line=dict(color="lightcoral", width=1),
                                                legendgroup="long_term",
                                                showlegend=(i == 0)
                                            ))
                                        
                                        # Add average EMAs
                                        fig.add_trace(go.Scatter(
                                            x=display_data.index,
                                            y=display_data['avg_short_ema'],
                                            mode="lines",
                                            name="Avg Short-term EMAs",
                                            line=dict(color="blue", width=2, dash='dot'),
                                        ))
                                        
                                        fig.add_trace(go.Scatter(
                                            x=display_data.index,
                                            y=display_data['avg_long_ema'],
                                            mode="lines",
                                            name="Avg Long-term EMAs",
                                            line=dict(color="red", width=2, dash='dot'),
                                        ))
                                        
                                        # Mark crossover signals
                                        crossover_dates = display_data[display_data['crossover']].index
                                        for date in crossover_dates:
                                            price_at_crossover = display_data.loc[date, 'close']
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
                                            title=f"{ticker} - {name} GMMA 图表 ({selected_history} 历史数据)",
                                            xaxis_title="日期",
                                            yaxis_title="价格",
                                            legend_title="图例",
                                            hovermode="x unified",
                                            template="plotly_white",
                                            height=800
                                        )
                                        
                                        # Display the plot
                                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Check if we found enough stocks
                        if len(crossover_stocks) >= max_stocks:
                            status_text.text(f"已找到足够数量的股票 ({max_stocks}只)，提前停止扫描...")
                            break
                            
                        # Force garbage collection to free memory
                        gc.collect()
                    
                    # Final update
                    progress_bar.progress(1.0)
                    status_text.empty()
                    
                    # Final message
                    if len(crossover_stocks) == 0:
                        st.warning(f"没有找到在最近 {days_to_check} 天内出现买入信号的股票。")
                    else:
                        scan_scope = f"所选 {len(selected_industries)} 个行业" if scan_mode == "按行业板块" else "全部 A 股"
                        st.success(f"扫描完成：在{scan_scope}中共找到 {len(crossover_stocks)} 只在最近 {days_to_check} 天内出现买入信号的股票。")
                
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
            st.info("请点击'开始扫描'按钮以查找最近出现买入信号的股票。\n\n注意：全部A股扫描可能需要较长时间，建议先尝试按行业板块扫描。")
