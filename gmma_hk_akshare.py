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
    page_title="GMMA 港股股票分析工具",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("顾比多重移动平均线 (GMMA) 港股图表")
st.markdown("""
此应用程序显示使用 akshare 数据的中国港股票的顾比多重移动平均线 (GMMA) 图表。  
可以分析单个股票或自动扫描最近出现买入信号的股票。
""")

# Function to get start date based on selected duration
def get_start_date(duration):
    today = datetime.today()
    if duration == "10年":
        return (today - timedelta(days=365*10)).strftime('%Y%m%d')
    elif duration == "5年":
        return (today - timedelta(days=365*5)).strftime('%Y%m%d')
    elif duration == "3年":
        return (today - timedelta(days=365*3)).strftime('%Y%m%d')
    elif duration == "2年":
        return (today - timedelta(days=365*2)).strftime('%Y%m%d')
    elif duration == "1年":
        return (today - timedelta(days=365)).strftime('%Y%m%d')
    elif duration == "6个月":
        return (today - timedelta(days=180)).strftime('%Y%m%d')
    elif duration == "3个月":
        return (today - timedelta(days=90)).strftime('%Y%m%d')
    else:
        return (today - timedelta(days=180)).strftime('%Y%m%d')  # Default to 6 months

# Function to check if a stock has a recent crossover
def has_recent_crossover(ticker, days_to_check=3, market="A", duration="6个月"):
    try:
        # Calculate date range based on selected duration
        end_date = datetime.today().strftime('%Y%m%d')
        start_date = get_start_date(duration)
        
        # For accurate EMA calculation, we need additional historical data
        # Calculate a buffer period to ensure proper initialization of longer period EMAs
        max_ema_period = 60  # The longest EMA period used
        today = datetime.today()
        
        # Convert start_date to datetime for adjustment
        start_date_dt = datetime.strptime(start_date, '%Y%m%d')
        
        # Add buffer period for EMA calculation (3x the max EMA period for proper initialization)
        buffer_days = max_ema_period * 3
        extended_start_date = (start_date_dt - timedelta(days=buffer_days)).strftime('%Y%m%d')
        
        # Fetch stock data using akshare based on market with extended period
        if market == "HK":
            stock_data = ak.stock_hk_hist(symbol=ticker, period="daily", 
                                         start_date=extended_start_date, end_date=end_date, adjust="")
        else:
            stock_data = ak.stock_zh_a_hist(symbol=ticker, period="daily", 
                                         start_date=extended_start_date, end_date=end_date, adjust="")
                                         
        if stock_data.empty:
            return False, None
            
        # Rename columns and process data
        stock_data.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open'}, inplace=True)
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data.set_index('date', inplace=True)
        stock_data.sort_index(inplace=True)
        
        # Calculate EMAs with proper initialization period
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
                stock_data.loc[stock_data.index[i], 'crossover'] = True
        
        # Filter to only include the originally requested date range for display/analysis
        display_data = stock_data[stock_data.index >= pd.to_datetime(start_date)]
        
        # Check if there's a crossover in the last 'days_to_check' days
        recent_data = display_data.iloc[-days_to_check:]
        has_crossover = recent_data['crossover'].any()
        
        return has_crossover, display_data if has_crossover else None
    except Exception as e:
        print(f"Error checking {ticker}: {str(e)}")
        return False, None


# Sidebar options
st.sidebar.title("分析模式")
analysis_mode = st.sidebar.radio("选择模式", ["港股扫描买入信号","单一股票分析", "知名港股全显示" ])

# Add duration selection to sidebar (common for both modes)
st.sidebar.title("历史数据范围")
data_duration = st.sidebar.selectbox(
    "选择历史数据时长",
    ["10年", "5年", "3年", "2年", "1年", "6个月", "3个月"],
    index=5  # Default to 6 months
)

if analysis_mode == "港股扫描买入信号":
    st.sidebar.title("港股扫描买入信号设置")
    hk_days_to_check = st.sidebar.slider("检查最近几天内的信号", 1, 7, 1)
    hk_max_stocks = st.sidebar.slider("最多显示股票数量", 1, 200, 200)
    
    if st.sidebar.button("开始扫描港股"):
        with st.spinner("正在扫描港股买入信号，这可能需要一些时间..."):
            try:
                # Get all HK stocks using akshare
                hk_stocks_df = ak.stock_hk_famous_spot_em()
                # hk_stocks_df = ak.stock_hk_spot()
                # print the length of the dataframe
                print(len(hk_stocks_df))
                
                st.info(f"准备扫描 {len(hk_stocks_df)} 只港股...")
                
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Container for results
                crossover_stocks = []
                
                # Loop through all HK stocks
                for i, row in hk_stocks_df.iterrows():
                    # Update progress
                    progress_bar.progress(min((i+1)/len(hk_stocks_df), 1.0))
                    
                    ticker = row['代码']
                    name = row['名称']
                    
                    # Check for crossover using our modified function with HK market parameter and duration
                    has_crossover, display_data = has_recent_crossover(ticker, hk_days_to_check, market="HK", duration=data_duration)
                    
                    if has_crossover:
                        # Add to crossover list
                        crossover_stocks.append((ticker, name, display_data))
                        
                        # Create expander for this stock
                        with st.expander(f"{ticker} - {name}", expanded=True):
                            # Create GMMA chart
                            fig = go.Figure()
                            
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
                            for j, period in enumerate([3, 5, 8, 10, 12, 15]):
                                fig.add_trace(go.Scatter(
                                    x=display_data.index,
                                    y=display_data[f"EMA{period}"],
                                    mode="lines",
                                    name=f"EMA{period}",
                                    line=dict(color="skyblue", width=1),
                                    legendgroup="short_term",
                                    showlegend=(j == 0)
                                ))
                            
                            # Add long-term EMAs (red)
                            for j, period in enumerate([30, 35, 40, 45, 50, 60]):
                                fig.add_trace(go.Scatter(
                                    x=display_data.index,
                                    y=display_data[f"EMA{period}"],
                                    mode="lines",
                                    name=f"EMA{period}",
                                    line=dict(color="lightcoral", width=1),
                                    legendgroup="long_term",
                                    showlegend=(j == 0)
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
                                title=f"{ticker} - {name} 港股 GMMA 图表",
                                xaxis_title="日期",
                                yaxis_title="价格",
                                legend_title="图例",
                                hovermode="x unified",
                                template="plotly_white",
                                height=800
                            )
                            
                            # Display the plot
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Check if we have found enough stocks
                    if len(crossover_stocks) >= hk_max_stocks:
                        break
                
                # Final update
                progress_bar.progress(1.0)
                
                # Display summary
                if len(crossover_stocks) == 0:
                    st.warning(f"没有找到在最近 {hk_days_to_check} 天内出现买入信号的港股。")
                else:
                    st.success(f"找到 {len(crossover_stocks)} 只在最近 {hk_days_to_check} 天内出现买入信号的港股。")
                    
                    # Create a summary table
                    summary_df = pd.DataFrame(
                        [(t, n) for t, n, _ in crossover_stocks], 
                        columns=["代码", "名称"]
                    )
                    st.subheader("港股买入信号列表")
                    st.table(summary_df)
            
            except Exception as e:
                st.error(f"扫描港股过程中出错: {str(e)}")
    else:
        st.info("请点击'开始扫描港股'按钮以查找最近出现买入信号的港股。")

elif analysis_mode == "单一股票分析":
    # Single stock analysis mode - with market selection
    st.sidebar.title("市场选择")
    market_type = st.sidebar.radio("选择市场", ["香港股市(HK)"])
    
    st.sidebar.title("股票输入")
    if market_type == "香港股市(HK)":
        ticker = st.sidebar.text_input("输入港股代码（例如，00001、00700）", "00700")
        ticker_placeholder = "输入港股代码"
        ticker_example = "如：00700 (腾讯控股)"
    
    st.sidebar.caption(ticker_example)
    
    st.sidebar.title("显示选项")
    show_short_term = st.sidebar.checkbox("显示短期 EMA", value=True)
    show_long_term = st.sidebar.checkbox("显示长期 EMA", value=True)
    
    # Calculate date range based on selected duration
    end_date = datetime.today().strftime('%Y%m%d')
    start_date = get_start_date(data_duration)
    
    # Fetch and process stock data
    with st.spinner("获取数据中..."):
        try:
            # Different validation rules based on market
            is_valid_ticker = False
            if market_type == "香港股市(HK)":
                # For HK stocks, expect 4-5 digit codes
                ticker = ticker.split('.')[0].zfill(5)  # Format to 5 digits with leading zeros
                if ticker.isdigit() and (len(ticker) == 4 or len(ticker) == 5):
                    is_valid_ticker = True
                
                # For accurate EMA calculation, we need additional historical data
                # Calculate a buffer period to ensure proper initialization of longer period EMAs
                max_ema_period = 60  # The longest EMA period used
                
                # Convert start_date to datetime for adjustment
                start_date_dt = datetime.strptime(start_date, '%Y%m%d')
                
                # Add buffer period for EMA calculation (3x the max EMA period for proper initialization)
                buffer_days = max_ema_period * 3
                extended_start_date = (start_date_dt - timedelta(days=buffer_days)).strftime('%Y%m%d')
                        
                # Fetch stock data using akshare based on market type with extended period
                stock_data = ak.stock_hk_hist(symbol=ticker, period="daily", 
                                              start_date=extended_start_date, end_date=end_date, adjust="")
                    
                if stock_data.empty:
                    market_name = "港股" 
                    st.error(f"未找到所输入{market_name}代码的数据。请检查代码并重试。")
                else:
                    # Rename columns from Chinese to English
                    stock_data.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open'}, inplace=True)
                    # Set 'date' as index and sort by date
                    stock_data['date'] = pd.to_datetime(stock_data['date'])
                    stock_data.set_index('date', inplace=True)
                    stock_data.sort_index(inplace=True)
                    
                    # Calculate Exponential Moving Averages (EMAs) with proper initialization period
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
                    
                    # Filter to only include the originally requested date range for display/analysis
                    display_data = stock_data[stock_data.index >= pd.to_datetime(start_date)]
                    
                    # Create Plotly figure
                    fig = go.Figure()
                    
                    # Add candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=display_data.index,
                        open=display_data["open"],
                        high=display_data[["open", "close"]].max(axis=1),
                        low=display_data[["open", "close"]].min(axis=1),
                        close=display_data["close"],
                        increasing_line_color='red',  # Red for increasing in Asian markets
                        decreasing_line_color='green',  # Green for decreasing in Asian markets
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
                                line=dict(color="blue", width=1),
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
                                line=dict(color="red", width=1),
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
                    
                    # Get market name for title
                    market_name = "港股" if market_type == "香港股市(HK)" else "A股"
                    
                    # Customize plot layout
                    fig.update_layout(
                        title=f"{market_name} {ticker} GMMA 图表 (标记: 短期EMA从下方穿过长期EMA)",
                        xaxis_title="日期",
                        yaxis_title="价格",
                        legend_title="图例",
                        hovermode="x unified",
                        template="plotly_white",
                        height=800
                    )
                    
                    # Display the plot in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display crossover days in a table
                    if len(crossover_dates) > 0:
                        st.subheader("买入信号日期")
                        # Fix the datetime conversion error
                        signal_dates = [date.strftime('%Y-%m-%d') for date in crossover_dates]
                        signal_df = pd.DataFrame(signal_dates, columns=["日期"])
                        st.table(signal_df)
        except Exception as e:
            st.error(f"获取数据时出错: {str(e)}")

elif analysis_mode == "知名港股全显示":
    st.sidebar.title("知名港股设置")
    
    # Load famous HK stocks
    with st.spinner("正在加载知名港股列表..."):
        try:
            # Get all famous HK stocks using akshare
            hk_famous_stocks_df = ak.stock_hk_famous_spot_em()
            
            # Create a dictionary of stock code and name for selection
            stock_options = {f"{row['代码']} - {row['名称']}": row['代码'] for _, row in hk_famous_stocks_df.iterrows()}
            
            # Multi-select for stock selection
            selected_stocks = st.sidebar.multiselect(
                "选择要显示的知名港股",
                options=list(stock_options.keys()),
                default=list(stock_options.keys()),  # Default to all famous stocks
                help="可以选择多只股票一起分析"
            )
            
            # Button to trigger analysis
            if st.sidebar.button("开始分析所选股票"):
                if not selected_stocks:
                    st.warning("请至少选择一只股票进行分析。")
                else:
                    st.success(f"已选择 {len(selected_stocks)} 只股票进行分析")
                    
                    # Calculate date range based on selected duration
                    end_date = datetime.today().strftime('%Y%m%d')
                    start_date = get_start_date(data_duration)
                    
                    # For accurate EMA calculation, we need additional historical data
                    max_ema_period = 60  # The longest EMA period used
                    start_date_dt = datetime.strptime(start_date, '%Y%m%d')
                    buffer_days = max_ema_period * 3
                    extended_start_date = (start_date_dt - timedelta(days=buffer_days)).strftime('%Y%m%d')
                    
                    # Process each selected stock
                    for selected_stock in selected_stocks:
                        ticker = stock_options[selected_stock]
                        stock_name = selected_stock.split(" - ")[1]
                        
                        with st.spinner(f"正在分析 {ticker} - {stock_name}..."):
                            try:
                                # Fetch stock data
                                stock_data = ak.stock_hk_hist(
                                    symbol=ticker, 
                                    period="daily", 
                                    start_date=extended_start_date, 
                                    end_date=end_date, 
                                    adjust=""
                                )
                                
                                if stock_data.empty:
                                    st.warning(f"未找到 {ticker} - {stock_name} 的数据。")
                                    continue
                                
                                # Rename columns and process data
                                stock_data.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open'}, inplace=True)
                                stock_data['date'] = pd.to_datetime(stock_data['date'])
                                stock_data.set_index('date', inplace=True)
                                stock_data.sort_index(inplace=True)
                                
                                # Calculate EMAs
                                for period in [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]:
                                    stock_data[f"EMA{period}"] = stock_data["close"].ewm(span=period, adjust=False).mean()
                                
                                # Define short-term and long-term EMAs
                                short_terms = [3, 5, 8, 10, 12, 15]
                                long_terms = [30, 35, 40, 45, 50, 60]
                                
                                # Calculate average EMAs
                                stock_data['avg_short_ema'] = stock_data[[f'EMA{period}' for period in short_terms]].mean(axis=1)
                                stock_data['avg_long_ema'] = stock_data[[f'EMA{period}' for period in long_terms]].mean(axis=1)
                                
                                # Detect crossovers
                                stock_data['short_above_long'] = stock_data['avg_short_ema'] > stock_data['avg_long_ema']
                                stock_data['crossover'] = False
                                
                                # Find crossover points
                                for i in range(1, len(stock_data)):
                                    if not stock_data['short_above_long'].iloc[i-1] and stock_data['short_above_long'].iloc[i]:
                                        stock_data.loc[stock_data.index[i], 'crossover'] = True
                                
                                # Filter to only include the originally requested date range
                                display_data = stock_data[stock_data.index >= pd.to_datetime(start_date)]
                                
                                # Create expander for this stock
                                with st.expander(f"{ticker} - {stock_name}", expanded=True):
                                    # Create GMMA chart
                                    fig = go.Figure()
                                    
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
                                    for j, period in enumerate([3, 5, 8, 10, 12, 15]):
                                        fig.add_trace(go.Scatter(
                                            x=display_data.index,
                                            y=display_data[f"EMA{period}"],
                                            mode="lines",
                                            name=f"EMA{period}",
                                            line=dict(color="skyblue", width=1),
                                            legendgroup="short_term",
                                            showlegend=(j == 0)
                                        ))
                                    
                                    # Add long-term EMAs (red)
                                    for j, period in enumerate([30, 35, 40, 45, 50, 60]):
                                        fig.add_trace(go.Scatter(
                                            x=display_data.index,
                                            y=display_data[f"EMA{period}"],
                                            mode="lines",
                                            name=f"EMA{period}",
                                            line=dict(color="lightcoral", width=1),
                                            legendgroup="long_term",
                                            showlegend=(j == 0)
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
                                    
                                    # Layout
                                    fig.update_layout(
                                        title=f"{ticker} - {stock_name} 港股 GMMA 图表",
                                        xaxis_title="日期",
                                        yaxis_title="价格",
                                        legend_title="图例",
                                        hovermode="x unified",
                                        template="plotly_white",
                                        height=600
                                    )
                                    
                                    # Display the plot
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display crossover days in a table if any exist
                                    if len(crossover_dates) > 0:
                                        st.subheader("买入信号日期")
                                        signal_dates = [date.strftime('%Y-%m-%d') for date in crossover_dates]
                                        signal_df = pd.DataFrame(signal_dates, columns=["日期"])
                                        st.table(signal_df)
                            
                            except Exception as e:
                                st.error(f"分析 {ticker} - {stock_name} 时出错: {str(e)}")
            else:
                st.info("请从侧边栏选择知名港股并点击'开始分析所选股票'按钮。")
                
        except Exception as e:
            st.error(f"加载知名港股列表时出错: {str(e)}")
