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
    page_title="GMMA 基金分析工具",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("顾比多重移动平均线 (GMMA) 基金图表")
st.markdown("""
此应用程序显示使用 akshare 数据的中国基金的顾比多重移动平均线 (GMMA) 图表。  
可以分析单个股票或自动扫描最近出现买入信号的股票。
""")

# Function to check if a stock has a recent crossover
def has_recent_crossover(ticker, days_to_check=3, market="A"):
    try:
        # Calculate date range for the past 2 months (enough data to calculate EMAs)
        end_date = datetime.today().strftime('%Y%m%d')
        start_date = (datetime.today() - timedelta(days=120)).strftime('%Y%m%d')
        
        # Fetch stock data using akshare based on market
        stock_data = ak.fund_etf_hist_em(symbol=ticker, period="daily", 
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
                stock_data.loc[stock_data.index[i], 'crossover'] = True
        
        # Check if there's a crossover in the last 'days_to_check' days
        recent_data = stock_data.iloc[-days_to_check:]
        has_crossover = recent_data['crossover'].any()
        
        return has_crossover, stock_data if has_crossover else None
    except Exception as e:
        print(f"Error checking {ticker}: {str(e)}")
        return False, None


# Sidebar options
st.sidebar.title("分析模式")
analysis_mode = st.sidebar.radio("选择模式", ["基金全扫描","单一基金分析" ])

if analysis_mode == "基金全扫描":
    st.sidebar.title("基金扫描设置")
    hk_days_to_check = st.sidebar.slider("检查最近几天内的信号", 1, 7, 4)
    hk_max_stocks = st.sidebar.slider("最多显示基金数量", 1, 500, 500)
    
    if st.sidebar.button("开始扫描基金"):
        with st.spinner("正在扫描基金买入信号，这可能需要一些时间..."):
            try:
                # etf_stocks_df = ak.fund_name_em()
                etf_stocks_df = ak.fund_info_index_em(symbol="沪深指数", indicator="增强指数型")
                # hk_stocks_df = ak.stock_hk_spot()
                # print the length of the dataframe
                print(len(etf_stocks_df))
                
                st.info(f"准备扫描 {len(etf_stocks_df)} 只基金...")
                
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Container for results
                crossover_stocks = []
                
                # Loop through all HK stocks
                for i, row in etf_stocks_df.iterrows():
                    # Update progress
                    progress_bar.progress(min((i+1)/len(etf_stocks_df), 1.0))
                    
                    ticker = row['基金代码']
                    name = row['基金名称']
                    
                    # Check for crossover using our modified function with HK market parameter
                    has_crossover, stock_data = has_recent_crossover(ticker, hk_days_to_check, market="A")
                    
                    if has_crossover:
                        # Add to crossover list
                        crossover_stocks.append((ticker, name, stock_data))
                        
                        # Create expander for this stock
                        with st.expander(f"{ticker} - {name}", expanded=True):
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
                            for j, period in enumerate([3, 5, 8, 10, 12, 15]):
                                fig.add_trace(go.Scatter(
                                    x=stock_data.index,
                                    y=stock_data[f"EMA{period}"],
                                    mode="lines",
                                    name=f"EMA{period}",
                                    line=dict(color="blue", width=1),
                                    legendgroup="short_term",
                                    showlegend=(j == 0)
                                ))
                            
                            # Add long-term EMAs (red)
                            for j, period in enumerate([30, 35, 40, 45, 50, 60]):
                                fig.add_trace(go.Scatter(
                                    x=stock_data.index,
                                    y=stock_data[f"EMA{period}"],
                                    mode="lines",
                                    name=f"EMA{period}",
                                    line=dict(color="red", width=1),
                                    legendgroup="long_term",
                                    showlegend=(j == 0)
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
                    
                    # Check if we have found enough stocks
                    if len(crossover_stocks) >= hk_max_stocks:
                        break
                
                # Final update
                progress_bar.progress(1.0)
                
                # Display summary
                if len(crossover_stocks) == 0:
                    st.warning(f"没有找到在最近 {hk_days_to_check} 天内出现买入信号的基金。")
                else:
                    st.success(f"找到 {len(crossover_stocks)} 只在最近 {hk_days_to_check} 天内出现买入信号的基金。")
                    
                    # Create a summary table
                    summary_df = pd.DataFrame(
                        [(t, n) for t, n, _ in crossover_stocks], 
                        columns=["基金代码", "基金名称"]
                    )
                    st.subheader("基金买入信号列表")
                    st.table(summary_df)
            
            except Exception as e:
                st.error(f"扫描基金过程中出错: {str(e)}")
    else:
        st.info("请点击'开始扫描基金'按钮以查找最近出现买入信号的基金。")

elif analysis_mode == "单一基金分析":
    # Single stock analysis mode - with market selection
    st.sidebar.title("市场选择")
    market_type = st.sidebar.radio("选择市场", ["A股"])
    
    st.sidebar.title("基金输入")
    if market_type == "A股":
        ticker = st.sidebar.text_input("输入基金代码（例如，006329）", "006329")
        ticker_placeholder = "输入基金代码"
        ticker_example = "如：006329 (易方达中证海外50ETF联接美元A估值图基金吧)"
    
    st.sidebar.caption(ticker_example)
    
    st.sidebar.title("显示选项")
    show_short_term = st.sidebar.checkbox("显示短期 EMA", value=True)
    show_long_term = st.sidebar.checkbox("显示长期 EMA", value=True)
    
    # Calculate date range for the past 6 months
    end_date = datetime.today().strftime('%Y%m%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y%m%d')
    
    # Fetch and process stock data
    with st.spinner("获取数据中..."):
        try:
            # Different validation rules based on market
            is_valid_ticker = False
            if market_type == "A股":
                # For HK stocks, expect 4-5 digit codes
                ticker = ticker.split('.')[0].zfill(6)  # Format to 5 digits with leading zeros
                if ticker.isdigit() and (len(ticker) == 4 or len(ticker) == 5):
                    is_valid_ticker = True
                        
                # Fetch stock data using akshare based on market type
                stock_data = ak.fund_etf_hist_em(symbol=ticker, period="daily", 
                                              start_date=start_date, end_date=end_date, adjust="")
                    
                if stock_data.empty:
                    market_name = "A股" 
                    st.error(f"未找到所输入{market_name}代码的数据。请检查代码并重试。")
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
                        increasing_line_color='red',  # Red for increasing in Asian markets
                        decreasing_line_color='green',  # Green for decreasing in Asian markets
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
                    
                    # Get market name for title
                    market_name = "A股"
                    
                    # Customize plot layout
                    fig.update_layout(
                        title=f"{market_name} {ticker} GMMA 图表 (标记: 短期EMA从下方穿过长期EMA)",
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
                        # Fix the datetime conversion error
                        signal_dates = [date.strftime('%Y-%m-%d') for date in crossover_dates]
                        signal_df = pd.DataFrame(signal_dates, columns=["日期"])
                        st.table(signal_df)
        except Exception as e:
            st.error(f"获取数据时出错: {str(e)}")
