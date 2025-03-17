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
        
        # Detect crossovers (short-term crossing above/below long-term)
        stock_data['short_above_long'] = stock_data['avg_short_ema'] > stock_data['avg_long_ema']
        stock_data['buy_signal'] = False
        stock_data['sell_signal'] = False
        
        # Find both buy and sell signals
        for i in range(1, len(stock_data)):
            # Buy signal: short-term crosses above long-term
            if not stock_data['short_above_long'].iloc[i-1] and stock_data['short_above_long'].iloc[i]:
                stock_data.loc[stock_data.index[i], 'buy_signal'] = True
            # Sell signal: short-term crosses below long-term AND price is below EMA8
            elif stock_data['short_above_long'].iloc[i-1] and not stock_data['short_above_long'].iloc[i] and stock_data['close'].iloc[i] < stock_data['EMA8'].iloc[i]:
                stock_data.loc[stock_data.index[i], 'sell_signal'] = True
        
        # Check if there's a crossover in the last 'days_to_check' days
        recent_data = stock_data.iloc[-days_to_check:]
        has_crossover = recent_data['buy_signal'].any() or recent_data['sell_signal'].any()
        
        return has_crossover, stock_data if has_crossover else None
    except Exception as e:
        print(f"Error checking {ticker}: {str(e)}")
        return False, None

# Function to perform back testing on buy/sell signals
def perform_back_testing(stock_data, units=100):
    """
    Perform back testing based on buy/sell signals in the stock data.
    Args:
        stock_data (DataFrame): DataFrame with 'close', 'buy_signal', and 'sell_signal' columns
        units (int): Number of units to buy/sell on each signal
    Returns:
        dict: Dictionary containing back testing results
    """
    # Initialize variables
    initial_cash = 100000  # Starting with 100,000 units of currency
    cash = initial_cash
    position = 0  # Number of units held
    trades = []
    
    # Sort data by date to ensure chronological processing
    stock_data = stock_data.sort_index()
    
    # Process each day in the data
    for date, row in stock_data.iterrows():
        price = row['close']
        
        # Process buy signal
        if row['buy_signal'] and cash >= price * units:
            # Buy units
            cost = price * units
            cash -= cost
            position += units
            trades.append({
                'date': date.strftime('%Y-%m-%d'),
                'action': '买入',
                'price': price,
                'units': units,
                'cost': cost,
                'cash': cash,
                'position_value': position * price,
                'total_value': cash + (position * price)
            })
        
        # Process sell signal
        elif row['sell_signal'] and position >= units:
            # Sell units
            proceeds = price * units
            cash += proceeds
            position -= units
            trades.append({
                'date': date.strftime('%Y-%m-%d'),
                'action': '卖出',
                'price': price,
                'units': units,
                'proceeds': proceeds,
                'cash': cash,
                'position_value': position * price,
                'total_value': cash + (position * price)
            })
    
    # Calculate final results
    final_price = stock_data['close'].iloc[-1]
    final_position_value = position * final_price
    final_value = cash + final_position_value
    
    # Buy and hold comparison
    buy_and_hold_units = initial_cash // stock_data['close'].iloc[0]
    buy_and_hold_value = buy_and_hold_units * final_price
    
    # Calculate returns
    signal_return_pct = ((final_value - initial_cash) / initial_cash) * 100
    buy_hold_return_pct = ((buy_and_hold_value - initial_cash) / initial_cash) * 100
    
    # Results
    results = {
        'initial_cash': initial_cash,
        'final_cash': cash,
        'final_position': position,
        'final_position_value': final_position_value,
        'final_value': final_value,
        'signal_return_pct': signal_return_pct,
        'buy_hold_units': buy_and_hold_units,
        'buy_hold_value': buy_and_hold_value,
        'buy_hold_return_pct': buy_hold_return_pct,
        'trades': trades
    }
    
    return results

# Sidebar options
st.sidebar.title("分析模式")
analysis_mode = st.sidebar.radio("选择模式", ["指定基金分析", "基金全扫描"], index=0)

# Add backtest operations units input to sidebar
st.sidebar.title("回测设置")
backtest_units = st.sidebar.number_input("回测操作单位数", min_value=1, max_value=10000, value=1000, 
                                          help="每次买入或卖出信号触发时的操作单位数")

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
                            
                            # Mark buy and sell signals on the chart
                            buy_dates = stock_data[stock_data['buy_signal']].index
                            sell_dates = stock_data[stock_data['sell_signal']].index
                            
                            # Add buy signals
                            for date in buy_dates:
                                price_at_signal = stock_data.loc[date, 'close']
                                # Add buy annotation - arrow pointing upward from below
                                fig.add_annotation(
                                    x=date,
                                    y=price_at_signal * 1.08,  # Move text higher
                                    text=f"买入信号 {date.strftime('%Y-%m-%d')}",
                                    showarrow=True,
                                    arrowhead=1,
                                    arrowcolor="green",
                                    arrowsize=1,
                                    arrowwidth=2,
                                    font=dict(color="green", size=12),
                                    ax=0,  # No horizontal shift
                                    ay=-40  # Move arrow start point down
                                )
                            
                            # Add sell signals
                            for date in sell_dates:
                                price_at_signal = stock_data.loc[date, 'close']
                                # Add sell annotation - arrow pointing downward from above
                                fig.add_annotation(
                                    x=date,
                                    y=price_at_signal * 0.92,  # Move text lower
                                    text=f"卖出信号 {date.strftime('%Y-%m-%d')}",
                                    showarrow=True,
                                    arrowhead=1,
                                    arrowcolor="red",
                                    arrowsize=1,
                                    arrowwidth=2,
                                    font=dict(color="red", size=12),
                                    ax=0,  # No horizontal shift
                                    ay=40  # Move arrow start point up
                                )
                            
                            # Count and display the number of signals
                            buy_count = len(buy_dates)
                            sell_count = len(sell_dates)
                            last_buy = buy_dates[-1].strftime('%Y-%m-%d') if buy_count > 0 else "None"
                            last_sell = sell_dates[-1].strftime('%Y-%m-%d') if sell_count > 0 else "None"
                            
                            signal_info = (
                                f"**买入信号**: 共 {buy_count} 个, 最近信号日期: {last_buy}<br>"
                                f"**卖出信号**: 共 {sell_count} 个, 最近信号日期: {last_sell}"
                            )
                            
                            fig.add_annotation(
                                x=0.02,
                                y=0.98,
                                xref="paper",
                                yref="paper",
                                text=signal_info,
                                showarrow=False,
                                font=dict(size=14),
                                bgcolor="white",
                                bordercolor="black",
                                borderwidth=1,
                                align="left"
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
                            
                            # Display both buy and sell signal dates in tables
                            if len(buy_dates) > 0 or len(sell_dates) > 0:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("买入信号日期")
                                    if len(buy_dates) > 0:
                                        buy_signal_dates = [date.strftime('%Y-%m-%d') for date in buy_dates]
                                        buy_df = pd.DataFrame(buy_signal_dates, columns=["日期"])
                                        st.table(buy_df)
                                    else:
                                        st.write("无买入信号")
                                
                                with col2:
                                    st.subheader("卖出信号日期")
                                    if len(sell_dates) > 0:
                                        sell_signal_dates = [date.strftime('%Y-%m-%d') for date in sell_dates]
                                        sell_df = pd.DataFrame(sell_signal_dates, columns=["日期"])
                                        st.table(sell_df)
                                    else:
                                        st.write("无卖出信号")
                            
                            # Add back testing section
                            st.subheader("回归测试")
                            st.markdown(f"""该回归测试模拟了严格按照买入和卖出信号操作的结果，每次操作购买或卖出{backtest_units}单位，以验证信号的有效性。""")
                            
                            # Perform back testing with user-defined units
                            backtest_results = perform_back_testing(stock_data, units=backtest_units)
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    label="信号策略最终价值", 
                                    value=f"¥{backtest_results['final_value']:,.2f}",
                                    delta=f"{backtest_results['signal_return_pct']:.2f}%"
                                )
                                
                            with col2:
                                st.metric(
                                    label="买入并持有策略", 
                                    value=f"¥{backtest_results['buy_hold_value']:,.2f}",
                                    delta=f"{backtest_results['buy_hold_return_pct']:.2f}%"
                                )
                                
                            with col3:
                                delta = backtest_results['signal_return_pct'] - backtest_results['buy_hold_return_pct']
                                st.metric(
                                    label="信号vs买入持有", 
                                    value=f"{delta:.2f}%",
                                    delta=delta
                                )
                            
                            # Display trades table
                            if backtest_results['trades']:
                                st.subheader("交易记录")
                                trades_df = pd.DataFrame(backtest_results['trades'])
                                st.dataframe(trades_df, use_container_width=True)
                            else:
                                st.warning("回测期间没有产生交易。")
                    
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

elif analysis_mode == "指定基金分析":
    # Single stock analysis mode - with market selection
    st.sidebar.title("市场选择")
    market_type = st.sidebar.radio("选择市场", ["A股"])
    
    st.sidebar.title("基金输入")
    default_funds = "510300,510050,512100,588000,512010,512200"
    if market_type == "A股":
        funds_input = st.sidebar.text_area("输入基金代码（多个代码用逗号分隔）", 
                                         value=default_funds,
                                         height=100)
        ticker_example = "示例：510300 (沪深300ETF), 510050 (上证50ETF)"
    
    st.sidebar.caption(ticker_example)
    
    st.sidebar.title("显示选项")
    show_short_term = st.sidebar.checkbox("显示短期 EMA", value=True)
    show_long_term = st.sidebar.checkbox("显示长期 EMA", value=True)
    
    # Process the input funds
    fund_list = [fund.strip() for fund in funds_input.split(",") if fund.strip()]
    
    # Calculate date range for the past 6 months
    end_date = datetime.today().strftime('%Y%m%d')
    start_date = (datetime.today() - timedelta(days=920)).strftime('%Y%m%d')
    
    # Create tabs for each fund
    tabs = st.tabs(fund_list)
    
    # Analyze each fund in its own tab
    for idx, ticker in enumerate(fund_list):
        with tabs[idx]:
            with st.spinner(f"获取 {ticker} 数据中..."):
                try:
                    # Format ticker
                    ticker = ticker.split('.')[0].zfill(6)
                    
                    # Fetch stock data using akshare
                    stock_data = ak.fund_etf_hist_em(symbol=ticker, period="daily", 
                                                  start_date=start_date, end_date=end_date, adjust="")
                    
                    if stock_data.empty:
                        st.error(f"未找到基金代码 {ticker} 的数据。请检查代码并重试。")
                        continue
                    
                    # Rename columns from Chinese to English
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
                    stock_data['buy_signal'] = False
                    stock_data['sell_signal'] = False
                    
                    # Find signals
                    for i in range(1, len(stock_data)):
                        if not stock_data['short_above_long'].iloc[i-1] and stock_data['short_above_long'].iloc[i]:
                            stock_data.loc[stock_data.index[i], 'buy_signal'] = True
                        elif stock_data['short_above_long'].iloc[i-1] and not stock_data['short_above_long'].iloc[i] and stock_data['close'].iloc[i] < stock_data['EMA8'].iloc[i]:
                            stock_data.loc[stock_data.index[i], 'sell_signal'] = True
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add candlestick
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
                    
                    # Add EMAs
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
                    
                    # Add signals
                    buy_dates = stock_data[stock_data['buy_signal']].index
                    sell_dates = stock_data[stock_data['sell_signal']].index
                    
                    # Add buy signals
                    for date in buy_dates:
                        price_at_signal = stock_data.loc[date, 'close']
                        fig.add_annotation(
                            x=date,
                            y=price_at_signal * 1.08,
                            text=f"买入信号 {date.strftime('%Y-%m-%d')}",
                            showarrow=True,
                            arrowhead=1,
                            arrowcolor="green",
                            arrowsize=1,
                            arrowwidth=2,
                            font=dict(color="green", size=12),
                            ax=0,
                            ay=-40
                        )
                    
                    # Add sell signals
                    for date in sell_dates:
                        price_at_signal = stock_data.loc[date, 'close']
                        fig.add_annotation(
                            x=date,
                            y=price_at_signal * 0.92,
                            text=f"卖出信号 {date.strftime('%Y-%m-%d')}",
                            showarrow=True,
                            arrowhead=1,
                            arrowcolor="red",
                            arrowsize=1,
                            arrowwidth=2,
                            font=dict(color="red", size=12),
                            ax=0,
                            ay=40
                        )
                    
                    # Signal summary
                    buy_count = len(buy_dates)
                    sell_count = len(sell_dates)
                    last_buy = buy_dates[-1].strftime('%Y-%m-%d') if buy_count > 0 else "None"
                    last_sell = sell_dates[-1].strftime('%Y-%m-%d') if sell_count > 0 else "None"
                    
                    signal_info = (
                        f"**买入信号**: 共 {buy_count} 个, 最近信号日期: {last_buy}<br>"
                        f"**卖出信号**: 共 {sell_count} 个, 最近信号日期: {last_sell}"
                    )
                    
                    fig.add_annotation(
                        x=0.02,
                        y=0.98,
                        xref="paper",
                        yref="paper",
                        text=signal_info,
                        showarrow=False,
                        font=dict(size=14),
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        align="left"
                    )
                    
                    # Layout
                    fig.update_layout(
                        title=f"A股 {ticker} GMMA 图表",
                        xaxis_title="日期",
                        yaxis_title="价格",
                        legend_title="图例",
                        hovermode="x unified",
                        template="plotly_white"
                    )
                    
                    # Display plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display signal tables
                    if len(buy_dates) > 0 or len(sell_dates) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("买入信号日期")
                            if len(buy_dates) > 0:
                                buy_signal_dates = [date.strftime('%Y-%m-%d') for date in buy_dates]
                                buy_df = pd.DataFrame(buy_signal_dates, columns=["日期"])
                                st.table(buy_df)
                            else:
                                st.write("无买入信号")
                        
                        with col2:
                            st.subheader("卖出信号日期")
                            if len(sell_dates) > 0:
                                sell_signal_dates = [date.strftime('%Y-%m-%d') for date in sell_dates]
                                sell_df = pd.DataFrame(sell_signal_dates, columns=["日期"])
                                st.table(sell_df)
                            else:
                                st.write("无卖出信号")
                        
                        # Add back testing section
                        st.subheader("回归测试")
                        st.markdown(f"""该回归测试模拟了严格按照买入和卖出信号操作的结果，每次操作购买或卖出{backtest_units}单位，以验证信号的有效性。""")
                        
                        # Perform back testing with user-defined units
                        backtest_results = perform_back_testing(stock_data, units=backtest_units)
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                label="信号策略最终价值", 
                                value=f"¥{backtest_results['final_value']:,.2f}",
                                delta=f"{backtest_results['signal_return_pct']:.2f}%"
                            )
                            
                        with col2:
                            st.metric(
                                label="买入并持有策略", 
                                value=f"¥{backtest_results['buy_hold_value']:,.2f}",
                                delta=f"{backtest_results['buy_hold_return_pct']:.2f}%"
                            )
                            
                        with col3:
                            delta = backtest_results['signal_return_pct'] - backtest_results['buy_hold_return_pct']
                            st.metric(
                                label="信号vs买入持有", 
                                value=f"{delta:.2f}%",
                                delta=delta
                            )
                        
                        # Display trades table
                        if backtest_results['trades']:
                            st.subheader("交易记录")
                            trades_df = pd.DataFrame(backtest_results['trades'])
                            st.dataframe(trades_df, use_container_width=True)
                        else:
                            st.warning("回测期间没有产生交易。")
                except Exception as e:
                    st.error(f"分析基金 {ticker} 时出错: {str(e)}")
