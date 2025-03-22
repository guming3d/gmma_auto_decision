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

# Sidebar options
st.sidebar.title("分析模式")
analysis_mode = st.sidebar.radio("选择模式", ["指定基金分析", "基金全扫描"], index=0)

# Add short-term EMA selection for sell signal
st.sidebar.title("信号设置")
sell_signal_ema = st.sidebar.selectbox(
    "卖出信号比较的短期EMA", 
    options=["EMA3", "EMA5", "EMA8", "EMA10"],
    index=2,  # Default to EMA8
    help="当价格低于所选EMA时，可能触发卖出信号"
)

# Add backtest operations units input to sidebar
# st.sidebar.title("回测设置")
backtest_units = 100

# Add back-testing strategy selection
st.sidebar.title("回测设置")
backtest_strategy = st.sidebar.radio(
    "回测策略",
    options=["常规策略", "百分比策略"],
    index=0,
    help="常规策略: 固定单位买卖; 百分比策略: 按资金比例投资，保留30%现金"
)

# Add historical data period selection
history_period = st.sidebar.selectbox(
    "历史数据周期",
    options=["25年", "20年", "15年", "10年", "8年", "6年", "4年", "3年", "2年", "1年", "6个月", "3个月"],
    index=7,  # Default to 3年
    help="选择用于分析和回测的历史数据范围"
)

# Convert selected period to days
period_days = {
    "25年": 365 * 25,
    "20年": 365 * 20,
    "15年": 365 * 15,
    "10年": 365 * 10,
    "8年": 365 * 8,
    "6年": 365 * 6,
    "4年": 365 * 4,
    "3年": 365 * 3,
    "2年": 365 * 2,
    "1年": 365,
    "6个月": 180,
    "3个月": 90
}
history_days = period_days[history_period]

# Display current settings
st.sidebar.markdown(f"**当前卖出信号设置**: 价格低于**{sell_signal_ema}**时产生卖出信号")

# Function to check if a stock has a recent crossover
def has_recent_crossover(ticker, days_to_check=3, market="A", ema_for_sell=None):
    try:
        # Calculate date range with extra 60 days for EMA calculation
        end_date = datetime.today().strftime('%Y%m%d')
        start_date = (datetime.today() - timedelta(days=120 + 60)).strftime('%Y%m%d')  # 120 days + 60 days warmup
        
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
        
        # Track if we're in a position (bought but not yet sold)
        in_position = False
        last_buy_price = None
        last_buy_index = -1
        
        # Extract the number from ema_for_sell if provided
        ema_number = int(ema_for_sell.replace("EMA", "")) if ema_for_sell else 8
        
        # Find both buy and sell signals
        for i in range(1, len(stock_data)):
            # Buy signal: short-term crosses above long-term
            if not stock_data['short_above_long'].iloc[i-1] and stock_data['short_above_long'].iloc[i] and not in_position:
                stock_data.loc[stock_data.index[i], 'buy_signal'] = True
                last_buy_price = stock_data['close'].iloc[i]
                last_buy_index = i
                in_position = True
            
            # Sell signal - use the selected EMA from the sidebar
            elif in_position and (
                (last_buy_price is not None) and 
                (stock_data['close'].iloc[i] < stock_data[f'EMA{ema_number}'].iloc[i])
            ):
                stock_data.loc[stock_data.index[i], 'sell_signal'] = True
                in_position = False  # Reset position status after selling
        
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
        units (int): Number of units to buy/sell on each signal (not used in the updated logic)
    Returns:
        dict: Dictionary containing back testing results
    """
    # Initialize variables
    initial_cash = 100000  # Starting with 100,000 units of currency
    cash = initial_cash
    position = 0  # Number of units held
    trades = []
    
    # Variables to track purchase information
    last_buy_price = None
    last_buy_units = 0
    
    # Sort data by date to ensure chronological processing
    stock_data = stock_data.sort_index()
    
    # Process each day in the data
    for date, row in stock_data.iterrows():
        price = row['close']
        
        # Process buy signal (only if not already in a position)
        if row['buy_signal'] and cash > 0 and position == 0:
            # Buy as many units as possible with available cash
            max_units = cash // price
            if max_units > 0:
                cost = price * max_units
                cash -= cost
                position += max_units
                
                # Store buy price and units for later gain/loss calculation
                last_buy_price = price
                last_buy_units = max_units
                
                trades.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'action': '买入',
                    'price': price,
                    'units': max_units,
                    'cost': cost,
                    'cash': cash,
                    'position_value': position * price,
                    'total_value': cash + (position * price)
                })
        
        # Process sell signal (only if we have a position)
        elif row['sell_signal'] and position > 0:
            # Sell all units
            proceeds = price * position
            cash += proceeds
            
            # Calculate gain/loss information
            gain_loss = 0
            gain_loss_pct = 0
            if last_buy_price is not None:
                gain_loss = (price - last_buy_price) * position
                gain_loss_pct = ((price / last_buy_price) - 1) * 100
            
            trades.append({
                'date': date.strftime('%Y-%m-%d'),
                'action': '卖出',
                'price': price,
                'units': position,
                'proceeds': proceeds,
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct,
                'cash': cash,
                'position_value': 0,
                'total_value': cash
            })
            
            # Reset position after selling all
            position = 0
            last_buy_price = None
            last_buy_units = 0
    
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

# Function to perform back testing on buy/sell signals with percentage strategy
def perform_back_testing_percentage(stock_data):
    """
    Perform back testing using a percentage-based strategy:
    1. Initial invest money is 100000, keep at least 30% of money at hand
    2. Use 10% of invest money to buy stocks on each buy signal
    3. If left money is less than 10%, waiting for the selling_signal to sell 50%
    4. Continue until latest trading day
    
    Args:
        stock_data (DataFrame): DataFrame with 'close', 'buy_signal', and 'sell_signal' columns
    
    Returns:
        dict: Dictionary containing back testing results
    """
    # Initialize variables
    initial_cash = 100000  # Starting with 100,000 units of currency
    cash = initial_cash
    min_cash_reserve = initial_cash * 0.3  # Keep 30% of initial cash as reserve
    min_cash_threshold = initial_cash * 0.1  # Threshold for when to consider selling (10% of initial)
    buy_percentage = 0.1  # Use 10% of invest money for each buy
    position = 0  # Number of units held
    trades = []
    
    # Variables to track purchase information
    last_buy_price = None
    position_value = 0
    position_history = {}  # To track buys at different prices
    
    # Sort data by date to ensure chronological processing
    stock_data = stock_data.sort_index()
    
    # Process each day in the data
    for date, row in stock_data.iterrows():
        price = row['close']
        current_value = cash + (position * price)
        
        # Process buy signal (if we have enough cash)
        if row['buy_signal']:
            # Calculate available money for this purchase
            available_money = cash - min_cash_reserve
            
            # Only buy if we have enough money (at least 10% of initial investment)
            if available_money >= (initial_cash * 0.1):
                # Calculate money to spend (10% of current value)
                buy_amount = current_value * buy_percentage
                
                # Make sure we don't go below our cash reserve
                if (cash - buy_amount) < min_cash_reserve:
                    buy_amount = cash - min_cash_reserve
                
                if buy_amount > 0:
                    # Calculate units to buy
                    units_to_buy = int(buy_amount // price)
                    
                    if units_to_buy > 0:
                        cost = price * units_to_buy
                        cash -= cost
                        position += units_to_buy
                        
                        # Track this purchase in position history
                        if price not in position_history:
                            position_history[price] = 0
                        position_history[price] += units_to_buy
                        
                        trades.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'action': '买入',
                            'price': price,
                            'units': units_to_buy,
                            'cost': cost,
                            'cash': cash,
                            'position_value': position * price,
                            'total_value': cash + (position * price)
                        })
        
        # Process sell signal only if:
        # 1. There is a sell signal
        # 2. We have a position
        # 3. Cash is below the 10% threshold of initial investment
        elif row['sell_signal'] and position > 0 and cash < min_cash_threshold:
            # Sell 50% of current position
            units_to_sell = position // 2
            if units_to_sell > 0:
                proceeds = price * units_to_sell
                cash += proceeds
                
                # Calculate weighted average buy price of current position
                if position_history:
                    weighted_buy_price = sum(p * q for p, q in position_history.items()) / sum(position_history.values())
                else:
                    weighted_buy_price = 0
                
                # Calculate gain/loss information
                gain_loss = (price - weighted_buy_price) * units_to_sell
                gain_loss_pct = ((price / weighted_buy_price) - 1) * 100 if weighted_buy_price > 0 else 0
                
                # Update position
                position -= units_to_sell
                
                # Update position history - sell proportionally from all buys
                sell_factor = units_to_sell / sum(position_history.values())
                for p in list(position_history.keys()):
                    units_sold_at_this_price = int(position_history[p] * sell_factor)
                    position_history[p] -= units_sold_at_this_price
                    if position_history[p] <= 0:
                        del position_history[p]
                
                trades.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'action': '卖出',
                    'price': price,
                    'units': units_to_sell,
                    'proceeds': proceeds,
                    'gain_loss': gain_loss,
                    'gain_loss_pct': gain_loss_pct,
                    'cash': cash,
                    'position_value': position * price,
                    'total_value': cash + (position * price)
                })
    
    # Calculate final results
    final_price = stock_data['close'].iloc[-1]
    final_position_value = position * final_price
    final_value = cash + final_position_value
    
    # Buy and hold comparison
    max_buy_and_hold_units = initial_cash // stock_data['close'].iloc[0]
    buy_and_hold_value = max_buy_and_hold_units * final_price
    
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
        'buy_hold_units': max_buy_and_hold_units,
        'buy_hold_value': buy_and_hold_value,
        'buy_hold_return_pct': buy_hold_return_pct,
        'trades': trades
    }
    
    return results

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
                    
                    # Calculate date range using the selected history period with extra 60 days
                    end_date = datetime.today().strftime('%Y%m%d')
                    display_start_date = (datetime.today() - timedelta(days=history_days)).strftime('%Y%m%d')
                    fetch_start_date = (datetime.today() - timedelta(days=history_days + 60)).strftime('%Y%m%d')
                    
                    # Check for crossover using our modified function with HK market parameter and selected EMA
                    has_crossover, stock_data = has_recent_crossover(ticker, hk_days_to_check, market="A", ema_for_sell=sell_signal_ema)
                    
                    if has_crossover:
                        # Add to crossover list
                        crossover_stocks.append((ticker, name, stock_data))
                        
                        # Filter stock_data for display
                        display_start_date_dt = pd.to_datetime(display_start_date)
                        display_data = stock_data[stock_data.index >= display_start_date_dt]
                        
                        # Create GMMA chart using display_data
                        fig = go.Figure()
                        
                        # Add candlestick chart with display_data
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
                            
                            # Mark buy and sell signals on the chart
                            buy_dates = display_data[display_data['buy_signal']].index
                            sell_dates = display_data[display_data['sell_signal']].index
                            
                            # Add buy signals
                            for date in buy_dates:
                                price_at_signal = display_data.loc[date, 'close']
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
                                price_at_signal = display_data.loc[date, 'close']
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
                                height=800
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
                            
                            # Display notification about which EMA is used for sell signals
                            st.info(f"当前卖出信号条件: 价格低于**{sell_signal_ema}**")
                            
                            # Add back testing section
                            st.subheader("回归测试")
                            
                            # Select the appropriate back testing function based on user selection
                            if backtest_strategy == "常规策略":
                                st.markdown(f"""该回归测试模拟了严格按照买入和卖出信号操作的结果，每次操作购买或卖出{backtest_units}单位，以验证信号的有效性。""")
                                backtest_results = perform_back_testing(stock_data, units=backtest_units)
                            else:
                                st.markdown("""该回归测试模拟了按比例投资的策略：
                                1. 初始资金10万，至少保留30%现金
                                2. 每次买入信号使用当前总资产的10%购买股票
                                3. 当现金不足10%时，等待卖出信号卖出50%持仓
                                """)
                                backtest_results = perform_back_testing_percentage(stock_data)
                            
                            # Before displaying backtest results:
                            # Filter backtest results to only include trades from the display period
                            filtered_trades = [trade for trade in backtest_results['trades'] 
                                              if pd.to_datetime(trade['date']) >= display_start_date_dt]

                            # Update the displayed trades
                            backtest_results['trades'] = filtered_trades
                            
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
                                
                                # Format gain/loss columns
                                if 'gain_loss' in trades_df.columns:
                                    # Function to color-code gain/loss values
                                    def color_gain_loss(val):
                                        if pd.isna(val):
                                            return ''
                                        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                                        return f'color: {color}'
                                    
                                    # First apply styling to the numeric data
                                    styled_df = trades_df.style.map(
                                        color_gain_loss, 
                                        subset=['gain_loss', 'gain_loss_pct']
                                    )
                                    
                                    # Then format the display values (this doesn't affect the styling)
                                    styled_df = styled_df.format({
                                        'gain_loss': lambda x: f"¥{x:,.2f}" if not pd.isna(x) else "",
                                        'gain_loss_pct': lambda x: f"{x:.2f}%" if not pd.isna(x) else ""
                                    })
                                    
                                    st.dataframe(styled_df, use_container_width=True)
                                else:
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
    
    # Update the sell signal info message
    st.info(f"当前卖出信号条件: 价格低于**{sell_signal_ema}**")
    
    # Process the input funds
    fund_list = [fund.strip() for fund in funds_input.split(",") if fund.strip()]
    
    # Calculate date range using the selected history period
    end_date = datetime.today().strftime('%Y%m%d')
    start_date = (datetime.today() - timedelta(days=history_days)).strftime('%Y%m%d')
    
    # Create tabs for each fund
    tabs = st.tabs(fund_list)
    
    # Analyze each fund in its own tab
    for idx, ticker in enumerate(fund_list):
        with tabs[idx]:
            with st.spinner(f"获取 {ticker} 数据中..."):
                try:
                    # Format ticker
                    ticker = ticker.split('.')[0].zfill(6)
                    
                    # Calculate date range using the selected history period
                    # Add 60 days to the history days for EMA calculation warmup
                    end_date = datetime.today().strftime('%Y%m%d')
                    display_start_date = (datetime.today() - timedelta(days=history_days)).strftime('%Y%m%d')
                    fetch_start_date = (datetime.today() - timedelta(days=history_days + 60)).strftime('%Y%m%d')
                    
                    # Fetch stock data using akshare with extended date range
                    stock_data = ak.fund_etf_hist_em(symbol=ticker, period="daily", 
                                                 start_date=fetch_start_date, end_date=end_date, adjust="")
                    
                    if stock_data.empty:
                        st.error(f"未找到基金代码 {ticker} 的数据。请检查代码并重试。")
                        continue
                    
                    # Rename columns from Chinese to English
                    stock_data.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open'}, inplace=True)
                    stock_data['date'] = pd.to_datetime(stock_data['date'])
                    stock_data.set_index('date', inplace=True)
                    stock_data.sort_index(inplace=True)
                    
                    # Calculate EMAs using full dataset
                    for period in [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]:
                        stock_data[f"EMA{period}"] = stock_data["close"].ewm(span=period, adjust=False).mean()
                    
                    # Define short-term and long-term EMAs
                    short_terms = [3, 5, 8, 10, 12, 15]
                    long_terms = [30, 35, 40, 45, 50, 60]
                    
                    # Calculate average EMAs
                    stock_data['avg_short_ema'] = stock_data[[f'EMA{period}' for period in short_terms]].mean(axis=1)
                    stock_data['avg_long_ema'] = stock_data[[f'EMA{period}' for period in long_terms]].mean(axis=1)
                    
                    # Detect crossovers using full dataset
                    stock_data['short_above_long'] = stock_data['avg_short_ema'] > stock_data['avg_long_ema']
                    stock_data['buy_signal'] = False
                    stock_data['sell_signal'] = False
                    
                    # Track position status and buy price
                    in_position = False
                    last_buy_price = None
                    last_buy_index = -1
                    
                    # Extract the number from the selected EMA
                    ema_number = int(sell_signal_ema.replace("EMA", ""))
                    
                    # Find signals using full dataset
                    for i in range(1, len(stock_data)):
                        if not stock_data['short_above_long'].iloc[i-1] and stock_data['short_above_long'].iloc[i] and not in_position:
                            stock_data.loc[stock_data.index[i], 'buy_signal'] = True
                            last_buy_price = stock_data['close'].iloc[i]
                            last_buy_index = i
                            in_position = True
                        
                        # Sell signal - only if we're in a position and price is lower than the selected EMA
                        elif in_position and (
                            stock_data['close'].iloc[i] < stock_data[f'EMA{ema_number}'].iloc[i]
                        ):
                            stock_data.loc[stock_data.index[i], 'sell_signal'] = True
                            in_position = False  # Reset position status after selling
                    
                    # Now filter to only show the user-selected date range
                    display_start_date_dt = pd.to_datetime(display_start_date)
                    display_data = stock_data[stock_data.index >= display_start_date_dt]
                    
                    # Create figure using the filtered data for display
                    fig = go.Figure()
                    
                    # Add candlestick
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
                    
                    # Add EMAs using filtered data
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
                    
                    # Add signals
                    buy_dates = display_data[display_data['buy_signal']].index
                    sell_dates = display_data[display_data['sell_signal']].index
                    
                    # Add buy signals
                    for date in buy_dates:
                        price_at_signal = display_data.loc[date, 'close']
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
                        price_at_signal = display_data.loc[date, 'close']
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
                        template="plotly_white",
                        height=800
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
                        
                        # Select the appropriate back testing function based on user selection
                        if backtest_strategy == "常规策略":
                            st.markdown(f"""该回归测试模拟了严格按照买入和卖出信号操作的结果，每次操作购买或卖出{backtest_units}单位，以验证信号的有效性。""")
                            backtest_results = perform_back_testing(stock_data, units=backtest_units)
                        else:
                            st.markdown("""该回归测试模拟了按比例投资的策略：
                            1. 初始资金10万，至少保留30%现金
                            2. 每次买入信号使用当前总资产的10%购买股票
                            3. 当现金不足10%时，等待卖出信号卖出50%持仓
                            """)
                            backtest_results = perform_back_testing_percentage(stock_data)
                        
                        # Before displaying backtest results:
                        # Filter backtest results to only include trades from the display period
                        filtered_trades = [trade for trade in backtest_results['trades'] 
                                          if pd.to_datetime(trade['date']) >= display_start_date_dt]

                        # Update the displayed trades
                        backtest_results['trades'] = filtered_trades
                        
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
                            
                            # Format gain/loss columns
                            if 'gain_loss' in trades_df.columns:
                                # Function to color-code gain/loss values
                                def color_gain_loss(val):
                                    if pd.isna(val):
                                        return ''
                                    color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                                    return f'color: {color}'
                                
                                # First apply styling to the numeric data
                                styled_df = trades_df.style.map(
                                    color_gain_loss, 
                                    subset=['gain_loss', 'gain_loss_pct']
                                )
                                
                                # Then format the display values (this doesn't affect the styling)
                                styled_df = styled_df.format({
                                    'gain_loss': lambda x: f"¥{x:,.2f}" if not pd.isna(x) else "",
                                    'gain_loss_pct': lambda x: f"{x:.2f}%" if not pd.isna(x) else ""
                                })
                                
                                st.dataframe(styled_df, use_container_width=True)
                            else:
                                st.dataframe(trades_df, use_container_width=True)
                        else:
                            st.warning("回测期间没有产生交易。")
                except Exception as e:
                    st.error(f"分析基金 {ticker} 时出错: {str(e)}")
