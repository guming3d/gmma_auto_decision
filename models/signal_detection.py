"""
Signal detection module for detecting buy/sell signals in fund data.
"""
import pandas as pd
from datetime import datetime, timedelta
from utils.data_utils import fetch_and_process_fund_data

def detect_crossover_signals(stock_data, sell_signal_ema="EMA8"):
    """
    Detect buy and sell signals based on GMMA crossovers.
    
    Args:
        stock_data (DataFrame): Processed fund data with EMAs
        sell_signal_ema (str): EMA to use for sell signal detection
        
    Returns:
        DataFrame: Data with buy/sell signal columns added
    """
    # Flag for crossovers (short-term crossing above/below long-term)
    stock_data['short_above_long'] = stock_data['avg_short_ema'] > stock_data['avg_long_ema']
    stock_data['buy_signal'] = False
    stock_data['sell_signal'] = False
    
    # Track if we're in a position (bought but not yet sold)
    in_position = False
    last_buy_price = None
    last_buy_index = -1
    
    # Find both buy and sell signals
    for i in range(1, len(stock_data)):
        # Buy signal: short-term crosses above long-term
        if not stock_data['short_above_long'].iloc[i-1] and stock_data['short_above_long'].iloc[i] and not in_position:
            stock_data.iloc[i, stock_data.columns.get_loc('buy_signal')] = True
            last_buy_price = stock_data['close'].iloc[i]
            last_buy_index = i
            in_position = True
        
        # Sell signal - only if we're in a position and one of these conditions is met:
        # 1. Close price is lower than previous close price for the most recent "buying_signal"
        # 2. Close price is lower than specified EMA
        elif in_position and (
            (last_buy_price is not None and i > last_buy_index and stock_data['close'].iloc[i] < last_buy_price) or 
            (stock_data['close'].iloc[i] < stock_data[sell_signal_ema].iloc[i])
        ):
            stock_data.iloc[i, stock_data.columns.get_loc('sell_signal')] = True
            in_position = False  # Reset position status after selling
    
    return stock_data

def has_recent_crossover(ticker, days_to_check=3, days_back=120, ema_for_sell="EMA8"):
    """
    Check if a fund has a recent crossover (buy or sell signal).
    
    Args:
        ticker (str): Fund ticker symbol
        days_to_check (int): Number of days to check for recent signals
        days_back (int): Number of days of data to fetch
        ema_for_sell (str): EMA to use for sell signal detection
        
    Returns:
        tuple: (has_crossover, stock_data) - Boolean indicating if crossover found, and data if found
    """
    try:
        # Fetch and process stock data
        stock_data = fetch_and_process_fund_data(ticker, days_back)
        
        if stock_data is None or stock_data.empty:
            return False, None
        
        # Detect signals
        stock_data = detect_crossover_signals(stock_data, ema_for_sell)
        
        # Check if there's a crossover in the last 'days_to_check' days
        recent_data = stock_data.iloc[-days_to_check:]
        has_crossover = recent_data['buy_signal'].any() or recent_data['sell_signal'].any()
        
        return has_crossover, stock_data if has_crossover else None
    except Exception as e:
        print(f"Error checking {ticker}: {str(e)}")
        return False, None

def scan_for_signals(funds_df, days_to_check=3, history_days=365*3, ema_for_sell="EMA8", max_funds=500):
    """
    Scan multiple funds for recent crossover signals.
    
    Args:
        funds_df (DataFrame): DataFrame containing funds to scan
        days_to_check (int): Number of days to check for recent signals
        history_days (int): Number of days of history to analyze
        ema_for_sell (str): EMA to use for sell signal detection
        max_funds (int): Maximum number of funds to return
        
    Returns:
        list: List of tuples (ticker, name, stock_data) for funds with signals
    """
    crossover_funds = []
    
    for i, row in funds_df.iterrows():
        ticker = row['基金代码']
        name = row['基金名称']
        
        # Check for crossover
        has_crossover, stock_data = has_recent_crossover(
            ticker, 
            days_to_check=days_to_check,
            days_back=history_days,
            ema_for_sell=ema_for_sell
        )
        
        if has_crossover:
            # Add to crossover list
            crossover_funds.append((ticker, name, stock_data))
        
        # Check if we have found enough funds
        if len(crossover_funds) >= max_funds:
            break
    
    return crossover_funds 