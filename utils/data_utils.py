"""
Data utilities for fetching and processing fund data.
"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache

def format_ticker(ticker):
    """Format ticker symbols to ensure consistent format."""
    return ticker.split('.')[0].zfill(6)

def get_date_range(days_back):
    """
    Get the date range for data fetching.
    
    Args:
        days_back (int): Number of days to look back
        
    Returns:
        tuple: (start_date, end_date) formatted as YYYYMMDD
    """
    end_date = datetime.today().strftime('%Y%m%d')
    start_date = (datetime.today() - timedelta(days=days_back)).strftime('%Y%m%d')
    return start_date, end_date

def fetch_fund_data(ticker, days_back=365*3):
    """
    Fetch fund data using akshare.
    
    Args:
        ticker (str): Fund ticker symbol
        days_back (int): Number of days to look back
        
    Returns:
        DataFrame: Processed fund data
    """
    # Format ticker
    ticker = format_ticker(ticker)
    
    # Get date range
    start_date, end_date = get_date_range(days_back)
    
    # Fetch data
    try:
        stock_data = ak.fund_etf_hist_em(
            symbol=ticker, 
            period="daily", 
            start_date=start_date, 
            end_date=end_date, 
            adjust=""
        )
        
        if stock_data.empty:
            return None
        
        # Rename columns and process data
        stock_data.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open'}, inplace=True)
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data.set_index('date', inplace=True)
        stock_data.sort_index(inplace=True)
        
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_emas(stock_data, periods=None):
    """
    Calculate EMAs for the given periods.
    
    Args:
        stock_data (DataFrame): Fund price data
        periods (list): List of periods for EMA calculation
        
    Returns:
        DataFrame: Data with EMA columns added
    """
    if periods is None:
        from config import SHORT_TERM_PERIODS, LONG_TERM_PERIODS
        periods = SHORT_TERM_PERIODS + LONG_TERM_PERIODS
    
    for period in periods:
        stock_data[f"EMA{period}"] = stock_data["close"].ewm(span=period, adjust=False).mean()
    
    return stock_data

def calculate_average_emas(stock_data):
    """
    Calculate average of short-term and long-term EMAs.
    
    Args:
        stock_data (DataFrame): Fund data with EMAs calculated
        
    Returns:
        DataFrame: Data with average EMAs added
    """
    from config import SHORT_TERM_PERIODS, LONG_TERM_PERIODS
    
    # Calculate average EMAs
    stock_data['avg_short_ema'] = stock_data[[f'EMA{period}' for period in SHORT_TERM_PERIODS]].mean(axis=1)
    stock_data['avg_long_ema'] = stock_data[[f'EMA{period}' for period in LONG_TERM_PERIODS]].mean(axis=1)
    
    return stock_data

def fetch_and_process_fund_data(ticker, days_back=365*3):
    """
    Fetch and process fund data with all EMAs calculated.
    
    Args:
        ticker (str): Fund ticker symbol
        days_back (int): Number of days to look back
        
    Returns:
        DataFrame: Fully processed fund data with EMAs
    """
    # Fetch basic data
    stock_data = fetch_fund_data(ticker, days_back)
    
    if stock_data is None or stock_data.empty:
        return None
    
    # Calculate EMAs
    stock_data = calculate_emas(stock_data)
    
    # Calculate average EMAs
    stock_data = calculate_average_emas(stock_data)
    
    return stock_data

def fetch_funds_list(indicator="增强指数型"):
    """
    Fetch list of funds based on indicator.
    
    Args:
        indicator (str): Fund indicator type
        
    Returns:
        DataFrame: List of funds
    """
    try:
        return ak.fund_info_index_em(symbol="沪深指数", indicator=indicator)
    except Exception as e:
        print(f"Error fetching funds list: {str(e)}")
        return pd.DataFrame() 