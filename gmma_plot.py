import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Function to fetch stock data with retries and longer delay
def fetch_stock_data(stock_symbol, retries=5, delay=30):
    for attempt in range(retries):
        try:
            # Download stock data with auto_adjust=True (default now)
            data = yf.download(stock_symbol, period="6mo", interval="1d", auto_adjust=True)
            if not data.empty:
                return data
            else:
                st.warning(f"Data for {stock_symbol} is empty. Retrying...")
                time.sleep(delay)
        except yf.YFRateLimitError:
            # If rate limited, wait and try again
            st.warning(f"Rate limited for {stock_symbol}. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            # Handle other exceptions
            st.error(f"Error fetching data for {stock_symbol}: {e}")
            break
    return None  # Return None if all attempts fail

# Function to compute EMAs for GMMA
def compute_gmma(data):
    short_emas = [3, 5, 8, 10, 12, 15]
    long_emas = [30, 35, 40, 45, 50, 60]
    
    for period in short_emas:
        data[f'Short_EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
    
    for period in long_emas:
        data[f'Long_EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
    
    return data

# Streamlit UI
st.title('Guppy Multiple Moving Average (GMMA) Plot')

# Input for stock symbol
stock_symbol = st.text_input('Enter Stock Symbol (e.g., AAPL, MSFT, TSLA)', 'AAPL')

if stock_symbol:
    # Fetch stock data
    data = fetch_stock_data(stock_symbol)
    
    if data is not None:
        # Compute GMMA
        data_with_gmma = compute_gmma(data)
        
        # Plotting the stock prices and GMMA
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(data_with_gmma.index, data_with_gmma['Close'], label='Close Price', color='black', linewidth=1)
        
        # Plot Short EMAs
        for period in [3, 5, 8, 10, 12, 15]:
            ax.plot(data_with_gmma.index, data_with_gmma[f'Short_EMA_{period}'], label=f'Short EMA {period}', linestyle='-', linewidth=1)
        
        # Plot Long EMAs
        for period in [30, 35, 40, 45, 50, 60]:
            ax.plot(data_with_gmma.index, data_with_gmma[f'Long_EMA_{period}'], label=f'Long EMA {period}', linestyle='--', linewidth=1)
        
        # Add title and labels
        ax.set_title(f'{stock_symbol} - Guppy Multiple Moving Average (GMMA)', fontsize=16)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper left', fontsize=10)

        # Display the plot
        st.pyplot(fig)
    else:
        st.error("Failed to fetch stock data. Please try again later.")
