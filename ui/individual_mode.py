"""
Individual fund analysis mode functionality.
"""
import streamlit as st
import pandas as pd
from utils.data_utils import fetch_and_process_fund_data
from models.signal_detection import detect_crossover_signals
from utils.chart_utils import create_gmma_chart
from models.backtest import perform_standard_backtest, perform_percentage_backtest
from ui.components import display_signal_tables, display_backtest_results
from config import DEFAULT_BACKTEST_UNITS

def run_individual_mode(settings):
    """
    Run the individual fund analysis mode with the specified settings.
    
    Args:
        settings (dict): Dictionary of UI settings
    """
    # Process the input funds
    fund_list = [fund.strip() for fund in settings['funds_input'].split(",") if fund.strip()]
    
    # Update the sell signal info message
    st.info(f"当前卖出信号条件: 1) 价格低于买入信号时的价格，或 2) 价格低于{settings['sell_signal_ema']}")
    
    # Create tabs for each fund
    if fund_list:
        tabs = st.tabs(fund_list)
        
        # Analyze each fund in its own tab
        for idx, ticker in enumerate(fund_list):
            with tabs[idx]:
                analyze_individual_fund(ticker, settings)

def analyze_individual_fund(ticker, settings):
    """
    Analyze an individual fund and display results.
    
    Args:
        ticker (str): Fund ticker symbol
        settings (dict): UI settings dictionary
    """
    with st.spinner(f"获取 {ticker} 数据中..."):
        try:
            # Fetch and process fund data
            stock_data = fetch_and_process_fund_data(ticker, settings['history_days'])
            
            if stock_data is None or stock_data.empty:
                st.error(f"未找到基金代码 {ticker} 的数据。请检查代码并重试。")
                return
            
            # Detect buy and sell signals
            stock_data = detect_crossover_signals(stock_data, settings['sell_signal_ema'])
            
            # Create GMMA chart
            fig = create_gmma_chart(
                stock_data, 
                ticker=ticker,
                show_short_term=settings['show_short_term'],
                show_long_term=settings['show_long_term']
            )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display signal tables
            buy_dates = stock_data[stock_data['buy_signal']].index
            sell_dates = stock_data[stock_data['sell_signal']].index
            
            if len(buy_dates) > 0 or len(sell_dates) > 0:
                display_signal_tables(buy_dates, sell_dates)
                
                # Run backtesting
                if settings['backtest_strategy'] == "常规策略":
                    backtest_results = perform_standard_backtest(stock_data, units=DEFAULT_BACKTEST_UNITS)
                    display_backtest_results(
                        backtest_results, 
                        strategy="常规策略", 
                        units=DEFAULT_BACKTEST_UNITS
                    )
                else:
                    backtest_results = perform_percentage_backtest(stock_data)
                    display_backtest_results(backtest_results, strategy="百分比策略")
            else:
                st.warning("该基金在分析期间内未检测到任何买入或卖出信号。")
        
        except Exception as e:
            st.error(f"分析基金 {ticker} 时出错: {str(e)}") 