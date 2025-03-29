"""
Fund scanning mode functionality.
"""
import streamlit as st
import pandas as pd
from models.signal_detection import scan_for_signals
from utils.data_utils import fetch_funds_list
from utils.chart_utils import create_gmma_chart
from models.backtest import perform_standard_backtest, perform_percentage_backtest
from ui.components import display_signal_tables, display_backtest_results, display_fund_summary
from config import DEFAULT_BACKTEST_UNITS

def run_scan_mode(settings):
    """
    Run the fund scanning mode with the specified settings.
    
    Args:
        settings (dict): Dictionary of UI settings
    """
    if not settings.get('start_scan'):
        st.info("请点击'开始扫描基金'按钮以查找最近出现买入信号的基金。")
        return
    
    with st.spinner("正在扫描基金买入信号，这可能需要一些时间..."):
        try:
            # Fetch funds list
            etf_stocks_df = fetch_funds_list(indicator="增强指数型")
            
            # Show info about scanning
            st.info(f"准备扫描 {len(etf_stocks_df)} 只基金...")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Scan for signals
            crossover_funds = scan_for_signals(
                etf_stocks_df,
                days_to_check=settings['scan_days_to_check'],
                history_days=settings['history_days'],
                ema_for_sell=settings['sell_signal_ema'],
                max_funds=settings['scan_max_funds']
            )
            
            # Final update and display results
            progress_bar.progress(1.0)
            
            if len(crossover_funds) == 0:
                st.warning(f"没有找到在最近 {settings['scan_days_to_check']} 天内出现买入信号的基金。")
                return
            
            st.success(f"找到 {len(crossover_funds)} 只在最近 {settings['scan_days_to_check']} 天内出现买入信号的基金。")
            
            # Display summary table
            display_fund_summary(crossover_funds)
            
            # Display detailed results for each fund
            for ticker, name, stock_data in crossover_funds:
                display_fund_details(ticker, name, stock_data, settings)
        
        except Exception as e:
            st.error(f"扫描基金过程中出错: {str(e)}")

def display_fund_details(ticker, name, stock_data, settings):
    """
    Display detailed analysis for a single fund.
    
    Args:
        ticker (str): Fund ticker symbol
        name (str): Fund name
        stock_data (DataFrame): Processed fund data with signals
        settings (dict): UI settings
    """
    with st.expander(f"{ticker} - {name}", expanded=True):
        # Create GMMA chart
        fig = create_gmma_chart(stock_data, ticker, name)
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Get buy and sell signal dates
        buy_dates = stock_data[stock_data['buy_signal']].index
        sell_dates = stock_data[stock_data['sell_signal']].index
        
        # Display signal tables if there are any signals
        if len(buy_dates) > 0 or len(sell_dates) > 0:
            display_signal_tables(buy_dates, sell_dates)
        
        # Display notification about which EMA is used for sell signals
        st.info(f"当前卖出信号条件: 1) 价格低于买入信号时的价格，或 2) 价格低于**{settings['sell_signal_ema']}**")
        
        # Run backtesting
        if settings['backtest_strategy'] == "常规策略":
            backtest_results = perform_standard_backtest(stock_data, units=DEFAULT_BACKTEST_UNITS)
            display_backtest_results(backtest_results, strategy="常规策略", units=DEFAULT_BACKTEST_UNITS)
        else:
            backtest_results = perform_percentage_backtest(stock_data)
            display_backtest_results(backtest_results, strategy="百分比策略") 