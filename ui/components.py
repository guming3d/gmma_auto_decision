"""
UI components for the GMMA application.
"""
import streamlit as st
import pandas as pd
from config import PERIOD_DAYS, DEFAULT_FUNDS

def setup_page():
    """
    Setup the page configuration and title.
    """
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

def setup_sidebar():
    """
    Set up the sidebar with all input options.
    
    Returns:
        dict: Dictionary containing all sidebar settings
    """
    settings = {}
    
    # Analysis mode selection
    st.sidebar.title("分析模式")
    settings['analysis_mode'] = st.sidebar.radio("选择模式", ["指定基金分析", "基金全扫描"], index=0)
    
    # Signal settings
    st.sidebar.title("信号设置")
    settings['sell_signal_ema'] = st.sidebar.selectbox(
        "卖出信号比较的短期EMA", 
        options=["EMA3", "EMA5", "EMA8", "EMA10"],
        index=2,  # Default to EMA8
        help="当价格低于所选EMA时，可能触发卖出信号"
    )
    
    # Backtest settings
    st.sidebar.title("回测设置")
    settings['backtest_strategy'] = st.sidebar.radio(
        "回测策略",
        options=["常规策略", "百分比策略"],
        index=0,
        help="常规策略: 固定单位买卖; 百分比策略: 按资金比例投资，保留30%现金"
    )
    
    # Historical data period selection
    settings['history_period'] = st.sidebar.selectbox(
        "历史数据周期",
        options=list(PERIOD_DAYS.keys()),
        index=7,  # Default to 3年
        help="选择用于分析和回测的历史数据范围"
    )
    
    # Convert selected period to days
    settings['history_days'] = PERIOD_DAYS[settings['history_period']]
    
    # Display current settings
    st.sidebar.markdown(f"**当前卖出信号设置**: 当价格低于买入信号时的价格，或价格低于**{settings['sell_signal_ema']}**时产生卖出信号")
    
    # Additional settings based on mode
    if settings['analysis_mode'] == "基金全扫描":
        setup_scan_settings(settings)
    else:  # "指定基金分析"
        setup_individual_analysis_settings(settings)
    
    return settings

def setup_scan_settings(settings):
    """
    Set up settings specific to fund scanning mode.
    
    Args:
        settings (dict): Settings dictionary to update
    """
    st.sidebar.title("基金扫描设置")
    settings['scan_days_to_check'] = st.sidebar.slider("检查最近几天内的信号", 1, 7, 4)
    settings['scan_max_funds'] = st.sidebar.slider("最多显示基金数量", 1, 500, 500)
    settings['start_scan'] = st.sidebar.button("开始扫描基金")

def setup_individual_analysis_settings(settings):
    """
    Set up settings specific to individual fund analysis mode.
    
    Args:
        settings (dict): Settings dictionary to update
    """
    st.sidebar.title("市场选择")
    settings['market_type'] = st.sidebar.radio("选择市场", ["A股"])
    
    st.sidebar.title("基金输入")
    default_funds = DEFAULT_FUNDS
    if settings['market_type'] == "A股":
        settings['funds_input'] = st.sidebar.text_area(
            "输入基金代码（多个代码用逗号分隔）", 
            value=default_funds,
            height=100
        )
        ticker_example = "示例：510300 (沪深300ETF), 510050 (上证50ETF)"
    
    st.sidebar.caption(ticker_example)
    
    st.sidebar.title("显示选项")
    settings['show_short_term'] = st.sidebar.checkbox("显示短期 EMA", value=True)
    settings['show_long_term'] = st.sidebar.checkbox("显示长期 EMA", value=True)

def display_signal_tables(buy_dates, sell_dates):
    """
    Display buy and sell signal dates in tables.
    
    Args:
        buy_dates (Index): Dates of buy signals
        sell_dates (Index): Dates of sell signals
    """
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

def display_backtest_results(backtest_results, strategy="常规策略", units=100):
    """
    Display backtest results including metrics and trades.
    
    Args:
        backtest_results (dict): Results from backtesting
        strategy (str): Strategy description
        units (int): Number of units for standard strategy
    """
    from utils.chart_utils import create_backtest_metrics, format_trades_dataframe
    
    st.subheader("回归测试")
    
    # Display strategy description
    if strategy == "常规策略":
        st.markdown(f"""该回归测试模拟了严格按照买入和卖出信号操作的结果，每次操作购买或卖出{units}单位，以验证信号的有效性。""")
    else:
        st.markdown("""该回归测试模拟了按比例投资的策略：
        1. 初始资金10万，至少保留30%现金
        2. 每次买入信号使用当前总资产的10%购买股票
        3. 当现金不足10%时，等待卖出信号卖出50%持仓
        """)
    
    # Display metrics in columns
    metrics = create_backtest_metrics(backtest_results)
    cols = st.columns(3)
    
    for i, metric in enumerate(metrics):
        with cols[i]:
            st.metric(
                label=metric['label'], 
                value=metric['value'],
                delta=metric['delta']
            )
    
    # Display trades table
    if backtest_results['trades']:
        st.subheader("交易记录")
        trades_df = pd.DataFrame(backtest_results['trades'])
        
        # Format the dataframe
        styled_df = format_trades_dataframe(trades_df)
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.warning("回测期间没有产生交易。")

def display_fund_summary(crossover_funds):
    """
    Display a summary table of funds with signals.
    
    Args:
        crossover_funds (list): List of tuples (ticker, name, stock_data)
    """
    if not crossover_funds:
        return
    
    # Create a summary table
    summary_df = pd.DataFrame(
        [(t, n) for t, n, _ in crossover_funds], 
        columns=["基金代码", "基金名称"]
    )
    st.subheader("基金买入信号列表")
    st.table(summary_df) 