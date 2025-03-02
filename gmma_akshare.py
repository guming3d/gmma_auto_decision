import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import time

# App title and description
st.title("顾比多重移动平均线 (GMMA) 图表")
st.markdown("""
此应用程序显示使用 akshare 数据的中国 A 股股票的古普利多重移动平均线 (GMMA) 图表。  
可以分析单个股票或自动扫描最近出现买入信号的股票。
""")

# Function to check if a stock has a recent crossover
def has_recent_crossover(ticker, days_to_check=3):
    try:
        # Calculate date range for the past 2 months (enough data to calculate EMAs)
        end_date = datetime.today().strftime('%Y%m%d')
        start_date = (datetime.today() - timedelta(days=120)).strftime('%Y%m%d')
        
        # Fetch stock data using akshare
        stock_data = ak.stock_zh_a_hist(symbol=ticker, period="daily", 
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
                # Replace: stock_data['crossover'].iloc[i] = True
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
analysis_mode = st.sidebar.radio("选择模式", ["自动扫描买入信号","单一股票分析"])

if analysis_mode == "单一股票分析":
    # Single stock analysis mode - similar to the original code
    st.sidebar.title("股票输入")
    ticker = st.sidebar.text_input("输入 6 位股票代码（例如，000001）", "000001")
    
    st.sidebar.title("显示选项")
    show_short_term = st.sidebar.checkbox("显示短期 EMA", value=True)
    show_long_term = st.sidebar.checkbox("显示长期 EMA", value=True)
    
    # Calculate date range for the past 6 months
    end_date = datetime.today().strftime('%Y%m%d')
    start_date = (datetime.today() - timedelta(days=180)).strftime('%Y%m%d')
    
    # Fetch and process stock data
    with st.spinner("获取数据中..."):
        try:
            # Remove exchange suffix if present (e.g., '000001.SZ' -> '000001')
            ticker = ticker.split('.')[0]
            if not ticker.isdigit() or len(ticker) != 6:
                st.error("请输入有效的 6 位股票代码。")
            else:
                # Fetch stock data using akshare
                stock_data = ak.stock_zh_a_hist(symbol=ticker, period="daily", start_date=start_date, end_date=end_date, adjust="")
                if stock_data.empty:
                    st.error("未找到所输入股票代码的数据。请检查代码并重试。")
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
                            # Replace: stock_data['crossover'].iloc[i] = True
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
                        increasing_line_color='green',
                        decreasing_line_color='red',
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
                    
                    # Customize plot layout
                    fig.update_layout(
                        title=f"股票 {ticker} GMMA 图表 (标记: 短期EMA从下方穿过长期EMA)",
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
                        signal_df = pd.DataFrame(crossover_dates, columns=["日期"])
                        signal_df["日期"] = signal_df["日期"].dt.strftime('%Y-%m-%d')
                        st.table(signal_df)
        except Exception as e:
            st.error(f"获取数据时出错: {str(e)}")

else:  # Auto scan mode
    st.sidebar.title("扫描设置")
    days_to_check = st.sidebar.slider("检查最近几天内的信号", 1, 7, 3)
    max_stocks = st.sidebar.slider("最多显示股票数量", 1, 200, 10)
    
    # Add industry selection option
    scan_mode = st.sidebar.radio("扫描范围", ["按行业板块","全部 A 股"])
    
    selected_industry = None
    
    # Industry board selection
    if scan_mode == "按行业板块":
        try:
            # Display a loading message in sidebar (no spinner available in sidebar)
            st.sidebar.text("正在加载行业板块...")
            
            # Fetch industry board list with proper spinner in main area
            with st.spinner("获取行业板块..."):
                industry_df = ak.stock_board_industry_name_em()
                industry_list = industry_df["板块名称"].tolist()
            
            # Remove loading message by replacing it with empty space
            st.sidebar.text("")
            
            # Replace selectbox with multiselect to allow multiple industry selections
            selected_industries = st.sidebar.multiselect(
                "选择行业板块 (可多选)",
                options=industry_list,
                default=[industry_list[0]] if industry_list else []
            )
            
            # Show the selected industry info
            if selected_industries:
                st.sidebar.info(f"已选择: {', '.join(selected_industries)}")
        except Exception as e:
            st.sidebar.error(f"获取行业板块失败: {str(e)}")

    if st.sidebar.button("开始扫描"):
        with st.spinner("正在扫描买入信号，这可能需要一些时间..."):
            try:
                # Variable to track if we have valid stocks to scan
                have_stocks_to_scan = True
                
                # Get stock list based on scan mode
                if scan_mode == "按行业板块" and selected_industries:
                    # Create an empty DataFrame to store all industry stocks
                    all_industry_stocks = pd.DataFrame()
                    
                    # Get stocks from each selected industry
                    for industry in selected_industries:
                        with st.spinner(f"获取 {industry} 行业的股票列表..."):
                            try:
                                industry_stocks_df = ak.stock_board_industry_cons_em(symbol=industry)
                                if not industry_stocks_df.empty:
                                    # Process industry stocks
                                    industry_stocks = industry_stocks_df[["代码", "名称"]].rename(
                                        columns={"代码": "code", "名称": "name"}
                                    )
                                    # Add to combined DataFrame
                                    all_industry_stocks = pd.concat([all_industry_stocks, industry_stocks])
                            except Exception as e:
                                st.warning(f"获取 {industry} 行业股票列表失败: {str(e)}")
                                continue
                    
                    # Remove duplicates (stocks that belong to multiple industries)
                    if not all_industry_stocks.empty:
                        stock_info_df = all_industry_stocks.drop_duplicates(subset=["code"])
                        # Remove any exchange suffix and ensure 6 digits
                        stock_info_df["code"] = stock_info_df["code"].apply(
                            lambda x: x.split('.')[0].zfill(6) if isinstance(x, str) else str(x).zfill(6)
                        )
                    else:
                        st.error("未能获取所选行业的股票列表。")
                        have_stocks_to_scan = False  # Set flag instead of using return
                else:
                    # Get all A-share stock codes and names
                    stock_info_df = ak.stock_info_a_code_name()
                
                # Only proceed if we have stocks to scan
                if have_stocks_to_scan:
                    # Show how many stocks will be scanned
                    stock_count = len(stock_info_df)
                    st.info(f"准备扫描 {stock_count} 只股票...")
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Container for results
                    results_container = st.container()
                    results_title = st.empty()
                    
                    # Counter for stocks with crossover
                    crossover_stocks = []
                    
                    # Create industry mapping dictionary for multiple industry case
                    industry_mapping = {}
                    
                    # If in industry mode, map stock codes to their industries
                    if scan_mode == "按行业板块":
                        for industry in selected_industries:
                            try:
                                industry_stocks = ak.stock_board_industry_cons_em(symbol=industry)
                                for _, row in industry_stocks.iterrows():
                                    stock_code = row["代码"].split('.')[0].zfill(6)
                                    # Use a dictionary with stock code as key and industry as value
                                    # If a stock belongs to multiple industries, use the last one (will be overwritten)
                                    industry_mapping[stock_code] = industry
                            except:
                                continue
                    
                    # Loop through selected stocks
                    for i, row in enumerate(stock_info_df.itertuples()):
                        # Update progress
                        progress_bar.progress(min((i+1)/stock_count, 1.0))
                        
                        ticker = row.code
                        name = row.name
                        
                        # Skip stocks with special prefixes only if scanning all stocks
                        if scan_mode == "全部 A 股" and ticker.startswith(('688', '300', '8', '4')):
                            continue
                            
                        # Check for crossover
                        has_crossover, stock_data = has_recent_crossover(ticker, days_to_check)
                        
                        if has_crossover:
                            # Get industry information for the stock
                            if scan_mode == "按行业板块":
                                # Use the mapped industry from our dictionary
                                industry = industry_mapping.get(ticker, "未知")
                            else:
                                # For all A-shares mode, try to get industry info directly
                                try:
                                    # First try with stock_individual_info_em
                                    stock_info = ak.stock_individual_info_em(symbol=ticker)
                                    # Extract industry info from the dataframe
                                    industry = stock_info.loc[stock_info['item'] == '所属行业', 'value'].iloc[0]
                                except:
                                    # If failed, use a placeholder
                                    industry = "未知"
                            
                            # Include industry in the crossover_stocks list
                            crossover_stocks.append((ticker, name, industry, stock_data))
                            
                            # Create tab for this stock
                            with st.expander(f"{ticker} - {name} ({industry})", expanded=True):
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
                                
                                # Mark crossover signals
                                crossover_dates = stock_data[stock_data['crossover']].index
                                for date in crossover_dates:
                                    price_at_crossover = stock_data.loc[date, 'close']
                                    # fig.add_shape(
                                    #     type="line",
                                    #     x0=date,
                                    #     y0=price_at_crossover * 0.97,
                                    #     x1=date,
                                    #     y1=price_at_crossover * 1.03,
                                    #     line=dict(color="orange", width=3),
                                    # )
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
                        
                        # Check if we found enough stocks
                        if len(crossover_stocks) >= max_stocks:
                            break
                    
                    # Final update
                    progress_bar.progress(1.0)
                    
                    # Final message
                    if len(crossover_stocks) == 0:
                        st.warning(f"没有找到在最近 {days_to_check} 天内出现买入信号的股票。")
                    else:
                        scan_scope = f"所选 {len(selected_industries)} 个行业" if scan_mode == "按行业板块" else "全部 A 股"
                        st.success(f"在{scan_scope}中找到 {len(crossover_stocks)} 只在最近 {days_to_check} 天内出现买入信号的股票。")
                        
                        # Create a summary table with industry information
                        summary_df = pd.DataFrame(
                            [(t, n, ind) for t, n, ind, _ in crossover_stocks], 
                            columns=["代码", "名称", "所属行业"]
                        )
                        st.subheader("买入信号股票列表")
                        st.table(summary_df)
                
            except Exception as e:
                st.error(f"扫描过程中出错: {str(e)}")
    else:
        if scan_mode == "按行业板块":
            if selected_industries:
                industry_count = len(selected_industries)
                industries_text = f"{industry_count} 个行业" if industry_count > 1 else selected_industries[0]
                st.info(f"请点击'开始扫描'按钮以查找 {industries_text} 中最近出现买入信号的股票。")
            else:
                st.info("请先选择至少一个行业板块，然后点击'开始扫描'按钮。")
        else:
            st.info("请点击'开始扫描'按钮以查找最近出现买入信号的股票。")
