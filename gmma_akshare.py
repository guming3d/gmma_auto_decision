import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# App title and description
st.title("顾比多重移动平均线 (GMMA) 图表")
st.markdown("""
此应用程序显示使用 akshare 数据的中国 A 股股票的古普利多重移动平均线 (GMMA) 图表。  
在侧边栏输入 6 位股票代码（例如，平安银行的 '000001'）并选择要显示的 EMA。
""")

# Sidebar for user inputs
st.sidebar.title("股票输入")
ticker = st.sidebar.text_input("输入 6 位股票代码（例如，000001）", "000001")

st.sidebar.title("显示选项")
show_short_term = st.sidebar.checkbox("显示短期 EMA", value=True)
show_long_term = st.sidebar.checkbox("显示长期 EMA", value=True)

# Calculate date range for the past 6 months
end_date = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=180)).strftime('%Y%m%d')

# Fetch and process stock data
with st.spinner("Fetching data..."):
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
                        stock_data['crossover'].iloc[i] = True
                
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
