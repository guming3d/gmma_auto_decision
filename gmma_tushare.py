import streamlit as st
import tushare as ts
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# App title and description
st.title("Guppy Multiple Moving Average (GMMA) Plot")
st.markdown("""
This app displays the Guppy Multiple Moving Average (GMMA) plot for a given Chinese stock using Tushare data.  
Enter a stock code (e.g., '600000.SH' for Shanghai stocks or '000001.SZ' for Shenzhen stocks) in the sidebar and select which EMAs to display.
""")

# Sidebar for inputs
st.sidebar.title("Stock Input")
ticker = st.sidebar.text_input("Enter Chinese stock code (e.g., 600000.SH)", "600000.SH")

st.sidebar.title("Display Options")
show_short_term = st.sidebar.checkbox("Show Short-term EMAs", value=True)
show_long_term = st.sidebar.checkbox("Show Long-term EMAs", value=True)

# Set Tushare token using Streamlit secrets
token = st.secrets["tushare"]["token"]
ts.set_token("74eea4b1a71952d5a08f4495f9d375f67e9354585e2b2ad82338b956")
pro = ts.pro_api("74eea4b1a71952d5a08f4495f9d375f67e9354585e2b2ad82338b956")

# Calculate date range (past 6 months)
end_date = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=180)).strftime('%Y%m%d')

# Fetch stock data
with st.spinner("Fetching data..."):
    try:
        # Fetch daily stock data
        #stock_data = pro.daily(ts_code=ticker, start_date=start_date, end_date=end_date)
        stock_data = pro.query('daily', ts_code='000001.SZ', start_date='20180701', end_date='20180718')
        if stock_data.empty:
            st.error("No data found for the given stock code. Please check the code and try again.")
        else:
            # Convert trade_date to datetime and set as index
            stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
            stock_data.set_index('trade_date', inplace=True)
            stock_data.sort_index(inplace=True)

            # Fetch stock name (optional)
            stock_info = pro.stock_basic(ts_code=ticker, fields='ts_code,name')
            stock_name = stock_info['name'].values[0] if not stock_info.empty else "Unknown"
            st.header(f"{stock_name} ({ticker})")

            # Calculate EMAs
            for period in [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]:
                stock_data[f"EMA{period}"] = stock_data["close"].ewm(span=period, adjust=False).mean()

            # Create Plotly figure
            fig = go.Figure()

            # Add closing price
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data["close"],
                mode="lines",
                name="Close Price",
                line=dict(color="black")
            ))

            # Add short-term EMAs
            if show_short_term:
                for i, period in enumerate([3, 5, 8, 10, 12, 15]):
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data[f"EMA{period}"],
                        mode="lines",
                        name="Short-term EMAs",
                        line=dict(color="blue"),
                        legendgroup="short_term",
                        showlegend=(i == 0)
                    ))

            # Add long-term EMAs
            if show_long_term:
                for i, period in enumerate([30, 35, 40, 45, 50, 60]):
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data[f"EMA{period}"],
                        mode="lines",
                        name="Long-term EMAs",
                        line=dict(color="red"),
                        legendgroup="long_term",
                        showlegend=(i == 0)
                    ))

            # Customize plot layout
            fig.update_layout(
                title=f"{stock_name} GMMA Plot",
                xaxis_title="Date",
                yaxis_title="Price",
                legend_title="Legend",
                hovermode="x unified",
                template="plotly_white"
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}. Please ensure your token is valid and the stock code is correct.")

