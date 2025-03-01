import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# App title and description
st.title("Guppy Multiple Moving Average (GMMA) Plot")
st.markdown("""
This app displays the Guppy Multiple Moving Average (GMMA) plot for a given stock ticker. 
Enter the stock ticker in the sidebar and choose which moving averages to display.
""")

# Sidebar for inputs
st.sidebar.title("Stock Input")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL for Apple)", "AAPL")

st.sidebar.title("Display Options")
show_short_term = st.sidebar.checkbox("Show Short-term EMAs", value=True)
show_long_term = st.sidebar.checkbox("Show Long-term EMAs", value=True)

# Fetch stock data
with st.spinner("Fetching data..."):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)  # Approximately 6 months
    try:
        # Download stock data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            st.error("No data found for the given ticker.")
        else:
            # Get stock name for display
            stock_info = yf.Ticker(ticker).info
            stock_name = stock_info.get("longName", "Unknown")
            st.header(f"{stock_name} ({ticker})")

            # Calculate EMAs
            for period in [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]:
                stock_data[f"EMA{period}"] = stock_data["Close"].ewm(span=period, adjust=False).mean()

            # Create Plotly figure
            fig = go.Figure()

            # Add closing price
            fig.add_trace(go.Scatter(
                x=stock_data.index, 
                y=stock_data["Close"], 
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

            # Customize the plot layout
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
        st.error(f"Error fetching data: {e}")
