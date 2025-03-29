"""
Chart utility functions for creating GMMA charts and visualizations.
"""
import plotly.graph_objects as go
import pandas as pd
from config import SHORT_TERM_PERIODS, LONG_TERM_PERIODS

def create_gmma_chart(stock_data, ticker=None, name=None, show_short_term=True, show_long_term=True):
    """
    Create a GMMA (Guppy Multiple Moving Average) chart using Plotly.
    
    Args:
        stock_data (DataFrame): Processed fund data with EMAs and signals
        ticker (str, optional): Fund ticker symbol
        name (str, optional): Fund name
        show_short_term (bool): Whether to show short-term EMAs
        show_long_term (bool): Whether to show long-term EMAs
        
    Returns:
        plotly.graph_objects.Figure: The plotly figure object
    """
    # Create figure
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
    if show_short_term:
        for i, period in enumerate(SHORT_TERM_PERIODS):
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data[f"EMA{period}"],
                mode="lines",
                name=f"EMA{period}",
                line=dict(color="skyblue", width=1),
                legendgroup="short_term",
                showlegend=(i == 0)
            ))
    
    # Add long-term EMAs (red)
    if show_long_term:
        for i, period in enumerate(LONG_TERM_PERIODS):
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data[f"EMA{period}"],
                mode="lines",
                name=f"EMA{period}",
                line=dict(color="lightcoral", width=1),
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
    
    # Add buy and sell signals
    add_signal_annotations(fig, stock_data)
    
    # Add signal summary annotation
    add_signal_summary(fig, stock_data)
    
    # Set layout
    title = f"GMMA 图表"
    if ticker:
        title = f"{ticker}" + (f" - {name}" if name else "") + f" {title}"
    
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="价格",
        legend_title="图例",
        hovermode="x unified",
        template="plotly_white",
        height=800
    )
    
    return fig

def add_signal_annotations(fig, stock_data):
    """
    Add buy and sell signal annotations to the chart.
    
    Args:
        fig (plotly.graph_objects.Figure): The chart figure
        stock_data (DataFrame): Processed fund data with signals
        
    Returns:
        plotly.graph_objects.Figure: The updated figure
    """
    # Find buy and sell signal dates
    buy_dates = stock_data[stock_data['buy_signal']].index
    sell_dates = stock_data[stock_data['sell_signal']].index
    
    # Add buy signals
    for date in buy_dates:
        price_at_signal = stock_data.loc[date, 'close']
        # Add buy annotation - arrow pointing upward from below
        fig.add_annotation(
            x=date,
            y=price_at_signal * 1.08,  # Move text higher
            text=f"买入信号 {date.strftime('%Y-%m-%d')}",
            showarrow=True,
            arrowhead=1,
            arrowcolor="green",
            arrowsize=1,
            arrowwidth=2,
            font=dict(color="green", size=12),
            ax=0,  # No horizontal shift
            ay=-40  # Move arrow start point down
        )
    
    # Add sell signals
    for date in sell_dates:
        price_at_signal = stock_data.loc[date, 'close']
        # Add sell annotation - arrow pointing downward from above
        fig.add_annotation(
            x=date,
            y=price_at_signal * 0.92,  # Move text lower
            text=f"卖出信号 {date.strftime('%Y-%m-%d')}",
            showarrow=True,
            arrowhead=1,
            arrowcolor="red",
            arrowsize=1,
            arrowwidth=2,
            font=dict(color="red", size=12),
            ax=0,  # No horizontal shift
            ay=40  # Move arrow start point up
        )
    
    return fig

def add_signal_summary(fig, stock_data):
    """
    Add a summary of signals to the chart.
    
    Args:
        fig (plotly.graph_objects.Figure): The chart figure
        stock_data (DataFrame): Processed fund data with signals
        
    Returns:
        plotly.graph_objects.Figure: The updated figure
    """
    # Count buy and sell signals
    buy_dates = stock_data[stock_data['buy_signal']].index
    sell_dates = stock_data[stock_data['sell_signal']].index
    
    buy_count = len(buy_dates)
    sell_count = len(sell_dates)
    last_buy = buy_dates[-1].strftime('%Y-%m-%d') if buy_count > 0 else "None"
    last_sell = sell_dates[-1].strftime('%Y-%m-%d') if sell_count > 0 else "None"
    
    signal_info = (
        f"**买入信号**: 共 {buy_count} 个, 最近信号日期: {last_buy}<br>"
        f"**卖出信号**: 共 {sell_count} 个, 最近信号日期: {last_sell}"
    )
    
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=signal_info,
        showarrow=False,
        font=dict(size=14),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="left"
    )
    
    return fig

def create_backtest_metrics(backtest_results):
    """
    Create metrics data for a backtest.
    
    Args:
        backtest_results (dict): Results from a backtest
        
    Returns:
        list: List of metrics dictionaries for Streamlit columns
    """
    metrics = []
    
    # Strategy final value metric
    metrics.append({
        'label': "信号策略最终价值",
        'value': f"¥{backtest_results['final_value']:,.2f}",
        'delta': f"{backtest_results['signal_return_pct']:.2f}%"
    })
    
    # Buy and hold metric
    metrics.append({
        'label': "买入并持有策略",
        'value': f"¥{backtest_results['buy_hold_value']:,.2f}",
        'delta': f"{backtest_results['buy_hold_return_pct']:.2f}%"
    })
    
    # Comparison metric
    delta = backtest_results['signal_return_pct'] - backtest_results['buy_hold_return_pct']
    metrics.append({
        'label': "信号vs买入持有",
        'value': f"{delta:.2f}%",
        'delta': delta
    })
    
    return metrics

def format_trades_dataframe(trades_df):
    """
    Format a trades dataframe for display.
    
    Args:
        trades_df (DataFrame): Dataframe of trades
        
    Returns:
        styled DataFrame: Styled dataframe for display
    """
    if 'gain_loss' in trades_df.columns:
        # Function to color-code gain/loss values
        def color_gain_loss(val):
            if pd.isna(val):
                return ''
            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
            return f'color: {color}'
        
        # First apply styling to the numeric data
        styled_df = trades_df.style.map(
            color_gain_loss, 
            subset=['gain_loss', 'gain_loss_pct']
        )
        
        # Then format the display values (this doesn't affect the styling)
        styled_df = styled_df.format({
            'price': lambda x: f"{x:.6f}" if not pd.isna(x) else "",
            'units': lambda x: f"{x:.2f}" if not pd.isna(x) else "",
            'cost': lambda x: f"{x:.6f}" if not pd.isna(x) else "",
            'cash': lambda x: f"{x:.6f}" if not pd.isna(x) else "",
            'position_value': lambda x: f"{x:.6f}" if not pd.isna(x) else "",
            'total_value': lambda x: f"{x:.6f}" if not pd.isna(x) else "",
            'proceeds': lambda x: f"{x:.6f}" if not pd.isna(x) else "",
            'gain_loss': lambda x: f"¥{x:,.2f}" if not pd.isna(x) else "",
            'gain_loss_pct': lambda x: f"{x:.2f}%" if not pd.isna(x) else ""
        })
        
        return styled_df
    else:
        # If no gain/loss columns, still format the other numeric columns
        if not trades_df.empty:
            styled_df = trades_df.style.format({
                'price': lambda x: f"{x:.6f}" if not pd.isna(x) else "",
                'units': lambda x: f"{x:.2f}" if not pd.isna(x) else "",
                'cost': lambda x: f"{x:.6f}" if not pd.isna(x) else "",
                'cash': lambda x: f"{x:.6f}" if not pd.isna(x) else "",
                'position_value': lambda x: f"{x:.6f}" if not pd.isna(x) else "",
                'total_value': lambda x: f"{x:.6f}" if not pd.isna(x) else "",
                'proceeds': lambda x: f"{x:.6f}" if not pd.isna(x) else "",
            })
            return styled_df
        return trades_df 