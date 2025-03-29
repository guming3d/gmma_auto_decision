"""
Backtesting module for evaluating trading signals.
"""
import pandas as pd
import numpy as np
from config import INITIAL_CASH, MIN_CASH_RESERVE_PCT, MIN_CASH_THRESHOLD_PCT, BUY_PERCENTAGE

def perform_standard_backtest(stock_data, units=100):
    """
    Perform back testing based on buy/sell signals using a fixed unit strategy.
    
    Args:
        stock_data (DataFrame): DataFrame with 'close', 'buy_signal', and 'sell_signal' columns
        units (int): Number of units to buy/sell on each signal
        
    Returns:
        dict: Dictionary containing back testing results
    """
    # Initialize variables
    initial_cash = INITIAL_CASH
    cash = initial_cash
    position = 0  # Number of units held
    trades = []
    
    # Variables to track purchase information
    last_buy_price = None
    last_buy_units = 0
    
    # Sort data by date to ensure chronological processing
    stock_data = stock_data.sort_index()
    
    # Process each day in the data
    for date, row in stock_data.iterrows():
        price = row['close']
        
        # Process buy signal (only if not already in a position)
        if row['buy_signal'] and cash > 0 and position == 0:
            # Buy as many units as possible with available cash
            max_units = cash // price
            if max_units > 0:
                cost = price * max_units
                cash -= cost  # Deduct cost from cash
                position += max_units
                
                # Store buy price and units for later gain/loss calculation
                last_buy_price = price
                last_buy_units = max_units
                
                trades.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'action': '买入',
                    'price': price,
                    'units': max_units,
                    'cost': cost,
                    'cash': cash,  # This is cash AFTER the purchase
                    'position_value': position * price,
                    'total_value': cash + (position * price)
                })
        
        # Process sell signal (only if we have a position)
        elif row['sell_signal'] and position > 0:
            # Sell all units
            proceeds = price * position
            cash += proceeds  # Add proceeds to cash
            
            # Calculate gain/loss information
            gain_loss = 0
            gain_loss_pct = 0
            if last_buy_price is not None:
                gain_loss = (price - last_buy_price) * position
                gain_loss_pct = ((price / last_buy_price) - 1) * 100
            
            trades.append({
                'date': date.strftime('%Y-%m-%d'),
                'action': '卖出',
                'price': price,
                'units': position,
                'proceeds': proceeds,
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct,
                'cash': cash,
                'position_value': 0,
                'total_value': cash
            })
            
            # Reset position after selling all
            position = 0
            last_buy_price = None
            last_buy_units = 0
    
    # Calculate final results
    final_price = stock_data['close'].iloc[-1]
    final_position_value = position * final_price
    final_value = cash + final_position_value
    
    # Buy and hold comparison
    buy_and_hold_units = initial_cash // stock_data['close'].iloc[0]
    buy_and_hold_value = buy_and_hold_units * final_price
    
    # Calculate returns
    signal_return_pct = ((final_value - initial_cash) / initial_cash) * 100
    buy_hold_return_pct = ((buy_and_hold_value - initial_cash) / initial_cash) * 100
    
    # Results
    results = {
        'initial_cash': initial_cash,
        'final_cash': cash,
        'final_position': position,
        'final_position_value': final_position_value,
        'final_value': final_value,
        'signal_return_pct': signal_return_pct,
        'buy_hold_units': buy_and_hold_units,
        'buy_hold_value': buy_and_hold_value,
        'buy_hold_return_pct': buy_hold_return_pct,
        'trades': trades
    }
    
    return results

def perform_percentage_backtest(stock_data):
    """
    Perform back testing using a percentage-based strategy:
    1. Initial invest money is 100000, keep at least 30% of money at hand
    2. Use 10% of invest money to buy stocks on each buy signal
    3. If left money is less than 10%, waiting for the selling_signal to sell 50%
    4. Continue until latest trading day
    
    Args:
        stock_data (DataFrame): DataFrame with 'close', 'buy_signal', and 'sell_signal' columns
    
    Returns:
        dict: Dictionary containing back testing results
    """
    # Initialize variables
    initial_cash = INITIAL_CASH
    cash = initial_cash
    min_cash_reserve = initial_cash * MIN_CASH_RESERVE_PCT
    min_cash_threshold = initial_cash * MIN_CASH_THRESHOLD_PCT
    position = 0  # Number of units held
    trades = []
    
    # Variables to track purchase information
    position_history = {}  # To track buys at different prices
    
    # Sort data by date to ensure chronological processing
    stock_data = stock_data.sort_index()
    
    # Process each day in the data
    for date, row in stock_data.iterrows():
        price = row['close']
        current_value = cash + (position * price)
        
        # Process buy signal (if we have enough cash)
        if row['buy_signal']:
            # Calculate available money for this purchase
            available_money = cash - min_cash_reserve
            
            # Only buy if we have enough money (at least 10% of initial investment)
            if available_money >= (initial_cash * MIN_CASH_THRESHOLD_PCT):
                # Calculate money to spend (10% of current value)
                buy_amount = current_value * BUY_PERCENTAGE
                
                # Make sure we don't go below our cash reserve
                if (cash - buy_amount) < min_cash_reserve:
                    buy_amount = cash - min_cash_reserve
                
                if buy_amount > 0:
                    # Calculate units to buy
                    units_to_buy = int(buy_amount // price)
                    
                    if units_to_buy > 0:
                        cost = price * units_to_buy
                        cash -= cost  # Deduct cost from cash
                        position += units_to_buy
                        
                        # Track this purchase in position history
                        if price not in position_history:
                            position_history[price] = 0
                        position_history[price] += units_to_buy
                        
                        trades.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'action': '买入',
                            'price': price,
                            'units': units_to_buy,
                            'cost': cost,
                            'cash': cash,  # This is cash AFTER the purchase
                            'position_value': position * price,
                            'total_value': cash + (position * price)
                        })
        
        # Process sell signal 
        elif row['sell_signal'] and position > 0 and cash < min_cash_threshold:
            # Sell 50% of current position
            units_to_sell = position // 2
            if units_to_sell > 0:
                proceeds = price * units_to_sell
                cash += proceeds  # Add proceeds to cash
                
                # Calculate weighted average buy price of current position
                if position_history:
                    weighted_buy_price = sum(p * q for p, q in position_history.items()) / sum(position_history.values())
                else:
                    weighted_buy_price = 0
                
                # Calculate gain/loss information
                gain_loss = (price - weighted_buy_price) * units_to_sell
                gain_loss_pct = ((price / weighted_buy_price) - 1) * 100 if weighted_buy_price > 0 else 0
                
                # Update position
                position -= units_to_sell
                
                # Update position history - sell proportionally from all buys
                sell_factor = units_to_sell / sum(position_history.values())
                for p in list(position_history.keys()):
                    units_sold_at_this_price = int(position_history[p] * sell_factor)
                    position_history[p] -= units_sold_at_this_price
                    if position_history[p] <= 0:
                        del position_history[p]
                
                trades.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'action': '卖出',
                    'price': price,
                    'units': units_to_sell,
                    'proceeds': proceeds,
                    'gain_loss': gain_loss,
                    'gain_loss_pct': gain_loss_pct,
                    'cash': cash,
                    'position_value': position * price,
                    'total_value': cash + (position * price)
                })
    
    # Calculate final results
    final_price = stock_data['close'].iloc[-1]
    final_position_value = position * final_price
    final_value = cash + final_position_value
    
    # Buy and hold comparison
    max_buy_and_hold_units = initial_cash // stock_data['close'].iloc[0]
    buy_and_hold_value = max_buy_and_hold_units * final_price
    
    # Calculate returns
    signal_return_pct = ((final_value - initial_cash) / initial_cash) * 100
    buy_hold_return_pct = ((buy_and_hold_value - initial_cash) / initial_cash) * 100
    
    # Results
    results = {
        'initial_cash': initial_cash,
        'final_cash': cash,
        'final_position': position,
        'final_position_value': final_position_value,
        'final_value': final_value,
        'signal_return_pct': signal_return_pct,
        'buy_hold_units': max_buy_and_hold_units,
        'buy_hold_value': buy_and_hold_value,
        'buy_hold_return_pct': buy_hold_return_pct,
        'trades': trades
    }
    
    return results 