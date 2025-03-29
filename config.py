"""
Configuration settings for the GMMA application.
"""

# Short-term EMA periods
SHORT_TERM_PERIODS = [3, 5, 8, 10, 12, 15]

# Long-term EMA periods
LONG_TERM_PERIODS = [30, 35, 40, 45, 50, 60]

# Default funds for analysis
DEFAULT_FUNDS = "510300,510050,512100,588000,512010,512200"

# Initial cash for backtesting
INITIAL_CASH = 100000

# Default backtest units
DEFAULT_BACKTEST_UNITS = 100

# Percentage strategy constants
MIN_CASH_RESERVE_PCT = 0.3  # Keep 30% of initial cash as reserve
MIN_CASH_THRESHOLD_PCT = 0.1  # Threshold for when to consider selling (10% of initial)
BUY_PERCENTAGE = 0.1  # Use 10% of invest money for each buy

# Period days mapping
PERIOD_DAYS = {
    "25年": 365 * 25,
    "20年": 365 * 20,
    "15年": 365 * 15,
    "10年": 365 * 10,
    "8年": 365 * 8,
    "6年": 365 * 6,
    "4年": 365 * 4,
    "3年": 365 * 3,
    "2年": 365 * 2,
    "1年": 365,
    "6个月": 180,
    "3个月": 90
} 