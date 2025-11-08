#!/usr/bin/env python3
"""Recreate the provided Tushare interactive shell session."""

import tushare as ts

TOKEN = "74eea4b1a71952d5a08f4495f9d375f67e9354585e2b2ad82338b956"
TRADE_CAL_KWARGS = {
    "exchange": "",
    "start_date": "20180901",
    "end_date": "20181001",
    "fields": "exchange,cal_date,is_open,pretrade_date",
    "is_open": "0",
}
DAILY_KWARGS = {
    "ts_code": "000001.SZ",
    "start_date": "20180701",
    "end_date": "20180718",
}


def attempt_trade_calendar(label: str, client) -> None:
    print(f"\n[{label}] trade_cal request")
    try:
        df = client.trade_cal(**TRADE_CAL_KWARGS)
    except Exception as exc:  # Tushare raises a plain Exception on API errors
        print(f"Request failed: {exc}")
    else:
        print(df)


def fetch_daily_bars(label: str, client) -> None:
    print(f"\n[{label}] daily request")
    df = client.daily(**DAILY_KWARGS)
    print(df)


def main() -> None:
    authorized_client = ts.pro_api(TOKEN)
    attempt_trade_calendar("Authorized client", authorized_client)
    fetch_daily_bars("Authorized client", authorized_client)


if __name__ == "__main__":
    main()
