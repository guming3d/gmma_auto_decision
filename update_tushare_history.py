#!/usr/bin/env python3
"""Incrementally update all_daily_bars_compact.parquet with latest Tushare data.

The script resumes from the most recent trade_date stored in
`history/all_daily_bars_compact.parquet`, fetches missing daily bars through the
Tushare `pro.daily` endpoint, and rewrites the parquet with an added `industry`
column for downstream aggregations.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tushare as ts
from tqdm import tqdm

RATE_LIMIT_MSG = "每分钟最多访问"
DEFAULT_HISTORY_PATH = Path("history") / "all_daily_bars_compact.parquet"
DEFAULT_STOCK_CACHE_DIR = Path("cache")
DEFAULT_MAX_WORKERS = 6
DEFAULT_RETRIES = 12
DEFAULT_BASE_DELAY = 1.0
DEFAULT_BACKOFF = 1.5
DEFAULT_PER_MINUTE_LIMIT = 45
THREAD_LOCAL = threading.local()

KEEP_COLUMNS = ["ts_code", "trade_date", "close"]
FLOAT_COLUMNS = ["close"]
OUTPUT_COLUMNS = ["trade_date", "close", "ts_code", "code_int", "industry"]


class RateLimiter:
    """Thread-safe limiter to cap requests per rolling window."""

    def __init__(self, max_calls: int, period: float = 60.0) -> None:
        self.max_calls = max_calls
        self.period = period
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._min_interval = self.period / float(self.max_calls)

    def wait(self) -> None:
        while True:
            now = time.monotonic()
            with self._lock:
                while self._timestamps and now - self._timestamps[0] >= self.period:
                    self._timestamps.popleft()
                if len(self._timestamps) >= self.max_calls:
                    window_block = self.period - (now - self._timestamps[0])
                else:
                    window_block = 0.0
                pace_block = max(0.0, self._next_allowed - now)
                if window_block <= 0 and pace_block <= 0:
                    self._timestamps.append(now)
                    self._next_allowed = now + self._min_interval
                    return
                sleep_for = max(window_block, pace_block)
            if sleep_for > 0:
                time.sleep(sleep_for)


def normalize_date_string(value: str | None) -> str | None:
    """Extract digits and return YYYYMMDD or None."""
    if not value:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) != 8:
        return None
    return digits


def today_yyyymmdd() -> str:
    return datetime.today().strftime("%Y%m%d")


def resolve_token(explicit_token: str | None) -> str:
    token = (
        explicit_token
        or os.getenv("TUSHARE_TOKEN")
        or os.getenv("TS_TOKEN")
        or os.getenv("TUSHARE_PRO_TOKEN")
    )
    if not token:
        raise RuntimeError("Missing Tushare token. Provide --token or set TUSHARE_TOKEN/TS_TOKEN.")
    return token


def call_with_retry(func, *, retries: int, base_delay: float, backoff: float, label: str):
    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as exc:  # Tushare raises plain Exception
            wait_time = base_delay * (backoff ** (attempt - 1))
            if attempt == retries:
                logging.error("API %s failed after %d attempts: %s", label, attempt, exc)
                raise
            message = str(exc)
            level = logging.warning if RATE_LIMIT_MSG in message else logging.info
            level("API %s failed (attempt %d/%d): %s. Retrying in %.1fs", label, attempt, retries, message, wait_time)
            time.sleep(wait_time)


def get_thread_client(token: str):
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = ts.pro_api(token)
        THREAD_LOCAL.client = client
    return client


def find_latest_stock_cache(cache_dir: Path) -> Path:
    candidates = sorted(cache_dir.glob("stock_basic_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No stock_basic_*.pkl files found under {cache_dir}")
    return candidates[0]


def load_stock_list(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stock list cache not found: {path}")
    df = pd.read_pickle(path)
    if df is None or df.empty:
        raise RuntimeError(f"Stock list cache {path} is empty.")
    required = {"ts_code", "code", "name", "industry", "market", "list_date"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Stock list cache missing columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["ts_code"] = df["ts_code"].astype(str)
    df["code_int"] = pd.to_numeric(df["code"], errors="coerce")
    df.dropna(subset=["code_int"], inplace=True)
    df["code_int"] = df["code_int"].astype(np.uint32)
    df["industry"] = df["industry"].fillna("").astype(str)
    df["list_date"] = df["list_date"].fillna("")
    return df


def load_existing_history(path: Path, code_to_ts: dict[int, str], industry_map: dict[str, str]) -> pd.DataFrame:
    if not path.exists():
        logging.info("History file %s not found; starting fresh.", path)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.read_parquet(path)
    if df is None or df.empty:
        logging.info("History file %s is empty; starting fresh.", path)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df.dropna(subset=["trade_date"], inplace=True)
    if "close" in df.columns:
        df["close"] = pd.to_numeric(df["close"], errors="coerce").astype(np.float32)
    ts_series = df["ts_code"] if "ts_code" in df.columns else pd.Series("", index=df.index)
    if "code_int" not in df.columns:
        df["code_int"] = pd.to_numeric(ts_series.astype(str).str.slice(0, 6), errors="coerce")
    df["code_int"] = pd.to_numeric(df["code_int"], errors="coerce")
    df.dropna(subset=["code_int"], inplace=True)
    df["code_int"] = df["code_int"].astype(np.uint32)
    if "ts_code" not in df.columns:
        df["ts_code"] = df["code_int"].map(code_to_ts)
    df["industry"] = df["ts_code"].map(industry_map).fillna("").astype(str)
    return df


def prepare_daily_frame(raw_df: pd.DataFrame, industry_map: dict[str, str]) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    keep = [col for col in KEEP_COLUMNS if col in raw_df.columns]
    df = raw_df[keep].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df.dropna(subset=["trade_date"], inplace=True)
    for col in FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    df["ts_code"] = df["ts_code"].astype(str)
    df["code_int"] = pd.to_numeric(df["ts_code"].str.slice(0, 6), errors="coerce")
    df.dropna(subset=["code_int"], inplace=True)
    df["code_int"] = df["code_int"].astype(np.uint32)
    df["industry"] = df["ts_code"].map(industry_map).fillna("").astype(str)
    return df


def fetch_daily(
    ts_code: str,
    *,
    start_date: str,
    end_date: str,
    token: str,
    retries: int,
    base_delay: float,
    backoff: float,
    industry_map: dict[str, str],
    rate_limiter: RateLimiter,
) -> pd.DataFrame:
    client = get_thread_client(token)

    def _call():
        rate_limiter.wait()
        return client.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

    df = call_with_retry(
        _call,
        retries=retries,
        base_delay=base_delay,
        backoff=backoff,
        label=f"pro.daily {ts_code}",
    )
    return prepare_daily_frame(df, industry_map)


def compute_start_date(history_df: pd.DataFrame, explicit_start: str | None, stock_df: pd.DataFrame) -> str:
    if explicit_start:
        return explicit_start
    if not history_df.empty:
        last_date = history_df["trade_date"].max()
        if pd.isna(last_date):
            raise RuntimeError("Existing history has invalid trade_date values.")
        return (last_date + timedelta(days=1)).strftime("%Y%m%d")
    list_dates = pd.to_datetime(stock_df["list_date"], errors="coerce")
    earliest = list_dates.min()
    if pd.isna(earliest):
        return "19900101"
    return earliest.strftime("%Y%m%d")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update history/all_daily_bars_compact.parquet with latest Tushare daily data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="Path to the existing parquet file to update.",
    )
    parser.add_argument(
        "--stock-cache",
        type=Path,
        default=None,
        help="Path to stock_basic cache pickle. Defaults to the newest stock_basic_*.pkl under cache/.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Override fetch start date (YYYYMMDD). Defaults to last date in history + 1.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=today_yyyymmdd(),
        help="Fetch end date (YYYYMMDD). Defaults to today.",
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Concurrent download workers.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="API retry attempts per stock.")
    parser.add_argument("--base-delay", type=float, default=DEFAULT_BASE_DELAY, help="Initial sleep before retry.")
    parser.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF, help="Retry backoff multiplier.")
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Explicit Tushare token; otherwise reads TUSHARE_TOKEN/TS_TOKEN/TUSHARE_PRO_TOKEN.",
    )
    parser.add_argument(
        "--per-minute-limit",
        type=int,
        default=DEFAULT_PER_MINUTE_LIMIT,
        help="Max API calls allowed per 60s across all workers.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ...).")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    per_minute_limit = max(1, args.per_minute_limit)
    rate_limiter = RateLimiter(per_minute_limit)
    token = resolve_token(args.token)
    stock_cache = args.stock_cache or find_latest_stock_cache(DEFAULT_STOCK_CACHE_DIR)
    logging.info("Using stock cache: %s", stock_cache)
    stock_df = load_stock_list(stock_cache)
    code_to_ts = dict(zip(stock_df["code_int"], stock_df["ts_code"]))
    industry_map = dict(zip(stock_df["ts_code"], stock_df["industry"]))

    history_df = load_existing_history(args.history, code_to_ts, industry_map)
    explicit_start = normalize_date_string(args.start_date)
    start_date = compute_start_date(history_df, explicit_start, stock_df)
    end_date = normalize_date_string(args.end_date) or today_yyyymmdd()

    if start_date > end_date:
        logging.info("History is already up to date (start_date %s > end_date %s).", start_date, end_date)
        history_df.to_parquet(args.history, index=False)
        logging.info("Rewrote %s with industry metadata only.", args.history)
        return

    jobs = stock_df["ts_code"].tolist()
    logging.info(
        "Preparing to fetch %d stocks from %s to %s into %s", len(jobs), start_date, end_date, args.history
    )
    logging.info("Applying global rate limit: %d calls per 60s across %d workers.", per_minute_limit, args.max_workers)
    new_frames: list[pd.DataFrame] = []
    failures: list[str] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_ts = {
                executor.submit(
                    fetch_daily,
                    ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    token=token,
                    retries=args.retries,
                    base_delay=args.base_delay,
                    backoff=args.backoff,
                    industry_map=industry_map,
                    rate_limiter=rate_limiter,
                ): ts_code
                for ts_code in jobs
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_ts),
                total=len(future_to_ts),
                desc="Downloading",
                unit="stock",
            ):
                ts_code = future_to_ts[future]
                try:
                    df = future.result()
                except Exception as exc:  # noqa: PERF203
                    logging.exception("Failed to download %s: %s", ts_code, exc)
                    failures.append(ts_code)
                    continue
                if df.empty:
                    continue
                new_frames.append(df)
    finally:
        pass

    if new_frames:
        new_data = pd.concat(new_frames, ignore_index=True)
    else:
        new_data = pd.DataFrame(columns=OUTPUT_COLUMNS)

    combined = pd.concat([history_df, new_data], ignore_index=True)
    combined["industry"] = combined["industry"].fillna("").astype(str)
    combined["trade_date"] = pd.to_datetime(combined["trade_date"], errors="coerce")
    combined.dropna(subset=["trade_date", "code_int"], inplace=True)
    combined.sort_values(["code_int", "trade_date"], inplace=True, ignore_index=True)
    combined.drop_duplicates(subset=["code_int", "trade_date"], keep="last", inplace=True)
    combined = combined[OUTPUT_COLUMNS]

    args.history.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.history, index=False)

    logging.info("Updated history saved to %s (rows=%d, stocks=%d).", args.history, len(combined), combined["code_int"].nunique())
    if failures:
        logging.warning("Failed to download %d tickers: %s", len(failures), ", ".join(failures[:20]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user; partial progress (if any) was merged.")
