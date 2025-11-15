#!/usr/bin/env python3
"""
Bulk download and cache full Tushare daily history for all tracked stocks.

The script reads the cached stock universe from `cache/stock_basic_*.pkl`,
fetches missing daily candles through the Tushare `pro.daily` endpoint with
retry/backoff logic, and stores everything inside a single pandas DataFrame
pickle (default: `history/all_daily_bars.pkl`). Subsequent executions append
only the newly published trading days so that the app can stay within the
remote rate limits.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import tushare as ts
from tqdm import tqdm

RATE_LIMIT_MSG = "每分钟最多访问"
DEFAULT_START_DATE = "19900101"
DEFAULT_OUTPUT = Path("history") / "all_daily_bars.pkl"
DEFAULT_STOCK_CACHE = Path("cache") / "stock_basic_20251109103329.pkl"
THREAD_LOCAL = threading.local()


def normalize_symbol(ticker: str) -> str:
    """Ensure the ticker is stored as a 6-digit numeric string."""
    if not ticker:
        return ""
    ticker = ticker.strip()
    if "." in ticker:
        ticker = ticker.split(".", 1)[0]
    return ticker.zfill(6)


def to_ts_code(symbol: str) -> str:
    """Convert numeric code to Tushare's ts_code format."""
    if not symbol:
        return ""
    if "." in symbol:
        base, suffix = symbol.split(".", 1)
        return f"{base.zfill(6)}.{suffix.upper()}"
    normalized = normalize_symbol(symbol)
    if normalized.startswith(("6", "9")):
        market = "SH"
    elif normalized.startswith(("4", "8")):
        market = "BJ"
    else:
        market = "SZ"
    return f"{normalized}.{market}"


def normalize_date_string(value: str | None) -> str | None:
    """Extract digits from an input date string and ensure YYYYMMDD."""
    if not value:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) != 8:
        return None
    return digits


def parse_date(value: str | None) -> pd.Timestamp | None:
    """Convert YYYYMMDD strings into pandas timestamps."""
    normalized = normalize_date_string(value)
    if not normalized:
        return None
    return pd.Timestamp(datetime.strptime(normalized, "%Y%m%d"))


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
        raise RuntimeError(
            "Missing Tushare token. Provide --token or set TUSHARE_TOKEN/TS_TOKEN."
        )
    return token


def get_thread_client(token: str):
    """Create one Tushare client per thread to avoid shared state issues."""
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = ts.pro_api(token)
        THREAD_LOCAL.client = client
    return client


def call_with_retry(
    func, *, retries: int, base_delay: float, backoff: float, label: str
):
    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as exc:  # Tushare raises generic Exception
            wait_time = base_delay * (backoff ** (attempt - 1))
            if attempt == retries:
                logging.error(
                    "API %s failed after %d attempts: %s", label, attempt, exc
                )
                raise
            message = str(exc)
            if RATE_LIMIT_MSG in message:
                logging.warning(
                    "Rate limit hit for %s (attempt %d/%d); sleeping %.1fs",
                    label,
                    attempt,
                    retries,
                    wait_time,
                )
            else:
                logging.warning(
                    "API %s failed (attempt %d/%d): %s. Retrying in %.1fs",
                    label,
                    attempt,
                    retries,
                    message,
                    wait_time,
                )
            time.sleep(wait_time)


@dataclass(frozen=True)
class DownloadJob:
    ts_code: str
    stock_name: str
    industry: str
    market: str
    list_date: str | None
    start_date: str
    end_date: str


class HistoryStore:
    """Track cached candles and persist consolidated DataFrames."""

    def __init__(self, output_path: Path, flush_interval: int):
        self.output_path = output_path
        self.flush_interval = max(1, flush_interval)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame = self._load_existing()
        self.pending_frames: list[pd.DataFrame] = []
        self.last_dates = self._extract_last_dates(self.frame)

    def _load_existing(self) -> pd.DataFrame:
        if not self.output_path.exists():
            return pd.DataFrame()
        df = pd.read_pickle(self.output_path)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        df.sort_values(["ts_code", "trade_date"], inplace=True, ignore_index=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def _extract_last_dates(df: pd.DataFrame) -> dict[str, pd.Timestamp]:
        if df.empty or "trade_date" not in df.columns:
            return {}
        grouped = df.groupby("ts_code")["trade_date"].max()
        return grouped.to_dict()

    def compute_start_date(
        self,
        ts_code: str,
        list_date: str | None,
        explicit_start: str | None,
        force_refresh: bool,
        end_ts: pd.Timestamp,
    ) -> str | None:
        default_ts = (
            parse_date(explicit_start)
            or parse_date(list_date)
            or parse_date(DEFAULT_START_DATE)
        )
        if default_ts is None:
            default_ts = parse_date(DEFAULT_START_DATE)
        if not force_refresh:
            existing_last = self.last_dates.get(ts_code)
            if existing_last is not None:
                default_ts = max(default_ts, existing_last + pd.Timedelta(days=1))
        if default_ts > end_ts:
            return None
        return default_ts.strftime("%Y%m%d")

    def register_new_data(self, data: pd.DataFrame) -> None:
        if data is None or data.empty:
            return
        self.pending_frames.append(data)
        for ts_code, trade_dt in data.groupby("ts_code")["trade_date"].max().items():
            prev = self.last_dates.get(ts_code)
            if prev is None or trade_dt > prev:
                self.last_dates[ts_code] = trade_dt
        if len(self.pending_frames) >= self.flush_interval:
            self.flush()

    def flush(self, force: bool = False) -> None:
        if not self.pending_frames:
            if force and not self.frame.empty:
                self.frame.to_pickle(self.output_path)
            return
        combined = pd.concat(self.pending_frames, ignore_index=True)
        self.pending_frames.clear()
        if self.frame.empty:
            updated = combined
        else:
            updated = pd.concat([self.frame, combined], ignore_index=True)
        updated.sort_values(["ts_code", "trade_date"], inplace=True, ignore_index=True)
        updated.drop_duplicates(
            subset=["ts_code", "trade_date"], keep="last", inplace=True
        )
        updated.reset_index(drop=True, inplace=True)
        self.frame = updated
        self.frame.to_pickle(self.output_path)

    def summary(self) -> str:
        rows = len(self.frame) if not self.frame.empty else 0
        stocks = self.frame["ts_code"].nunique() if not self.frame.empty else 0
        return f"{rows} rows across {stocks} tickers cached at {self.output_path}"


def read_stock_list(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stock list cache not found: {path}")
    df = pd.read_pickle(path)
    if df is None or df.empty:
        raise RuntimeError(f"Stock list cache {path} is empty.")
    if "ts_code" not in df.columns:
        df = df.copy()
        df["ts_code"] = df["code"].apply(to_ts_code)
    required = {"ts_code", "name", "industry", "market", "list_date"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Stock list cache missing columns: {', '.join(sorted(missing))}"
        )
    df["ts_code"] = df["ts_code"].astype(str)
    df["name"] = df["name"].fillna("")
    df["industry"] = df["industry"].fillna("")
    df["market"] = df["market"].fillna("")
    df["list_date"] = df["list_date"].fillna("")
    return df


def build_jobs(
    stock_df: pd.DataFrame,
    store: HistoryStore,
    *,
    end_date: str,
    explicit_start: str | None,
    force_refresh: bool,
) -> list[DownloadJob]:
    end_ts = parse_date(end_date)
    if end_ts is None:
        raise ValueError(f"Invalid end date: {end_date}")
    jobs: list[DownloadJob] = []
    for row in stock_df.itertuples(index=False):
        start_date = store.compute_start_date(
            row.ts_code,
            getattr(row, "list_date", None),
            explicit_start,
            force_refresh,
            end_ts,
        )
        if not start_date:
            continue
        jobs.append(
            DownloadJob(
                ts_code=row.ts_code,
                stock_name=row.name,
                industry=row.industry,
                market=row.market,
                list_date=getattr(row, "list_date", None),
                start_date=start_date,
                end_date=end_date,
            )
        )
    return jobs


def download_job(
    job: DownloadJob,
    *,
    token: str,
    retries: int,
    base_delay: float,
    backoff: float,
) -> pd.DataFrame:
    client = get_thread_client(token)

    def _call():
        return client.daily(
            ts_code=job.ts_code,
            start_date=job.start_date,
            end_date=job.end_date,
        )

    df = call_with_retry(
        _call,
        retries=retries,
        base_delay=base_delay,
        backoff=backoff,
        label=f"pro.daily {job.ts_code}",
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"trade_date": "trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.sort_values("trade_date", inplace=True, ignore_index=True)
    df["stock_name"] = job.stock_name
    df["industry"] = job.industry
    df["market"] = job.market
    df["list_date"] = job.list_date or ""
    return df


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download full historical daily bars for every stock in the cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stock-cache",
        type=Path,
        default=DEFAULT_STOCK_CACHE,
        help="Path to stock_basic cache pickle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="File to persist the combined pandas DataFrame (.pkl).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Number of concurrent download workers.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=20,
        help="How many times to retry a failed Tushare call.",
    )
    parser.add_argument(
        "--base-delay",
        type=float,
        default=1.2,
        help="Initial sleep (seconds) before retrying.",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=1.5,
        help="Backoff multiplier applied to each retry delay.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Override earliest fetch date (YYYYMMDD). Defaults to each stock list_date.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=today_yyyymmdd(),
        help="Latest trade date to fetch (YYYYMMDD). Defaults to today.",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=200,
        help="Persist merged data every N successful downloads.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore local cache and refetch from the configured start date.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Explicit Tushare token; otherwise reads from environment.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    token = resolve_token(args.token)
    stock_df = read_stock_list(args.stock_cache)
    store = HistoryStore(args.output, flush_interval=args.flush_interval)
    explicit_start = normalize_date_string(args.start_date)
    end_date = normalize_date_string(args.end_date) or today_yyyymmdd()
    jobs = build_jobs(
        stock_df,
        store,
        end_date=end_date,
        explicit_start=explicit_start,
        force_refresh=args.force_refresh,
    )
    if not jobs:
        logging.info("No downloads needed. Cache already up to date.")
        return
    logging.info(
        "Prepared %d download jobs (end_date=%s, output=%s)",
        len(jobs),
        end_date,
        args.output,
    )
    successes = 0
    empty_returns = 0
    failures: list[str] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            future_to_job = {
                executor.submit(
                    download_job,
                    job,
                    token=token,
                    retries=args.retries,
                    base_delay=args.base_delay,
                    backoff=args.backoff,
                ): job
                for job in jobs
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_job),
                total=len(future_to_job),
                desc="Downloading",
                unit="stock",
            ):
                job = future_to_job[future]
                try:
                    df = future.result()
                except Exception as exc:
                    logging.exception(
                        "Failed to download %s (%s): %s",
                        job.ts_code,
                        job.stock_name,
                        exc,
                    )
                    failures.append(job.ts_code)
                    continue
                if df.empty:
                    empty_returns += 1
                    continue
                store.register_new_data(df)
                successes += 1
    finally:
        store.flush(force=True)
    logging.info(
        "Finished downloads: %d succeeded, %d empty, %d failed",
        successes,
        empty_returns,
        len(failures),
    )
    logging.info(store.summary())
    if failures:
        logging.warning("Failed tickers: %s", ", ".join(failures[:20]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user; partial data (if any) has been flushed.")
