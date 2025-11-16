import os
import random
import time
import traceback
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from threading import Lock

import numpy as np
import pandas as pd
import streamlit as st
import tushare as ts

# Set page layout to wide mode
st.set_page_config(
    page_title="A股市值变化排序器",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("A股市值变化排序工具")
st.markdown("""
此应用程序使用 Tushare 数据分析中国 A 股股票在指定时间区间内的总市值变化。
它会基于用户选择的起止日期并发扫描全市场，列出总市值增幅最大的前50只股票以及跌幅最大的前50只股票。
""")

# Create cache directory if it doesn't exist
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(cache_dir, exist_ok=True)

MAX_WORKERS = min(6, max(2, (os.cpu_count() or 2)))
DEFAULT_LOOKBACK_DAYS = 30
TOP_COUNT = 50
ERROR_LOG_DISPLAY_LIMIT = 100
ERROR_LOG_FILE = os.path.join(cache_dir, "value_sorting_errors.log")
STOCK_LIST_CACHE_FILE = os.path.join(cache_dir, "stock_list.csv")
STOCK_LIST_REFRESH_DAYS = 7
REQUEST_MAX_RETRIES = 3
REQUEST_MIN_INTERVAL = 0.08  # seconds between outbound requests
REQUEST_JITTER = 0.03        # random jitter to avoid fixed pattern
REQUEST_BACKOFF_FACTOR = 1.8
REQUEST_BACKOFF_MAX = 6.0
RATE_LIMIT_MSG = "每分钟最多访问"

_request_lock = Lock()
_next_available_time = 0.0


def _get_tushare_token_from_secrets() -> str | None:
    """从 Streamlit secrets 中读取 Tushare 授权令牌。"""
    secret_token = None
    tushare_section = st.secrets.get("tushare")
    if tushare_section and isinstance(tushare_section, Mapping):
        secret_token = (
            tushare_section.get("token")
            or tushare_section.get("api_token")
            or tushare_section.get("TUSHARE_TOKEN")
        )
    return secret_token or st.secrets.get("tushare_token") or st.secrets.get(
        "TUSHARE_TOKEN"
    )


TUSHARE_TOKEN = (
    _get_tushare_token_from_secrets()
    or os.getenv("TUSHARE_TOKEN")
    or os.getenv("TS_TOKEN")
    or os.getenv("TUSHARE_PRO_TOKEN")
)


def to_ts_code(code: str) -> str:
    """将 6 位股票代码转换为 Tushare ts_code 格式。"""
    code = str(code).zfill(6)
    if code.startswith(("6", "9")) or code.startswith(("688", "689")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SZ"


@st.cache_resource
def get_tushare_client():
    """缓存并返回 Tushare Pro 客户端实例。"""
    if not TUSHARE_TOKEN:
        raise RuntimeError("请在 TUSHARE_TOKEN/TS_TOKEN 环境变量或 st.secrets 中配置 Tushare 授权令牌。")
    return ts.pro_api(TUSHARE_TOKEN)


def call_tushare_api(func, *, api_label: str):
    """包装 Tushare API 调用并提供退避与错误日志。"""
    last_exception = None
    for attempt in range(1, REQUEST_MAX_RETRIES + 1):
        acquire_request_slot()
        try:
            return func()
        except Exception as exc:
            last_exception = exc
            message = str(exc)
            if attempt == REQUEST_MAX_RETRIES:
                break
            delay = backoff_delay(attempt)
            if RATE_LIMIT_MSG in message:
                append_error_log(
                    f"Tushare 限频: {api_label} 第 {attempt} 次调用失败，将在 {delay:.2f}s 后重试。错误: {message}"
                )
            else:
                append_error_log(
                    f"Tushare API {api_label} 第 {attempt} 次调用失败，将在 {delay:.2f}s 后重试。错误: {message}"
                )
            time.sleep(delay)
    if last_exception:
        raise last_exception
    return None


def prepare_stock_list_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """标准化股票列表数据结构，确保包含 code/name/ts_code 列。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=['code', 'name', 'ts_code'])
    normalised = df.copy()
    if 'code' in normalised.columns:
        normalised['code'] = (
            normalised['code']
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(6)
        )
    elif 'symbol' in normalised.columns:
        normalised['code'] = (
            normalised['symbol']
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(6)
        )
    else:
        normalised['code'] = normalised.index.astype(str).str.zfill(6)
    if 'name' not in normalised.columns:
        normalised['name'] = ""
    normalised['name'] = normalised['name'].fillna("").astype(str)
    if 'ts_code' not in normalised.columns:
        normalised['ts_code'] = normalised['code'].apply(to_ts_code)
    else:
        normalised['ts_code'] = normalised['ts_code'].fillna("").astype(str)
        missing_mask = normalised['ts_code'].eq("")
        normalised.loc[missing_mask, 'ts_code'] = normalised.loc[
            missing_mask, 'code'
        ].apply(to_ts_code)
    return normalised


# Cache for stock list to avoid repeated API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_list():
    """获取所有沪深A股的代码和名称，并在本地缓存七天"""
    try:
        use_cached_file = False
        if os.path.exists(STOCK_LIST_CACHE_FILE):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(STOCK_LIST_CACHE_FILE))
            if file_age <= timedelta(days=STOCK_LIST_REFRESH_DAYS):
                use_cached_file = True
        if use_cached_file:
            cached_df = pd.read_csv(STOCK_LIST_CACHE_FILE)
            return prepare_stock_list_df(cached_df)

        client = get_tushare_client()
        fields = "ts_code,symbol,name,market,list_date"
        stock_list_df = call_tushare_api(
            lambda: client.stock_basic(exchange="", list_status="L", fields=fields),
            api_label="stock_basic",
        )
        if stock_list_df is not None and not stock_list_df.empty:
            stock_list_df = stock_list_df.rename(columns={"symbol": "code"})
            stock_list_df = prepare_stock_list_df(stock_list_df)
            try:
                stock_list_df.to_csv(STOCK_LIST_CACHE_FILE, index=False, encoding="utf-8")
            except Exception as write_err:
                append_error_log(f"保存股票列表缓存失败: {write_err}")
        elif os.path.exists(STOCK_LIST_CACHE_FILE):
            cached_df = pd.read_csv(STOCK_LIST_CACHE_FILE)
            return prepare_stock_list_df(cached_df)
        return stock_list_df or pd.DataFrame(columns=['code', 'name', 'ts_code'])
    except RuntimeError as token_err:
        st.error(str(token_err))
        return pd.DataFrame(columns=['code', 'name', 'ts_code'])
    except Exception as e:
        append_error_log(f"获取股票列表失败: {e}")
        if os.path.exists(STOCK_LIST_CACHE_FILE):
            try:
                cached_df = pd.read_csv(STOCK_LIST_CACHE_FILE)
                return prepare_stock_list_df(cached_df)
            except Exception as read_err:
                append_error_log(f"读取本地股票列表缓存失败: {read_err}")
        st.error(f"获取股票列表失败: {str(e)}")
        return pd.DataFrame(columns=['code', 'name', 'ts_code'])

def append_error_log(message):
    """将错误信息写入日志文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass
    
def acquire_request_slot():
    """Rate limit outbound requests to避免触发远端限流"""
    global _next_available_time
    with _request_lock:
        now = time.monotonic()
        wait = _next_available_time - now
        if wait < 0:
            wait = 0.0
        _next_available_time = now + wait + REQUEST_MIN_INTERVAL
    if wait > 0:
        time.sleep(wait)
    if REQUEST_JITTER > 0:
        time.sleep(random.uniform(0, REQUEST_JITTER))
    
def backoff_delay(attempt):
    """Calculate exponential backoff delay for retries"""
    delay = REQUEST_BACKOFF_FACTOR ** attempt
    return min(delay, REQUEST_BACKOFF_MAX)

@st.cache_data(ttl=900, show_spinner=False)
def get_stock_market_value(symbol, start_date, end_date, *, ts_code=None, stock_name=None):
    """获取指定股票在给定日期范围的 Tushare 总市值与流通市值数据。"""
    symbol_no_prefix = str(symbol).zfill(6)
    ts_code = ts_code or to_ts_code(symbol_no_prefix)
    display_name = stock_name or symbol_no_prefix
    try:
        client = get_tushare_client()
    except RuntimeError as token_err:
        append_error_log(f"初始化 Tushare 客户端失败: {token_err}")
        return None
    fields = (
        "ts_code,trade_date,close,total_mv,circ_mv,total_share,float_share,free_share"
    )
    try:
        hist_df = call_tushare_api(
            lambda: client.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields=fields,
            ),
            api_label=f"daily_basic:{ts_code}",
        )
    except Exception as exc:
        append_error_log(f"获取 {symbol_no_prefix} Tushare 数据失败: {exc}")
        return None

    if hist_df is None or hist_df.empty:
        return None

    hist_df = hist_df.copy()
    if 'trade_date' not in hist_df.columns:
        append_error_log(f"{ts_code} 返回数据中缺少 trade_date 列")
        return None
    numeric_cols = [
        'close',
        'total_mv',
        'circ_mv',
        'total_share',
        'float_share',
        'free_share',
    ]
    for col in numeric_cols:
        if col in hist_df.columns:
            hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
    hist_df['日期'] = pd.to_datetime(hist_df['trade_date'], format='%Y%m%d', errors='coerce')
    hist_df = hist_df.dropna(subset=['日期']).sort_values('日期').reset_index(drop=True)
    if hist_df.empty:
        return None

    hist_df['名称'] = display_name
    if 'close' in hist_df.columns:
        hist_df['收盘'] = hist_df['close']

    # Convert value units (Tushare total/circ mv are in 1e4 RMB)
    if 'total_mv' in hist_df.columns:
        hist_df['总市值'] = hist_df['total_mv'] * 1e4
    if 'circ_mv' in hist_df.columns:
        hist_df['流通市值'] = hist_df['circ_mv'] * 1e4

    # Convert share counts (Tushare share columns use 万股单位)
    if 'total_share' in hist_df.columns:
        hist_df['总股本'] = hist_df['total_share'] * 1e4
    if 'float_share' in hist_df.columns:
        hist_df['流通股本'] = hist_df['float_share'] * 1e4
    elif 'free_share' in hist_df.columns:
        hist_df['流通股本'] = hist_df['free_share'] * 1e4

    # Reconstruct missing market values when necessary
    if '总市值' not in hist_df.columns or hist_df['总市值'].isna().all():
        if '收盘' in hist_df.columns and '总股本' in hist_df.columns:
            hist_df['总市值'] = hist_df['收盘'] * hist_df['总股本']
    if '流通市值' not in hist_df.columns or hist_df['流通市值'].isna().all():
        if '收盘' in hist_df.columns and '流通股本' in hist_df.columns:
            hist_df['流通市值'] = hist_df['收盘'] * hist_df['流通股本']

    needed_cols = ['日期', '名称']
    for col in ['流通市值', '总市值', '收盘', '总股本', '流通股本']:
        if col in hist_df.columns:
            needed_cols.append(col)
    return hist_df[needed_cols]

# Function to test available Tushare functions for historical data
def test_available_history_functions():
    """测试关键的 Tushare 数据接口可用性"""
    results = {}
    try:
        client = get_tushare_client()
    except RuntimeError as token_err:
        results["tushare_client"] = {
            "status": "error",
            "error": str(token_err),
        }
        return results

    checks = [
        {
            "name": "stock_basic",
            "callable": lambda: client.stock_basic(
                exchange="", list_status="L", fields="ts_code,name"
            ),
            "expected_column": "ts_code",
        },
        {
            "name": "daily_basic",
            "callable": lambda: client.daily_basic(
                ts_code="000001.SZ",
                start_date="20230101",
                end_date="20230105",
                fields="ts_code,trade_date,total_mv,circ_mv,close",
            ),
            "expected_column": "total_mv",
        },
    ]

    for item in checks:
        api_name = item["name"]
        try:
            df = call_tushare_api(item["callable"], api_label=api_name)
            if df is not None and not df.empty:
                result_entry = {
                    "status": "success",
                    "rows": len(df),
                    "columns": list(df.columns),
                    "sample": df.head(1).to_dict('records')[0],
                }
                expected_col = item.get("expected_column")
                if expected_col and expected_col not in df.columns:
                    result_entry["warning"] = f"缺少 {expected_col} 列"
                results[api_name] = result_entry
            else:
                results[api_name] = {
                    "status": "empty_result",
                    "columns": [],
                }
        except Exception as exc:
            results[api_name] = {
                "status": "error",
                "error": str(exc),
            }

    return results

# Function to get market value for specific dates
def get_market_value_for_dates(symbol, start_date, end_date, *, ts_code=None, stock_name=None):
    """获取指定股票在起始日和结束日的市值数据"""
    try:
        df = get_stock_market_value(
            symbol,
            start_date,
            end_date,
            ts_code=ts_code,
            stock_name=stock_name,
        )
        if df is None or df.empty:
            return None, None, None, None, None, "历史行情为空"
        
        # Ensure we have at least one record
        if len(df) < 1:
            return None, None, None, None, None, "历史记录数量不足"
        
        df = df.sort_values('日期').reset_index(drop=True)
        
        name = df.iloc[-1].get('名称', stock_name or symbol)
        circ_series = df['流通市值'] if '流通市值' in df.columns else pd.Series(dtype=float)
        total_series = df['总市值'] if '总市值' in df.columns else pd.Series(dtype=float)
        
        def extract_first_last(series):
            if series.empty:
                return None, None
            valid = series.dropna()
            if valid.empty:
                return None, None
            first_value = float(valid.iloc[0])
            last_value = float(valid.iloc[-1])
            return first_value, last_value
        
        start_circ_mv, end_circ_mv = extract_first_last(circ_series)
        start_total_mv, end_total_mv = extract_first_last(total_series)
        
        if start_total_mv is None or end_total_mv is None:
            # Attempt to reconstruct total market value from closing price and share count
            if '收盘' in df.columns and '总股本' in df.columns:
                reconstructed_total_series = (
                    pd.to_numeric(df['收盘'], errors='coerce')
                    * pd.to_numeric(df['总股本'], errors='coerce')
                )
                start_total_mv, end_total_mv = extract_first_last(reconstructed_total_series)

        if start_circ_mv is None or end_circ_mv is None:
            if '收盘' in df.columns and '流通股本' in df.columns:
                reconstructed_circ_series = (
                    pd.to_numeric(df['收盘'], errors='coerce')
                    * pd.to_numeric(df['流通股本'], errors='coerce')
                )
                start_circ_mv, end_circ_mv = extract_first_last(reconstructed_circ_series)
        
        if start_total_mv is None or end_total_mv is None:
            return None, None, None, None, None, "无法获取或重建总市值数据"
        
        return name, start_circ_mv, end_circ_mv, start_total_mv, end_total_mv, None
    except Exception as e:
        return None, None, None, None, None, f"异常: {e}"

# Function to calculate date range formatted for API
def get_formatted_date_range(days_ago):
    """计算从当前日期向前推算的日期，格式化为API所需格式"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_ago)
    
    # Format dates to YYYYMMDD
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    return start_date_str, end_date_str

# Function to test API with a single stock
def test_api_connectivity(start_date_str, end_date_str):
    """测试与API的连接和数据获取"""
    test_stocks = [
        {"code": "000001", "ts_code": "000001.SZ"},  # 平安银行
        {"code": "600000", "ts_code": "600000.SH"},  # 浦发银行
    ]
    results = []
    
    # First, test which functions are available
    function_test_results = test_available_history_functions()
    
    # Now test actual data retrieval for specific stocks
    for stock in test_stocks:
        try:
            code = stock["code"]
            ts_code = stock.get("ts_code") or to_ts_code(code)
            hist_df = get_stock_market_value(
                code,
                start_date_str,
                end_date_str,
                ts_code=ts_code,
                stock_name=None,
            )
            
            if hist_df is not None and not hist_df.empty:
                sample_data = hist_df.head(1).to_dict('records')[0]
                results.append({
                    'stock': code,
                    'ts_code': ts_code,
                    'status': 'success',
                    'columns': list(hist_df.columns),
                    'sample': sample_data
                })
            else:
                results.append({
                    'stock': code,
                    'ts_code': ts_code,
                    'status': 'empty_response',
                    'columns': [],
                    'sample': {}
                })
        except Exception as e:
            results.append({
                'stock': stock["code"],
                'ts_code': stock.get("ts_code") or to_ts_code(stock["code"]),
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    return {
        'stock_tests': results,
        'function_tests': function_test_results
    }

# Main functionality
def main():
    # Sidebar options
    st.sidebar.title("分析设置")
    
    today = datetime.now().date()
    default_start = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    
    start_date = st.sidebar.date_input(
        "开始日期",
        value=default_start,
        max_value=today
    )
    
    end_date = st.sidebar.date_input(
        "结束日期",
        value=today,
        min_value=start_date,
        max_value=today
    )
    
    if start_date > end_date:
        st.sidebar.error("结束日期需晚于或等于开始日期")
        return
    
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    
    
    # Start analysis button
    if st.sidebar.button("开始分析", type="primary"):
        try:
            # Display selected parameters
            st.subheader("分析参数")
            st.write(f"- 时间区间: {start_date_str} 至 {end_date_str}")
            st.write("- 排除 ST/*ST 股票以避免异常数据")
            st.write(f"- 固定输出总市值涨跌榜 TOP/BOTTOM {TOP_COUNT}")
            
            # Get stock list
            with st.spinner("正在获取 A 股股票列表..."):
                stock_list_df = get_stock_list()
                if stock_list_df.empty:
                    st.error("无法获取股票列表，请稍后重试")
                    return
                stock_list_df['code'] = stock_list_df['code'].astype(str).str.zfill(6)
                if 'ts_code' in stock_list_df.columns:
                    stock_list_df['ts_code'] = stock_list_df['ts_code'].fillna("").astype(str)
                    missing_ts = stock_list_df['ts_code'].eq("")
                    stock_list_df.loc[missing_ts, 'ts_code'] = stock_list_df.loc[missing_ts, 'code'].apply(to_ts_code)
                else:
                    stock_list_df['ts_code'] = stock_list_df['code'].apply(to_ts_code)
                stock_list_df = stock_list_df.drop_duplicates(subset=['code'])
                initial_count = len(stock_list_df)
                stock_list_df = stock_list_df[~stock_list_df['name'].astype(str).str.contains('ST', case=False, na=False)]
                removed = initial_count - len(stock_list_df)
                if removed > 0:
                    st.info(f"已自动排除 {removed} 只 ST/*ST 股票")
                
                st.success(f"共获取到 {len(stock_list_df)} 只 A 股股票")
            
            if stock_list_df.empty:
                st.warning("筛选后没有可用股票，请调整条件后重试")
                return
            
            stock_records = stock_list_df.to_dict('records')
            total_stocks = len(stock_records)

            try:
                with open(ERROR_LOG_FILE, "w", encoding="utf-8") as log_file:
                    log_file.write(f"=== 市值排序运行开始 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    log_file.write(f"时间区间: {start_date_str} 至 {end_date_str}\n")
                    log_file.write(f"股票总数: {total_stocks}\n")
            except Exception:
                pass
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            results = []
            error_logs = []
            
            def process_stock(record):
                code = str(record.get('code')).zfill(6)
                fallback_name = record.get('name') or code
                ts_code = record.get('ts_code') or to_ts_code(code)
                try:
                    name, _, _, start_total_mv, end_total_mv, reason = get_market_value_for_dates(
                        code,
                        start_date_str,
                        end_date_str,
                        ts_code=ts_code,
                        stock_name=fallback_name,
                    )
                    stock_name = name or fallback_name
                    if reason:
                        message = f"股票 {code} ({stock_name}) 总市值数据缺失: {reason}"
                        append_error_log(message)
                        return None, message
                    change = end_total_mv - start_total_mv
                    if pd.notna(start_total_mv) and start_total_mv != 0:
                        change_percent = change / start_total_mv * 100
                    else:
                        change_percent = np.nan
                    return {
                        "code": code,
                        "name": stock_name,
                        "start_total_mv": float(start_total_mv),
                        "end_total_mv": float(end_total_mv),
                        "total_mv_change": float(change),
                        "total_mv_change_percent": float(change_percent) if not pd.isna(change_percent) else np.nan
                    }, None
                except Exception as exc:
                    message = f"股票 {code} ({fallback_name}) 异常: {exc}"
                    append_error_log(message)
                    return None, message
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_record = {
                    executor.submit(process_stock, record): record for record in stock_records
                }
                for idx, future in enumerate(as_completed(future_to_record), start=1):
                    record = future_to_record[future]
                    code = str(record.get('code')).zfill(6)
                    fallback_name = record.get('name') or code
                    try:
                        data, error = future.result()
                        if data:
                            results.append(data)
                        if error:
                            error_logs.append(error)
                            append_error_log(error)
                    except Exception as exc:
                        message = f"股票 {code} ({fallback_name}) 线程异常: {exc}"
                        error_logs.append(message)
                        append_error_log(message)
                    progress_bar.progress(min(idx / total_stocks, 1.0))
                    status_text.text(
                        f"已处理: {idx}/{total_stocks} | 有效: {len(results)} | 错误: {len(error_logs)}"
                    )
            
            progress_bar.progress(1.0)
            
            with st.expander("错误日志 (展开查看详情)", expanded=False):
                if error_logs:
                    st.write(f"处理过程中遇到 {len(error_logs)} 个异常：")
                    for i, log in enumerate(error_logs[:ERROR_LOG_DISPLAY_LIMIT], start=1):
                        st.text(f"{i}. {log}")
                    remaining = len(error_logs) - ERROR_LOG_DISPLAY_LIMIT
                    if remaining > 0:
                        st.text(f"... 还有 {remaining} 条错误未显示")
                    st.caption(f"完整错误详情已写入 {ERROR_LOG_FILE}")
                else:
                    st.write("处理过程未发现错误")
            
            if not results:
                st.error("未能获取有效数据，请尝试其他时间区间或检查网络连接")
                return
            
            results_df = pd.DataFrame(results)
            if results_df.empty:
                st.error("暂无可用的市值变化结果")
                return
            
            available_count = len(results_df)
            st.info(f"成功收集了 {available_count} 只股票的总市值数据")
            
            top_k = min(TOP_COUNT, available_count)
            if top_k == 0:
                st.warning("没有足够的数据用于排名展示")
                return
            
            top_increase = results_df.nlargest(top_k, 'total_mv_change').copy()
            top_decrease = results_df.nsmallest(top_k, 'total_mv_change').copy()
            
            for df in (top_increase, top_decrease):
                df['start_total_mv_亿'] = df['start_total_mv'] / 1e8
                df['end_total_mv_亿'] = df['end_total_mv'] / 1e8
                df['change_total_mv_亿'] = df['total_mv_change'] / 1e8
                df['change_percent'] = df['total_mv_change_percent']
            
            display_columns = [
                'code', 'name', 'start_total_mv_亿', 'end_total_mv_亿', 'change_total_mv_亿', 'change_percent'
            ]
            
            display_increase = top_increase[display_columns].copy()
            display_increase.columns = [
                '股票代码', '股票名称', '起始总市值(亿元)', '结束总市值(亿元)', '总市值变化(亿元)', '变化百分比(%)'
            ]
            display_increase['起始总市值(亿元)'] = display_increase['起始总市值(亿元)'].round(2)
            display_increase['结束总市值(亿元)'] = display_increase['结束总市值(亿元)'].round(2)
            display_increase['总市值变化(亿元)'] = display_increase['总市值变化(亿元)'].round(2)
            display_increase['变化百分比(%)'] = display_increase['变化百分比(%)'].round(2)
            
            display_decrease = top_decrease[display_columns].copy()
            display_decrease.columns = [
                '股票代码', '股票名称', '起始总市值(亿元)', '结束总市值(亿元)', '总市值变化(亿元)', '变化百分比(%)'
            ]
            display_decrease['起始总市值(亿元)'] = display_decrease['起始总市值(亿元)'].round(2)
            display_decrease['结束总市值(亿元)'] = display_decrease['结束总市值(亿元)'].round(2)
            display_decrease['总市值变化(亿元)'] = display_decrease['总市值变化(亿元)'].round(2)
            display_decrease['变化百分比(%)'] = display_decrease['变化百分比(%)'].round(2)
            
            st.subheader(f"总市值增加最多的前 {top_k} 只股票")
            st.dataframe(display_increase, use_container_width=True)
            
            st.subheader(f"总市值减少最多的前 {top_k} 只股票")
            st.dataframe(display_decrease, use_container_width=True)
            
            st.subheader("数据下载")
            csv_increase = display_increase.to_csv(index=False)
            csv_decrease = display_decrease.to_csv(index=False)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="下载总市值增加榜单",
                    data=csv_increase,
                    file_name=f"top_increase_total_mv_{start_date_str}_to_{end_date_str}.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label="下载总市值减少榜单",
                    data=csv_decrease,
                    file_name=f"top_decrease_total_mv_{start_date_str}_to_{end_date_str}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"分析过程中发生错误: {str(e)}")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
