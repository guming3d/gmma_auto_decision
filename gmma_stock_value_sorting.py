import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re
import time
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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
此应用程序使用 akshare 数据分析中国 A 股股票在指定时间区间内的总市值变化。
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

_request_lock = Lock()
_next_available_time = 0.0

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
            cached_df = pd.read_csv(STOCK_LIST_CACHE_FILE, dtype={"code": str, "name": str})
            cached_df['code'] = cached_df['code'].str.zfill(6)
            return cached_df

        stock_list_df = ak.stock_info_a_code_name()
        if stock_list_df is not None and not stock_list_df.empty:
            stock_list_df['code'] = stock_list_df['code'].astype(str).str.zfill(6)
            try:
                stock_list_df.to_csv(STOCK_LIST_CACHE_FILE, index=False, encoding="utf-8")
            except Exception as write_err:
                append_error_log(f"保存股票列表缓存失败: {write_err}")
        elif os.path.exists(STOCK_LIST_CACHE_FILE):
            # API 失败但仍有旧缓存可用
            cached_df = pd.read_csv(STOCK_LIST_CACHE_FILE, dtype={"code": str, "name": str})
            cached_df['code'] = cached_df['code'].str.zfill(6)
            return cached_df
        return stock_list_df or pd.DataFrame(columns=['code', 'name'])
    except Exception as e:
        append_error_log(f"获取股票列表失败: {e}")
        if os.path.exists(STOCK_LIST_CACHE_FILE):
            try:
                cached_df = pd.read_csv(STOCK_LIST_CACHE_FILE, dtype={"code": str, "name": str})
                cached_df['code'] = cached_df['code'].str.zfill(6)
                return cached_df
            except Exception as read_err:
                append_error_log(f"读取本地股票列表缓存失败: {read_err}")
        st.error(f"获取股票列表失败: {str(e)}")
        return pd.DataFrame(columns=['code', 'name'])

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
    
# Helper to normalise Chinese number strings like "123.4亿"
def parse_numeric_value(value):
    """将带单位的中文数字字符串转换为浮点数"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan
    # Remove common wrapper characters
    for ch in ("(", ")", "（", "）", " "):
        text = text.replace(ch, "")
    if text in {"", "--", "None"}:
        return np.nan
    suffix_multipliers = [
        ("万亿元", 1e12),
        ("亿元", 1e8),
        ("亿股", 1e8),
        ("亿份", 1e8),
        ("亿手", 1e8 * 100),
        ("亿", 1e8),
        ("万股", 1e4),
        ("万份", 1e4),
        ("万手", 1e4 * 100),
        ("万元", 1e4),
        ("万", 1e4),
        ("千股", 1e3),
        ("千份", 1e3),
        ("千手", 1e3 * 100),
        ("百股", 1e2),
        ("百份", 1e2),
        ("百手", 1e2 * 100),
        ("股", 1.0),
        ("份", 1.0),
        ("手", 100.0),
        ("元", 1.0)
    ]
    multiplier = 1.0
    for suffix, factor in suffix_multipliers:
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            multiplier = factor
            break
    # Extract numeric component
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return np.nan
    try:
        return float(match.group()) * multiplier
    except ValueError:
        return np.nan

# Cache stock basic info to avoid repeated API hits
@st.cache_data(ttl=3600)
def get_stock_basic_info(symbol_no_prefix):
    """获取单只股票的基础信息（总股本、流通股本、市值等）"""
    try:
        info_df = ak.stock_individual_info_em(symbol=symbol_no_prefix)
    except Exception:
        return {}
    if info_df is None or info_df.empty:
        return {}
    info_dict = dict(zip(info_df["item"], info_df["value"]))
    return info_dict

# Function to get historical market value data for a stock
@st.cache_data(ttl=900, show_spinner=False)
def get_stock_market_value(symbol, start_date, end_date):
    """获取指定股票在给定日期范围的市值数据"""
    try:
        symbol_no_prefix = str(symbol).zfill(6)
        hist_df = None
        last_exception = None
        
        for attempt in range(REQUEST_MAX_RETRIES):
            if attempt:
                time.sleep(backoff_delay(attempt))
            acquire_request_slot()
            try:
                hist_df = ak.stock_zh_a_hist(
                    symbol=symbol_no_prefix,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=""
                )
                break
            except Exception as exc:
                last_exception = exc
                continue
        
        if hist_df is None:
            reason = last_exception or "未知错误"
            append_error_log(
                f"获取 {symbol_no_prefix} 数据失败(重试 {REQUEST_MAX_RETRIES} 次后放弃): {reason}"
            )
            return None
        
        if hist_df.empty:
            return None
        
        # Ensure we always have a name column for downstream usage
        if '名称' not in hist_df.columns:
            basic_info = get_stock_basic_info(symbol_no_prefix)
            stock_name = basic_info.get('证券简称') or symbol_no_prefix
            hist_df['名称'] = stock_name
        
        # Normalise date column
        hist_df['日期'] = pd.to_datetime(hist_df['日期'])
        
        # Calculate market value when not provided directly
        has_circ_mv = '流通市值' in hist_df.columns
        has_total_mv = '总市值' in hist_df.columns
        if not has_circ_mv or not has_total_mv:
            basic_info = get_stock_basic_info(symbol_no_prefix)
            float_shares = parse_numeric_value(basic_info.get('流通股本'))
            total_shares = parse_numeric_value(basic_info.get('总股本'))
            latest_circ_mv = parse_numeric_value(basic_info.get('流通市值'))
            latest_total_mv = parse_numeric_value(basic_info.get('总市值'))
            
            # Use latest market value to back-calc shares if share counts missing
            close_candidates = [col for col in hist_df.columns if any(key in col for key in ['收盘', 'close'])]
            close_col = None
            if '收盘' in hist_df.columns:
                close_col = '收盘'
            elif '收盘价' in hist_df.columns:
                close_col = '收盘价'
            elif close_candidates:
                close_col = close_candidates[0]
            elif 'close' in hist_df.columns:
                close_col = 'close'
            
            if close_col is not None:
                end_price = parse_numeric_value(hist_df.iloc[-1][close_col])
            else:
                end_price = np.nan
            
            if np.isnan(float_shares) and not np.isnan(latest_circ_mv) and end_price not in (None, 0):
                float_shares = latest_circ_mv / end_price if end_price else np.nan
            if np.isnan(total_shares) and not np.isnan(latest_total_mv) and end_price not in (None, 0):
                total_shares = latest_total_mv / end_price if end_price else np.nan
            
            if close_col is not None:
                if not has_circ_mv and not np.isnan(float_shares):
                    hist_df['流通市值'] = hist_df[close_col].apply(lambda x: parse_numeric_value(x) * float_shares if pd.notna(x) else np.nan)
                    has_circ_mv = True
                if not has_total_mv and not np.isnan(total_shares):
                    hist_df['总市值'] = hist_df[close_col].apply(lambda x: parse_numeric_value(x) * total_shares if pd.notna(x) else np.nan)
                    has_total_mv = True
        
        for col in ['流通市值', '总市值']:
            if col in hist_df.columns:
                hist_df[col] = hist_df[col].apply(parse_numeric_value)
        
        # Return only necessary columns if market value available
        needed_cols = ['日期', '名称']
        if has_circ_mv:
            needed_cols.append('流通市值')
        if has_total_mv:
            needed_cols.append('总市值')
        return hist_df[needed_cols]
    except Exception as e:
        append_error_log(f"获取 {symbol} 数据失败: {e}")
        return None

# Function to test available AkShare functions for historical data
def test_available_history_functions():
    """测试可用的历史数据函数"""
    test_stock = "000001"  # 平安银行
    results = {}
    
    # List of potential functions to try
    functions_to_try = [
        {"name": "stock_zh_a_hist", "params": {"symbol": test_stock, "period": "daily", "start_date": "20230101", "end_date": "20230102", "adjust": ""}},
        {"name": "stock_zh_a_daily", "params": {"symbol": test_stock}},
        {"name": "stock_zh_a_spot_em", "params": {}},
        {"name": "stock_individual_info_em", "params": {"symbol": test_stock}}
    ]
    
    for func in functions_to_try:
        try:
            function_name = func["name"]
            function = getattr(ak, function_name)
            result = function(**func["params"])
            
            if not result.empty:
                # Check if market value data is available
                has_market_value = any('市值' in col for col in result.columns)
                columns = list(result.columns)
                
                results[function_name] = {
                    "status": "success",
                    "has_market_value": has_market_value,
                    "columns": columns,
                    "sample": result.head(1).to_dict('records')[0] if not result.empty else {}
                }
            else:
                results[function_name] = {
                    "status": "empty_result",
                    "has_market_value": False,
                    "columns": []
                }
        except Exception as e:
            results[function_name] = {
                "status": "error",
                "error": str(e)
            }
    
    return results

# Function to add exchange prefix to stock code
def add_exchange_prefix(code):
    """根据股票代码添加交易所前缀"""
    code = str(code).zfill(6)
    if code.startswith(('6', '688', '900')):
        return f"sh{code}"
    else:
        return f"sz{code}"

# Function to get market value for specific dates
def get_market_value_for_dates(symbol, start_date, end_date):
    """获取指定股票在起始日和结束日的市值数据"""
    try:
        symbol_no_prefix = str(symbol).zfill(6)
        # Get all data in range
        df = get_stock_market_value(symbol, start_date, end_date)
        if df is None or df.empty:
            return None, None, None, None, None, "历史行情为空"
        
        # Ensure we have at least one record
        if len(df) < 1:
            return None, None, None, None, None, "历史记录数量不足"
        
        df = df.sort_values('日期').reset_index(drop=True)
        
        name = df.iloc[-1].get('名称', symbol)
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
            close_column_candidates = ["收盘", "收盘价", "close", "Close", "收盘(元)", "收盘价(元)", "收盘价(元/股)"]
            close_col = next((col for col in close_column_candidates if col in df.columns), None)
            if close_col:
                closing_prices = df[close_col].apply(parse_numeric_value)
                basic_info = get_stock_basic_info(symbol_no_prefix)
                total_shares = parse_numeric_value(basic_info.get('总股本'))
                if pd.notna(total_shares) and total_shares not in (0, np.nan):
                    reconstructed_total_series = closing_prices * total_shares
                    start_total_mv, end_total_mv = extract_first_last(reconstructed_total_series)
        
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
    test_stocks = ['000001', '600000']  # Test with both SZ and SH markets
    results = []
    
    # First, test which functions are available
    function_test_results = test_available_history_functions()
    
    # Now test actual data retrieval for specific stocks
    for stock in test_stocks:
        try:
            hist_df = get_stock_market_value(stock, start_date_str, end_date_str)
            
            if hist_df is not None and not hist_df.empty:
                sample_data = hist_df.head(1).to_dict('records')[0]
                results.append({
                    'stock': stock,
                    'status': 'success',
                    'columns': list(hist_df.columns),
                    'sample': sample_data
                })
            else:
                results.append({
                    'stock': stock,
                    'status': 'empty_response',
                    'columns': [],
                    'sample': {}
                })
        except Exception as e:
            results.append({
                'stock': stock,
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
                try:
                    name, _, _, start_total_mv, end_total_mv, reason = get_market_value_for_dates(
                        code, start_date_str, end_date_str
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
