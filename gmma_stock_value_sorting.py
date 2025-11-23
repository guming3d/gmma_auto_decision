from __future__ import annotations
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Iterable
import gc
import json

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="A股市值变化排序器",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("A股市值变化排序工具")
st.markdown(
    """
该工具对比**本地 Excel 报价**与历史行情（`history/all_daily_bars.pkl`）,
计算所有 A 股的总市值变化并给出涨跌榜。上传最新的 A 股报价表，
选择一个历史交易日后即可在离线环境完成排序，无需访问 Tushare。
"""
)

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = APP_DIR / "history" / "all_daily_bars_compact.parquet"
HISTORY_CACHE_FILE = CACHE_DIR / "value-sorting-history.parquet"
HISTORY_CACHE_META = CACHE_DIR / "value-sorting-history.meta.json"
SAMPLE_SNAPSHOT_FILE = APP_DIR / "A股报价.xls"
DEFAULT_TOP_COUNT = 50
TOP_COUNT_OPTIONS = [50, 100, 200]
SCAN_HISTORY_FILE = CACHE_DIR / "value-sorting-scan-history.json"
SCAN_HISTORY_LIMIT = 10
SORT_OPTIONS = {
    "总市值变化(绝对值)": "total_mv_change",
    "总市值变化百分比": "total_mv_change_pct",
}
SCOPE_OPTIONS = {
    "按个股": "stock",
    "按行业": "industry",
}

AUTH_SESSION_KEY = "gmma_is_authenticated"
AUTH_USER_KEY = "gmma_authenticated_user"
LOGIN_FORM_KEY = "gmma_login_form"

CODE_COLUMNS = ["代码", "股票代码", "证券代码", "symbol", "code", "ts_code"]
NAME_COLUMNS = ["名称", "股票名称", "证券简称", "name"]
PRICE_COLUMNS = ["最新", "现价", "收盘价", "close", "价格"]
TOTAL_SHARE_COLUMNS = ["总股本", "总股本(股)", "总股本(万股)", "total_share"]
TOTAL_MV_COLUMNS = [
    "总市值",
    "总市值(元)",
    "总市值(人民币)",
    "总市值(万元)",
    "总市值(亿元)",
    "market_cap",
    "total_mv",
    "circ_mv",
]
HISTORY_SHARE_COLUMNS = ["total_share", "float_share", "free_share"]
HISTORY_MV_COLUMNS = ["total_mv", "circ_mv"]


def _resolve_history_source(primary: Path | None = None) -> Path:
    candidates: list[Path] = []
    if primary:
        candidates.append(primary)
        if primary.suffix:
            candidates.append(primary.with_suffix(""))
    default_candidates = [
        APP_DIR / "history" / "all_daily_bars_compact.parquet",
        APP_DIR / "history" / "all_daily_bars.pkl",
    ]
    candidates.extend(default_candidates)
    seen: set[str] = set()
    missing_paths: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate if candidate.is_absolute() else (APP_DIR / candidate).resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return resolved
        missing_paths.append(key)
    raise FileNotFoundError(
        "未找到历史行情文件，请确认以下路径之一存在:\n" + "\n".join(missing_paths)
    )


def _get_auth_credentials():
    username = None
    password = None
    auth_section = st.secrets.get("auth")
    if auth_section and isinstance(auth_section, Mapping):
        username = auth_section.get("username") or auth_section.get("user")
        password = auth_section.get("password")
    username = username or st.secrets.get("auth_username") or st.secrets.get("AUTH_USERNAME")
    password = password or st.secrets.get("auth_password") or st.secrets.get("AUTH_PASSWORD")
    return username, password


def _trigger_rerun():
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun:
        rerun()


def ensure_authenticated():
    stored_username, stored_password = _get_auth_credentials()
    if not stored_username or not stored_password:
        st.error("请在 st.secrets 中配置 auth.username 和 auth.password。")
        st.stop()
    if st.session_state.get(AUTH_SESSION_KEY):
        return True
    st.subheader("登录验证")
    st.caption("请输入用户名和密码以继续访问 GMMA 工具。")
    with st.form(LOGIN_FORM_KEY, clear_on_submit=False):
        username_input = st.text_input("用户名", key="auth_username_input")
        password_input = st.text_input("密码", type="password", key="auth_password_input")
        submitted = st.form_submit_button("登录")
    entered_username = username_input.strip()
    entered_password = password_input
    if submitted:
        if entered_username == stored_username and entered_password == stored_password:
            st.session_state[AUTH_SESSION_KEY] = True
            st.session_state[AUTH_USER_KEY] = entered_username
            st.success("登录成功，正在加载应用……")
            _trigger_rerun()
        else:
            st.error("用户名或密码错误，请重试。")
    st.stop()


ensure_authenticated()


def normalize_stock_code(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if "." in text:
        text = text.split(".", 1)[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6) if digits else ""


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized = {col: str(col).strip() for col in df.columns}
    canonical = [str(c).strip() for c in candidates]
    for col, label in normalized.items():
        if label in canonical:
            return col
    lower_map = {col: label.lower() for col, label in normalized.items()}
    lower_targets = [c.lower() for c in canonical]
    for col, label in lower_map.items():
        if label in lower_targets:
            return col
    for target in canonical:
        for col, label in normalized.items():
            if target and target in label:
                return col
    return None


def convert_share_units(series: pd.Series | None, column_name: str | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    values = pd.to_numeric(series, errors="coerce")
    column_name = (column_name or "").lower()
    if column_name in {"total_share", "float_share", "free_share"} or "万股" in column_name:
        return values * 1e4
    return values


def convert_market_value(series: pd.Series | None, column_name: str | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    values = pd.to_numeric(series, errors="coerce")
    label = (column_name or "").lower()
    if label in {"total_mv", "circ_mv"}:
        return values * 1e4
    if "万亿" in label:
        return values * 1e12
    if "亿元" in label:
        return values * 1e8
    if "万元" in label:
        return values * 1e4
    return values


def prepare_snapshot_dataframe(df: pd.DataFrame, *, source_label: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"{source_label} 没有可用的行情数据。")
    working = df.copy()
    working.columns = [str(col).strip() for col in working.columns]
    code_col = find_column(working, CODE_COLUMNS)
    price_col = find_column(working, PRICE_COLUMNS)
    if not code_col or not price_col:
        raise ValueError("无法识别行情文件中的 `股票代码` / `现价` 列，请确认列名。")
    name_col = find_column(working, NAME_COLUMNS)
    share_col = find_column(working, TOTAL_SHARE_COLUMNS)
    mv_col = find_column(working, TOTAL_MV_COLUMNS)

    result = pd.DataFrame()
    result["code"] = working[code_col].apply(normalize_stock_code)
    result["name"] = (
        working[name_col].astype(str).str.strip()
        if name_col
        else result["code"]
    )
    result["current_price"] = pd.to_numeric(working[price_col], errors="coerce")
    result["current_total_share"] = (
        convert_share_units(working[share_col], share_col) if share_col else np.nan
    )
    result["current_total_mv"] = (
        convert_market_value(working[mv_col], mv_col) if mv_col else np.nan
    )

    result = result[result["code"].ne("")].copy()
    code_numeric = pd.to_numeric(result["code"], errors="coerce")
    result = result[code_numeric.notna()].copy()
    result["code_int"] = code_numeric.astype(np.uint32)
    result = result[result["current_price"].notna()]
    result.drop_duplicates(subset=["code"], keep="last", inplace=True)

    share_missing = result["current_total_share"].isna()
    mv_available = result["current_total_mv"].notna()
    price_available = result["current_price"].notna()
    result.loc[
        share_missing & mv_available & price_available, "current_total_share"
    ] = (
        result["current_total_mv"] / result["current_price"]
    )

    mv_missing = result["current_total_mv"].isna()
    share_available = result["current_total_share"].notna()
    result.loc[
        mv_missing & share_available & price_available, "current_total_mv"
    ] = (
        result["current_total_share"] * result["current_price"]
    )

    result = result[result["current_total_mv"].notna()].copy()
    result["source"] = source_label
    return result.reset_index(drop=True)


def read_snapshot_from_source(uploaded_file, fallback_path: Path | None) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        uploaded_file.seek(0)
        try:
            df = pd.read_excel(uploaded_file, engine=None)
        except Exception as exc:
            raise RuntimeError(f"读取上传的 Excel 文件失败: {exc}") from exc
        parsed = prepare_snapshot_dataframe(df, source_label=uploaded_file.name or "uploaded")
        return parsed, uploaded_file.name or "uploaded"
    if fallback_path and fallback_path.exists():
        try:
            df = pd.read_excel(fallback_path, engine=None)
        except Exception as exc:
            raise RuntimeError(f"读取示例文件失败: {exc}") from exc
        parsed = prepare_snapshot_dataframe(df, source_label=fallback_path.name)
        return parsed, fallback_path.name
    raise RuntimeError("请上传当日报价 Excel 文件。")


def _history_cache_is_valid(source: Path) -> bool:
    if not HISTORY_CACHE_FILE.exists() or not HISTORY_CACHE_META.exists():
        return False
    try:
        meta = json.loads(HISTORY_CACHE_META.read_text(encoding="utf-8"))
    except Exception:
        return False
    recorded_mtime = meta.get("source_mtime")
    try:
        source_mtime = source.stat().st_mtime
    except OSError:
        return False
    return isinstance(recorded_mtime, (int, float)) and abs(recorded_mtime - source_mtime) < 1


def _persist_history_cache(df: pd.DataFrame, source: Path) -> None:
    try:
        df.to_parquet(HISTORY_CACHE_FILE, index=False)
        HISTORY_CACHE_META.write_text(
            json.dumps({"source_mtime": source.stat().st_mtime}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        # 缓存失败不影响主流程
        pass


def _prepare_history_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["trade_date", "close", "ts_code", "code_int", "industry"]
    for col in HISTORY_SHARE_COLUMNS + HISTORY_MV_COLUMNS:
        if col in raw_df.columns:
            keep_cols.append(col)
    keep_cols = list(dict.fromkeys(keep_cols))
    available_cols = [col for col in keep_cols if col in raw_df.columns]
    if "trade_date" not in available_cols or "close" not in available_cols:
        raise RuntimeError("历史行情缺少 trade_date 或 close 字段，无法加载。")
    history = raw_df[available_cols].copy()
    del raw_df
    gc.collect()

    history["trade_date"] = pd.to_datetime(history["trade_date"], errors="coerce")
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    history.dropna(subset=["trade_date", "close"], inplace=True)

    for col in HISTORY_SHARE_COLUMNS + HISTORY_MV_COLUMNS:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce")
    if "industry" in history.columns:
        history["industry"] = history["industry"].fillna("").astype(str)

    if "code_int" in history.columns:
        history["code_int"] = pd.to_numeric(history["code_int"], errors="coerce")
    else:
        if "ts_code" not in history.columns:
            raise RuntimeError("历史行情缺少 ts_code 或 code_int 字段，无法推导股票代码。")
        code_series = history["ts_code"].astype(str).str.slice(0, 6)
        code_numeric = pd.to_numeric(code_series, errors="coerce")
        history["code_int"] = code_numeric
    history.dropna(subset=["code_int"], inplace=True)
    history["code_int"] = history["code_int"].astype(np.uint32)
    history.drop(columns=["ts_code"], inplace=True, errors="ignore")

    history["close"] = history["close"].astype(np.float32)
    history.sort_values(["code_int", "trade_date"], inplace=True)
    history.reset_index(drop=True, inplace=True)
    return history


@st.cache_resource(show_spinner=True)
def load_history_bars(path: Path = HISTORY_FILE) -> pd.DataFrame:
    source_path = _resolve_history_source(path)
    if _history_cache_is_valid(source_path):
        try:
            cached = pd.read_parquet(HISTORY_CACHE_FILE)
            cached["trade_date"] = pd.to_datetime(cached["trade_date"])
            if "code_int" in cached.columns:
                cached["code_int"] = cached["code_int"].astype(np.uint32)
            return cached
        except Exception:
            HISTORY_CACHE_FILE.unlink(missing_ok=True)
            HISTORY_CACHE_META.unlink(missing_ok=True)
    suffix = source_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(source_path)
    else:
        df = pd.read_pickle(source_path)
    if df is None or df.empty:
        raise RuntimeError(f"{source_path} 没有可用的历史数据。")
    history = _prepare_history_dataframe(df)
    _persist_history_cache(history, source_path)
    return history


def build_history_snapshot(history_df: pd.DataFrame, codes: list[str], target_date: datetime) -> pd.DataFrame:
    if history_df.empty or not codes:
        return pd.DataFrame()
    code_ints = pd.to_numeric(pd.Series(codes), errors="coerce").dropna().astype(np.uint32)
    if code_ints.empty:
        return pd.DataFrame()
    subset = history_df[
        history_df["code_int"].isin(set(code_ints.tolist()))
        & (history_df["trade_date"] <= target_date)
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    subset.sort_values(["code_int", "trade_date"], inplace=True)
    snapshot = subset.groupby("code_int", as_index=False).tail(1)
    if "industry" in snapshot.columns:
        snapshot["industry"] = snapshot["industry"].fillna("").astype(str)
    for col in HISTORY_SHARE_COLUMNS:
        if col in snapshot.columns:
            snapshot[col] = convert_share_units(snapshot[col], col)
    for col in HISTORY_MV_COLUMNS:
        if col in snapshot.columns:
            snapshot[col] = convert_market_value(snapshot[col], col)
    rename_map = {
        "trade_date": "history_trade_date",
        "close": "history_close",
    }
    for col in HISTORY_SHARE_COLUMNS + HISTORY_MV_COLUMNS:
        if col in snapshot.columns:
            rename_map[col] = f"history_{col}"
    snapshot.rename(columns=rename_map, inplace=True)
    return snapshot.reset_index(drop=True)


def combine_series(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    combined = pd.Series(np.nan, index=df.index, dtype=float)
    for col in candidates:
        if col in df.columns:
            combined = combined.fillna(pd.to_numeric(df[col], errors="coerce"))
    return combined


def compute_market_value_changes(
    current_df: pd.DataFrame,
    history_snapshot: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    merged = current_df.merge(history_snapshot, on="code_int", how="left")
    missing_codes = merged.loc[merged["history_close"].isna(), "code"].tolist()
    merged = merged[merged["history_close"].notna()].copy()
    if merged.empty:
        return pd.DataFrame(), missing_codes

    merged["history_trade_date"] = pd.to_datetime(merged["history_trade_date"])

    merged["history_total_share"] = combine_series(
        merged,
        ["history_total_share", "history_float_share", "history_free_share"],
    )
    merged["history_total_mv"] = combine_series(
        merged,
        ["history_total_mv", "history_circ_mv"],
    )

    fallback_share_mask = merged["history_total_share"].isna() & merged["current_total_share"].notna()
    merged.loc[fallback_share_mask, "history_total_share"] = merged.loc[fallback_share_mask, "current_total_share"]

    calc_history_mv_mask = (
        merged["history_total_mv"].isna()
        & merged["history_total_share"].notna()
        & merged["history_close"].notna()
    )
    merged.loc[calc_history_mv_mask, "history_total_mv"] = (
        merged.loc[calc_history_mv_mask, "history_total_share"]
        * merged.loc[calc_history_mv_mask, "history_close"]
    )

    calc_current_mv_mask = (
        merged["current_total_mv"].isna()
        & merged["current_total_share"].notna()
        & merged["current_price"].notna()
    )
    merged.loc[calc_current_mv_mask, "current_total_mv"] = (
        merged.loc[calc_current_mv_mask, "current_total_share"]
        * merged.loc[calc_current_mv_mask, "current_price"]
    )

    merged = merged[
        merged["history_total_mv"].notna()
        & merged["current_total_mv"].notna()
        & (merged["history_total_mv"] > 0)
    ].copy()
    if merged.empty:
        return pd.DataFrame(), missing_codes

    if "industry" not in merged.columns:
        merged["industry"] = ""
    merged["industry"] = merged["industry"].fillna("").astype(str)

    merged["total_mv_change"] = merged["current_total_mv"] - merged["history_total_mv"]
    merged["total_mv_change_pct"] = merged["total_mv_change"] / merged["history_total_mv"] * 100

    merged["history_total_mv_亿"] = merged["history_total_mv"] / 1e8
    merged["current_total_mv_亿"] = merged["current_total_mv"] / 1e8
    merged["total_mv_change_亿"] = merged["total_mv_change"] / 1e8

    merged["history_trade_date_str"] = merged["history_trade_date"].dt.strftime("%Y-%m-%d")
    return merged, missing_codes


def aggregate_by_industry(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    working = results_df.copy()
    working["industry"] = working["industry"].replace("", "未提供行业").fillna("未提供行业")
    grouped = (
        working.groupby("industry", as_index=False)
        .agg(
            history_total_mv=("history_total_mv", "sum"),
            current_total_mv=("current_total_mv", "sum"),
            total_mv_change=("total_mv_change", "sum"),
            stock_count=("code_int", "nunique"),
        )
    )
    grouped = grouped[grouped["history_total_mv"] > 0].copy()
    grouped["total_mv_change_pct"] = grouped["total_mv_change"] / grouped["history_total_mv"] * 100
    grouped["history_total_mv_亿"] = grouped["history_total_mv"] / 1e8
    grouped["current_total_mv_亿"] = grouped["current_total_mv"] / 1e8
    grouped["total_mv_change_亿"] = grouped["total_mv_change"] / 1e8
    return grouped


def load_scan_history() -> list[dict]:
    if not SCAN_HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(SCAN_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _write_scan_history(records: list[dict]) -> None:
    try:
        SCAN_HISTORY_FILE.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # 历史记录写入失败不影响主流程
        pass


def record_scan_history(entry: dict) -> list[dict]:
    history = load_scan_history()
    history.insert(0, entry)
    del history[SCAN_HISTORY_LIMIT:]
    _write_scan_history(history)
    return history


def render_scan_history_tab(records: list[dict]) -> None:
    if not records:
        st.info("暂无历史扫描记录。")
        return
    for idx, record in enumerate(records):
        scope_value = record.get("scope") or "stock"
        scope_label = record.get("scope_label") or ("按行业" if scope_value == "industry" else "按个股")
        unit_label = "个行业" if scope_value == "industry" else "只股票"
        header = (
            f"{record.get('history_date', '未知日期')} · "
            f"{record.get('metric_label', '未知排序')} · "
            f"{scope_label} · "
            f"前 {record.get('top_count', '?')} {unit_label}"
        )
        with st.expander(header, expanded=(idx == 0)):
            st.caption(
                f"扫描时间：{record.get('run_ts', '未知')} · "
                f"数据源：{record.get('source', '未记录')}"
            )
            top_table = pd.DataFrame(record.get("top_table", []))
            bottom_table = pd.DataFrame(record.get("bottom_table", []))
            if not top_table.empty:
                st.write("总市值增加榜")
                st.dataframe(top_table, use_container_width=True)
            else:
                st.write("总市值增加榜记录缺失。")
            if not bottom_table.empty:
                st.write("总市值减少榜")
                st.dataframe(bottom_table, use_container_width=True)
            else:
                st.write("总市值减少榜记录缺失。")


def build_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    display = df[
        [
            "code",
            "name",
            "history_trade_date_str",
            "history_total_mv_亿",
            "current_total_mv_亿",
            "total_mv_change_亿",
            "total_mv_change_pct",
        ]
    ].copy()
    display.columns = [
        "股票代码",
        "股票名称",
        "历史交易日",
        "历史总市值(亿元)",
        "当前总市值(亿元)",
        "总市值变化(亿元)",
        "变化百分比(%)",
    ]
    for col in [
        "历史总市值(亿元)",
        "当前总市值(亿元)",
        "总市值变化(亿元)",
        "变化百分比(%)",
    ]:
        display[col] = display[col].astype(float).round(2)
    return display


def build_industry_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    display = df[
        [
            "industry",
            "stock_count",
            "history_total_mv_亿",
            "current_total_mv_亿",
            "total_mv_change_亿",
            "total_mv_change_pct",
        ]
    ].copy()
    display.columns = [
        "行业",
        "覆盖股票数",
        "历史总市值(亿元)",
        "当前总市值(亿元)",
        "总市值变化(亿元)",
        "变化百分比(%)",
    ]
    for col in [
        "历史总市值(亿元)",
        "当前总市值(亿元)",
        "总市值变化(亿元)",
        "变化百分比(%)",
    ]:
        display[col] = display[col].astype(float).round(2)
    display["覆盖股票数"] = display["覆盖股票数"].astype(int)
    return display


def render_stock_tables(
    results_df: pd.DataFrame,
    metric_key: str,
    top_k: int,
    history_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available = len(results_df)
    if available == 0:
        st.error("没有足够的数据用于展示，请调整日期或确认行情文件。")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    top_k = min(top_k, available)
    sorted_df = results_df.sort_values(metric_key, ascending=False)
    top_df = sorted_df.head(top_k)
    bottom_df = sorted_df.tail(top_k).sort_values(metric_key, ascending=True)

    display_top = build_display_frame(top_df)
    display_bottom = build_display_frame(bottom_df)

    st.subheader(f"总市值增加最多的前 {top_k} 只股票")
    st.dataframe(display_top, use_container_width=True)

    st.subheader(f"总市值减少最多的前 {top_k} 只股票")
    st.dataframe(display_bottom, use_container_width=True)

    st.subheader("数据下载")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="下载涨幅榜 CSV",
            data=display_top.to_csv(index=False),
            file_name=f"top_increase_total_mv_{history_label}.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            label="下载跌幅榜 CSV",
            data=display_bottom.to_csv(index=False),
            file_name=f"top_decrease_total_mv_{history_label}.csv",
            mime="text/csv",
        )
    return top_df, bottom_df, display_top, display_bottom


def render_industry_tables(
    industry_df: pd.DataFrame,
    metric_key: str,
    top_k: int,
    history_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available = len(industry_df)
    if available == 0:
        st.error("没有足够的行业数据用于展示。")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    top_k = min(top_k, available)
    sorted_df = industry_df.sort_values(metric_key, ascending=False)
    top_df = sorted_df.head(top_k)
    bottom_df = sorted_df.tail(top_k).sort_values(metric_key, ascending=True)

    display_top = build_industry_display_frame(top_df)
    display_bottom = build_industry_display_frame(bottom_df)

    st.subheader(f"总市值增加最多的前 {top_k} 个行业")
    st.dataframe(display_top, use_container_width=True)

    st.subheader(f"总市值减少最多的前 {top_k} 个行业")
    st.dataframe(display_bottom, use_container_width=True)

    st.subheader("数据下载")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="下载行业涨幅榜 CSV",
            data=display_top.to_csv(index=False),
            file_name=f"industry_top_increase_total_mv_{history_label}.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            label="下载行业跌幅榜 CSV",
            data=display_bottom.to_csv(index=False),
            file_name=f"industry_top_decrease_total_mv_{history_label}.csv",
            mime="text/csv",
        )
    return top_df, bottom_df, display_top, display_bottom


def main():
    st.sidebar.title("分析设置")
    today = datetime.now().date()
    history_input_key = "history_compare_date"
    history_seed_key = "_history_compare_date_seed"
    if st.session_state.get(history_seed_key) != today:
        # Force the widget to reinitialize with today's date when the system date changes
        st.session_state[history_seed_key] = today
        st.session_state.pop(history_input_key, None)
    history_date = st.sidebar.date_input(
        "历史比较日",
        value=today,
        max_value=today,
        key=history_input_key,
    )
    scope_label = st.sidebar.radio(
        "排序范围",
        list(SCOPE_OPTIONS.keys()),
        index=0,
    )
    scope = SCOPE_OPTIONS[scope_label]
    metric_label = st.sidebar.radio(
        "排序依据",
        list(SORT_OPTIONS.keys()),
        index=0,
    )
    top_count = st.sidebar.selectbox(
        "展示股票数量",
        options=TOP_COUNT_OPTIONS,
        index=TOP_COUNT_OPTIONS.index(DEFAULT_TOP_COUNT),
        format_func=lambda x: f"{x} 只",
    )
    exclude_st = st.sidebar.checkbox("排除 ST/*ST 股票", value=True)
    uploaded_file = st.sidebar.file_uploader("上传当日 A 股报价 (xls)", type=["xls", "xlsx"])
    if uploaded_file is None and SAMPLE_SNAPSHOT_FILE.exists():
        st.sidebar.info(f"未上传文件，将默认使用示例 {SAMPLE_SNAPSHOT_FILE.name}")
    elif uploaded_file is None:
        st.sidebar.warning("请上传当日报价 Excel 文件（xls/xlsx）。")

    analyze = st.sidebar.button("开始分析", type="primary")
    history_records = load_scan_history()
    result_tab, history_tab = st.tabs(["本次结果", "历史记录"])
    try:
        with result_tab:
            if not analyze:
                st.info("请在左侧上传文件并点击“开始分析”")
                return

            try:
                current_df, source_name = read_snapshot_from_source(uploaded_file, SAMPLE_SNAPSHOT_FILE)
            except Exception as exc:
                st.error(str(exc))
                return

            if current_df.empty:
                st.error("行情文件没有可用数据。")
                return

            st.success(f"已加载 {len(current_df)} 只股票的当日行情（来源：{source_name}）")
            if exclude_st:
                before = len(current_df)
                current_df = current_df[~current_df["name"].str.contains("ST", case=False, na=False)].copy()
                removed = before - len(current_df)
                if removed > 0:
                    st.info(f"已根据名称排除 {removed} 只 ST/*ST 股票。")

            try:
                with st.spinner("正在载入历史行情... 该过程可能耗时一分钟"):
                    history_df = load_history_bars()
            except Exception as exc:
                st.error(f"无法读取历史行情: {exc}")
                return

            history_ts = datetime.combine(history_date, datetime.min.time())
            with st.spinner("正在匹配历史交易日..."):
                snapshot = build_history_snapshot(history_df, current_df["code_int"].tolist(), history_ts)

            if snapshot.empty:
                st.error("在指定日期之前未找到任何历史行情记录，请选择更晚的日期。")
                return

            results_df, missing_codes = compute_market_value_changes(current_df, snapshot)
            if results_df.empty:
                st.error("无法计算市值变化，请检查上传文件与历史日期。")
                return

            hist_dates = sorted(results_df["history_trade_date"].dt.date.unique())
            if len(hist_dates) == 1:
                st.info(f"历史比较日匹配到 {hist_dates[0]}")
            else:
                st.info(
                    f"请求日期 {history_date} 对应到 {hist_dates[0]}~{hist_dates[-1]} "
                    "（部分新股会使用首个可用交易日）"
                )

            metric_key = SORT_OPTIONS[metric_label]
            history_label = history_date.strftime("%Y%m%d")
            if scope == "industry":
                industry_df = aggregate_by_industry(results_df)
                top_df, bottom_df, display_top, display_bottom = render_industry_tables(
                    industry_df,
                    metric_key,
                    top_count,
                    history_label,
                )
            else:
                top_df, bottom_df, display_top, display_bottom = render_stock_tables(
                    results_df,
                    metric_key,
                    top_count,
                    history_label,
                )
            if top_df.empty and bottom_df.empty:
                return

            history_entry = {
                "run_ts": datetime.now().isoformat(timespec="seconds"),
                "history_date": history_date.isoformat(),
                "metric_label": metric_label,
                "scope": scope,
                "scope_label": scope_label,
                "top_count": top_count,
                "source": source_name,
                "top_table": display_top.to_dict(orient="records"),
                "bottom_table": display_bottom.to_dict(orient="records"),
            }
            history_records = record_scan_history(history_entry)

            if missing_codes:
                with st.expander(f"缺少历史数据的股票（共 {len(missing_codes)} 只）", expanded=False):
                    st.write(", ".join(sorted(set(missing_codes))[:200]))
    finally:
        with history_tab:
            render_scan_history_tab(history_records)


if __name__ == "__main__":
    main()
