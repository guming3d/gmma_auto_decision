import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import time
from functools import lru_cache
import os
import json
import traceback

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
此应用程序使用 akshare 数据分析中国 A 股市场股票在指定时间区间内的总市值/流通市值变化。
它可以排名出总市值/流通市值增加最多的前100只股票和减少最多的前100只股票。
""")

# Create cache directory if it doesn't exist
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(cache_dir, exist_ok=True)

# Cache for stock list to avoid repeated API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_list():
    """获取所有沪深A股的代码和名称"""
    try:
        stock_list_df = ak.stock_info_a_code_name()
        return stock_list_df
    except Exception as e:
        st.error(f"获取股票列表失败: {str(e)}")
        return pd.DataFrame(columns=['code', 'name'])

# Function to get historical market value data for a stock
def get_stock_market_value(symbol, start_date, end_date):
    """获取指定股票在给定日期范围的市值数据"""
    try:
        # For akshare stock_zh_a_hist, the symbol should not have sh/sz prefix
        symbol_no_prefix = symbol
        if symbol.startswith('sh') or symbol.startswith('sz'):
            symbol_no_prefix = symbol[2:]
            
        # Use the standard AkShare history function that includes market value data
        hist_df = ak.stock_zh_a_hist(symbol=symbol_no_prefix, period="daily", 
                                  start_date=start_date, end_date=end_date, adjust="")
        
        if hist_df.empty:
            return None
        
        # Check if required columns exist - rename columns from Chinese to standardized names
        if '流通市值' not in hist_df.columns or '总市值' not in hist_df.columns:
            # Try to find alternative columns that might contain market value data
            market_value_columns = [col for col in hist_df.columns if '市值' in col]
            if market_value_columns:
                st.warning(f"股票 {symbol} 数据列名称不标准，找到可能的市值列: {market_value_columns}")
            
            # For debugging - show what columns we actually have
            st.info(f"股票 {symbol} 可用列: {list(hist_df.columns)}")
            
            # If we don't have the required columns, try to get market value data from another API
            try:
                # Use stock_individual_info_em to get current market value
                stock_info = ak.stock_individual_info_em(symbol=symbol_no_prefix)
                
                if not stock_info.empty:
                    # Extract total and circulating market value if available
                    market_value_row = stock_info[stock_info['item'] == '总市值'].iloc[0]['value'] if '总市值' in stock_info['item'].values else None
                    circ_value_row = stock_info[stock_info['item'] == '流通市值'].iloc[0]['value'] if '流通市值' in stock_info['item'].values else None
                    
                    # Create a new DataFrame with the market value data for the specified dates
                    market_value_df = pd.DataFrame({
                        '日期': [start_date, end_date],
                        '名称': [symbol_no_prefix, symbol_no_prefix],
                        '流通市值': [circ_value_row, circ_value_row],
                        '总市值': [market_value_row, market_value_row]
                    })
                    
                    return market_value_df
            except Exception as e:
                st.warning(f"尝试获取 {symbol} 市值数据的备选方法失败: {str(e)}")
                return None
        
        # If we have the correct columns, return them
        return hist_df[['日期', '名称', '流通市值', '总市值']]
    except Exception as e:
        st.warning(f"获取 {symbol} 数据失败: {str(e)}")
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
        # Get all data in range
        df = get_stock_market_value(symbol, start_date, end_date)
        if df is None or df.empty:
            return None, None, None, None, None
        
        # Convert date to datetime for comparison
        df['日期'] = pd.to_datetime(df['日期'])
        
        # Ensure we have at least one record
        if len(df) < 1:
            return None, None, None, None, None
        
        # Get start date record (first available)
        start_record = df.iloc[0]
        
        # Ensure we have at least two different days of data
        if len(df) < 2 and start_date != end_date:
            # If we only have one record but requested a range, use that record for both start and end
            end_record = start_record
        else:
            end_record = df.iloc[-1]
        
        # Extract values - ensure numeric values
        try:
            name = start_record['名称']
            start_circ_mv = pd.to_numeric(start_record['流通市值'], errors='coerce')
            end_circ_mv = pd.to_numeric(end_record['流通市值'], errors='coerce')
            start_total_mv = pd.to_numeric(start_record['总市值'], errors='coerce')
            end_total_mv = pd.to_numeric(end_record['总市值'], errors='coerce')
            
            # Check for NaN values
            if pd.isna(start_circ_mv) or pd.isna(end_circ_mv) or pd.isna(start_total_mv) or pd.isna(end_total_mv):
                return None, None, None, None, None
                
            return name, start_circ_mv, end_circ_mv, start_total_mv, end_total_mv
        except (KeyError, TypeError) as e:
            return None, None, None, None, None
    except Exception as e:
        return None, None, None, None, None

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
            symbol = add_exchange_prefix(stock)
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
    
    # Date range selection
    st.sidebar.subheader("时间区间")
    date_range_option = st.sidebar.radio(
        "选择时间区间",
        ["过去7天", "过去14天", "过去30天", "过去90天", "过去180天", "过去365天", "自定义"]
    )
    
    # Handle date range selection
    if date_range_option == "自定义":
        today = datetime.now()
        default_start = today - timedelta(days=30)
        
        start_date = st.sidebar.date_input(
            "开始日期",
            value=default_start,
            max_value=today - timedelta(days=1)
        )
        
        end_date = st.sidebar.date_input(
            "结束日期",
            value=today,
            min_value=start_date,
            max_value=today
        )
        
        # Format dates for API
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
    else:
        # Calculate days based on selection
        days_map = {
            "过去7天": 7,
            "过去14天": 14,
            "过去30天": 30,
            "过去90天": 90,
            "过去180天": 180,
            "过去365天": 365
        }
        days_ago = days_map[date_range_option]
        start_date_str, end_date_str = get_formatted_date_range(days_ago)
    
    # Market value type selection
    market_value_type = st.sidebar.radio(
        "选择市值类型",
        ["流通市值", "总市值"]
    )
    
    # Number of stocks to display
    top_n = st.sidebar.slider(
        "显示每组前N只股票",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )
    
    # Add a test API button
    if st.sidebar.button("测试API连接", help="点击测试与API的连接状态"):
        with st.spinner("正在测试API连接和可用函数..."):
            try:
                api_test_results = test_api_connectivity(start_date_str, end_date_str)
                
                st.subheader("API函数测试结果")
                for func_name, result in api_test_results['function_tests'].items():
                    if result['status'] == 'success':
                        market_value_status = "包含" if result.get('has_market_value', False) else "不包含"
                        st.success(f"函数 {func_name} 调用成功，{market_value_status}市值数据")
                        st.write(f"可用列: {', '.join(result.get('columns', []))}")
                        if 'sample' in result:
                            with st.expander(f"{func_name} 数据样例"):
                                st.write(result['sample'])
                    else:
                        st.error(f"函数 {func_name} 调用失败: {result.get('error', '未知错误')}")
                
                st.subheader("股票数据测试结果")
                for result in api_test_results['stock_tests']:
                    if result['status'] == 'success':
                        st.success(f"成功获取 {result['stock']} 的数据")
                        st.write(f"可用列: {', '.join(result['columns'])}")
                        with st.expander("数据样例"):
                            st.write(result['sample'])
                    else:
                        st.error(f"获取 {result['stock']} 的数据失败: {result.get('error', '未知错误')}")
                        if 'traceback' in result:
                            with st.expander("查看详细错误信息"):
                                st.code(result['traceback'])
            except Exception as e:
                st.error(f"测试API连接时出错: {str(e)}")
                st.code(traceback.format_exc())
    
    # Start analysis button
    if st.sidebar.button("开始分析", type="primary"):
        try:
            # Display selected parameters
            st.subheader("分析参数")
            st.write(f"- 时间区间: {start_date_str} 至 {end_date_str}")
            st.write(f"- 市值类型: {market_value_type}")
            st.write(f"- 显示每组前: {top_n} 只股票")
            
            # Get stock list
            with st.spinner("正在获取 A 股股票列表..."):
                stock_list_df = get_stock_list()
                if stock_list_df.empty:
                    st.error("无法获取股票列表，请稍后重试")
                    return
                st.success(f"共获取到 {len(stock_list_df)} 只 A 股股票")
            
            # Initialize results dataframe
            results = []
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create container for detailed error logs
            error_log_container = st.container()
            with error_log_container:
                error_expander = st.expander("错误日志 (展开查看详情)", expanded=False)
            
            # Process each stock
            total_stocks = len(stock_list_df)
            processed = 0
            errors = 0
            error_logs = []
            
            # Limit to first 20 stocks for testing if needed - comment out in production
            # stock_list_df = stock_list_df.head(20)
            
            for _, row in stock_list_df.iterrows():
                try:
                    # Update progress
                    processed += 1
                    if processed % 10 == 0:
                        progress_bar.progress(min(processed / total_stocks, 1.0))
                        status_text.text(f"已处理: {processed}/{total_stocks} (错误: {errors})")
                    
                    # Get stock code and add exchange prefix
                    code = str(row['code']).zfill(6)
                    
                    # Get market value data
                    name, start_circ_mv, end_circ_mv, start_total_mv, end_total_mv = get_market_value_for_dates(
                        code, start_date_str, end_date_str
                    )
                    
                    # Skip if data is missing
                    if None in (name, start_circ_mv, end_circ_mv, start_total_mv, end_total_mv):
                        errors += 1
                        # Add to error log
                        if name is None:
                            name = row.get('name', '未知')
                        error_logs.append(f"股票 {code} ({name}): 数据不完整或缺失")
                        continue
                    
                    # Calculate changes
                    circ_mv_change = end_circ_mv - start_circ_mv
                    total_mv_change = end_total_mv - start_total_mv
                    
                    # Use try-except for percentage calculations to handle division by zero
                    try:
                        circ_mv_change_percent = (circ_mv_change / start_circ_mv * 100) if start_circ_mv > 0 else 0
                    except:
                        circ_mv_change_percent = 0
                        
                    try:
                        total_mv_change_percent = (total_mv_change / start_total_mv * 100) if start_total_mv > 0 else 0
                    except:
                        total_mv_change_percent = 0
                    
                    # Add to results
                    results.append({
                        'code': code,
                        'name': name,
                        'start_circ_mv': start_circ_mv,
                        'end_circ_mv': end_circ_mv,
                        'circ_mv_change': circ_mv_change,
                        'circ_mv_change_percent': circ_mv_change_percent,
                        'start_total_mv': start_total_mv,
                        'end_total_mv': end_total_mv,
                        'total_mv_change': total_mv_change,
                        'total_mv_change_percent': total_mv_change_percent
                    })
                except Exception as e:
                    errors += 1
                    # Add to error log with more details
                    error_detail = f"股票 {code} ({row.get('name', '未知')}): {str(e)}"
                    error_logs.append(error_detail)
                    continue
            
            # Update error log in UI
            with error_expander:
                if error_logs:
                    st.write(f"处理过程中遇到 {len(error_logs)} 个错误:")
                    for i, log in enumerate(error_logs[:100]):  # Limit to first 100 errors
                        st.text(f"{i+1}. {log}")
                    if len(error_logs) > 100:
                        st.text(f"... 还有 {len(error_logs) - 100} 个错误未显示")
                else:
                    st.write("处理过程未发现错误")
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.text(f"分析完成! 总共处理: {processed}/{total_stocks} (错误: {errors})")
            
            # Convert results to DataFrame
            if not results:
                st.error("未能获取有效数据，请尝试其他时间区间或检查网络连接")
                return
                
            results_df = pd.DataFrame(results)
            
            # Show summary of data collected
            st.info(f"成功收集了 {len(results_df)} 只股票的市值数据")
            
            # Determine which columns to use based on selected market value type
            if market_value_type == "流通市值":
                value_col = 'circ_mv_change'
                percent_col = 'circ_mv_change_percent'
                start_col = 'start_circ_mv'
                end_col = 'end_circ_mv'
            else:  # "总市值"
                value_col = 'total_mv_change'
                percent_col = 'total_mv_change_percent'
                start_col = 'start_total_mv'
                end_col = 'end_total_mv'
            
            # Sort for top increasing
            top_increase = results_df.sort_values(by=value_col, ascending=False).head(top_n).copy()
            
            # Sort for top decreasing
            top_decrease = results_df.sort_values(by=value_col, ascending=True).head(top_n).copy()
            
            # Format numbers for display (convert to 亿元)
            for df in [top_increase, top_decrease]:
                df['start_value_亿'] = df[start_col] / 100000000
                df['end_value_亿'] = df[end_col] / 100000000
                df['change_value_亿'] = df[value_col] / 100000000
                df['change_percent'] = df[percent_col]
            
            # Display top increasing stocks
            st.subheader(f"{market_value_type}增加最多的前{top_n}只股票")
            
            # Format the display columns
            display_increase = top_increase[['code', 'name', 'start_value_亿', 'end_value_亿', 'change_value_亿', 'change_percent']].copy()
            display_increase.columns = ['股票代码', '股票名称', f'起始{market_value_type}(亿元)', f'结束{market_value_type}(亿元)', 
                                        f'{market_value_type}变化(亿元)', '变化百分比(%)']
            
            # Format decimal places
            display_increase[f'起始{market_value_type}(亿元)'] = display_increase[f'起始{market_value_type}(亿元)'].round(2)
            display_increase[f'结束{market_value_type}(亿元)'] = display_increase[f'结束{market_value_type}(亿元)'].round(2)
            display_increase[f'{market_value_type}变化(亿元)'] = display_increase[f'{market_value_type}变化(亿元)'].round(2)
            display_increase['变化百分比(%)'] = display_increase['变化百分比(%)'].round(2)
            
            st.dataframe(display_increase, use_container_width=True)
            
            # Create bar chart for top increases
            fig_increase = go.Figure()
            fig_increase.add_trace(go.Bar(
                x=display_increase['股票名称'],
                y=display_increase[f'{market_value_type}变化(亿元)'],
                marker_color='red',
                text=display_increase[f'{market_value_type}变化(亿元)'].round(2),
                textposition='auto'
            ))
            fig_increase.update_layout(
                title=f"{market_value_type}增加最多的前{top_n}只股票",
                xaxis_title="股票名称",
                yaxis_title=f"{market_value_type}变化 (亿元)",
                height=600
            )
            st.plotly_chart(fig_increase, use_container_width=True)
            
            # Display top decreasing stocks
            st.subheader(f"{market_value_type}减少最多的前{top_n}只股票")
            
            # Format the display columns
            display_decrease = top_decrease[['code', 'name', 'start_value_亿', 'end_value_亿', 'change_value_亿', 'change_percent']].copy()
            display_decrease.columns = ['股票代码', '股票名称', f'起始{market_value_type}(亿元)', f'结束{market_value_type}(亿元)', 
                                        f'{market_value_type}变化(亿元)', '变化百分比(%)']
            
            # Format decimal places
            display_decrease[f'起始{market_value_type}(亿元)'] = display_decrease[f'起始{market_value_type}(亿元)'].round(2)
            display_decrease[f'结束{market_value_type}(亿元)'] = display_decrease[f'结束{market_value_type}(亿元)'].round(2)
            display_decrease[f'{market_value_type}变化(亿元)'] = display_decrease[f'{market_value_type}变化(亿元)'].round(2)
            display_decrease['变化百分比(%)'] = display_decrease['变化百分比(%)'].round(2)
            
            st.dataframe(display_decrease, use_container_width=True)
            
            # Create bar chart for top decreases
            fig_decrease = go.Figure()
            fig_decrease.add_trace(go.Bar(
                x=display_decrease['股票名称'],
                y=display_decrease[f'{market_value_type}变化(亿元)'],
                marker_color='green',
                text=display_decrease[f'{market_value_type}变化(亿元)'].round(2),
                textposition='auto'
            ))
            fig_decrease.update_layout(
                title=f"{market_value_type}减少最多的前{top_n}只股票",
                xaxis_title="股票名称",
                yaxis_title=f"{market_value_type}变化 (亿元)",
                height=600
            )
            st.plotly_chart(fig_decrease, use_container_width=True)
            
            # Add download buttons for the data
            st.subheader("数据下载")
            
            # Convert to CSV for download
            csv_increase = display_increase.to_csv(index=False)
            csv_decrease = display_decrease.to_csv(index=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label=f"下载{market_value_type}增加最多的股票数据",
                    data=csv_increase,
                    file_name=f"top_increase_{market_value_type}_{start_date_str}_to_{end_date_str}.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label=f"下载{market_value_type}减少最多的股票数据",
                    data=csv_decrease,
                    file_name=f"top_decrease_{market_value_type}_{start_date_str}_to_{end_date_str}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"分析过程中发生错误: {str(e)}")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main() 