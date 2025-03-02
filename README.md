# GMMA 股票分析工具

这是一个基于 Streamlit 的 A 股 GMMA (Guppy Multiple Moving Average) 技术分析工具，用于快速发现和分析股票的买入信号。

## 📊 功能特点

- **单一股票分析**：输入股票代码查看详细的 GMMA 指标和买入信号
- **自动扫描**：自动扫描全市场或指定行业中的买入信号
- **行业筛选**：按行业板块筛选股票
- **可视化图表**：直观显示股票价格、EMA 指标和买入信号点

## 📸 应用界面

![GMMA应用界面截图](images/Screenshot_2-3-2025_151929_gmmaautodecision.streamlit.app.jpeg)
*GMMA应用界面 - 显示股票GMMA交叉买入信号分析*

## 🚀 安装指南

1. **克隆仓库**

   ```bash
   git clone [仓库地址]
   cd GMMA
   ```

2. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

3. **运行应用**

   ```bash
   streamlit run gmma_akshare.py
   ```

## 📝 使用说明

### 单一股票分析

1. 在左侧边栏选择"单一股票分析"模式
2. 输入 6 位股票代码（例如：000001 代表平安银行）
3. 选择是否显示短期 EMA 和长期 EMA
4. 系统将自动加载并分析该股票的 GMMA 数据
5. 图表上用绿色垂直线标记买入信号（短期 EMA 从下方穿过长期 EMA）

### 自动扫描买入信号

1. 在左侧边栏选择"自动扫描买入信号"模式
2. 设置扫描参数：
   - 检查最近几天内的信号（1-7天）
   - 最多显示股票数量（1-20只）
   - 选择扫描范围（全部 A 股或按行业板块）
3. 如选择按行业板块，从下拉菜单中选择感兴趣的行业
4. 点击"开始扫描"按钮
5. 系统将自动扫描并显示符合条件的股票列表和详细图表

## 📈 GMMA 指标说明

GMMA (Guppy Multiple Moving Average) 是由澳大利亚交易员 Daryl Guppy 开发的技术分析工具，使用多条指数移动平均线 (EMA) 来分析市场趋势：

- **短期均线组**：由 3、5、8、10、12、15 日 EMA 组成，反映短期交易者行为
- **长期均线组**：由 30、35、40、45、50、60 日 EMA 组成，反映长期投资者行为

当短期均线从下方穿过长期均线时，通常被视为买入信号。

## 📊 数据来源

本应用使用 [AKShare](https://github.com/akfamily/akshare) 获取 A 股实时数据。

## 🔍 注意事项

- 该应用仅供参考，不构成任何投资建议
- 投资有风险，需谨慎决策

## 🔧 技术栈

- Streamlit
- AKShare
- Pandas
- Plotly
- NumPy
