# LLM-Quant with FinGPT — 新闻驱动 + 技术指标 的 5 分钟量化策略

> 一个可运行的回测项目：结合 **FinGPT 情绪与事件评分** + **技术指标确认** + **风控（移动止损/分批止盈/日终清仓）**，针对美股大盘股进行 5 分钟级别回测；并预留 Alpaca 实盘接口。

---

## ✨ 项目特性（Highlights）

- **新闻→情绪→事件评分→交易信号**：FinGPT（Llama2-13B LoRA）对新闻做三元组输出（情绪/重要性/理由），并与技术因子融合，生成高置信度的 `BUY/SELL/HOLD` 信号。  
- **5 分钟级别回测引擎**：支持 **冷却时间、移动止损、分批止盈、最大日亏**、是否允许隔夜等风控开关。  
- **批量股票池**：默认含 AAPL/MSFT/NVDA/TSLA/AMZN/GOOGL/META，可通过 `universe_builder.py` 与配置扩展。  
- **可视化 + 详细日志**：输出权益曲线、交易明细、信号记录与事件日志到 `output/` 与 `logs/`。  
- **预留实盘**：`portfolio_manager.py` 集成 Alpaca 的下单逻辑入口（可切换回测/实盘）。

---

## 🗂 目录结构（建议）

```
.
├── backtest_engine.py                 # 5分钟回测引擎（移动止损/分批止盈/日终清仓/动态仓位）
├── data_collector.py                  # Tiingo/Finnhub 数据与新闻采集 + 映射
├── enhanced_run_strategy_backtest.py  # FinGPT增强版本主脚本（含参数优化/图表/综合分析）
├── factor_model_optimized.py          # FinGPT Llama2-13B LoRA 因子工程（情绪/事件/热度/技术因子）
├── portfolio_manager.py               # 组合与下单执行层（含事件评分加权、风控）
├── run_strategy_backtest.py           # 基础优化版主脚本（无FinGPT重模型）
├── setup_environment.py               # 环境自检与初始化
├── signal_generator.py                # 信号生成（阈值/冷却/成交量确认/技术确认/事件评分）
├── universe_builder.py                # 股票池筛选与映射（可接入Alpaca借券/新闻覆盖检查）
├── requirements.txt                   # 依赖（建议添加）
├── .env                               # 放置 API Keys（见下）
├── logs/                              # 运行日志
└── output/                            # 回测输出（图表/报表/JSON等）
```

---

## ⚙️ 环境准备

**1) Python & 虚拟环境**
```bash
conda create -n llm-quant python=3.10 -y
conda activate llm-quant
# 或者使用 venv：python -m venv venv && source venv/bin/activate
```

**2) 安装依赖**
- 推荐将依赖写入 `requirements.txt` 并执行：
```bash
pip install -r requirements.txt
```
- 若暂时没有 `requirements.txt`，可先安装常用包：
```bash
pip install pandas numpy matplotlib plotly seaborn tqdm python-dotenv requests             tiingo finnhub-python alpaca-trade-api transformers peft bitsandbytes             sentence-transformers scikit-learn scipy
```

**3) 可选：环境自检**
```bash
python setup_environment.py
```
该脚本会检查 **Python 版本 / GPU / 磁盘空间 / 目录** 是否就绪，并尝试安装依赖。

**4) GPU 建议**
- 运行 `factor_model_optimized.py` 中的 **Llama2-13B + LoRA** 推理，建议 **≥16GB 显存**（可 8bit 量化）。

---

## 🔐 配置 `.env`（根目录新建）

```ini
# 市场与新闻
TIINGO_API_KEY=你的TiingoKey
TIINGO_PLAN=free        # 或 power/enterprise
FINNHUB_API_KEY=你的FinnhubKey

# （可选）实盘/行情路由
ALPACA_API_KEY_ID=你的AlpacaKey
ALPACA_SECRET_KEY=你的AlpacaSecret
```

> 小贴士：免费计划会有速率/功能限制；若仅回测历史数据，主要受限在拉取频率与粒度。

---

## 🚀 快速运行

### 方式 A：基础优化版（不加载大模型）
```bash
python run_strategy_backtest.py
```
- 默认标的：`AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META`  
- 默认区间：`2024-09-01` ~ `2025-07-01`  
- 默认风控：**2%止损 / 3%止盈 / 2%移动止损 / 5%最大日亏 / 日终清仓 / 分批止盈 50%**

### 方式 B：FinGPT 增强版（加载 Llama2-13B LoRA）
```bash
python enhanced_run_strategy_backtest.py
```
- 在生成信号时叠加：**情绪得分、事件评分（1-5）、新闻热度、技术确认**  
- 内置 **参数优化**、**综合可视化**，输出至 `output/backtest/fingpt_enhanced/`。

运行完成后，查看：
- `output/`：权益曲线、回测指标、交易/信号明细(JSON/图表)  
- `logs/`：引擎/信号/组合等模块日志

---

## 🔧 关键参数（示例）

- **信号阈值**：`pos_thresh=0.7`, `neg_thresh=-0.7`, `novel_thresh=0.6`, `cooldown=300s`  
- **风控**：`stop_loss_pct=0.02`, `take_profit_pct=0.03`, `trailing_stop_pct=0.02`, `max_daily_loss_pct=0.05`, `allow_overnight=False`, `partial_take_profit=True`, `partial_take_ratio=0.5`  
- **事件增强**（增强版）：`min_event_score`, `min_confidence`, `event_position_multiplier`, `high_event_stop_loss_pct` 等

> 参数可在 `run_strategy_backtest.py / enhanced_run_strategy_backtest.py` 中直接修改；增强版还提供网格 **参数优化**。

---

## 🧱 模块说明（如何串起来）

1. **数据采集 `data_collector.py`**  
   - 拉取 **Tiingo** K线、**Finnhub** 新闻，维护 **ticker ↔ 公司/关键词** 映射与新闻去重；支持定时轮询统计新闻热度。

2. **因子工程 `factor_model_optimized.py`**  
   - 使用 **FinGPT Llama2-13B LoRA** 对新闻生成：**情绪(label/score/prob)**、**事件评分(1–5)**、**简要理由**。  
   - 叠加 **技术指标**（MACD/RSI/布林带/成交量异常/波动率等）→ 产出统一结构 `EnhancedNewsFactor`。

3. **信号生成 `signal_generator.py`**  
   - 基于阈值、**事件评分门槛**、**成交量确认**、**新闻热度** 与 **技术确认** 决定 `BUY/SELL/HOLD`，并带 `confidence`。  
   - 内置 **冷却机制**，避免新闻连发导致过度交易。

4. **组合执行 `portfolio_manager.py`**  
   - 根据 `TradeSignal` 动态控制 **仓位**（事件评分越高，仓位可放大），并执行 **移动止损 / 分批止盈 / 日终清仓** 等规则。  
   - 预留 **Alpaca** 实盘接口。

5. **回测引擎 `backtest_engine.py`**  
   - 管理价格/新闻缓存、持仓、权益曲线与全链路事件日志，输出 **绩效指标** + **交易/信号历史**。

6. **主脚本**  
   - `run_strategy_backtest.py`：基础优化版一键回测与可视化。  
   - `enhanced_run_strategy_backtest.py`：FinGPT 增强版（含参数优化、综合图表）。

7. **股票池 `universe_builder.py`**  
   - 维护默认大盘股池；可扩展至自定义列表，支持按 **新闻覆盖/活跃度/波动率** 等筛选。

8. **环境脚本 `setup_environment.py`**  
   - 对 **Python/GPU/磁盘/目录/依赖** 做一次性检查，便于在云端（如 AWS EC2）快速落地。

---

## 📊 输出与指标

- **performance**：`total_return / annual_return / sharpe_ratio / max_drawdown / win_rate`  
- **trade_stats**：`total_trades / winning_trades / avg_win / avg_loss / profit_factor`  
- **time series**：`equity_curve / positions_history / trades_history / signals_history`（JSON/CSV/图）

---

## 💡 自定义与扩展

- **更换/扩充股票池**：编辑 `universe_builder.py` 对应映射或接入自有配置。  
- **接入更多新闻源**：在 `data_collector.py` 添加拉取逻辑；与 `factor_model_optimized.py` 的解析接口对齐即可。  
- **新增技术因子/过滤器**：在因子工程与信号生成模块扩展字段和规则。  
- **切换实盘**：在 `portfolio_manager.py` 中将 `live=False` 切为 `True` 并提供 Alpaca Keys；务必先小额沙盒验证。

---

## ❓ 常见问题（FAQ）

- **Alpaca 提示 401 unauthorized**：检查 `.env` 的 `ALPACA_API_KEY_ID/ALPACA_SECRET_KEY`；确认是否使用 **paper** 还是 **live** 的 Base URL。  
- **显存不足（加载 13B）**：开启 `use_8bit=True`，或换用更小基座模型；尽量使用带 16GB+ 显存的 GPU 实例。  
- **Tiingo/Finnhub 调用受限**：免费计划速率低，建议缓存历史数据，或升级计划。  
- **回测过度交易**：提高 `cooldown`、提高 `min_event_score / min_confidence`、提高 `novel_thresh`，或启用更严格的成交量/技术确认。

---

## 📜 许可与致谢

- 建议开源协议：MIT / Apache-2.0（按需选择并在仓库根目录添加 `LICENSE`）  
- 模型与数据来源：致谢 FinGPT 团队与公开数据源（Tiingo/Finnhub/Alpaca 等）。

---

## 🏁 一键开始（Cheatsheet）

```bash
# 1) 环境
conda create -n llm-quant python=3.10 -y
conda activate llm-quant
pip install -r requirements.txt

# 2) 写入 .env（API Keys）

# 3) 运行：基础优化版
python run_strategy_backtest.py

# 或 运行：FinGPT 增强版
python enhanced_run_strategy_backtest.py

# 4) 查看输出
ls output/ logs/
```
