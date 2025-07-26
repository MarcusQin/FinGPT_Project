"""
run_strategy_backtest.py - LLM-Quant策略回测主脚本（优化版）
针对7只大型科技股进行2024.9.1-2025.7.1的回测
应用了更严格的信号过滤、增强的风控机制
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入回测引擎
from backtest_engine import (
    BacktestConfig, BacktestEngine, BacktestResult, 
    create_backtest_config, run_strategy_backtest
)

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/main_backtest.log', encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StrategyBacktester:
    """策略回测器（优化版）"""
    
    def __init__(self):
        """初始化回测器"""
        # 定义目标股票
        self.target_symbols = [
            "GOOGL",  # Alphabet
            "AMZN",   # Amazon
            "AAPL",   # Apple
            "META",   # Meta Platforms
            "MSFT",   # Microsoft
            "NVDA",   # Nvidia
            "TSLA"    # Tesla
        ]
        
        # 回测时间范围
        self.start_date = "2024-09-01"
        self.end_date = "2025-07-01"
        
        # 策略参数（优化版）
        self.strategy_params = {
            # 资金管理
            'initial_capital': 100000.0,
            'max_position_percent': 0.2,  # 单股最大仓位20%
            'commission': 0.0,  # 假设无手续费
            
            # 信号阈值（更严格）
            'pos_thresh': 0.7,      # 正面情绪阈值（从0.5提高到0.7）
            'neg_thresh': -0.7,     # 负面情绪阈值（从-0.5提高到-0.7）
            'novel_thresh': 0.6,    # 新颖度阈值（从0.5提高到0.6）
            'cooldown': 300,        # 5分钟冷却
            
            # 风控参数（增强版）
            'stop_loss_pct': 0.02,          # 2%止损
            'take_profit_pct': 0.03,        # 3%止盈
            'trailing_stop_pct': 0.02,      # 2%移动止损（新增）
            'max_daily_loss_pct': 0.05,     # 日内最大亏损5%（从0.1降低）
            'allow_overnight': False,        # 不允许隔夜持仓（新增）
            'partial_take_profit': True,     # 启用分批止盈（新增）
            'partial_take_ratio': 0.5,       # 分批止盈比例50%（新增）
            
            # 数据和计算
            'data_frequency': "5Min",   # 5分钟K线
            'use_async': True,          # 使用异步处理
            'batch_size': 16            # 批处理大小
        }
        
        # 输出目录
        self.output_dir = "output/backtest/optimized"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"策略回测器初始化: {len(self.target_symbols)}只股票")
        logger.info("使用优化参数: 更严格的信号阈值、移动止损、分批止盈、日终清仓")
    
    def run_full_backtest(self) -> BacktestResult:
        """运行完整回测"""
        logger.info("="*80)
        logger.info("开始LLM-Quant策略回测（优化版）")
        logger.info(f"时间范围: {self.start_date} 至 {self.end_date}")
        logger.info(f"标的股票: {', '.join(self.target_symbols)}")
        logger.info(f"初始资金: ${self.strategy_params['initial_capital']:,.0f}")
        logger.info("关键优化:")
        logger.info(f"  - 情绪阈值: 正面{self.strategy_params['pos_thresh']}, 负面{self.strategy_params['neg_thresh']}")
        logger.info(f"  - 新颖度阈值: {self.strategy_params['novel_thresh']}")
        logger.info(f"  - 移动止损: {self.strategy_params['trailing_stop_pct']:.1%}")
        logger.info(f"  - 分批止盈: {'启用' if self.strategy_params['partial_take_profit'] else '禁用'}")
        logger.info(f"  - 隔夜持仓: {'允许' if self.strategy_params['allow_overnight'] else '禁止'}")
        logger.info("="*80)
        
        # 创建回测配置
        config = create_backtest_config(
            start_date=self.start_date,
            end_date=self.end_date,
            symbols=self.target_symbols,
            **self.strategy_params
        )
        
        # 运行回测
        logger.info("\n开始运行回测引擎...")
        start_time = datetime.now()
        
        result = run_strategy_backtest(config)
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n回测运行时间: {elapsed_time:.1f}秒")
        
        # 生成详细报告
        self.generate_detailed_report(result)
        
        # 可视化结果
        self.visualize_results(result)
        
        return result
    
    def run_parameter_sensitivity(self):
        """参数敏感性分析"""
        logger.info("\n运行参数敏感性分析...")
        
        # 定义要测试的参数范围
        param_grid = {
            'pos_thresh': [0.6, 0.7, 0.8],
            'neg_thresh': [-0.8, -0.7, -0.6],
            'novel_thresh': [0.5, 0.6, 0.7],
            'trailing_stop_pct': [0.01, 0.02, 0.03],
            'partial_take_ratio': [0.3, 0.5, 0.7]
        }
        
        # 简化测试：只测试部分组合
        test_configs = [
            # 基准配置
            {'pos_thresh': 0.7, 'neg_thresh': -0.7, 'novel_thresh': 0.6, 'trailing_stop_pct': 0.02},
            # 更严格的情绪阈值
            {'pos_thresh': 0.8, 'neg_thresh': -0.8, 'novel_thresh': 0.6, 'trailing_stop_pct': 0.02},
            # 更宽松的移动止损
            {'pos_thresh': 0.7, 'neg_thresh': -0.7, 'novel_thresh': 0.6, 'trailing_stop_pct': 0.03},
            # 更高的新颖度要求
            {'pos_thresh': 0.7, 'neg_thresh': -0.7, 'novel_thresh': 0.7, 'trailing_stop_pct': 0.02},
        ]
        
        results = []
        
        for i, test_params in enumerate(test_configs, 1):
            logger.info(f"\n测试配置 {i}/{len(test_configs)}: {test_params}")
            
            # 更新参数
            params = self.strategy_params.copy()
            params.update(test_params)
            
            # 创建配置
            config = create_backtest_config(
                start_date=self.start_date,
                end_date=self.end_date,
                symbols=self.target_symbols[:3],  # 简化：只用3只股票
                **params
            )
            
            # 运行回测
            engine = BacktestEngine(config)
            result = engine.run_backtest()
            
            # 记录结果
            results.append({
                'params': test_params,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'trades': result.total_trades
            })
        
        # 保存敏感性分析结果
        sensitivity_file = f"{self.output_dir}/parameter_sensitivity.json"
        with open(sensitivity_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n参数敏感性分析完成，结果保存至: {sensitivity_file}")
        
        # 打印最佳参数组合
        best_result = max(results, key=lambda x: x['sharpe_ratio'])
        logger.info(f"\n最佳参数组合（按夏普比率）:")
        logger.info(f"  参数: {best_result['params']}")
        logger.info(f"  夏普比率: {best_result['sharpe_ratio']:.2f}")
        logger.info(f"  总收益: {best_result['total_return']:.2%}")
        logger.info(f"  最大回撤: {best_result['max_drawdown']:.2%}")
        
        return results
    
    def generate_detailed_report(self, result: BacktestResult):
        """生成详细报告"""
        report = {
            'backtest_summary': {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'symbols': self.target_symbols,
                'initial_capital': self.strategy_params['initial_capital'],
                'data_frequency': self.strategy_params['data_frequency'],
                'optimization_features': [
                    '更严格的情绪和新颖度阈值',
                    '移动止损机制',
                    '分批止盈策略',
                    '日终强制清仓',
                    '降低的日内亏损限制'
                ]
            },
            'performance_metrics': result.to_dict()['performance'],
            'trade_statistics': result.to_dict()['trade_stats'],
            'risk_metrics': {
                'max_drawdown': result.max_drawdown,
                'daily_volatility': result.equity_curve.pct_change().std() * np.sqrt(252),
                'downside_deviation': result.equity_curve.pct_change()[
                    result.equity_curve.pct_change() < 0
                ].std() * np.sqrt(252) if len(result.equity_curve) > 1 else 0
            },
            'signal_analysis': self._analyze_signals(result),
            'position_analysis': self._analyze_positions(result),
            'best_worst_trades': self._analyze_trades(result),
            'risk_control_effectiveness': self._analyze_risk_controls(result)
        }
        
        # 保存报告
        report_file = f"{self.output_dir}/detailed_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成可读报告
        readable_report = f"{self.output_dir}/report_summary.txt"
        with open(readable_report, 'w', encoding='utf-8') as f:
            f.write("LLM-Quant策略回测报告（优化版）\n")
            f.write("="*60 + "\n\n")
            
            f.write("1. 回测概要\n")
            f.write(f"   时间范围: {self.start_date} 至 {self.end_date}\n")
            f.write(f"   测试股票: {', '.join(self.target_symbols)}\n")
            f.write(f"   初始资金: ${self.strategy_params['initial_capital']:,.0f}\n")
            f.write(f"   数据频率: {self.strategy_params['data_frequency']}\n\n")
            
            f.write("2. 优化特性\n")
            for feature in report['backtest_summary']['optimization_features']:
                f.write(f"   - {feature}\n")
            f.write("\n")
            
            f.write("3. 绩效指标\n")
            f.write(f"   总收益率: {result.total_return:.2%}\n")
            f.write(f"   年化收益率: {result.annual_return:.2%}\n")
            f.write(f"   夏普比率: {result.sharpe_ratio:.2f}\n")
            f.write(f"   最大回撤: {result.max_drawdown:.2%}\n\n")
            
            f.write("4. 交易统计\n")
            f.write(f"   总交易次数: {result.total_trades}\n")
            f.write(f"   胜率: {result.win_rate:.2%}\n")
            f.write(f"   平均盈利: ${result.avg_win:.2f}\n")
            f.write(f"   平均亏损: ${result.avg_loss:.2f}\n")
            f.write(f"   盈亏比: {result.profit_factor:.2f}\n")
        
        logger.info(f"详细报告已生成: {report_file}")
    
    def _analyze_signals(self, result: BacktestResult) -> Dict:
        """分析信号"""
        signal_counts = defaultdict(int)
        signal_by_ticker = defaultdict(lambda: defaultdict(int))
        confidence_distribution = []
        
        for signal in result.signals_history:
            signal_counts[signal['signal']] += 1
            signal_by_ticker[signal['ticker']][signal['signal']] += 1
            confidence_distribution.append(signal['confidence'])
        
        # 分析高置信度信号
        high_confidence_signals = [s for s in result.signals_history if s['confidence'] > 0.7]
        
        return {
            'total_signals': len(result.signals_history),
            'signal_distribution': dict(signal_counts),
            'signals_by_ticker': dict(signal_by_ticker),
            'confidence_stats': {
                'mean': np.mean(confidence_distribution) if confidence_distribution else 0,
                'std': np.std(confidence_distribution) if confidence_distribution else 0,
                'min': np.min(confidence_distribution) if confidence_distribution else 0,
                'max': np.max(confidence_distribution) if confidence_distribution else 0
            },
            'high_confidence_signals': {
                'count': len(high_confidence_signals),
                'percentage': len(high_confidence_signals) / len(result.signals_history) * 100 if result.signals_history else 0
            }
        }
    
    def _analyze_positions(self, result: BacktestResult) -> Dict:
        """分析持仓"""
        # 从交易历史推断持仓情况
        position_times = defaultdict(list)
        current_positions = {}
        
        for trade in result.trades_history:
            ticker = trade['ticker']
            if trade['action'] in ['BUY', 'SELL_SHORT']:
                # 开仓
                current_positions[ticker] = trade['timestamp']
            elif trade['action'] in ['SELL', 'BUY_TO_COVER']:
                # 平仓
                if ticker in current_positions:
                    holding_time = (
                        datetime.fromisoformat(trade['timestamp']) - 
                        datetime.fromisoformat(current_positions[ticker])
                    ).total_seconds() / 60  # 分钟
                    position_times[ticker].append(holding_time)
                    del current_positions[ticker]
        
        # 计算平均持仓时间
        avg_holding_times = {}
        for ticker, times in position_times.items():
            if times:
                avg_holding_times[ticker] = np.mean(times)
        
        # 统计日内交易比例（持仓时间<390分钟，即6.5小时）
        all_times = [t for times in position_times.values() for t in times]
        intraday_trades = sum(1 for t in all_times if t < 390)
        
        return {
            'avg_holding_time_minutes': avg_holding_times,
            'total_positions_opened': len(result.trades_history) // 2,  # 粗略估计
            'positions_by_ticker': dict(position_times),
            'intraday_trade_ratio': intraday_trades / len(all_times) if all_times else 0
        }
    
    def _analyze_trades(self, result: BacktestResult) -> Dict:
        """分析最佳和最差交易"""
        # 匹配开仓和平仓交易
        trades_by_ticker = defaultdict(list)
        for trade in result.trades_history:
            trades_by_ticker[trade['ticker']].append(trade)
        
        trade_pnls = []
        
        for ticker, trades in trades_by_ticker.items():
            i = 0
            while i < len(trades) - 1:
                if trades[i]['action'] in ['BUY', 'SELL_SHORT']:
                    # 找到对应的平仓交易
                    for j in range(i + 1, len(trades)):
                        if trades[j]['action'] in ['SELL', 'BUY_TO_COVER']:
                            # 计算盈亏
                            if trades[i]['action'] == 'BUY':
                                pnl = (trades[j]['price'] - trades[i]['price']) * trades[i]['quantity']
                            else:  # SELL_SHORT
                                pnl = (trades[i]['price'] - trades[j]['price']) * trades[i]['quantity']
                            
                            trade_pnls.append({
                                'ticker': ticker,
                                'entry': trades[i],
                                'exit': trades[j],
                                'pnl': pnl,
                                'return_pct': pnl / (trades[i]['price'] * trades[i]['quantity'])
                            })
                            i = j
                            break
                i += 1
        
        # 排序找出最佳和最差
        trade_pnls.sort(key=lambda x: x['pnl'], reverse=True)
        
        return {
            'best_trades': trade_pnls[:5] if len(trade_pnls) >= 5 else trade_pnls,
            'worst_trades': trade_pnls[-5:] if len(trade_pnls) >= 5 else trade_pnls,
            'total_round_trips': len(trade_pnls),
            'average_pnl': np.mean([t['pnl'] for t in trade_pnls]) if trade_pnls else 0,
            'pnl_std': np.std([t['pnl'] for t in trade_pnls]) if trade_pnls else 0
        }
    
    def _analyze_risk_controls(self, result: BacktestResult) -> Dict:
        """分析风控机制效果"""
        # 从事件日志中统计各种风控触发次数
        risk_events = {
            'stop_loss': 0,
            'take_profit': 0,
            'trailing_stop': 0,
            'partial_take_profit': 0,
            'daily_loss_limit': 0,
            'end_of_day_close': 0
        }
        
        for event in result.event_log:
            if '[止损]' in event:
                risk_events['stop_loss'] += 1
            elif '[移动止损]' in event:
                risk_events['trailing_stop'] += 1
            elif '[分批止盈]' in event:
                risk_events['partial_take_profit'] += 1
            elif '[日终清仓]' in event:
                risk_events['end_of_day_close'] += 1
            elif '[风控]' in event and '日内亏损' in event:
                risk_events['daily_loss_limit'] += 1
        
        return {
            'risk_control_triggers': risk_events,
            'total_risk_events': sum(risk_events.values()),
            'most_common_trigger': max(risk_events.items(), key=lambda x: x[1])[0] if any(risk_events.values()) else None
        }
    
    def visualize_results(self, result: BacktestResult):
        """可视化回测结果（增强版）"""
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('LLM-Quant策略回测结果（优化版）', fontsize=16)
        
        # 1. 权益曲线
        ax1 = axes[0, 0]
        result.equity_curve.plot(ax=ax1, color='blue', linewidth=2)
        ax1.set_title('权益曲线')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('账户权益 ($)')
        ax1.grid(True, alpha=0.3)
        
        # 添加初始资金线
        ax1.axhline(y=self.strategy_params['initial_capital'], 
                   color='red', linestyle='--', alpha=0.5, label='初始资金')
        ax1.legend()
        
        # 2. 日收益率分布
        ax2 = axes[0, 1]
        daily_returns = result.equity_curve.pct_change().dropna()
        daily_returns.hist(ax=ax2, bins=30, alpha=0.7, color='green')
        ax2.set_title('日收益率分布')
        ax2.set_xlabel('日收益率')
        ax2.set_ylabel('频次')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # 添加统计信息
        ax2.text(0.05, 0.95, f'均值: {daily_returns.mean():.3%}\n标准差: {daily_returns.std():.3%}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. 信号分布
        ax3 = axes[0, 2]
        signal_counts = defaultdict(int)
        for signal in result.signals_history:
            signal_counts[signal['signal']] += 1
        
        if signal_counts:
            labels = list(signal_counts.keys())
            counts = list(signal_counts.values())
            colors = ['green', 'red', 'gray'][:len(labels)]
            ax3.bar(labels, counts, color=colors, alpha=0.7)
            ax3.set_title('交易信号分布')
            ax3.set_xlabel('信号类型')
            ax3.set_ylabel('数量')
            
            # 添加数值标签
            for i, (label, count) in enumerate(zip(labels, counts)):
                ax3.text(i, count + 0.5, str(count), ha='center')
        
        # 4. 累计收益率 vs 回撤
        ax4 = axes[1, 0]
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        drawdown = self._calculate_drawdown_series(result.equity_curve)
        
        ax4_twin = ax4.twinx()
        cumulative_returns.plot(ax=ax4, color='blue', label='累计收益率', linewidth=2)
        (drawdown * 100).plot(ax=ax4_twin, color='red', label='回撤', linewidth=1, alpha=0.7)
        
        ax4.set_title('累计收益率 vs 回撤')
        ax4.set_xlabel('日期')
        ax4.set_ylabel('累计收益率', color='blue')
        ax4_twin.set_ylabel('回撤 (%)', color='red')
        ax4.tick_params(axis='y', labelcolor='blue')
        ax4_twin.tick_params(axis='y', labelcolor='red')
        ax4.grid(True, alpha=0.3)
        
        # 添加图例
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 5. 信号置信度分布
        ax5 = axes[1, 1]
        confidences = [s['confidence'] for s in result.signals_history if s['signal'] != 'HOLD']
        if confidences:
            ax5.hist(confidences, bins=20, alpha=0.7, color='purple')
            ax5.set_title('信号置信度分布')
            ax5.set_xlabel('置信度')
            ax5.set_ylabel('频次')
            ax5.axvline(x=np.mean(confidences), color='red', linestyle='--', 
                       label=f'平均: {np.mean(confidences):.3f}')
            ax5.legend()
        
        # 6. 风控触发统计
        ax6 = axes[1, 2]
        risk_analysis = self._analyze_risk_controls(result)
        risk_triggers = risk_analysis['risk_control_triggers']
        
        if any(risk_triggers.values()):
            trigger_names = list(risk_triggers.keys())
            trigger_counts = list(risk_triggers.values())
            
            # 只显示有触发的风控类型
            active_triggers = [(n, c) for n, c in zip(trigger_names, trigger_counts) if c > 0]
            if active_triggers:
                names, counts = zip(*active_triggers)
                ax6.bar(range(len(names)), counts, alpha=0.7, color='orange')
                ax6.set_xticks(range(len(names)))
                ax6.set_xticklabels(names, rotation=45, ha='right')
                ax6.set_title('风控机制触发次数')
                ax6.set_ylabel('触发次数')
                
                # 添加数值标签
                for i, count in enumerate(counts):
                    ax6.text(i, count + 0.1, str(count), ha='center')
        
        plt.tight_layout()
        
        # 保存图形
        plot_file = f"{self.output_dir}/backtest_results.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"可视化结果已保存: {plot_file}")
        
        # 生成额外的分析图
        self._create_additional_plots(result)
    
    def _calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """计算回撤序列"""
        cumulative = equity_curve / equity_curve.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    def _create_additional_plots(self, result: BacktestResult):
        """创建额外的分析图表"""
        # 1. 按股票的收益贡献
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 计算每只股票的盈亏
        ticker_pnl = defaultdict(float)
        for trade in result.trades_history:
            # 简化计算（实际应该匹配开平仓）
            if trade['action'] in ['SELL', 'BUY_TO_COVER']:
                ticker_pnl[trade['ticker']] += trade.get('pnl', 0)
        
        if ticker_pnl:
            tickers = list(ticker_pnl.keys())
            pnls = list(ticker_pnl.values())
            colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
            
            ax.bar(tickers, pnls, color=colors, alpha=0.7)
            ax.set_title('各股票收益贡献')
            ax.set_xlabel('股票代码')
            ax.set_ylabel('盈亏 ($)')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/ticker_pnl.png", dpi=300)
        
        plt.close()
        
        # 2. 新闻热度与信号成功率关系
        if result.signals_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 将信号按新闻热度分组
            heat_groups = defaultdict(list)
            for signal in result.signals_history:
                heat = signal.get('news_heat', 0)
                if heat < 0.3:
                    heat_level = 'Low'
                elif heat < 0.7:
                    heat_level = 'Medium'
                else:
                    heat_level = 'High'
                heat_groups[heat_level].append(signal)
            
            # 统计每组的信号数量
            heat_levels = ['Low', 'Medium', 'High']
            signal_counts = [len(heat_groups[level]) for level in heat_levels]
            
            if any(signal_counts):
                ax.bar(heat_levels, signal_counts, alpha=0.7, color=['blue', 'orange', 'red'])
                ax.set_title('信号分布 vs 新闻热度')
                ax.set_xlabel('新闻热度等级')
                ax.set_ylabel('信号数量')
                
                for i, count in enumerate(signal_counts):
                    ax.text(i, count + 0.5, str(count), ha='center')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/heat_vs_signals.png", dpi=300)
            plt.close()


def main():
    """主函数"""
    print("\n" + "="*80)
    print("LLM-Quant 美股日内高频交易策略回测（优化版）")
    print("="*80)
    print(f"执行时间: {datetime.now()}")
    print("="*80 + "\n")
    
    # 创建回测器
    backtester = StrategyBacktester()
    
    # 1. 运行完整回测
    print("\n[1] 运行完整策略回测...")
    result = backtester.run_full_backtest()
    
    # 2. 参数敏感性分析（可选）
    print("\n[2] 运行参数敏感性分析...")
    sensitivity_results = backtester.run_parameter_sensitivity()
    
    print("\n" + "="*80)
    print("回测完成!")
    print(f"结果保存在: {backtester.output_dir}")
    print("="*80)
    
    # 返回结果供进一步分析
    return result


if __name__ == "__main__":
    # 运行主程序
    result = main()
    
    # 打印关键结果
    print(f"\n关键绩效指标:")
    print(f"- 总收益率: {result.total_return:.2%}")
    print(f"- 年化收益率: {result.annual_return:.2%}")
    print(f"- 夏普比率: {result.sharpe_ratio:.2f}")
    print(f"- 最大回撤: {result.max_drawdown:.2%}")
    print(f"- 胜率: {result.win_rate:.2%}")
    print(f"- 总交易次数: {result.total_trades}")
    print(f"- 盈亏比: {result.profit_factor:.2f}")
    
    # 打印优化效果提示
    print("\n优化效果预期:")
    print("✓ 更严格的信号过滤应减少噪音交易")
    print("✓ 移动止损应提高盈亏比")
    print("✓ 分批止盈应平滑收益曲线")
    print("✓ 日终清仓应降低隔夜风险")
    print("✓ 更低的日内亏损限制应减少极端回撤")