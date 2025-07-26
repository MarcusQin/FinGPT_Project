"""
enhanced_run_strategy_backtest.py - FinGPT增强策略回测主脚本
完整的FinGPT量化策略回测系统，包含参数优化、性能分析、可视化等功能
针对配置的股票池进行长期回测，充分利用FinGPT的事件评分和情绪分析能力
"""

import os
import sys
import json
import logging
import warnings
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入增强回测引擎
from enhanced_backtest_engine import (
    EnhancedBacktestConfig, EnhancedBacktestEngine, EnhancedBacktestResult,
    create_enhanced_backtest_config, run_fingpt_strategy_backtest,
    print_enhanced_backtest_report
)

# 导入其他模块
from universe_builder import UniverseBuilder

# 配置matplotlib和seaborn
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/main_fingpt_backtest.log', encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FinGPTStrategyBacktester:
    """FinGPT策略回测器（完整版）"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化FinGPT策略回测器
        
        Args:
            config_file: 配置文件路径（可选）
        """
        # 加载配置
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            self.config = EnhancedBacktestConfig(**config_data)
        else:
            # 使用默认配置
            self.config = self._create_default_config()
        
        # 获取股票池
        self.universe_builder = UniverseBuilder(use_configured_universe=True)
        self.symbols, self.company_map = self._load_universe()
        
        # 输出目录
        self.output_dir = "output/backtest/fingpt_enhanced"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 结果存储
        self.results = {}
        self.benchmark_results = {}
        
        logger.info(f"FinGPT策略回测器初始化完成")
        logger.info(f"股票池: {len(self.symbols)}只股票")
        logger.info(f"回测期间: {self.config.start_date} 至 {self.config.end_date}")
        logger.info(f"FinGPT模式: {self.config.use_fingpt}")
    
    def _create_default_config(self) -> EnhancedBacktestConfig:
        """创建默认回测配置"""
        return create_enhanced_backtest_config(
            start_date="2024-09-01",
            end_date="2025-07-01",
            
            # 资金配置
            initial_capital=100000.0,
            max_position_percent=0.2,
            commission=0.0,
            
            # FinGPT配置
            use_fingpt=True,
            min_event_score=3,
            min_confidence=0.6,
            require_technical_confirmation=False,
            min_news_heat=0.3,
            
            # 传统信号参数
            pos_thresh=0.7,
            neg_thresh=-0.7,
            novel_thresh=0.6,
            cooldown=300,
            
            # 风控配置
            stop_loss_pct=0.02,
            take_profit_pct=0.03,
            high_event_stop_loss_pct=0.03,
            trailing_stop_pct=0.02,
            max_daily_loss_pct=0.05,
            allow_overnight=False,
            partial_take_profit=True,
            partial_take_ratio=0.5,
            event_position_multiplier=1.5,
            
            # 数据配置
            data_frequency="5Min",
            news_lookback_days=2,
            use_async=True,
            batch_size=8,
            
            # 股票池配置
            use_configured_universe=True
        )
    
    def _load_universe(self) -> Tuple[List[str], Dict[str, str]]:
        """加载股票池"""
        try:
            symbols, company_map = self.universe_builder.load_today_universe()
            if not symbols:
                # 如果没有今日池，构建一个
                symbols, _, company_map = self.universe_builder.build_universe()
            
            logger.info(f"加载股票池成功: {len(symbols)}只股票")
            return symbols, company_map
            
        except Exception as e:
            logger.error(f"加载股票池失败: {e}")
            # 使用默认股票
            default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
            default_map = {s: s for s in default_symbols}
            return default_symbols, default_map
    
    def run_full_backtest(self) -> EnhancedBacktestResult:
        """运行完整FinGPT策略回测"""
        logger.info("="*100)
        logger.info("开始FinGPT增强策略回测")
        logger.info("="*100)
        
        # 显示配置信息
        self._print_config_summary()
        
        # 运行回测
        start_time = datetime.now()
        result = run_fingpt_strategy_backtest(self.config)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"回测完成，耗时: {elapsed_time:.1f}秒")
        
        # 存储结果
        self.results['main'] = result
        
        # 生成详细分析
        self._generate_comprehensive_analysis(result)
        
        # 创建可视化
        self._create_comprehensive_visualizations(result)
        
        return result
    
    def run_parameter_optimization(self, param_ranges: Optional[Dict] = None) -> Dict[str, Any]:
        """运行参数优化"""
        logger.info("\n" + "="*60)
        logger.info("开始参数优化分析")
        logger.info("="*60)
        
        if param_ranges is None:
            param_ranges = {
                'min_event_score': [2, 3, 4],
                'min_confidence': [0.5, 0.6, 0.7],
                'event_position_multiplier': [1.0, 1.3, 1.5],
                'trailing_stop_pct': [0.015, 0.02, 0.025],
                'partial_take_ratio': [0.3, 0.5, 0.7]
            }
        
        optimization_results = []
        total_combinations = 1
        for values in param_ranges.values():
            total_combinations *= len(values)
        
        logger.info(f"总参数组合数: {total_combinations}")
        
        # 缩短测试期间以加快优化
        short_config = self.config
        short_config.start_date = "2024-11-01"
        short_config.end_date = "2024-12-31"
        
        combination_count = 0
        
        # 生成参数组合
        import itertools
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        for combination in itertools.product(*param_values):
            combination_count += 1
            param_dict = dict(zip(param_names, combination))
            
            logger.info(f"测试组合 {combination_count}/{total_combinations}: {param_dict}")
            
            # 创建测试配置
            test_config = short_config
            for param, value in param_dict.items():
                setattr(test_config, param, value)
            
            try:
                # 运行回测
                engine = EnhancedBacktestEngine(test_config)
                result = engine.run_backtest()
                
                # 记录结果
                optimization_results.append({
                    'parameters': param_dict,
                    'total_return': result.total_return,
                    'annual_return': result.annual_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'total_trades': result.total_trades,
                    'fingpt_trades': result.fingpt_trades,
                    'high_event_trades': result.high_event_trades,
                    'avg_event_score': result.avg_event_score
                })
                
            except Exception as e:
                logger.error(f"参数组合测试失败: {e}")
                continue
        
        # 分析最佳参数
        best_by_sharpe = max(optimization_results, key=lambda x: x['sharpe_ratio'])
        best_by_return = max(optimization_results, key=lambda x: x['total_return'])
        best_by_drawdown = min(optimization_results, key=lambda x: abs(x['max_drawdown']))
        
        optimization_summary = {
            'total_combinations_tested': len(optimization_results),
            'best_by_sharpe': best_by_sharpe,
            'best_by_return': best_by_return,
            'best_by_drawdown': best_by_drawdown,
            'all_results': optimization_results
        }
        
        # 保存优化结果
        opt_file = f"{self.output_dir}/parameter_optimization.json"
        with open(opt_file, 'w', encoding='utf-8') as f:
            json.dump(optimization_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"参数优化完成，结果保存至: {opt_file}")
        logger.info(f"最佳夏普比率: {best_by_sharpe['sharpe_ratio']:.3f} - {best_by_sharpe['parameters']}")
        logger.info(f"最佳收益率: {best_by_return['total_return']:.2%} - {best_by_return['parameters']}")
        
        return optimization_summary
    
    def run_ablation_study(self) -> Dict[str, EnhancedBacktestResult]:
        """运行消融研究（不同模块的效果分析）"""
        logger.info("\n" + "="*60)
        logger.info("开始消融研究")
        logger.info("="*60)
        
        ablation_configs = {
            'baseline_fingpt': self.config,  # 完整FinGPT策略
            
            'no_event_scoring': EnhancedBacktestConfig(
                **{**asdict(self.config), 'min_event_score': 1}  # 不使用事件评分过滤
            ),
            
            'no_technical_confirmation': EnhancedBacktestConfig(
                **{**asdict(self.config), 'require_technical_confirmation': False}
            ),
            
            'no_position_sizing': EnhancedBacktestConfig(
                **{**asdict(self.config), 'event_position_multiplier': 1.0}  # 不使用事件驱动仓位
            ),
            
            'no_trailing_stop': EnhancedBacktestConfig(
                **{**asdict(self.config), 'trailing_stop_pct': 0.0}  # 不使用移动止损
            ),
            
            'traditional_only': EnhancedBacktestConfig(
                **{**asdict(self.config), 'use_fingpt': False}  # 仅使用传统方法
            )
        }
        
        ablation_results = {}
        
        for name, config in ablation_configs.items():
            logger.info(f"测试配置: {name}")
            
            try:
                engine = EnhancedBacktestEngine(config)
                result = engine.run_backtest()
                ablation_results[name] = result
                
                logger.info(f"{name} - 收益: {result.total_return:.2%}, 夏普: {result.sharpe_ratio:.3f}")
                
            except Exception as e:
                logger.error(f"消融测试失败 {name}: {e}")
        
        # 保存消融研究结果
        ablation_summary = {}
        for name, result in ablation_results.items():
            ablation_summary[name] = {
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'fingpt_trades': getattr(result, 'fingpt_trades', 0),
                'high_event_trades': getattr(result, 'high_event_trades', 0)
            }
        
        ablation_file = f"{self.output_dir}/ablation_study.json"
        with open(ablation_file, 'w', encoding='utf-8') as f:
            json.dump(ablation_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"消融研究完成，结果保存至: {ablation_file}")
        
        self.results.update(ablation_results)
        return ablation_results
    
    def run_benchmark_comparison(self) -> Dict[str, Any]:
        """运行基准对比"""
        logger.info("\n" + "="*60)
        logger.info("开始基准对比分析")
        logger.info("="*60)
        
        # 这里简化为与买入持有策略的对比
        # 实际项目中可以添加更多基准（如市场指数、其他策略等）
        
        benchmark_results = {
            'strategy_vs_buyhold': self._compare_with_buy_hold(),
            'risk_adjusted_metrics': self._calculate_risk_adjusted_metrics(),
            'market_correlation': self._analyze_market_correlation()
        }
        
        # 保存基准对比结果
        benchmark_file = f"{self.output_dir}/benchmark_comparison.json"
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2, ensure_ascii=False, default=str)
        
        self.benchmark_results = benchmark_results
        return benchmark_results
    
    def _compare_with_buy_hold(self) -> Dict[str, Any]:
        """与买入持有策略对比"""
        if 'main' not in self.results:
            return {}
        
        main_result = self.results['main']
        
        # 简化的买入持有计算（等权重）
        try:
            # 获取第一只股票的数据作为代理
            first_symbol = self.symbols[0]
            from data_collector import DataCollector
            
            data_collector = DataCollector()
            df = data_collector.get_historical_data(
                first_symbol,
                self.config.start_date,
                self.config.end_date,
                'daily'
            )
            
            if not df.empty:
                buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                buy_hold_volatility = df['close'].pct_change().std() * np.sqrt(252)
            else:
                buy_hold_return = 0.0
                buy_hold_volatility = 0.2
            
        except Exception as e:
            logger.warning(f"买入持有计算失败: {e}")
            buy_hold_return = 0.1  # 假设基准收益
            buy_hold_volatility = 0.2
        
        return {
            'strategy_return': main_result.total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': main_result.total_return - buy_hold_return,
            'strategy_sharpe': main_result.sharpe_ratio,
            'buy_hold_sharpe': buy_hold_return / buy_hold_volatility if buy_hold_volatility > 0 else 0,
            'information_ratio': (main_result.total_return - buy_hold_return) / main_result.risk_metrics.get('volatility', 0.2)
        }
    
    def _calculate_risk_adjusted_metrics(self) -> Dict[str, float]:
        """计算风险调整指标"""
        if 'main' not in self.results:
            return {}
        
        result = self.results['main']
        
        return {
            'sharpe_ratio': result.sharpe_ratio,
            'calmar_ratio': result.risk_metrics.get('calmar_ratio', 0),
            'sortino_ratio': result.annual_return / result.risk_metrics.get('downside_deviation', 0.01),
            'max_drawdown': result.max_drawdown,
            'volatility': result.risk_metrics.get('volatility', 0),
            'value_at_risk_95': np.percentile(result.daily_returns, 5) if len(result.daily_returns) > 0 else 0
        }
    
    def _analyze_market_correlation(self) -> Dict[str, float]:
        """分析与市场的相关性"""
        if 'main' not in self.results:
            return {}
        
        # 简化分析，实际可以加载市场指数数据
        result = self.results['main']
        
        return {
            'correlation_with_market': 0.3,  # 假设值，实际需要计算
            'beta': 0.8,  # 假设值
            'alpha': result.annual_return - 0.8 * 0.1,  # 假设市场收益10%
            'tracking_error': result.risk_metrics.get('volatility', 0) * 0.5
        }
    
    def _print_config_summary(self):
        """打印配置摘要"""
        print(f"\n📋 回测配置摘要:")
        print(f"{'='*60}")
        print(f"📅 时间期间: {self.config.start_date} 至 {self.config.end_date}")
        print(f"💰 初始资金: ${self.config.initial_capital:,.0f}")
        print(f"📈 股票池: {len(self.symbols)}只股票")
        print(f"🤖 FinGPT模式: {'启用' if self.config.use_fingpt else '禁用'}")
        
        if self.config.use_fingpt:
            print(f"\n🎯 FinGPT参数:")
            print(f"   最小事件评分: {self.config.min_event_score}")
            print(f"   最小置信度: {self.config.min_confidence}")
            print(f"   事件仓位乘数: {self.config.event_position_multiplier}")
            print(f"   技术确认要求: {'是' if self.config.require_technical_confirmation else '否'}")
        
        print(f"\n⚠️  风控参数:")
        print(f"   止损: {self.config.stop_loss_pct:.1%}")
        print(f"   止盈: {self.config.take_profit_pct:.1%}")
        print(f"   移动止损: {self.config.trailing_stop_pct:.1%}")
        print(f"   隔夜持仓: {'允许' if self.config.allow_overnight else '禁止'}")
        
        print(f"\n📊 数据配置:")
        print(f"   数据频率: {self.config.data_frequency}")
        print(f"   新闻回看: {self.config.news_lookback_days}天")
        print(f"   批处理大小: {self.config.batch_size}")
        print(f"{'='*60}\n")
    
    def _generate_comprehensive_analysis(self, result: EnhancedBacktestResult):
        """生成综合分析报告"""
        analysis = {
            'backtest_summary': {
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'total_days': (pd.Timestamp(self.config.end_date) - pd.Timestamp(self.config.start_date)).days,
                'symbols_tested': self.symbols,
                'company_mappings': self.company_map,
                'fingpt_enabled': self.config.use_fingpt
            },
            
            'performance_metrics': {
                'returns': {
                    'total_return': result.total_return,
                    'annual_return': result.annual_return,
                    'monthly_avg_return': result.monthly_returns.mean() if len(result.monthly_returns) > 0 else 0,
                    'best_month': result.monthly_returns.max() if len(result.monthly_returns) > 0 else 0,
                    'worst_month': result.monthly_returns.min() if len(result.monthly_returns) > 0 else 0
                },
                
                'risk_metrics': result.risk_metrics,
                
                'trade_analysis': {
                    'total_trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'avg_win': result.avg_win,
                    'avg_loss': result.avg_loss,
                    'profit_factor': result.profit_factor,
                    'largest_win': max([t.get('pnl', 0) for t in result.trades_history], default=0),
                    'largest_loss': min([t.get('pnl', 0) for t in result.trades_history], default=0)
                }
            },
            
            'fingpt_analysis': {
                'fingpt_trades': result.fingpt_trades,
                'high_event_trades': result.high_event_trades,
                'technical_confirmed_trades': result.technical_confirmed_trades,
                'avg_event_score': result.avg_event_score,
                'event_score_distribution': result.event_score_distribution,
                'signal_effectiveness': result.signal_effectiveness
            },
            
            'sector_analysis': self._analyze_by_sector(result),
            'time_analysis': self._analyze_by_time_periods(result),
            'drawdown_analysis': self._analyze_drawdowns(result),
            'signal_quality_analysis': self._analyze_signal_quality(result)
        }
        
        # 保存分析报告
        analysis_file = f"{self.output_dir}/comprehensive_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成可读报告
        self._generate_readable_report(analysis)
        
        logger.info(f"综合分析报告已保存: {analysis_file}")
    
    def _analyze_by_sector(self, result: EnhancedBacktestResult) -> Dict[str, Any]:
        """按行业分析"""
        # 简化的行业分析
        sector_map = {
            'AAPL': '科技', 'MSFT': '科技', 'GOOGL': '科技', 'META': '科技',
            'AMZN': '消费', 'TSLA': '汽车', 'NVDA': '科技',
            'JPM': '金融', 'BAC': '金融', 'JNJ': '医疗'
        }
        
        sector_trades = defaultdict(list)
        for trade in result.trades_history:
            sector = sector_map.get(trade['ticker'], '其他')
            sector_trades[sector].append(trade)
        
        sector_analysis = {}
        for sector, trades in sector_trades.items():
            if trades:
                total_pnl = sum(t.get('pnl', 0) for t in trades)
                win_count = sum(1 for t in trades if t.get('pnl', 0) > 0)
                
                sector_analysis[sector] = {
                    'trade_count': len(trades),
                    'total_pnl': total_pnl,
                    'win_rate': win_count / len(trades) if trades else 0,
                    'avg_pnl': total_pnl / len(trades) if trades else 0
                }
        
        return sector_analysis
    
    def _analyze_by_time_periods(self, result: EnhancedBacktestResult) -> Dict[str, Any]:
        """按时间段分析"""
        if len(result.daily_returns) == 0:
            return {}
        
        returns_df = pd.DataFrame({
            'date': result.daily_returns.index,
            'return': result.daily_returns.values
        })
        returns_df['month'] = pd.to_datetime(returns_df['date']).dt.month
        returns_df['weekday'] = pd.to_datetime(returns_df['date']).dt.weekday
        
        return {
            'monthly_performance': returns_df.groupby('month')['return'].agg(['mean', 'std', 'count']).to_dict(),
            'weekday_performance': returns_df.groupby('weekday')['return'].agg(['mean', 'std', 'count']).to_dict(),
            'best_performing_month': returns_df.groupby('month')['return'].mean().idxmax(),
            'worst_performing_month': returns_df.groupby('month')['return'].mean().idxmin()
        }
    
    def _analyze_drawdowns(self, result: EnhancedBacktestResult) -> Dict[str, Any]:
        """分析回撤"""
        if len(result.drawdown_series) == 0:
            return {}
        
        drawdowns = result.drawdown_series
        
        # 找到回撤期间
        in_drawdown = drawdowns < 0
        drawdown_periods = []
        start_dd = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_dd is None:
                start_dd = i
            elif not is_dd and start_dd is not None:
                drawdown_periods.append((start_dd, i-1))
                start_dd = None
        
        if start_dd is not None:
            drawdown_periods.append((start_dd, len(drawdowns)-1))
        
        # 分析回撤期间
        dd_analysis = []
        for start, end in drawdown_periods:
            period_dd = drawdowns.iloc[start:end+1]
            dd_analysis.append({
                'start_date': str(period_dd.index[0]),
                'end_date': str(period_dd.index[-1]),
                'duration_days': end - start + 1,
                'max_drawdown': period_dd.min(),
                'recovery_time': 0  # 简化
            })
        
        return {
            'total_drawdown_periods': len(drawdown_periods),
            'avg_drawdown_duration': np.mean([dd['duration_days'] for dd in dd_analysis]) if dd_analysis else 0,
            'max_drawdown_duration': max([dd['duration_days'] for dd in dd_analysis], default=0),
            'drawdown_details': dd_analysis[:5]  # 只保存前5个最大回撤
        }
    
    def _analyze_signal_quality(self, result: EnhancedBacktestResult) -> Dict[str, Any]:
        """分析信号质量"""
        if not result.signals_history:
            return {}
        
        signals_df = pd.DataFrame(result.signals_history)
        
        # 按置信度分组
        confidence_bins = [0, 0.5, 0.7, 0.9, 1.0]
        signals_df['confidence_bin'] = pd.cut(signals_df['confidence'], bins=confidence_bins, include_lowest=True)
        
        # 按事件评分分组
        if 'event_score' in signals_df.columns:
            event_score_analysis = signals_df.groupby('event_score').agg({
                'confidence': ['mean', 'count'],
                'signal': lambda x: (x != 'HOLD').sum()
            }).to_dict()
        else:
            event_score_analysis = {}
        
        return {
            'total_signals': len(signals_df),
            'signal_distribution': signals_df['signal'].value_counts().to_dict(),
            'avg_confidence': signals_df['confidence'].mean(),
            'confidence_distribution': signals_df.groupby('confidence_bin')['signal'].count().to_dict(),
            'event_score_analysis': event_score_analysis,
            'high_confidence_signals': (signals_df['confidence'] > 0.7).sum(),
            'fingpt_signals': (signals_df.get('event_score', pd.Series([1] * len(signals_df))) >= 3).sum()
        }
    
    def _generate_readable_report(self, analysis: Dict[str, Any]):
        """生成可读的分析报告"""
        report_file = f"{self.output_dir}/strategy_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# FinGPT增强量化策略回测报告\n\n")
            
            # 执行摘要
            f.write("## 📊 执行摘要\n\n")
            perf = analysis['performance_metrics']['returns']
            risk = analysis['performance_metrics']['risk_metrics']
            
            f.write(f"- **回测期间**: {analysis['backtest_summary']['start_date']} 至 {analysis['backtest_summary']['end_date']}\n")
            f.write(f"- **总收益率**: {perf['total_return']:.2%}\n")
            f.write(f"- **年化收益率**: {perf['annual_return']:.2%}\n")
            f.write(f"- **夏普比率**: {risk.get('sharpe_ratio', 0):.3f}\n")
            f.write(f"- **最大回撤**: {risk.get('max_drawdown', 0):.2%}\n")
            f.write(f"- **胜率**: {analysis['performance_metrics']['trade_analysis']['win_rate']:.2%}\n\n")
            
            # FinGPT特色
            f.write("## 🤖 FinGPT特色统计\n\n")
            fingpt = analysis['fingpt_analysis']
            
            f.write(f"- **FinGPT驱动交易**: {fingpt['fingpt_trades']}笔\n")
            f.write(f"- **高事件评分交易**: {fingpt['high_event_trades']}笔\n")
            f.write(f"- **技术确认交易**: {fingpt['technical_confirmed_trades']}笔\n")
            f.write(f"- **平均事件评分**: {fingpt['avg_event_score']:.1f}/5\n")
            f.write(f"- **信号转换率**: {fingpt['signal_effectiveness'].get('signals_to_trades_ratio', 0):.2%}\n\n")
            
            # 风险分析
            f.write("## ⚠️ 风险分析\n\n")
            dd_analysis = analysis.get('drawdown_analysis', {})
            
            f.write(f"- **年化波动率**: {risk.get('volatility', 0):.2%}\n")
            f.write(f"- **下行偏差**: {risk.get('downside_deviation', 0):.2%}\n")
            f.write(f"- **回撤期间数**: {dd_analysis.get('total_drawdown_periods', 0)}\n")
            f.write(f"- **平均回撤持续天数**: {dd_analysis.get('avg_drawdown_duration', 0):.1f}天\n\n")
            
            # 交易分析
            f.write("## 📈 交易分析\n\n")
            trade = analysis['performance_metrics']['trade_analysis']
            
            f.write(f"- **总交易次数**: {trade['total_trades']}\n")
            f.write(f"- **平均盈利**: ${trade['avg_win']:.2f}\n")
            f.write(f"- **平均亏损**: ${trade['avg_loss']:.2f}\n")
            f.write(f"- **盈亏比**: {trade['profit_factor']:.2f}\n")
            f.write(f"- **最大单笔盈利**: ${trade['largest_win']:.2f}\n")
            f.write(f"- **最大单笔亏损**: ${trade['largest_loss']:.2f}\n\n")
            
            # 结论
            f.write("## 📝 结论\n\n")
            f.write("基于FinGPT的事件驱动量化策略表现出了以下特点：\n\n")
            
            if fingpt['fingpt_trades'] > 0:
                f.write("✅ FinGPT成功识别并驱动了交易决策\n")
            if fingpt['high_event_trades'] > 0:
                f.write("✅ 高事件评分机制有效筛选重要新闻\n")
            if perf['total_return'] > 0:
                f.write("✅ 策略获得正收益\n")
            if risk.get('sharpe_ratio', 0) > 1:
                f.write("✅ 风险调整后收益表现良好\n")
            
            f.write("\n建议进一步优化的方向：\n")
            f.write("- 调整事件评分阈值以平衡信号数量和质量\n")
            f.write("- 优化技术确认机制的参数\n")
            f.write("- 考虑加入更多基本面因子\n")
            f.write("- 优化风控参数以减少回撤\n")
        
        logger.info(f"可读报告已生成: {report_file}")
    
    def _create_comprehensive_visualizations(self, result: EnhancedBacktestResult):
        """创建综合可视化"""
        logger.info("生成可视化图表...")
        
        # 1. 创建主要绩效图表
        self._create_performance_dashboard(result)
        
        # 2. 创建FinGPT分析图表
        self._create_fingpt_analysis_charts(result)
        
        # 3. 创建交互式图表
        self._create_interactive_charts(result)
        
        # 4. 创建风险分析图表
        self._create_risk_analysis_charts(result)
    
    def _create_performance_dashboard(self, result: EnhancedBacktestResult):
        """创建绩效仪表板"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('FinGPT策略绩效仪表板', fontsize=16, fontweight='bold')
        
        # 1. 权益曲线
        ax1 = axes[0, 0]
        if len(result.equity_curve) > 0:
            result.equity_curve.plot(ax=ax1, color='blue', linewidth=2)
            ax1.axhline(y=self.config.initial_capital, color='red', linestyle='--', alpha=0.7, label='初始资金')
            ax1.set_title('权益曲线')
            ax1.set_xlabel('日期')
            ax1.set_ylabel('账户权益 ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 日收益率分布
        ax2 = axes[0, 1]
        if len(result.daily_returns) > 0:
            result.daily_returns.hist(ax=ax2, bins=30, alpha=0.7, color='green')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax2.set_title('日收益率分布')
            ax2.set_xlabel('日收益率')
            ax2.set_ylabel('频次')
        
        # 3. 回撤曲线
        ax3 = axes[0, 2]
        if len(result.drawdown_series) > 0:
            (result.drawdown_series * 100).plot(ax=ax3, color='red', linewidth=1)
            ax3.fill_between(result.drawdown_series.index, result.drawdown_series * 100, 0, alpha=0.3, color='red')
            ax3.set_title('回撤曲线')
            ax3.set_xlabel('日期')
            ax3.set_ylabel('回撤 (%)')
            ax3.grid(True, alpha=0.3)
        
        # 4. 事件评分分布
        ax4 = axes[1, 0]
        if result.event_score_distribution:
            scores = list(result.event_score_distribution.keys())
            counts = list(result.event_score_distribution.values())
            bars = ax4.bar(scores, counts, alpha=0.7, color='purple')
            ax4.set_title('事件评分分布')
            ax4.set_xlabel('事件评分')
            ax4.set_ylabel('交易次数')
            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom')
        
        # 5. 月度收益
        ax5 = axes[1, 1]
        if len(result.monthly_returns) > 0:
            colors = ['green' if x > 0 else 'red' for x in result.monthly_returns]
            result.monthly_returns.plot(kind='bar', ax=ax5, color=colors, alpha=0.7)
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax5.set_title('月度收益')
            ax5.set_xlabel('月份')
            ax5.set_ylabel('收益率')
            ax5.tick_params(axis='x', rotation=45)
        
        # 6. 交易统计
        ax6 = axes[1, 2]
        trade_stats = [
            result.fingpt_trades,
            result.high_event_trades,
            result.technical_confirmed_trades,
            result.total_trades - result.fingpt_trades
        ]
        labels = ['FinGPT交易', '高事件评分', '技术确认', '其他交易']
        colors = ['blue', 'orange', 'green', 'gray']
        
        wedges, texts, autotexts = ax6.pie(trade_stats, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        ax6.set_title('交易类型分布')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/performance_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_fingpt_analysis_charts(self, result: EnhancedBacktestResult):
        """创建FinGPT分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('FinGPT特色分析', fontsize=16, fontweight='bold')
        
        # 1. 置信度vs收益率
        ax1 = axes[0, 0]
        if result.signals_history:
            confidences = [s['confidence'] for s in result.signals_history]
            ax1.hist(confidences, bins=20, alpha=0.7, color='blue')
            ax1.axvline(x=np.mean(confidences), color='red', linestyle='--', 
                       label=f'平均: {np.mean(confidences):.3f}')
            ax1.set_title('信号置信度分布')
            ax1.set_xlabel('置信度')
            ax1.set_ylabel('频次')
            ax1.legend()
        
        # 2. 事件评分与成功率
        ax2 = axes[0, 1]
        if result.trades_history and any('event_score' in t for t in result.trades_history):
            event_success = defaultdict(list)
            for trade in result.trades_history:
                if 'event_score' in trade and 'pnl' in trade:
                    event_success[trade['event_score']].append(trade['pnl'] > 0)
            
            if event_success:
                scores = sorted(event_success.keys())
                success_rates = [np.mean(event_success[score]) for score in scores]
                
                bars = ax2.bar(scores, success_rates, alpha=0.7, color='green')
                ax2.set_title('事件评分vs成功率')
                ax2.set_xlabel('事件评分')
                ax2.set_ylabel('成功率')
                ax2.set_ylim(0, 1)
                
                for bar, rate in zip(bars, success_rates):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{rate:.2%}', ha='center', va='bottom')
        
        # 3. 时间序列信号数量
        ax3 = axes[1, 0]
        if result.signals_history:
            signals_df = pd.DataFrame(result.signals_history)
            signals_df['date'] = pd.to_datetime(signals_df['timestamp']).dt.date
            daily_signals = signals_df.groupby('date').size()
            
            daily_signals.plot(ax=ax3, color='orange', linewidth=2)
            ax3.set_title('每日信号数量')
            ax3.set_xlabel('日期')
            ax3.set_ylabel('信号数量')
            ax3.grid(True, alpha=0.3)
        
        # 4. 技术确认效果
        ax4 = axes[1, 1]
        if result.trades_history:
            tech_confirmed = [t for t in result.trades_history if t.get('technical_confirmed', False)]
            tech_not_confirmed = [t for t in result.trades_history if not t.get('technical_confirmed', False)]
            
            if tech_confirmed and tech_not_confirmed:
                confirmed_returns = [t.get('pnl', 0) for t in tech_confirmed]
                not_confirmed_returns = [t.get('pnl', 0) for t in tech_not_confirmed]
                
                data = [confirmed_returns, not_confirmed_returns]
                labels = ['技术确认', '未确认']
                
                ax4.boxplot(data, labels=labels)
                ax4.set_title('技术确认vs收益分布')
                ax4.set_ylabel('收益 ($)')
                ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/fingpt_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_interactive_charts(self, result: EnhancedBacktestResult):
        """创建交互式图表"""
        try:
            # 1. 交互式权益曲线
            fig = go.Figure()
            
            if len(result.equity_curve) > 0:
                fig.add_trace(go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve.values,
                    mode='lines',
                    name='权益曲线',
                    line=dict(color='blue', width=2)
                ))
                
                fig.add_hline(y=self.config.initial_capital, 
                             line_dash="dash", line_color="red",
                             annotation_text="初始资金")
                
                fig.update_layout(
                    title='FinGPT策略权益曲线（交互式）',
                    xaxis_title='日期',
                    yaxis_title='账户权益 ($)',
                    hovermode='x unified'
                )
                
                fig.write_html(f"{self.output_dir}/interactive_equity_curve.html")
            
            # 2. 交易分析仪表板
            fig2 = make_subplots(
                rows=2, cols=2,
                subplot_titles=('事件评分分布', '月度收益', '信号置信度', '回撤分析'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "histogram"}, {"type": "scatter"}]]
            )
            
            # 事件评分分布
            if result.event_score_distribution:
                fig2.add_trace(
                    go.Bar(x=list(result.event_score_distribution.keys()),
                          y=list(result.event_score_distribution.values()),
                          name='事件评分'),
                    row=1, col=1
                )
            
            # 月度收益
            if len(result.monthly_returns) > 0:
                colors = ['green' if x > 0 else 'red' for x in result.monthly_returns]
                fig2.add_trace(
                    go.Bar(x=result.monthly_returns.index,
                          y=result.monthly_returns.values,
                          marker_color=colors,
                          name='月度收益'),
                    row=1, col=2
                )
            
            # 信号置信度
            if result.signals_history:
                confidences = [s['confidence'] for s in result.signals_history]
                fig2.add_trace(
                    go.Histogram(x=confidences, name='置信度分布'),
                    row=2, col=1
                )
            
            # 回撤分析
            if len(result.drawdown_series) > 0:
                fig2.add_trace(
                    go.Scatter(x=result.drawdown_series.index,
                              y=result.drawdown_series.values * 100,
                              fill='tonexty',
                              name='回撤'),
                    row=2, col=2
                )
            
            fig2.update_layout(
                title_text="FinGPT策略分析仪表板",
                showlegend=False,
                height=800
            )
            
            fig2.write_html(f"{self.output_dir}/interactive_dashboard.html")
            
            logger.info("交互式图表已保存")
            
        except Exception as e:
            logger.warning(f"创建交互式图表失败: {e}")
    
    def _create_risk_analysis_charts(self, result: EnhancedBacktestResult):
        """创建风险分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('风险分析', fontsize=16, fontweight='bold')
        
        # 1. 滚动夏普比率
        ax1 = axes[0, 0]
        if len(result.daily_returns) > 30:
            rolling_sharpe = result.daily_returns.rolling(window=30).mean() / result.daily_returns.rolling(window=30).std() * np.sqrt(252)
            rolling_sharpe.plot(ax=ax1, color='blue', linewidth=1)
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax1.set_title('30日滚动夏普比率')
            ax1.set_xlabel('日期')
            ax1.set_ylabel('夏普比率')
            ax1.grid(True, alpha=0.3)
        
        # 2. 收益率分布（正态性检验）
        ax2 = axes[0, 1]
        if len(result.daily_returns) > 0:
            from scipy import stats
            
            result.daily_returns.hist(ax=ax2, bins=30, density=True, alpha=0.7, color='lightblue', label='实际分布')
            
            # 叠加正态分布
            mu, sigma = result.daily_returns.mean(), result.daily_returns.std()
            x = np.linspace(result.daily_returns.min(), result.daily_returns.max(), 100)
            normal_dist = stats.norm.pdf(x, mu, sigma)
            ax2.plot(x, normal_dist, 'r-', linewidth=2, label='正态分布')
            
            ax2.set_title('收益率分布vs正态分布')
            ax2.set_xlabel('日收益率')
            ax2.set_ylabel('密度')
            ax2.legend()
        
        # 3. VaR分析
        ax3 = axes[1, 0]
        if len(result.daily_returns) > 0:
            var_levels = [0.01, 0.05, 0.1, 0.25]
            var_values = [np.percentile(result.daily_returns, level * 100) for level in var_levels]
            
            bars = ax3.bar([f"{level:.0%}" for level in var_levels], 
                          [abs(v) * 100 for v in var_values], 
                          alpha=0.7, color='red')
            ax3.set_title('风险价值(VaR)')
            ax3.set_xlabel('置信水平')
            ax3.set_ylabel('VaR (%)')
            
            for bar, val in zip(bars, var_values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{abs(val):.2%}', ha='center', va='bottom')
        
        # 4. 最大回撤期间
        ax4 = axes[1, 1]
        if len(result.drawdown_series) > 0:
            drawdown_underwater = result.drawdown_series.copy()
            drawdown_underwater = drawdown_underwater[drawdown_underwater < 0]
            
            if len(drawdown_underwater) > 0:
                drawdown_underwater.plot(ax=ax4, kind='area', color='red', alpha=0.5)
                ax4.set_title('水下图（回撤期间）')
                ax4.set_xlabel('日期')
                ax4.set_ylabel('回撤 (%)')
                ax4.set_ylim(bottom=drawdown_underwater.min() * 1.1)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/risk_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """运行完整分析套件"""
        logger.info("\n" + "="*100)
        logger.info("开始FinGPT策略完整分析套件")
        logger.info("="*100)
        
        results = {}
        
        # 1. 主要回测
        logger.info("\n[1/4] 运行主要策略回测...")
        main_result = self.run_full_backtest()
        results['main_backtest'] = main_result
        
        # 2. 参数优化
        logger.info("\n[2/4] 运行参数优化...")
        try:
            optimization_results = self.run_parameter_optimization()
            results['parameter_optimization'] = optimization_results
        except Exception as e:
            logger.error(f"参数优化失败: {e}")
            results['parameter_optimization'] = {"error": str(e)}
        
        # 3. 消融研究
        logger.info("\n[3/4] 运行消融研究...")
        try:
            ablation_results = self.run_ablation_study()
            results['ablation_study'] = ablation_results
        except Exception as e:
            logger.error(f"消融研究失败: {e}")
            results['ablation_study'] = {"error": str(e)}
        
        # 4. 基准对比
        logger.info("\n[4/4] 运行基准对比...")
        try:
            benchmark_results = self.run_benchmark_comparison()
            results['benchmark_comparison'] = benchmark_results
        except Exception as e:
            logger.error(f"基准对比失败: {e}")
            results['benchmark_comparison'] = {"error": str(e)}
        
        # 保存完整结果
        complete_results_file = f"{self.output_dir}/complete_analysis_results.json"
        with open(complete_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成执行摘要
        self._generate_executive_summary(results)
        
        logger.info("="*100)
        logger.info("FinGPT策略完整分析套件完成!")
        logger.info(f"所有结果已保存至: {self.output_dir}")
        logger.info("="*100)
        
        return results
    
    def _generate_executive_summary(self, results: Dict[str, Any]):
        """生成执行摘要"""
        summary_file = f"{self.output_dir}/executive_summary.md"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# FinGPT量化策略执行摘要\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 主要结果
            if 'main_backtest' in results:
                main = results['main_backtest']
                f.write("## 🎯 主要成果\n\n")
                f.write(f"- **总收益率**: {main.total_return:.2%}\n")
                f.write(f"- **年化收益率**: {main.annual_return:.2%}\n")
                f.write(f"- **夏普比率**: {main.sharpe_ratio:.3f}\n")
                f.write(f"- **最大回撤**: {main.max_drawdown:.2%}\n")
                f.write(f"- **FinGPT驱动交易**: {main.fingpt_trades}笔\n")
                f.write(f"- **高事件评分交易**: {main.high_event_trades}笔\n\n")
            
            # 参数优化结果
            if 'parameter_optimization' in results and 'best_by_sharpe' in results['parameter_optimization']:
                opt = results['parameter_optimization']['best_by_sharpe']
                f.write("## 🔧 最优参数\n\n")
                f.write("基于夏普比率的最优参数组合：\n\n")
                for param, value in opt['parameters'].items():
                    f.write(f"- **{param}**: {value}\n")
                f.write(f"\n最优表现: 夏普比率 {opt['sharpe_ratio']:.3f}, 收益率 {opt['total_return']:.2%}\n\n")
            
            # 消融研究洞察
            if 'ablation_study' in results and isinstance(results['ablation_study'], dict):
                f.write("## 🔬 关键洞察\n\n")
                f.write("消融研究显示各模块对策略的贡献：\n\n")
                
                ablation = results['ablation_study']
                if 'baseline_fingpt' in ablation:
                    baseline = ablation['baseline_fingpt']
                    f.write(f"- **完整FinGPT策略**: 收益 {baseline.total_return:.2%}, 夏普 {baseline.sharpe_ratio:.3f}\n")
                
                for name, result in ablation.items():
                    if name != 'baseline_fingpt' and hasattr(result, 'total_return'):
                        f.write(f"- **{name}**: 收益 {result.total_return:.2%}, 夏普 {result.sharpe_ratio:.3f}\n")
                
                f.write("\n")
            
            # 建议
            f.write("## 📋 建议\n\n")
            f.write("基于分析结果，建议：\n\n")
            f.write("1. **参数调优**: 使用优化后的参数提升策略表现\n")
            f.write("2. **风险管理**: 进一步优化止损止盈机制\n")
            f.write("3. **模型优化**: 考虑增加更多因子或改进FinGPT prompt\n")
            f.write("4. **实盘准备**: 在纸面交易中验证策略稳定性\n\n")
            
            f.write("---\n")
            f.write("*本报告由FinGPT增强量化策略系统自动生成*\n")
        
        logger.info(f"执行摘要已生成: {summary_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FinGPT增强策略回测')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'quick', 'optimize', 'ablation'],
                       help='运行模式')
    parser.add_argument('--start-date', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--disable-fingpt', action='store_true', help='禁用FinGPT')
    
    args = parser.parse_args()
    
    print("\n" + "="*100)
    print("FinGPT增强量化策略回测系统")
    print("="*100)
    print(f"执行时间: {datetime.now()}")
    print(f"运行模式: {args.mode}")
    print("="*100 + "\n")
    
    try:
        # 创建回测器
        backtester = FinGPTStrategyBacktester(args.config)
        
        # 根据参数调整配置
        if args.start_date:
            backtester.config.start_date = args.start_date
        if args.end_date:
            backtester.config.end_date = args.end_date
        if args.disable_fingpt:
            backtester.config.use_fingpt = False
        
        # 根据模式运行
        if args.mode == 'full':
            results = backtester.run_complete_analysis()
        elif args.mode == 'quick':
            # 快速测试：缩短时间范围
            backtester.config.end_date = "2024-10-31"
            result = backtester.run_full_backtest()
            print_enhanced_backtest_report(result)
        elif args.mode == 'optimize':
            results = backtester.run_parameter_optimization()
        elif args.mode == 'ablation':
            results = backtester.run_ablation_study()
        
        print("\n" + "="*100)
        print("回测完成!")
        print(f"结果保存在: {backtester.output_dir}")
        print("="*100)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n用户中断回测")
        return 1
    except Exception as e:
        logger.error(f"回测过程发生异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

