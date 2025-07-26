"""
enhanced_backtest_engine.py - FinGPT增强回测引擎
完全集成FinGPT事件评分、技术指标、动态仓位管理的高频回测系统
支持5分钟级别的新闻驱动策略回测
"""

from __future__ import annotations
import os
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
import json
import warnings

# 导入项目模块
from data_collector import DataCollector, NewsArticle
from factor_model_optimized import get_enhanced_extractor, EnhancedNewsFactorExtractor
from signal_generator import SignalGenerator, SignalType, TradeSignal
from portfolio_manager import FinGPTPortfolioManager, RiskConfig
from universe_builder import UniverseBuilder

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_backtest.log', encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建必要的目录
os.makedirs('logs', exist_ok=True)
os.makedirs('output/backtest', exist_ok=True)


@dataclass
class EnhancedBacktestConfig:
    """FinGPT增强回测配置"""
    # 时间范围
    start_date: str
    end_date: str
    
    # 标的配置
    use_configured_universe: bool = True  # 使用配置的标的池
    custom_symbols: Optional[List[str]] = None  # 自定义标的列表
    
    # 资金和仓位参数
    initial_capital: float = 100000.0
    max_position_percent: float = 0.2
    commission: float = 0.0
    
    # FinGPT信号参数（更严格）
    min_event_score: int = 3              # 最小事件评分阈值
    min_confidence: float = 0.6           # 最小信号置信度
    require_technical_confirmation: bool = False  # 是否要求技术确认
    min_news_heat: float = 0.3           # 最小新闻热度
    
    # 传统信号参数（作为备选）
    pos_thresh: float = 0.7
    neg_thresh: float = -0.7
    novel_thresh: float = 0.6
    cooldown: int = 300  # 5分钟
    
    # 风控参数（FinGPT优化版）
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.03
    high_event_stop_loss_pct: float = 0.03  # 高事件评分更宽松止损
    trailing_stop_pct: float = 0.02
    max_daily_loss_pct: float = 0.05
    allow_overnight: bool = False
    partial_take_profit: bool = True
    partial_take_ratio: float = 0.5
    event_position_multiplier: float = 1.5  # 事件评分仓位乘数
    
    # 数据频率和处理
    data_frequency: str = "5Min"
    news_lookback_days: int = 2
    use_async: bool = True
    batch_size: int = 8  # FinGPT处理批次大小
    
    # 模型配置
    use_fingpt: bool = True               # 是否使用FinGPT
    local_fingpt_path: Optional[str] = None  # 本地FinGPT模型路径
    
    def __post_init__(self):
        """配置后处理"""
        if self.custom_symbols is None:
            self.custom_symbols = []


@dataclass
class EnhancedBacktestResult:
    """增强回测结果"""
    # 基础绩效指标
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    
    # 交易统计
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # FinGPT特色统计
    fingpt_trades: int               # FinGPT驱动的交易数
    high_event_trades: int           # 高事件评分交易数
    technical_confirmed_trades: int  # 技术确认交易数
    avg_event_score: float          # 平均事件评分
    event_score_distribution: Dict[int, int]  # 事件评分分布
    
    # 时间序列数据
    equity_curve: pd.Series
    daily_returns: pd.Series
    drawdown_series: pd.Series
    
    # 详细记录
    positions_history: List[Dict]
    trades_history: List[Dict]
    signals_history: List[Dict]
    news_analysis_history: List[Dict]
    risk_events_history: List[Dict]
    
    # 性能分析
    monthly_returns: pd.Series
    risk_metrics: Dict[str, float]
    signal_effectiveness: Dict[str, Any]
    
    # 日志
    event_log: List[str]
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'performance': {
                'total_return': self.total_return,
                'annual_return': self.annual_return,
                'sharpe_ratio': self.sharpe_ratio,
                'max_drawdown': self.max_drawdown,
                'win_rate': self.win_rate
            },
            'trade_stats': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'profit_factor': self.profit_factor
            },
            'fingpt_stats': {
                'fingpt_trades': self.fingpt_trades,
                'high_event_trades': self.high_event_trades,
                'technical_confirmed_trades': self.technical_confirmed_trades,
                'avg_event_score': self.avg_event_score,
                'event_score_distribution': self.event_score_distribution
            },
            'risk_metrics': self.risk_metrics,
            'signal_effectiveness': self.signal_effectiveness
        }


class EnhancedBacktestEngine:
    """FinGPT增强回测引擎"""
    
    def __init__(self, config: EnhancedBacktestConfig):
        """初始化增强回测引擎"""
        self.config = config
        
        # 构建股票池和映射
        logger.info("初始化股票池和公司映射...")
        self.universe_builder = UniverseBuilder(use_configured_universe=config.use_configured_universe)
        
        if config.use_configured_universe:
            # 使用配置的标的池
            self.symbols, self.company_map = self.universe_builder.load_today_universe()
            if not self.symbols:
                # 如果没有今日池，构建一个
                self.symbols, _, self.company_map = self.universe_builder.build_universe()
        else:
            # 使用自定义标的
            self.symbols = config.custom_symbols or ["AAPL", "MSFT", "GOOGL"]
            self.company_map = {ticker: self.universe_builder.ticker_company_map.get(ticker, ticker) 
                              for ticker in self.symbols}
        
        # 获取关键词映射
        self.keywords_map = {ticker: self.universe_builder.get_news_keywords_for_ticker(ticker) 
                           for ticker in self.symbols}
        
        logger.info(f"股票池: {len(self.symbols)}只股票")
        logger.info(f"公司映射: {len(self.company_map)}个映射")
        
        # 初始化数据收集器（带映射）
        self.data_collector = DataCollector(
            company_map=self.company_map,
            keywords_map=self.keywords_map
        )
        
        # 初始化FinGPT因子提取器
        if config.use_fingpt:
            try:
                logger.info("初始化FinGPT因子提取器...")
                self.factor_extractor = get_enhanced_extractor(
                    data_collector=self.data_collector,
                    use_async=config.use_async,
                    batch_size=config.batch_size
                )
                logger.info("FinGPT因子提取器初始化成功")
            except Exception as e:
                logger.error(f"FinGPT初始化失败: {e}")
                logger.warning("回退到传统因子模式")
                self.factor_extractor = None
                config.use_fingpt = False
        else:
            self.factor_extractor = None
        
        # 初始化信号生成器（支持FinGPT模式）
        self.signal_generator = SignalGenerator(
            pos_thresh=config.pos_thresh,
            neg_thresh=config.neg_thresh,
            novel_thresh=config.novel_thresh,
            cooldown=config.cooldown,
            use_enhanced_factors=config.use_fingpt,  # 使用增强因子模式
            min_event_score=config.min_event_score,
            min_confidence=config.min_confidence,
            require_technical_confirmation=config.require_technical_confirmation
        )
        
        # 初始化风控配置
        risk_config = RiskConfig(
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            high_event_score_stop_loss_pct=config.high_event_stop_loss_pct,
            trailing_stop_pct=config.trailing_stop_pct,
            max_daily_loss_pct=config.max_daily_loss_pct,
            partial_take_profit=config.partial_take_profit,
            partial_take_ratio=config.partial_take_ratio,
            force_eod_close=not config.allow_overnight,
            event_score_position_multiplier=config.event_position_multiplier,
            max_positions=min(len(self.symbols), 5)  # 最多持仓数
        )
        
        # 初始化组合管理器（FinGPT优化版）
        self.portfolio_manager = FinGPTPortfolioManager(
            capital=config.initial_capital,
            max_position_percent=config.max_position_percent,
            commission=config.commission,
            risk_config=risk_config,
            live=False  # 回测模式
        )
        
        # 数据缓存
        self.price_data = {}
        self.news_data = {}
        self.current_prices = {}
        
        # 记录
        self.equity_curve = []
        self.daily_returns = []
        self.event_log = []
        self.signals_history = []
        self.news_analysis_history = []
        self.risk_events_history = []
        
        # 统计
        self.fingpt_trades = 0
        self.high_event_trades = 0
        self.technical_confirmed_trades = 0
        self.event_scores = []
        
        logger.info(
            f"FinGPT增强回测引擎初始化完成: "
            f"{config.start_date} 至 {config.end_date}, "
            f"FinGPT模式: {config.use_fingpt}"
        )
    
    def load_historical_data(self):
        """加载历史数据（增强版）"""
        logger.info("开始加载历史数据...")
        
        # 1. 批量加载价格数据
        logger.info(f"批量加载{len(self.symbols)}只股票的{self.config.data_frequency}数据")
        
        self.price_data = self.data_collector.get_multiple_stocks_data(
            self.symbols,
            self.config.start_date,
            self.config.end_date,
            self.config.data_frequency,
            max_workers=10
        )
        
        successful_symbols = [ticker for ticker, df in self.price_data.items() if not df.empty]
        failed_symbols = [ticker for ticker in self.symbols if ticker not in successful_symbols]
        
        if failed_symbols:
            logger.warning(f"以下股票价格数据加载失败: {failed_symbols}")
            # 从回测中移除失败的股票
            self.symbols = successful_symbols
        
        logger.info(f"成功加载{len(successful_symbols)}只股票的价格数据")
        
        # 2. 批量加载新闻数据
        logger.info("批量加载历史新闻数据...")
        
        # 扩展时间范围确保新闻覆盖
        news_start = (pd.Timestamp(self.config.start_date) - timedelta(days=self.config.news_lookback_days)).strftime('%Y-%m-%d')
        news_end = (pd.Timestamp(self.config.end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        
        self.news_data = self.data_collector.get_universe_news_batch(
            self.symbols,
            days_back=self.config.news_lookback_days + 1
        )
        
        # 过滤时间范围内的新闻
        filtered_news_data = {}
        total_news = 0
        
        for ticker, news_list in self.news_data.items():
            filtered_news = []
            for news in news_list:
                news_time = news.datetime if isinstance(news.datetime, datetime) else datetime.fromisoformat(news.datetime.replace('Z', '+00:00'))
                if self.config.start_date <= news_time.strftime('%Y-%m-%d') <= self.config.end_date:
                    filtered_news.append(news)
            
            filtered_news_data[ticker] = sorted(filtered_news, key=lambda x: x.datetime)
            total_news += len(filtered_news)
        
        self.news_data = filtered_news_data
        
        # 统计
        total_bars = sum(len(df) for df in self.price_data.values())
        logger.info(f"数据加载完成: {total_bars}条价格数据, {total_news}条新闻")
        
        # 数据质量检查
        self._validate_data_quality()
    
    def _validate_data_quality(self):
        """验证数据质量"""
        issues = []
        
        for ticker in self.symbols:
            # 检查价格数据
            if ticker in self.price_data:
                df = self.price_data[ticker]
                if df.empty:
                    issues.append(f"{ticker}: 价格数据为空")
                elif len(df) < 100:
                    issues.append(f"{ticker}: 价格数据不足({len(df)}条)")
                elif df['volume'].sum() == 0:
                    issues.append(f"{ticker}: 成交量数据异常")
            
            # 检查新闻数据
            if ticker in self.news_data:
                news_count = len(self.news_data[ticker])
                if news_count == 0:
                    issues.append(f"{ticker}: 无新闻数据")
                elif news_count < 5:
                    issues.append(f"{ticker}: 新闻数据稀少({news_count}条)")
        
        if issues:
            logger.warning("数据质量问题:")
            for issue in issues[:10]:  # 只显示前10个问题
                logger.warning(f"  - {issue}")
            if len(issues) > 10:
                logger.warning(f"  ... 还有{len(issues)-10}个问题")
    
    def get_price_at_time(self, ticker: str, timestamp: datetime) -> Optional[float]:
        """获取指定时间的价格"""
        if ticker not in self.price_data:
            return None
        
        df = self.price_data[ticker]
        
        # 找到时间戳之前最近的价格
        mask = df.index <= timestamp
        if mask.any():
            return float(df.loc[mask].iloc[-1]['close'])
        
        # 如果没有之前的数据，使用之后最近的
        if len(df) > 0:
            return float(df.iloc[0]['close'])
        
        return None
    
    def get_market_data_at_time(self, ticker: str, timestamp: datetime) -> Dict[str, float]:
        """获取指定时间的市场数据"""
        if ticker not in self.price_data:
            return {}
        
        df = self.price_data[ticker]
        
        # 找到时间戳之前最近的数据
        mask = df.index <= timestamp
        if mask.any():
            row = df.loc[mask].iloc[-1]
            return {
                'close': float(row['close']),
                'volume': float(row['volume']),
                'high': float(row.get('high', row['close'])),
                'low': float(row.get('low', row['close'])),
                'open': float(row.get('open', row['close']))
            }
        
        return {}
    
    def process_news_events(self, timestamp: datetime) -> List[TradeSignal]:
        """处理新闻事件（FinGPT增强版）"""
        signals = []
        
        # 收集当前时间窗口的新闻
        current_news = []
        time_window = timedelta(minutes=5)  # 5分钟窗口
        
        for ticker, news_list in self.news_data.items():
            for news in news_list:
                news_time = news.datetime if isinstance(news.datetime, datetime) else datetime.fromisoformat(news.datetime.replace('Z', '+00:00'))
                
                # 检查是否在当前时间窗口内
                if abs((news_time - timestamp).total_seconds()) <= time_window.total_seconds():
                    # 添加市场数据到新闻
                    market_data = self.get_market_data_at_time(ticker, timestamp)
                    news_dict = news.to_dict()
                    news_dict.update({
                        'current_price': market_data.get('close'),
                        'current_volume': market_data.get('volume')
                    })
                    current_news.append(news_dict)
        
        if not current_news:
            return signals
        
        # 使用FinGPT提取因子
        if self.config.use_fingpt and self.factor_extractor:
            try:
                # 转换为NewsArticle对象
                news_articles = []
                for news_dict in current_news:
                    article = NewsArticle(
                        ticker=news_dict['ticker'],
                        datetime=news_dict['datetime'],
                        headline=news_dict['headline'],
                        summary=news_dict['summary'],
                        source=news_dict['source'],
                        url=news_dict['url'],
                        company=news_dict.get('company'),
                        keywords_matched=news_dict.get('keywords_matched'),
                        relevance_score=news_dict.get('relevance_score')
                    )
                    news_articles.append(article)
                
                # FinGPT因子提取
                factors = self.factor_extractor.extract_factors_from_news_articles(news_articles)
                
                # 为每个因子生成信号
                for factor in factors:
                    signal = self.signal_generator.generate_signal_from_enhanced_factor(factor, timestamp)
                    
                    if signal.signal != SignalType.HOLD:
                        signals.append(signal)
                        
                        # 记录统计
                        if factor.event_score >= 4:
                            self.high_event_trades += 1
                        if signal.technical_confirmed:
                            self.technical_confirmed_trades += 1
                        
                        self.event_scores.append(factor.event_score)
                        
                        # 记录新闻分析历史
                        self.news_analysis_history.append({
                            'timestamp': timestamp.isoformat(),
                            'ticker': factor.ticker,
                            'headline': factor.headline[:100],
                            'sentiment_score': factor.sentiment_score,
                            'event_score': factor.event_score,
                            'event_impact': factor.event_impact,
                            'confidence': factor.confidence_composite,
                            'novelty': factor.novelty,
                            'news_heat': factor.news_heat,
                            'technical_confirmed': signal.technical_confirmed
                        })
                        
                        self.log_event(
                            f"[FinGPT新闻] {timestamp:%H:%M} {factor.ticker} "
                            f"事件={factor.event_score}({factor.event_impact}) "
                            f"情绪={factor.sentiment_score:.3f} → {signal.signal.value}"
                        )
            
            except Exception as e:
                logger.error(f"FinGPT因子提取失败: {e}")
                # 回退到传统方法
                signals.extend(self._process_news_fallback(current_news, timestamp))
        else:
            # 传统新闻处理方法
            signals.extend(self._process_news_fallback(current_news, timestamp))
        
        return signals
    
    def _process_news_fallback(self, news_list: List[Dict], timestamp: datetime) -> List[TradeSignal]:
        """传统新闻处理方法（回退）"""
        signals = []
        
        for news_dict in news_list:
            try:
                # 简单的关键词情绪分析
                headline = news_dict.get('headline', '').lower()
                summary = news_dict.get('summary', '').lower()
                text = f"{headline} {summary}"
                
                # 计算简单情绪分数
                positive_words = ['beat', 'strong', 'growth', 'profit', 'gain', 'rise', 'up', 'good', 'excellent']
                negative_words = ['miss', 'weak', 'decline', 'loss', 'fall', 'down', 'bad', 'poor', 'worry']
                
                pos_count = sum(1 for word in positive_words if word in text)
                neg_count = sum(1 for word in negative_words if word in text)
                
                if pos_count > neg_count and pos_count >= 2:
                    sentiment_score = 0.8
                    signal_type = SignalType.BUY
                elif neg_count > pos_count and neg_count >= 2:
                    sentiment_score = -0.8
                    signal_type = SignalType.SELL
                else:
                    continue  # 跳过中性新闻
                
                # 创建信号
                signal = TradeSignal(
                    ticker=news_dict['ticker'],
                    signal=signal_type,
                    confidence=0.6,  # 传统方法置信度较低
                    timestamp=timestamp,
                    factors={'sentiment_score': sentiment_score},
                    reason=f"传统新闻分析: {headline[:50]}...",
                    event_score=2,  # 默认事件评分
                    event_impact="low"
                )
                
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"传统新闻处理失败: {e}")
                continue
        
        return signals
    
    def update_market_prices(self, timestamp: datetime):
        """更新市场价格"""
        for ticker in self.symbols:
            price = self.get_price_at_time(ticker, timestamp)
            if price:
                self.current_prices[ticker] = price
                
                # 更新组合管理器中的价格
                if ticker in self.portfolio_manager.positions:
                    self.portfolio_manager.positions[ticker].update_price(price)
    
    def calculate_portfolio_metrics(self, timestamp: datetime) -> Dict[str, float]:
        """计算组合指标"""
        # 更新价格
        self.update_market_prices(timestamp)
        
        # 获取组合状态
        status = self.portfolio_manager.get_portfolio_status()
        
        return {
            'total_equity': status['total_equity'],
            'cash': status['cash'],
            'total_market_value': status['total_market_value'],
            'daily_return': status['daily_return'],
            'total_return': status['total_return']
        }
    
    def log_event(self, message: str):
        """记录事件"""
        self.event_log.append(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {message}")
        logger.info(message)
    
    async def run_backtest_async(self) -> EnhancedBacktestResult:
        """异步运行增强回测"""
        logger.info("="*80)
        logger.info("开始运行FinGPT增强回测...")
        logger.info("="*80)
        
        # 加载数据
        self.load_historical_data()
        
        if not self.symbols:
            raise ValueError("没有可用的股票数据进行回测")
        
        # 构建时间序列事件
        all_events = []
        
        # 添加价格更新事件
        for ticker, df in self.price_data.items():
            for idx, row in df.iterrows():
                all_events.append({
                    'type': 'price_update',
                    'ticker': ticker,
                    'timestamp': idx,
                    'data': {
                        'close': row['close'],
                        'volume': row['volume'],
                        'high': row.get('high', row['close']),
                        'low': row.get('low', row['close']),
                        'open': row.get('open', row['close'])
                    }
                })
        
        # 添加新闻事件
        for ticker, news_list in self.news_data.items():
            for news in news_list:
                news_time = news.datetime if isinstance(news.datetime, datetime) else datetime.fromisoformat(news.datetime.replace('Z', '+00:00'))
                all_events.append({
                    'type': 'news',
                    'ticker': ticker,
                    'timestamp': news_time,
                    'data': news
                })
        
        # 按时间排序
        all_events.sort(key=lambda x: x['timestamp'])
        logger.info(f"总事件数: {len(all_events)} (价格更新 + 新闻)")
        
        # 模拟交易
        current_date = None
        processed_events = 0
        
        with tqdm(total=len(all_events), desc="回测进度") as pbar:
            for event in all_events:
                timestamp = event['timestamp']
                
                # 检查是否新的一天
                event_date = timestamp.date()
                if current_date != event_date:
                    if current_date is not None:
                        # 日终处理
                        if not self.config.allow_overnight:
                            eod_trades = self.portfolio_manager.force_eod_close(timestamp)
                            if eod_trades:
                                self.log_event(f"[日终清仓] 平掉{len(eod_trades)}个仓位")
                        
                        # 记录日终权益
                        metrics = self.calculate_portfolio_metrics(timestamp)
                        self.equity_curve.append({
                            'date': current_date,
                            'equity': metrics['total_equity'],
                            'return': metrics['daily_return']
                        })
                    
                    # 新的一天
                    current_date = event_date
                    self.portfolio_manager.start_new_trading_day(timestamp)
                    self.log_event(f"[开盘] {current_date}")
                
                # 处理事件
                if event['type'] == 'price_update':
                    # 更新价格并检查风控
                    self.update_market_prices(timestamp)
                    
                    # 检查止损止盈（每次价格更新都检查）
                    risk_trades = self.portfolio_manager.check_stop_loss_take_profit(timestamp)
                    if risk_trades:
                        for trade in risk_trades:
                            self.risk_events_history.append({
                                'timestamp': timestamp.isoformat(),
                                'type': 'risk_control',
                                'action': trade.action,
                                'ticker': trade.ticker,
                                'reason': trade.reason
                            })
                
                elif event['type'] == 'news':
                    # 处理新闻事件
                    signals = self.process_news_events(timestamp)
                    
                    if signals:
                        # 构建信号字典和价格字典
                        signals_dict = {signal.ticker: signal for signal in signals}
                        prices_dict = {ticker: self.current_prices.get(ticker, 0) for ticker in signals_dict.keys()}
                        
                        # 批量处理信号
                        trades = self.portfolio_manager.process_signals(signals_dict, prices_dict, timestamp)
                        
                        if trades:
                            self.fingpt_trades += len(trades)
                            
                        # 记录信号历史
                        for signal in signals:
                            self.signals_history.append({
                                'timestamp': timestamp.isoformat(),
                                'ticker': signal.ticker,
                                'signal': signal.signal.value,
                                'confidence': signal.confidence,
                                'event_score': signal.event_score,
                                'event_impact': signal.event_impact,
                                'technical_confirmed': signal.technical_confirmed,
                                'reason': signal.reason
                            })
                
                processed_events += 1
                pbar.update(1)
                
                # 定期记录进度
                if processed_events % 1000 == 0:
                    metrics = self.calculate_portfolio_metrics(timestamp)
                    logger.debug(f"进度: {processed_events}/{len(all_events)}, 权益: ${metrics['total_equity']:.2f}")
        
        # 最终处理
        if current_date:
            final_timestamp = all_events[-1]['timestamp']
            
            # 最终清仓
            if not self.config.allow_overnight:
                final_trades = self.portfolio_manager.force_eod_close(final_timestamp)
                if final_trades:
                    self.log_event(f"[最终清仓] 平掉{len(final_trades)}个仓位")
            
            # 最终权益记录
            final_metrics = self.calculate_portfolio_metrics(final_timestamp)
            self.equity_curve.append({
                'date': current_date,
                'equity': final_metrics['total_equity'],
                'return': final_metrics['daily_return']
            })
        
        # 计算绩效
        result = self.calculate_enhanced_performance()
        
        logger.info("="*80)
        logger.info("FinGPT增强回测完成!")
        logger.info(f"FinGPT驱动交易: {self.fingpt_trades}")
        logger.info(f"高事件评分交易: {self.high_event_trades}")
        logger.info(f"技术确认交易: {self.technical_confirmed_trades}")
        logger.info("="*80)
        
        return result
    
    def run_backtest(self) -> EnhancedBacktestResult:
        """同步运行回测"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.run_backtest_async())
        finally:
            loop.close()
    
    def calculate_enhanced_performance(self) -> EnhancedBacktestResult:
        """计算增强绩效指标"""
        # 获取最终状态
        final_status = self.portfolio_manager.get_portfolio_status()
        trades = self.portfolio_manager.get_trade_history()
        
        # 转换权益曲线
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_series = equity_df.set_index('date')['equity']
            returns_series = equity_df.set_index('date')['return']
        else:
            equity_series = pd.Series([self.config.initial_capital])
            returns_series = pd.Series([0.0])
        
        # 基础绩效指标
        total_return = (final_status['total_equity'] - self.config.initial_capital) / self.config.initial_capital
        
        days = (pd.Timestamp(self.config.end_date) - pd.Timestamp(self.config.start_date)).days
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 夏普比率
        if len(returns_series) > 1 and returns_series.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns_series.mean() / returns_series.std()
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        drawdown_series = drawdown
        
        # 交易统计
        winning_trades = 0
        losing_trades = 0
        wins = []
        losses = []
        
        # 简化的盈亏计算
        for trade in trades:
            if 'pnl' in trade and trade['pnl'] is not None:
                if trade['pnl'] > 0:
                    winning_trades += 1
                    wins.append(trade['pnl'])
                else:
                    losing_trades += 1
                    losses.append(trade['pnl'])
        
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
        
        # FinGPT特色统计
        avg_event_score = np.mean(self.event_scores) if self.event_scores else 0
        event_score_distribution = {}
        for score in self.event_scores:
            event_score_distribution[score] = event_score_distribution.get(score, 0) + 1
        
        # 月度收益
        monthly_returns = returns_series.resample('M').sum() if len(returns_series) > 1 else pd.Series([0])
        
        # 风险指标
        risk_metrics = {
            'volatility': returns_series.std() * np.sqrt(252) if len(returns_series) > 1 else 0,
            'downside_deviation': returns_series[returns_series < 0].std() * np.sqrt(252) if len(returns_series) > 1 else 0,
            'max_drawdown': max_drawdown,
            'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        }
        
        # 信号有效性分析
        signal_effectiveness = {
            'total_signals': len(self.signals_history),
            'signals_to_trades_ratio': len(trades) / len(self.signals_history) if self.signals_history else 0,
            'avg_confidence': np.mean([s['confidence'] for s in self.signals_history]) if self.signals_history else 0,
            'high_confidence_signals': sum(1 for s in self.signals_history if s['confidence'] > 0.7),
            'fingpt_signal_rate': sum(1 for s in self.signals_history if s.get('event_score', 1) >= 3) / len(self.signals_history) if self.signals_history else 0
        }
        
        return EnhancedBacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            fingpt_trades=self.fingpt_trades,
            high_event_trades=self.high_event_trades,
            technical_confirmed_trades=self.technical_confirmed_trades,
            avg_event_score=avg_event_score,
            event_score_distribution=event_score_distribution,
            equity_curve=equity_series,
            daily_returns=returns_series,
            drawdown_series=drawdown_series,
            positions_history=[],
            trades_history=trades,
            signals_history=self.signals_history,
            news_analysis_history=self.news_analysis_history,
            risk_events_history=self.risk_events_history,
            monthly_returns=monthly_returns,
            risk_metrics=risk_metrics,
            signal_effectiveness=signal_effectiveness,
            event_log=self.event_log
        )
    
    def save_results(self, result: EnhancedBacktestResult, output_dir: str = "output/backtest"):
        """保存增强回测结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存综合报告
        report_file = f"{output_dir}/fingpt_enhanced_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        
        # 保存权益曲线
        equity_file = f"{output_dir}/equity_curve_{timestamp}.csv"
        result.equity_curve.to_csv(equity_file)
        
        # 保存交易历史
        trades_file = f"{output_dir}/trades_history_{timestamp}.json"
        with open(trades_file, 'w', encoding='utf-8') as f:
            json.dump(result.trades_history, f, indent=2, default=str)
        
        # 保存信号历史
        signals_file = f"{output_dir}/signals_history_{timestamp}.json"
        with open(signals_file, 'w', encoding='utf-8') as f:
            json.dump(result.signals_history, f, indent=2, default=str)
        
        # 保存新闻分析历史
        news_file = f"{output_dir}/news_analysis_{timestamp}.json"
        with open(news_file, 'w', encoding='utf-8') as f:
            json.dump(result.news_analysis_history, f, indent=2, default=str)
        
        # 保存事件日志
        log_file = f"{output_dir}/event_log_{timestamp}.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(result.event_log))
        
        logger.info(f"增强回测结果已保存到: {output_dir}")
        
        return {
            'report': report_file,
            'equity': equity_file,
            'trades': trades_file,
            'signals': signals_file,
            'news': news_file,
            'log': log_file
        }


# 便捷函数
def create_enhanced_backtest_config(
    start_date: str,
    end_date: str,
    **kwargs
) -> EnhancedBacktestConfig:
    """创建增强回测配置"""
    return EnhancedBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )


def run_fingpt_strategy_backtest(config: EnhancedBacktestConfig) -> EnhancedBacktestResult:
    """运行FinGPT策略回测"""
    engine = EnhancedBacktestEngine(config)
    result = engine.run_backtest()
    
    # 保存结果
    engine.save_results(result)
    
    return result


def print_enhanced_backtest_report(result: EnhancedBacktestResult):
    """打印增强回测报告"""
    print("\n" + "="*80)
    print("FinGPT增强策略回测报告")
    print("="*80)
    
    print("\n📊 基础绩效指标:")
    print(f"总收益率: {result.total_return:.2%}")
    print(f"年化收益率: {result.annual_return:.2%}")
    print(f"夏普比率: {result.sharpe_ratio:.2f}")
    print(f"最大回撤: {result.max_drawdown:.2%}")
    print(f"Calmar比率: {result.risk_metrics.get('calmar_ratio', 0):.2f}")
    
    print("\n📈 交易统计:")
    print(f"总交易次数: {result.total_trades}")
    print(f"胜率: {result.win_rate:.2%}")
    print(f"盈利次数: {result.winning_trades}")
    print(f"亏损次数: {result.losing_trades}")
    print(f"平均盈利: ${result.avg_win:.2f}")
    print(f"平均亏损: ${result.avg_loss:.2f}")
    print(f"盈亏比: {result.profit_factor:.2f}")
    
    print("\n🤖 FinGPT特色统计:")
    print(f"FinGPT驱动交易: {result.fingpt_trades}")
    print(f"高事件评分交易: {result.high_event_trades}")
    print(f"技术确认交易: {result.technical_confirmed_trades}")
    print(f"平均事件评分: {result.avg_event_score:.1f}")
    print(f"事件评分分布: {result.event_score_distribution}")
    
    print("\n📊 信号有效性:")
    print(f"总信号数: {result.signal_effectiveness['total_signals']}")
    print(f"信号转换率: {result.signal_effectiveness['signals_to_trades_ratio']:.2%}")
    print(f"平均置信度: {result.signal_effectiveness['avg_confidence']:.2f}")
    print(f"高置信度信号: {result.signal_effectiveness['high_confidence_signals']}")
    
    print("\n⚠️  风险指标:")
    print(f"年化波动率: {result.risk_metrics['volatility']:.2%}")
    print(f"下行偏差: {result.risk_metrics['downside_deviation']:.2%}")
    
    print("\n" + "="*80)


# 测试代码
if __name__ == "__main__":
    print("=== FinGPT增强回测引擎测试 ===")
    
    # 创建测试配置
    test_config = create_enhanced_backtest_config(
        start_date="2024-09-01",
        end_date="2024-09-15",  # 短期测试
        use_configured_universe=True,
        use_fingpt=True,
        min_event_score=3,
        min_confidence=0.6,
        allow_overnight=False
    )
    
    print(f"测试配置: {test_config.start_date} 至 {test_config.end_date}")
    print(f"FinGPT模式: {test_config.use_fingpt}")
    print(f"最小事件评分: {test_config.min_event_score}")
    
    try:
        # 运行回测
        result = run_fingpt_strategy_backtest(test_config)
        
        # 打印报告
        print_enhanced_backtest_report(result)
        
        print("\n✅ FinGPT增强回测引擎测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
