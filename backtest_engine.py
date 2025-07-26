"""
backtest_engine.py - 高频策略回测引擎（优化版）
专门针对5分钟K线数据，结合新闻情绪和技术指标的回测系统
新增功能：移动止损、分批止盈、日终清仓、动态仓位控制
"""

from __future__ import annotations
import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
import json

# 导入项目模块
from data_collector import DataCollector, NewsArticle
from factor_model_optimized import get_enhanced_extractor
from signal_generator import SignalGenerator, SignalType
from portfolio_manager import PortfolioManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest.log', encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建必要的目录
os.makedirs('logs', exist_ok=True)
os.makedirs('output/backtest', exist_ok=True)


@dataclass
class BacktestConfig:
    """回测配置（增强版）"""
    # 时间范围
    start_date: str
    end_date: str
    
    # 标的列表
    symbols: List[str]
    
    # 资金和仓位参数
    initial_capital: float = 100000.0
    max_position_percent: float = 0.2
    commission: float = 0.0  # 假设无手续费
    
    # 信号参数（更严格的阈值）
    pos_thresh: float = 0.7      # 从0.5提高到0.7
    neg_thresh: float = -0.7     # 从-0.5提高到-0.7
    novel_thresh: float = 0.6    # 从0.5提高到0.6
    cooldown: int = 300          # 5分钟冷却
    
    # 风控参数（增强版）
    stop_loss_pct: float = 0.02      # 2%止损
    take_profit_pct: float = 0.03    # 3%止盈
    trailing_stop_pct: float = 0.02  # 新增：2%移动止损回撤
    max_daily_loss_pct: float = 0.05 # 从0.1降低到0.05
    allow_overnight: bool = False    # 新增：是否允许隔夜持仓
    partial_take_profit: bool = True # 新增：是否启用分批止盈
    partial_take_ratio: float = 0.5  # 新增：分批止盈比例
    
    # 数据频率
    data_frequency: str = "5Min"
    
    # 其他
    use_async: bool = True
    batch_size: int = 16


@dataclass
class Position:
    """持仓信息（增强版）"""
    ticker: str
    quantity: int
    avg_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    peak_pnl_pct: float = 0.0  # 新增：持仓期间最高浮盈百分比
    is_partial_closed: bool = False  # 新增：是否已部分止盈
    
    def update_price(self, price: float):
        """更新当前价格和浮盈"""
        self.current_price = price
        if self.quantity > 0:
            self.unrealized_pnl = (price - self.avg_price) * self.quantity
            pnl_pct = (price - self.avg_price) / self.avg_price
            self.peak_pnl_pct = max(self.peak_pnl_pct, pnl_pct)
        elif self.quantity < 0:
            self.unrealized_pnl = (self.avg_price - price) * abs(self.quantity)
            pnl_pct = (self.avg_price - price) / self.avg_price
            self.peak_pnl_pct = max(self.peak_pnl_pct, pnl_pct)


@dataclass
class BacktestResult:
    """回测结果"""
    # 绩效指标
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
    
    # 时间序列数据
    equity_curve: pd.Series
    positions_history: List[Dict]
    trades_history: List[Dict]
    signals_history: List[Dict]
    
    # 详细日志
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
            }
        }


class BacktestEngine:
    """回测引擎类（优化版）"""
    
    def __init__(self, config: BacktestConfig):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置
        """
        self.config = config
        
        # 初始化组件
        self.data_collector = DataCollector()
        self.factor_extractor = get_enhanced_extractor(
            data_collector=self.data_collector,
            use_async=config.use_async,
            batch_size=config.batch_size
        )
        self.signal_generator = SignalGenerator(
            pos_thresh=config.pos_thresh,
            neg_thresh=config.neg_thresh,
            novel_thresh=config.novel_thresh,
            cooldown=config.cooldown,
            use_enhanced_factors=True  # 使用增强因子模式
        )
        self.portfolio_manager = PortfolioManager(
            capital=config.initial_capital,
            max_position_percent=config.max_position_percent,
            commission=config.commission,
            live=False  # 回测模式
        )
        
        # 数据缓存
        self.price_data = {}  # {ticker: DataFrame}
        self.news_data = {}   # {ticker: List[news]}
        self.current_prices = {}  # {ticker: price}
        self.positions = {}  # 持仓管理
        
        # 记录
        self.equity_curve = []
        self.event_log = []
        self.signals_history = []
        self.daily_start_equity = config.initial_capital
        
        logger.info(f"回测引擎初始化完成: {config.start_date} 至 {config.end_date}")
    
    def load_historical_data(self):
        """加载历史数据"""
        logger.info("开始加载历史数据...")
        
        # 1. 加载价格数据
        logger.info(f"加载{len(self.config.symbols)}只股票的{self.config.data_frequency}数据")
        
        for ticker in tqdm(self.config.symbols, desc="加载价格数据"):
            try:
                df = self.data_collector.get_historical_data(
                    ticker,
                    self.config.start_date,
                    self.config.end_date,
                    self.config.data_frequency
                )
                if not df.empty:
                    self.price_data[ticker] = df
                    logger.debug(f"{ticker}: 加载{len(df)}条{self.config.data_frequency}数据")
                else:
                    logger.warning(f"{ticker}: 价格数据为空")
            except Exception as e:
                logger.error(f"加载{ticker}价格数据失败: {e}")
        
        # 2. 加载新闻数据
        logger.info("加载历史新闻数据...")
        
        # 扩展日期范围以确保覆盖
        news_start = (pd.Timestamp(self.config.start_date) - timedelta(days=2)).strftime('%Y-%m-%d')
        news_end = (pd.Timestamp(self.config.end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        
        for ticker in tqdm(self.config.symbols, desc="加载新闻数据"):
            try:
                news_list = self.data_collector.finnhub_client.company_news(
                    ticker,
                    _from=news_start,
                    to=news_end
                )
                
                # 转换时间戳并过滤
                processed_news = []
                for news in news_list:
                    news_time = datetime.fromtimestamp(news.get('datetime', 0))
                    if self.config.start_date <= news_time.strftime('%Y-%m-%d') <= self.config.end_date:
                        news['ticker'] = ticker
                        news['datetime_obj'] = news_time
                        processed_news.append(news)
                
                self.news_data[ticker] = sorted(processed_news, key=lambda x: x['datetime_obj'])
                logger.debug(f"{ticker}: 加载{len(processed_news)}条新闻")
                
            except Exception as e:
                logger.error(f"加载{ticker}新闻数据失败: {e}")
                self.news_data[ticker] = []
        
        # 统计
        total_bars = sum(len(df) for df in self.price_data.values())
        total_news = sum(len(news) for news in self.news_data.values())
        logger.info(f"数据加载完成: {total_bars}条价格数据, {total_news}条新闻")
    
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
    
    def get_volume_at_time(self, ticker: str, timestamp: datetime) -> Optional[float]:
        """获取指定时间的成交量"""
        if ticker not in self.price_data:
            return None
        
        df = self.price_data[ticker]
        
        # 找到时间戳之前最近的成交量
        mask = df.index <= timestamp
        if mask.any():
            return float(df.loc[mask].iloc[-1]['volume'])
        
        return 0.0
    
    def get_avg_volume(self, ticker: str, timestamp: datetime, window: int = 60) -> Optional[float]:
        """获取平均成交量（用于计算volume_spike）"""
        if ticker not in self.price_data:
            return None
        
        df = self.price_data[ticker]
        
        # 找到时间戳之前window分钟的数据
        start_time = timestamp - timedelta(minutes=window)
        mask = (df.index > start_time) & (df.index <= timestamp)
        
        if mask.any():
            avg_vol = df.loc[mask]['volume'].mean()
            return float(avg_vol) if not pd.isna(avg_vol) else None
        
        return None
    
    def check_stop_loss_take_profit(self, timestamp: datetime) -> List[Any]:
        """检查止损止盈（增强版：包含移动止损和分批止盈）"""
        trades = []
        positions = self.positions.copy()
        
        for ticker, position in positions.items():
            if position.quantity == 0:
                continue
            
            # 获取当前价格
            current_price = self.get_price_at_time(ticker, timestamp)
            if current_price is None:
                continue
            
            # 更新持仓价格
            position.update_price(current_price)
            
            # 计算盈亏比例
            if position.quantity > 0:  # 多头
                pnl_pct = (current_price - position.avg_price) / position.avg_price
                
                # 检查固定止损
                if pnl_pct <= -self.config.stop_loss_pct:
                    trade = self.portfolio_manager.execute_trade(
                        ticker, "SELL", abs(position.quantity), current_price, timestamp
                    )
                    if trade:
                        trades.append(trade)
                        self.log_event(f"[止损] {ticker} @ ${current_price:.2f}, 亏损{pnl_pct:.2%}")
                        del self.positions[ticker]
                    continue
                
                # 检查分批止盈（优先于完全止盈）
                if (self.config.partial_take_profit and 
                    not position.is_partial_closed and 
                    pnl_pct >= self.config.take_profit_pct):
                    # 卖出部分仓位
                    sell_quantity = max(1, int(abs(position.quantity) * self.config.partial_take_ratio))
                    trade = self.portfolio_manager.execute_trade(
                        ticker, "SELL", sell_quantity, current_price, timestamp
                    )
                    if trade:
                        trades.append(trade)
                        position.quantity -= sell_quantity
                        position.is_partial_closed = True
                        # 将剩余仓位成本价调整为当前价（保本）
                        position.avg_price = current_price
                        self.log_event(f"[分批止盈] {ticker} @ ${current_price:.2f}, 盈利{pnl_pct:.2%}, 卖出{sell_quantity}股")
                    continue
                
                # 检查移动止损（对于已有盈利的仓位）
                if position.peak_pnl_pct >= self.config.take_profit_pct:
                    # 从峰值回落幅度
                    drawdown_from_peak = position.peak_pnl_pct - pnl_pct
                    if drawdown_from_peak >= self.config.trailing_stop_pct:
                        trade = self.portfolio_manager.execute_trade(
                            ticker, "SELL", abs(position.quantity), current_price, timestamp
                        )
                        if trade:
                            trades.append(trade)
                            self.log_event(f"[移动止损] {ticker} @ ${current_price:.2f}, 从峰值{position.peak_pnl_pct:.2%}回落{drawdown_from_peak:.2%}")
                            del self.positions[ticker]
                        continue
                        
            else:  # 空头
                pnl_pct = (position.avg_price - current_price) / position.avg_price
                
                # 检查止损
                if pnl_pct <= -self.config.stop_loss_pct:
                    trade = self.portfolio_manager.execute_trade(
                        ticker, "BUY_TO_COVER", abs(position.quantity), current_price, timestamp
                    )
                    if trade:
                        trades.append(trade)
                        self.log_event(f"[止损] 空头{ticker} @ ${current_price:.2f}, 亏损{pnl_pct:.2%}")
                        del self.positions[ticker]
                    continue
                
                # 检查分批止盈
                if (self.config.partial_take_profit and 
                    not position.is_partial_closed and 
                    pnl_pct >= self.config.take_profit_pct):
                    cover_quantity = max(1, int(abs(position.quantity) * self.config.partial_take_ratio))
                    trade = self.portfolio_manager.execute_trade(
                        ticker, "BUY_TO_COVER", cover_quantity, current_price, timestamp
                    )
                    if trade:
                        trades.append(trade)
                        position.quantity += cover_quantity  # 空头回补，数量增加（趋向0）
                        position.is_partial_closed = True
                        position.avg_price = current_price
                        self.log_event(f"[分批止盈] 空头{ticker} @ ${current_price:.2f}, 盈利{pnl_pct:.2%}, 回补{cover_quantity}股")
                    continue
                
                # 检查移动止损
                if position.peak_pnl_pct >= self.config.take_profit_pct:
                    drawdown_from_peak = position.peak_pnl_pct - pnl_pct
                    if drawdown_from_peak >= self.config.trailing_stop_pct:
                        trade = self.portfolio_manager.execute_trade(
                            ticker, "BUY_TO_COVER", abs(position.quantity), current_price, timestamp
                        )
                        if trade:
                            trades.append(trade)
                            self.log_event(f"[移动止损] 空头{ticker} @ ${current_price:.2f}, 从峰值{position.peak_pnl_pct:.2%}回落{drawdown_from_peak:.2%}")
                            del self.positions[ticker]
        
        return trades
    
    def check_daily_loss_limit(self, current_equity: float) -> bool:
        """检查日内亏损限制"""
        daily_loss = (self.daily_start_equity - current_equity) / self.daily_start_equity
        
        if daily_loss >= self.config.max_daily_loss_pct:
            self.log_event(f"[风控] 触发日内亏损限制: {daily_loss:.2%}")
            
            # 平掉所有仓位
            positions = self.positions.copy()
            for ticker, position in positions.items():
                if position.quantity != 0:
                    price = self.current_prices.get(ticker, position.current_price)
                    if position.quantity > 0:
                        self.portfolio_manager.execute_trade(
                            ticker, "SELL", abs(position.quantity), price
                        )
                    else:
                        self.portfolio_manager.execute_trade(
                            ticker, "BUY_TO_COVER", abs(position.quantity), price
                        )
                    self.log_event(f"[风控] 强制平仓 {ticker}")
                    del self.positions[ticker]
            
            return True  # 停止交易
        
        return False
    
    def close_all_positions(self, timestamp: datetime):
        """平掉所有仓位（用于日终清仓）"""
        positions = self.positions.copy()
        for ticker, position in positions.items():
            if position.quantity != 0:
                price = self.current_prices.get(ticker, position.current_price)
                action = "SELL" if position.quantity > 0 else "BUY_TO_COVER"
                trade = self.portfolio_manager.execute_trade(
                    ticker, action, abs(position.quantity), price, timestamp
                )
                if trade:
                    self.log_event(f"[日终清仓] 平掉{ticker}仓位{abs(position.quantity)}股 @ ${price:.2f}")
                del self.positions[ticker]
    
    def process_news_event(self, news: Dict, timestamp: datetime) -> Optional[Dict]:
        """处理单条新闻事件（优化版：计算真实技术指标）"""
        ticker = news['ticker']
        
        # 获取当前价格和成交量
        price = self.get_price_at_time(ticker, timestamp)
        volume = self.get_volume_at_time(ticker, timestamp)
        
        if price is None:
            logger.debug(f"无法获取{ticker}在{timestamp}的价格")
            return None
        
        # 更新当前价格缓存
        self.current_prices[ticker] = price
        
        # 计算技术指标因子
        # 1. 价格变动
        prev_price_5min = self.get_price_at_time(ticker, timestamp - timedelta(minutes=5))
        prev_price_15min = self.get_price_at_time(ticker, timestamp - timedelta(minutes=15))
        
        price_change_5min = 0.0
        price_change_15min = 0.0
        
        if prev_price_5min:
            price_change_5min = (price - prev_price_5min) / prev_price_5min
        if prev_price_15min:
            price_change_15min = (price - prev_price_15min) / prev_price_15min
        
        # 2. 成交量放大倍数
        avg_volume = self.get_avg_volume(ticker, timestamp, window=60)
        volume_spike = 1.0
        if volume and avg_volume and avg_volume > 0:
            volume_spike = volume / avg_volume
        
        # 提取新闻因子
        factor = self.factor_extractor.extract_factors_sync([news])
        if not factor:
            return None
        
        factor = factor[0]  # 取第一个
        
        # 更新因子的技术指标（覆盖默认值）
        factor.price_change_5min = price_change_5min
        factor.price_change_15min = price_change_15min
        factor.volume_spike = volume_spike
        
        # 生成信号（使用增强因子模式）
        signal = self.signal_generator.generate_signal_from_enhanced_factor(factor, timestamp)
        
        # 记录信号
        signal_dict = {
            'ticker': ticker,
            'timestamp': timestamp,
            'signal': signal.signal.value,
            'confidence': signal.confidence,
            'sentiment_score': factor.sentiment_score,
            'novelty': factor.novelty,
            'news_heat': factor.news_heat,
            'price_change_5min': price_change_5min,
            'volume_spike': volume_spike,
            'reason': signal.reason,
            'price': price
        }
        
        self.signals_history.append(signal_dict)
        
        # 执行交易
        if signal.signal != SignalType.HOLD and signal.confidence > 0:
            trade = self.portfolio_manager.update_position(
                ticker, signal, price, signal.confidence, timestamp
            )
            
            if trade:
                # 更新内部持仓记录
                if trade.action in ['BUY', 'SELL_SHORT']:
                    self.positions[ticker] = Position(
                        ticker=ticker,
                        quantity=trade.quantity if trade.action == 'BUY' else -trade.quantity,
                        avg_price=trade.price,
                        entry_time=timestamp,
                        current_price=trade.price
                    )
                elif trade.action in ['SELL', 'BUY_TO_COVER'] and ticker in self.positions:
                    # 完全平仓时删除
                    if self.portfolio_manager.positions.get(ticker) is None:
                        del self.positions[ticker]
                
                self.log_event(
                    f"[交易] {timestamp:%H:%M} {trade.action} {trade.quantity} {ticker} "
                    f"@ ${price:.2f} (情绪={factor.sentiment_score:.3f}, "
                    f"新闻热度={factor.news_heat:.3f}, 信心={signal.confidence:.2f})"
                )
        
        return signal_dict
    
    def calculate_portfolio_value(self, timestamp: datetime) -> float:
        """计算组合总价值"""
        # 更新所有持仓的最新价格
        for ticker, position in self.positions.items():
            if position.quantity != 0:
                price = self.get_price_at_time(ticker, timestamp)
                if price:
                    position.update_price(price)
                    self.current_prices[ticker] = price
        
        # 获取组合状态
        status = self.portfolio_manager.get_portfolio_status()
        return status['total_equity']
    
    def log_event(self, message: str):
        """记录事件"""
        self.event_log.append(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {message}")
        logger.info(message)
    
    async def run_backtest_async(self) -> BacktestResult:
        """异步运行回测"""
        logger.info("="*60)
        logger.info("开始运行回测...")
        logger.info("="*60)
        
        # 加载数据
        self.load_historical_data()
        
        # 合并所有事件（价格更新和新闻）
        all_events = []
        
        # 添加价格更新事件（每个K线）
        for ticker, df in self.price_data.items():
            for idx, row in df.iterrows():
                all_events.append({
                    'type': 'price',
                    'ticker': ticker,
                    'timestamp': idx,
                    'price': row['close'],
                    'volume': row['volume']
                })
        
        # 添加新闻事件
        for ticker, news_list in self.news_data.items():
            for news in news_list:
                all_events.append({
                    'type': 'news',
                    'ticker': ticker,
                    'timestamp': news['datetime_obj'],
                    'news': news
                })
        
        # 按时间排序
        all_events.sort(key=lambda x: x['timestamp'])
        
        logger.info(f"总事件数: {len(all_events)} (价格更新+新闻)")
        
        # 模拟交易
        current_date = None
        daily_stop = False
        
        for event in tqdm(all_events, desc="回测进度"):
            timestamp = event['timestamp']
            
            # 检查是否新的一天
            event_date = timestamp.date()
            if current_date != event_date:
                if current_date is not None:
                    # 日终处理
                    # 1. 如果不允许隔夜，清仓
                    if not self.config.allow_overnight:
                        self.close_all_positions(timestamp)
                    
                    # 2. 记录前一天收盘权益
                    eod_equity = self.calculate_portfolio_value(timestamp)
                    self.equity_curve.append({
                        'date': current_date,
                        'equity': eod_equity
                    })
                    self.log_event(f"[日结] {current_date} 权益: ${eod_equity:,.2f}")
                
                # 新的一天
                current_date = event_date
                self.daily_start_equity = self.calculate_portfolio_value(timestamp)
                daily_stop = False
                self.log_event(f"[开盘] {current_date} 初始权益: ${self.daily_start_equity:,.2f}")
            
            # 如果当日已停止交易，跳过
            if daily_stop:
                continue
            
            # 处理事件
            if event['type'] == 'price':
                # 更新价格
                self.current_prices[event['ticker']] = event['price']
                
                # 检查止损止盈（每次价格更新都检查）
                self.check_stop_loss_take_profit(timestamp)
                
            elif event['type'] == 'news':
                # 处理新闻
                signal = self.process_news_event(event['news'], timestamp)
                if signal:
                    self.log_event(
                        f"[新闻] {timestamp:%H:%M} {signal['ticker']} "
                        f"情绪={signal['sentiment_score']:.2f} "
                        f"新颖度={signal['novelty']:.2f} "
                        f"热度={signal['news_heat']:.2f} → {signal['signal']}"
                    )
            
            # 检查日内亏损限制
            current_equity = self.calculate_portfolio_value(timestamp)
            if self.check_daily_loss_limit(current_equity):
                daily_stop = True
        
        # 最后一天收盘
        if current_date:
            # 清仓所有持仓
            if not self.config.allow_overnight:
                self.close_all_positions(all_events[-1]['timestamp'])
            
            final_equity = self.calculate_portfolio_value(all_events[-1]['timestamp'])
            self.equity_curve.append({
                'date': current_date,
                'equity': final_equity
            })
        
        # 计算绩效指标
        result = self.calculate_performance()
        
        logger.info("="*60)
        logger.info("回测完成!")
        logger.info("="*60)
        
        return result
    
    def run_backtest(self) -> BacktestResult:
        """同步运行回测"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.run_backtest_async())
        finally:
            loop.close()
    
    def calculate_performance(self) -> BacktestResult:
        """计算绩效指标"""
        # 获取最终状态
        final_status = self.portfolio_manager.get_portfolio_status()
        trades = self.portfolio_manager.get_trade_history()
        
        # 转换权益曲线为Series
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_series = equity_df.set_index('date')['equity']
        else:
            equity_series = pd.Series([self.config.initial_capital])
        
        # 计算收益率
        total_return = (final_status['total_equity'] - self.config.initial_capital) / self.config.initial_capital
        
        # 计算年化收益
        days = (pd.Timestamp(self.config.end_date) - pd.Timestamp(self.config.start_date)).days
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 计算夏普比率
        if len(equity_series) > 1:
            daily_returns = equity_series.pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 计算最大回撤
        cumulative = (1 + equity_series.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 交易统计
        winning_trades = []
        losing_trades = []
        
        # 分析每笔交易的盈亏
        positions_closed = {}
        for trade in trades:
            if trade['action'] in ['SELL', 'BUY_TO_COVER']:
                # 这是平仓交易
                ticker = trade['ticker']
                if ticker in positions_closed:
                    # 计算这笔交易的盈亏
                    entry_trade = positions_closed[ticker]
                    if trade['action'] == 'SELL':
                        pnl = (trade['price'] - entry_trade['price']) * trade['quantity']
                    else:  # BUY_TO_COVER
                        pnl = (entry_trade['price'] - trade['price']) * trade['quantity']
                    
                    if pnl > 0:
                        winning_trades.append(pnl)
                    else:
                        losing_trades.append(pnl)
                    
                    del positions_closed[ticker]
            else:
                # 开仓交易
                positions_closed[trade['ticker']] = trade
        
        # 计算统计指标
        win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) if (winning_trades or losing_trades) else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=final_status['total_trades'],
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            equity_curve=equity_series,
            positions_history=[],  # 简化
            trades_history=trades,
            signals_history=self.signals_history,
            event_log=self.event_log
        )
    
    def save_results(self, result: BacktestResult, output_dir: str = "output/backtest"):
        """保存回测结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存绩效报告
        report_file = f"{output_dir}/report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 保存权益曲线
        equity_file = f"{output_dir}/equity_curve_{timestamp}.csv"
        result.equity_curve.to_csv(equity_file)
        
        # 保存交易历史
        trades_file = f"{output_dir}/trades_{timestamp}.json"
        with open(trades_file, 'w', encoding='utf-8') as f:
            json.dump(result.trades_history, f, indent=2, default=str)
        
        # 保存事件日志
        log_file = f"{output_dir}/event_log_{timestamp}.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(result.event_log))
        
        logger.info(f"回测结果已保存到: {output_dir}")
        
        return {
            'report': report_file,
            'equity': equity_file,
            'trades': trades_file,
            'log': log_file
        }


def print_backtest_report(result: BacktestResult):
    """打印回测报告"""
    print("\n" + "="*60)
    print("回测绩效报告")
    print("="*60)
    
    print("\n收益指标:")
    print(f"总收益率: {result.total_return:.2%}")
    print(f"年化收益率: {result.annual_return:.2%}")
    print(f"夏普比率: {result.sharpe_ratio:.2f}")
    print(f"最大回撤: {result.max_drawdown:.2%}")
    
    print("\n交易统计:")
    print(f"总交易次数: {result.total_trades}")
    print(f"胜率: {result.win_rate:.2%}")
    print(f"盈利次数: {result.winning_trades}")
    print(f"亏损次数: {result.losing_trades}")
    print(f"平均盈利: ${result.avg_win:.2f}")
    print(f"平均亏损: ${result.avg_loss:.2f}")
    print(f"盈亏比: {result.profit_factor:.2f}")
    
    print("\n信号统计:")
    signal_counts = defaultdict(int)
    for signal in result.signals_history:
        signal_counts[signal['signal']] += 1
    
    print(f"总信号数: {len(result.signals_history)}")
    for signal_type, count in signal_counts.items():
        print(f"{signal_type}: {count}")
    
    print("\n" + "="*60)


# 便捷函数
def create_backtest_config(
    start_date: str,
    end_date: str,
    symbols: List[str],
    **kwargs
) -> BacktestConfig:
    """创建回测配置"""
    return BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        **kwargs
    )


def run_strategy_backtest(config: BacktestConfig) -> BacktestResult:
    """运行策略回测"""
    engine = BacktestEngine(config)
    result = engine.run_backtest()
    
    # 打印报告
    print_backtest_report(result)
    
    # 保存结果
    engine.save_results(result)
    
    return result


# 测试代码
if __name__ == "__main__":
    # 创建测试配置
    test_config = create_backtest_config(
        start_date="2024-09-01",
        end_date="2024-09-30",  # 先测试一个月
        symbols=["AAPL", "MSFT", "GOOGL"],  # 测试3只股票
        initial_capital=100000,
        max_position_percent=0.2,
        pos_thresh=0.7,     # 使用更严格的阈值
        neg_thresh=-0.7,
        novel_thresh=0.6,
        trailing_stop_pct=0.02,  # 2%移动止损
        allow_overnight=False    # 不允许隔夜
    )
    
    # 运行回测
    result = run_strategy_backtest(test_config)
    
    print("\n回测完成!")