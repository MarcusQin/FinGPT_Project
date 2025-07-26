"""
portfolio_manager.py - FinGPT优化版组合管理模块
管理投资组合持仓，根据信号执行交易，支持回测和实盘模式
完全集成signal_generator的事件评分机制

核心增强功能：
1. 根据事件评分动态调整仓位大小
2. 高事件评分交易的特殊风控规则
3. 考虑技术确认和新闻热度的综合决策
4. 支持TradeSignal的所有字段（包括event_score和event_impact）

集成workflow：
factor_model_optimized -> signal_generator -> portfolio_manager -> backtest_engine
"""

from __future__ import annotations
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from enum import Enum

# 尝试导入Alpaca API（实盘模式需要）
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("alpaca-trade-api未安装，实盘功能将不可用")

# 导入信号类型（与signal_generator.py保持一致）
try:
    from signal_generator import SignalType, TradeSignal
    SIGNAL_GENERATOR_AVAILABLE = True
except ImportError:
    # 如果无法导入，定义兼容的类
    class SignalType(Enum):
        BUY = "BUY"
        SELL = "SELL"
        HOLD = "HOLD"
    
    @dataclass
    class TradeSignal:
        """兼容的TradeSignal类（包含event_score和event_impact）"""
        ticker: str
        signal: SignalType
        confidence: float
        timestamp: datetime
        factors: Dict[str, float]
        reason: str
        cooldown_blocked: bool = False
        technical_confirmed: bool = False
        news_heat_level: str = "low"
        event_score: int = 1  # 事件评分 (1-5)
        event_impact: str = "minimal"  # 事件影响等级
        
        def to_dict(self) -> Dict:
            return {
                'ticker': self.ticker,
                'signal': self.signal.value,
                'confidence': self.confidence,
                'timestamp': self.timestamp.isoformat(),
                'factors': self.factors,
                'reason': self.reason,
                'cooldown_blocked': self.cooldown_blocked,
                'technical_confirmed': self.technical_confirmed,
                'news_heat_level': self.news_heat_level,
                'event_score': self.event_score,
                'event_impact': self.event_impact
            }
    
    SIGNAL_GENERATOR_AVAILABLE = False
    logging.warning("signal_generator.py未找到，使用兼容模式")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/portfolio_manager.log', encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建必要的目录
os.makedirs('logs', exist_ok=True)


@dataclass
class RiskConfig:
    """风险控制配置（增强版，考虑事件评分）"""
    stop_loss_pct: float = 0.02           # 2% 固定止损
    take_profit_pct: float = 0.03         # 3% 初始止盈目标
    trailing_stop_pct: float = 0.02       # 2% 移动止损
    max_daily_loss_pct: float = 0.05      # 5% 日内最大亏损
    max_holding_time_min: int = 120       # 120分钟超时止损
    partial_take_profit: bool = True      # 启用分批止盈
    partial_take_ratio: float = 0.5       # 分批止盈比例（卖出50%）
    force_eod_close: bool = True          # 强制日终平仓
    max_positions: int = 5                # 最大同时持仓数
    volatility_adjustment: bool = True    # 启用波动率调整
    # 事件相关风控参数
    high_event_score_stop_loss_pct: float = 0.03  # 高事件评分交易更宽松的止损（3%）
    event_score_position_multiplier: float = 1.5  # 高事件评分仓位乘数


@dataclass
class Position:
    """增强版持仓数据结构（包含事件信息）"""
    ticker: str                           # 股票代码
    quantity: int                         # 持仓数量（正为多头，负为空头）
    avg_price: float                      # 平均成本价
    direction: str                        # "long"/"short"/"flat"
    entry_time: datetime                  # 开仓时间
    current_price: float = 0.0            # 当前价格
    unrealized_pnl: float = 0.0          # 未实现盈亏
    realized_pnl: float = 0.0            # 已实现盈亏
    
    # 移动止损相关字段
    peak_pnl_pct: float = 0.0            # 持仓期间最高盈利百分比
    trailing_stop_price: float = 0.0     # 移动止损价
    is_partial_closed: bool = False       # 是否已经分批止盈
    
    # 新增：事件相关信息
    entry_event_score: int = 1           # 开仓时的事件评分
    entry_event_impact: str = "minimal"  # 开仓时的事件影响
    entry_confidence: float = 0.5        # 开仓时的信号置信度
    entry_reason: str = ""               # 开仓原因
    technical_confirmed: bool = False    # 是否有技术确认
    news_heat_level: str = "low"        # 新闻热度等级
    
    @property
    def market_value(self) -> float:
        """持仓市值"""
        return abs(self.quantity) * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """持仓成本"""
        return abs(self.quantity) * self.avg_price
    
    @property
    def pnl_pct(self) -> float:
        """持仓盈亏百分比"""
        if self.avg_price == 0:
            return 0.0
        if self.quantity > 0:  # 多头
            return (self.current_price - self.avg_price) / self.avg_price
        elif self.quantity < 0:  # 空头
            return (self.avg_price - self.current_price) / self.avg_price
        return 0.0
    
    @property
    def holding_time(self) -> timedelta:
        """持仓时间"""
        return datetime.now() - self.entry_time
    
    def update_price(self, price: float):
        """更新当前价格和相关指标"""
        self.current_price = price
        
        # 更新未实现盈亏
        if self.quantity > 0:  # 多头
            self.unrealized_pnl = (price - self.avg_price) * self.quantity
        elif self.quantity < 0:  # 空头
            self.unrealized_pnl = (self.avg_price - price) * abs(self.quantity)
        else:
            self.unrealized_pnl = 0.0
        
        # 更新移动止损相关指标
        current_pnl_pct = self.pnl_pct
        if current_pnl_pct > self.peak_pnl_pct:
            self.peak_pnl_pct = current_pnl_pct
            # 更新移动止损价
            if self.quantity > 0:  # 多头
                self.trailing_stop_price = price * (1 - 0.02)  # 2%移动止损
            elif self.quantity < 0:  # 空头
                self.trailing_stop_price = price * (1 + 0.02)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'ticker': self.ticker,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'direction': self.direction,
            'entry_time': self.entry_time.isoformat(),
            'current_price': self.current_price,
            'market_value': self.market_value,
            'cost_basis': self.cost_basis,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'pnl_pct': self.pnl_pct,
            'peak_pnl_pct': self.peak_pnl_pct,
            'trailing_stop_price': self.trailing_stop_price,
            'holding_time_min': self.holding_time.total_seconds() / 60,
            'is_partial_closed': self.is_partial_closed,
            # 事件相关信息
            'entry_event_score': self.entry_event_score,
            'entry_event_impact': self.entry_event_impact,
            'entry_confidence': self.entry_confidence,
            'entry_reason': self.entry_reason,
            'technical_confirmed': self.technical_confirmed,
            'news_heat_level': self.news_heat_level
        }


@dataclass
class Trade:
    """交易记录（增强版）"""
    timestamp: datetime
    ticker: str
    action: str  # "BUY"/"SELL"/"BUY_TO_COVER"/"SELL_SHORT"
    quantity: int
    price: float
    commission: float = 0.0
    reason: str = ""
    order_id: Optional[str] = None  # Alpaca订单ID
    # 新增字段
    event_score: int = 1
    event_impact: str = "minimal"
    confidence: float = 0.5
    technical_confirmed: bool = False
    news_heat_level: str = "low"
    
    @property
    def cost(self) -> float:
        """交易金额（含佣金）"""
        return self.quantity * self.price + self.commission
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'ticker': self.ticker,
            'action': self.action,
            'quantity': self.quantity,
            'price': self.price,
            'cost': self.cost,
            'commission': self.commission,
            'reason': self.reason,
            'order_id': self.order_id,
            'event_score': self.event_score,
            'event_impact': self.event_impact,
            'confidence': self.confidence,
            'technical_confirmed': self.technical_confirmed,
            'news_heat_level': self.news_heat_level
        }


class FinGPTPortfolioManager:
    """FinGPT优化版投资组合管理器"""
    
    def __init__(self, 
                 capital: float = 100000.0,
                 max_position_percent: float = 0.2,
                 commission: float = 0.0,
                 live: bool = False,
                 alpaca_api_key: Optional[str] = None,
                 alpaca_secret_key: Optional[str] = None,
                 paper: bool = True,
                 risk_config: Optional[RiskConfig] = None):
        """
        初始化FinGPT优化版组合管理器
        
        Args:
            capital: 初始资金
            max_position_percent: 单股票最大仓位占比
            commission: 交易佣金（每笔固定费用）
            live: 是否实盘模式
            alpaca_api_key: Alpaca API密钥
            alpaca_secret_key: Alpaca密钥
            paper: 是否使用纸面账户
            risk_config: 风险控制配置
        """
        self.initial_capital = capital
        self.cash = capital
        self.max_position_percent = max_position_percent
        self.commission = commission
        self.live = live
        
        # 风险控制配置
        self.risk_config = risk_config or RiskConfig()
        
        # 持仓字典
        self.positions: Dict[str, Position] = {}
        
        # 交易历史
        self.trade_history: List[Trade] = []
        
        # 统计信息（增强版）
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.high_event_trades = 0  # 高事件评分交易数
        self.technical_confirmed_trades = 0  # 技术确认交易数
        
        # 日内风控
        self.daily_start_equity = capital
        self.current_date = None
        self.trading_stopped = False
        
        # 波动率缓存
        self.volatility_cache: Dict[str, float] = {}
        
        # 实盘模式初始化
        if self.live:
            if not ALPACA_AVAILABLE:
                raise ImportError("实盘模式需要安装alpaca-trade-api")
            
            api_key = alpaca_api_key or os.getenv('ALPACA_API_KEY_ID')
            secret_key = alpaca_secret_key or os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not secret_key:
                raise ValueError("实盘模式需要提供Alpaca API密钥")
            
            base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
            self.alpaca = tradeapi.REST(
                api_key,
                secret_key,
                base_url=base_url,
                api_version='v2'
            )
            
            self._sync_with_alpaca()
            logger.info(f"Alpaca实盘模式初始化成功 (纸面账户: {paper})")
        
        logger.info(
            f"FinGPT PortfolioManager初始化: "
            f"资金={capital}, 最大仓位={max_position_percent*100}%, "
            f"模式={'实盘' if live else '回测'}"
        )
    
    def _sync_with_alpaca(self):
        """与Alpaca账户同步（实盘模式）"""
        if not self.live:
            return
        
        try:
            account = self.alpaca.get_account()
            self.cash = float(account.cash)
            
            positions = self.alpaca.list_positions()
            for pos in positions:
                self.positions[pos.symbol] = Position(
                    ticker=pos.symbol,
                    quantity=int(pos.qty) if pos.side == 'long' else -int(pos.qty),
                    avg_price=float(pos.avg_entry_price),
                    direction=pos.side,
                    entry_time=datetime.now(),
                    current_price=float(pos.current_price or pos.avg_entry_price),
                    unrealized_pnl=float(pos.unrealized_pl or 0)
                )
            
            logger.info(f"同步Alpaca账户: 现金={self.cash}, 持仓数={len(self.positions)}")
            
        except Exception as e:
            logger.error(f"同步Alpaca账户失败: {e}")
    
    def get_volatility_factor(self, ticker: str) -> float:
        """获取股票波动率因子"""
        return self.volatility_cache.get(ticker, 1.0)
    
    def set_volatility_factor(self, ticker: str, volatility_factor: float):
        """设置股票波动率因子"""
        self.volatility_cache[ticker] = volatility_factor
    
    def get_current_equity(self) -> float:
        """计算当前总权益"""
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + total_market_value
    
    def check_daily_risk_limit(self) -> bool:
        """检查日内风险限制"""
        if self.trading_stopped:
            return False
        
        current_equity = self.get_current_equity()
        daily_loss_pct = (self.daily_start_equity - current_equity) / self.daily_start_equity
        
        if daily_loss_pct >= self.risk_config.max_daily_loss_pct:
            self.trading_stopped = True
            logger.warning(
                f"触发日内止损！当日亏损{daily_loss_pct:.2%} >= "
                f"限制{self.risk_config.max_daily_loss_pct:.2%}，停止交易"
            )
            return False
        
        return True
    
    def check_position_limit(self) -> bool:
        """检查持仓数量限制"""
        active_positions = sum(1 for pos in self.positions.values() if pos.quantity != 0)
        return active_positions < self.risk_config.max_positions
    
    def calculate_position_size(self, signal: TradeSignal, price: float) -> int:
        """
        计算仓位大小（基于事件评分和置信度）
        
        Args:
            signal: TradeSignal对象（包含事件评分等信息）
            price: 当前价格
            
        Returns:
            目标持仓数量
        """
        # 使用当前权益
        current_equity = self.get_current_equity()
        
        # 基础仓位 = 置信度 * 最大仓位百分比 * 当前权益
        base_value = signal.confidence * self.max_position_percent * current_equity
        
        # 事件评分调整（1-5分映射到0.5-1.5倍）
        event_multiplier = 0.5 + (signal.event_score - 1) * 0.25
        if signal.event_score >= 5:
            # 极高事件评分额外加成
            event_multiplier = self.risk_config.event_score_position_multiplier
        
        # 技术确认调整
        tech_multiplier = 1.1 if signal.technical_confirmed else 0.9
        
        # 新闻热度调整
        heat_multiplier = 1.0
        if signal.news_heat_level == "high":
            heat_multiplier = 1.15
        elif signal.news_heat_level == "medium":
            heat_multiplier = 1.05
        elif signal.news_heat_level == "low":
            heat_multiplier = 0.95
        
        # 波动率调整
        vol_multiplier = 1.0
        if self.risk_config.volatility_adjustment:
            vol_factor = self.get_volatility_factor(signal.ticker)
            if vol_factor > 1.5:
                vol_multiplier = 0.5
                logger.info(f"{signal.ticker}波动率高({vol_factor:.2f})，仓位减半")
            elif vol_factor < 0.5:
                vol_multiplier = 1.2
        
        # 计算最终仓位价值
        target_value = base_value * event_multiplier * tech_multiplier * heat_multiplier * vol_multiplier
        
        # 转换为股数
        target_quantity = int(target_value / price)
        
        logger.debug(
            f"仓位计算: {signal.ticker} "
            f"基础={base_value:.0f}, 事件乘数={event_multiplier:.2f}, "
            f"技术乘数={tech_multiplier:.2f}, 热度乘数={heat_multiplier:.2f}, "
            f"目标={target_quantity}股"
        )
        
        return max(1, target_quantity)
    
    def get_stop_loss_pct(self, position: Position) -> float:
        """获取止损百分比（根据事件评分动态调整）"""
        if position.entry_event_score >= 5:
            # 极高事件评分，使用更宽松的止损
            return self.risk_config.high_event_score_stop_loss_pct
        else:
            return self.risk_config.stop_loss_pct
    
    def check_stop_loss_take_profit(self, timestamp: datetime) -> List[Trade]:
        """检查止损止盈（考虑事件评分）"""
        trades = []
        
        for ticker, position in list(self.positions.items()):
            if position.quantity == 0:
                continue
            
            current_price = position.current_price
            pnl_pct = position.pnl_pct
            
            # 动态止损（基于事件评分）
            stop_loss_pct = self.get_stop_loss_pct(position)
            
            # 固定止损检查
            if pnl_pct <= -stop_loss_pct:
                action = "SELL" if position.quantity > 0 else "BUY_TO_COVER"
                trade = self.execute_trade(
                    ticker, action, abs(position.quantity), 
                    current_price, timestamp, 
                    f"固定止损(事件评分={position.entry_event_score})"
                )
                if trade:
                    trades.append(trade)
                    logger.info(
                        f"[止损] {ticker}亏损{pnl_pct:.2%}，触发止损"
                        f"(阈值={stop_loss_pct:.2%}，事件评分={position.entry_event_score})"
                    )
                continue
            
            # 超时止损检查（高事件评分交易给予更多时间）
            holding_minutes = position.holding_time.total_seconds() / 60
            max_holding_time = self.risk_config.max_holding_time_min
            if position.entry_event_score >= 4:
                max_holding_time *= 1.5  # 高事件评分给予50%更多时间
            
            if holding_minutes > max_holding_time and pnl_pct < 0:
                action = "SELL" if position.quantity > 0 else "BUY_TO_COVER"
                trade = self.execute_trade(
                    ticker, action, abs(position.quantity), 
                    current_price, timestamp, "超时止损"
                )
                if trade:
                    trades.append(trade)
                    logger.info(f"[超时止损] {ticker}持仓{holding_minutes:.0f}分钟无盈利，平仓")
                continue
            
            # 分批止盈检查
            if (self.risk_config.partial_take_profit and 
                not position.is_partial_closed and 
                pnl_pct >= self.risk_config.take_profit_pct):
                
                partial_quantity = max(1, int(abs(position.quantity) * self.risk_config.partial_take_ratio))
                action = "SELL" if position.quantity > 0 else "BUY_TO_COVER"
                
                trade = self.execute_trade(
                    ticker, action, partial_quantity, 
                    current_price, timestamp, "分批止盈"
                )
                if trade:
                    trades.append(trade)
                    if ticker in self.positions:
                        self.positions[ticker].is_partial_closed = True
                        self.positions[ticker].avg_price = current_price
                    logger.info(f"[分批止盈] {ticker}盈利{pnl_pct:.2%}，卖出{partial_quantity}股")
                continue
            
            # 移动止损检查
            if position.peak_pnl_pct > 0.02:
                if position.quantity > 0:  # 多头
                    if current_price <= position.trailing_stop_price:
                        trade = self.execute_trade(
                            ticker, "SELL", abs(position.quantity), 
                            current_price, timestamp, "移动止损"
                        )
                        if trade:
                            trades.append(trade)
                            logger.info(
                                f"[移动止损] {ticker}从峰值{position.peak_pnl_pct:.2%}回落"
                            )
                elif position.quantity < 0:  # 空头
                    if current_price >= position.trailing_stop_price:
                        trade = self.execute_trade(
                            ticker, "BUY_TO_COVER", abs(position.quantity), 
                            current_price, timestamp, "移动止损"
                        )
                        if trade:
                            trades.append(trade)
                            logger.info(
                                f"[移动止损] {ticker}空头从峰值{position.peak_pnl_pct:.2%}回落"
                            )
        
        return trades
    
    def force_eod_close(self, timestamp: datetime) -> List[Trade]:
        """强制日终平仓"""
        if not self.risk_config.force_eod_close:
            return []
        
        trades = []
        positions = self.positions.copy()
        
        for ticker, position in positions.items():
            if position.quantity != 0:
                current_price = position.current_price
                action = "SELL" if position.quantity > 0 else "BUY_TO_COVER"
                
                trade = self.execute_trade(
                    ticker, action, abs(position.quantity), 
                    current_price, timestamp, "日终清仓"
                )
                if trade:
                    trades.append(trade)
                    logger.info(f"[日终清仓] 平掉{ticker}仓位{abs(position.quantity)}股")
        
        return trades
    
    def start_new_trading_day(self, timestamp: datetime):
        """开始新的交易日"""
        current_date = timestamp.date()
        if self.current_date != current_date:
            self.current_date = current_date
            self.daily_start_equity = self.get_current_equity()
            self.trading_stopped = False
            logger.info(f"新交易日开始: {current_date}, 起始权益: ${self.daily_start_equity:.2f}")
    
    def update_position(self, ticker: str, signal: Union[str, SignalType, TradeSignal], 
                       price: float, confidence: float = 1.0,
                       timestamp: Optional[datetime] = None) -> Optional[Trade]:
        """
        根据信号更新持仓（FinGPT优化版）
        
        Args:
            ticker: 股票代码
            signal: 交易信号（优先支持TradeSignal对象）
            price: 当前价格
            confidence: 信号置信度（当signal为TradeSignal时会被覆盖）
            timestamp: 时间戳
            
        Returns:
            执行的交易记录
        """
        if timestamp:
            self.start_new_trading_day(timestamp)
        
        if not self.check_daily_risk_limit():
            return None
        
        # 处理TradeSignal对象（优先）
        if isinstance(signal, TradeSignal):
            # 确保ticker一致
            if signal.ticker != ticker:
                logger.warning(f"ticker不匹配: {ticker} vs {signal.ticker}")
                ticker = signal.ticker
            
            # 提取所有信息
            actual_signal = signal.signal
            confidence = signal.confidence
            signal_reason = signal.reason
            technical_confirmed = signal.technical_confirmed
            news_heat_level = signal.news_heat_level
            event_score = signal.event_score
            event_impact = signal.event_impact
            
            # 如果信号被冷却阻止，直接返回
            if signal.cooldown_blocked:
                logger.debug(f"{ticker}信号被冷却机制阻止")
                return None
                
        else:
            # 兼容旧格式
            if isinstance(signal, SignalType):
                actual_signal = signal
            elif isinstance(signal, str):
                try:
                    actual_signal = SignalType(signal.upper())
                except ValueError:
                    logger.warning(f"无效的信号字符串: {signal}")
                    return None
            else:
                logger.warning(f"不支持的信号类型: {type(signal)}")
                return None
            
            # 使用默认值
            signal_reason = ""
            technical_confirmed = False
            news_heat_level = "low"
            event_score = 1
            event_impact = "minimal"
        
        # 提取信号字符串
        signal_str = actual_signal.value
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # 获取当前持仓
        current_position = self.positions.get(ticker)
        
        # HOLD信号不做任何操作
        if signal_str == "HOLD":
            return None
        
        # 检查持仓数量限制
        if ((current_position is None or current_position.quantity == 0) and 
            not self.check_position_limit()):
            logger.warning(f"持仓数量已达上限{self.risk_config.max_positions}，跳过{ticker}信号")
            return None
        
        # 使用TradeSignal计算目标仓位（如果有）
        if isinstance(signal, TradeSignal):
            target_quantity = self.calculate_position_size(signal, price)
        else:
            # 创建临时TradeSignal对象用于计算
            temp_signal = TradeSignal(
                ticker=ticker,
                signal=actual_signal,
                confidence=confidence,
                timestamp=timestamp,
                factors={},
                reason=signal_reason,
                event_score=event_score,
                event_impact=event_impact,
                technical_confirmed=technical_confirmed,
                news_heat_level=news_heat_level
            )
            target_quantity = self.calculate_position_size(temp_signal, price)
        
        # 确定交易动作和数量
        if signal_str == "BUY":
            if current_position is None or current_position.quantity == 0:
                action = "BUY"
                quantity = target_quantity
            elif current_position.quantity < 0:
                cover_quantity = abs(current_position.quantity)
                action = "BUY_TO_COVER"
                quantity = cover_quantity + target_quantity
            else:
                logger.debug(f"{ticker}已有多头持仓，跳过买入信号")
                return None
                
        elif signal_str == "SELL":
            if current_position is None or current_position.quantity == 0:
                action = "SELL_SHORT"
                quantity = target_quantity
            elif current_position.quantity > 0:
                sell_quantity = current_position.quantity
                action = "SELL"
                quantity = sell_quantity + target_quantity
            else:
                logger.debug(f"{ticker}已有空头持仓，跳过卖出信号")
                return None
        else:
            logger.warning(f"未知信号类型: {signal_str}")
            return None
        
        # 检查资金是否充足
        if action in ["BUY", "BUY_TO_COVER"]:
            required_cash = quantity * price + self.commission
            if required_cash > self.cash:
                affordable_quantity = int((self.cash - self.commission) / price)
                if affordable_quantity <= 0:
                    logger.warning(f"资金不足，无法买入{ticker}")
                    return None
                logger.info(f"资金不足，调整{ticker}买入数量: {quantity} -> {affordable_quantity}")
                quantity = affordable_quantity
        
        # 构建交易原因
        trade_reason = f"事件={event_score}({event_impact})"
        if signal_reason:
            trade_reason += f" | {signal_reason}"
        if technical_confirmed:
            trade_reason += " [技术确认]"
        if news_heat_level != "low":
            trade_reason += f" [热度:{news_heat_level}]"
        
        # 执行交易（包含事件信息）
        trade = self.execute_trade(
            ticker, action, quantity, price, timestamp, trade_reason,
            event_score=event_score,
            event_impact=event_impact,
            confidence=confidence,
            technical_confirmed=technical_confirmed,
            news_heat_level=news_heat_level
        )
        
        if trade:
            # 更新持仓的事件信息
            if ticker in self.positions:
                position = self.positions[ticker]
                position.entry_event_score = event_score
                position.entry_event_impact = event_impact
                position.entry_confidence = confidence
                position.entry_reason = trade_reason
                position.technical_confirmed = technical_confirmed
                position.news_heat_level = news_heat_level
            
            logger.info(
                f"FinGPT信号执行: {ticker} {signal_str} "
                f"(事件={event_score}, 置信度={confidence:.3f}, 技术={technical_confirmed}) -> "
                f"{action} {quantity}股 @ ${price:.2f}"
            )
        
        return trade
    
    def execute_trade(self, ticker: str, action: str, quantity: int, 
                     price: float, timestamp: Optional[datetime] = None,
                     reason: str = "", event_score: int = 1,
                     event_impact: str = "minimal", confidence: float = 0.5,
                     technical_confirmed: bool = False,
                     news_heat_level: str = "low") -> Optional[Trade]:
        """
        执行交易（FinGPT增强版）
        """
        if quantity <= 0:
            logger.warning(f"无效的交易数量: {quantity}")
            return None
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # 创建交易记录
        trade = Trade(
            timestamp=timestamp,
            ticker=ticker,
            action=action,
            quantity=quantity,
            price=price,
            commission=self.commission,
            reason=reason,
            event_score=event_score,
            event_impact=event_impact,
            confidence=confidence,
            technical_confirmed=technical_confirmed,
            news_heat_level=news_heat_level
        )
        
        # 实盘模式下单
        if self.live:
            order_id = self._submit_alpaca_order(ticker, action, quantity)
            if not order_id:
                logger.error(f"Alpaca下单失败: {ticker} {action} {quantity}")
                return None
            trade.order_id = order_id
        
        # 更新持仓和资金
        self._update_portfolio(trade)
        
        # 记录交易
        self.trade_history.append(trade)
        self.total_trades += 1
        
        # 更新统计
        if event_score >= 4:
            self.high_event_trades += 1
        if technical_confirmed:
            self.technical_confirmed_trades += 1
        
        # 记录交易日志
        logger.info(
            f"[交易] {timestamp:%H:%M} {action} {quantity} {ticker} @ ${price:.2f} "
            f"(事件={event_score}, 置信度={confidence:.2f}, {reason})"
        )
        
        return trade
    
    def _submit_alpaca_order(self, ticker: str, action: str, quantity: int) -> Optional[str]:
        """提交Alpaca订单（实盘模式）"""
        if not self.live or not self.alpaca:
            return None
        
        try:
            if action in ["BUY", "BUY_TO_COVER"]:
                side = "buy"
            elif action in ["SELL", "SELL_SHORT"]:
                side = "sell"
            else:
                logger.error(f"未知的交易动作: {action}")
                return None
            
            order = self.alpaca.submit_order(
                symbol=ticker,
                qty=quantity,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            logger.info(f"Alpaca订单已提交: {order.id} - {side} {quantity} {ticker}")
            return order.id
            
        except Exception as e:
            logger.error(f"Alpaca下单失败: {e}")
            return None
    
    def _update_portfolio(self, trade: Trade):
        """更新投资组合（内部方法）"""
        ticker = trade.ticker
        current_position = self.positions.get(ticker)
        
        if trade.action == "BUY":
            if current_position is None:
                # 新建多头持仓
                self.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=trade.quantity,
                    avg_price=trade.price,
                    direction="long",
                    entry_time=trade.timestamp,
                    current_price=trade.price,
                    entry_event_score=trade.event_score,
                    entry_event_impact=trade.event_impact,
                    entry_confidence=trade.confidence,
                    entry_reason=trade.reason,
                    technical_confirmed=trade.technical_confirmed,
                    news_heat_level=trade.news_heat_level
                )
            else:
                # 加仓
                total_cost = (current_position.quantity * current_position.avg_price + 
                            trade.quantity * trade.price)
                total_quantity = current_position.quantity + trade.quantity
                current_position.quantity = total_quantity
                current_position.avg_price = total_cost / total_quantity if total_quantity > 0 else 0
                current_position.direction = "long" if total_quantity > 0 else "flat"
            
            self.cash -= trade.cost
            
        elif trade.action == "SELL":
            if current_position and current_position.quantity > 0:
                # 平多头
                sell_quantity = min(trade.quantity, current_position.quantity)
                
                # 计算已实现盈亏
                realized_pnl = (trade.price - current_position.avg_price) * sell_quantity - trade.commission
                current_position.realized_pnl += realized_pnl
                
                # 更新持仓
                current_position.quantity -= sell_quantity
                if current_position.quantity == 0:
                    del self.positions[ticker]
                    if realized_pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                
                # 如果还有剩余数量，开空
                remaining = trade.quantity - sell_quantity
                if remaining > 0:
                    self.positions[ticker] = Position(
                        ticker=ticker,
                        quantity=-remaining,
                        avg_price=trade.price,
                        direction="short",
                        entry_time=trade.timestamp,
                        current_price=trade.price,
                        entry_event_score=trade.event_score,
                        entry_event_impact=trade.event_impact,
                        entry_confidence=trade.confidence,
                        entry_reason=trade.reason,
                        technical_confirmed=trade.technical_confirmed,
                        news_heat_level=trade.news_heat_level
                    )
            else:
                # 直接开空
                if current_position is None:
                    self.positions[ticker] = Position(
                        ticker=ticker,
                        quantity=-trade.quantity,
                        avg_price=trade.price,
                        direction="short",
                        entry_time=trade.timestamp,
                        current_price=trade.price,
                        entry_event_score=trade.event_score,
                        entry_event_impact=trade.event_impact,
                        entry_confidence=trade.confidence,
                        entry_reason=trade.reason,
                        technical_confirmed=trade.technical_confirmed,
                        news_heat_level=trade.news_heat_level
                    )
                else:
                    # 加空
                    total_cost = (abs(current_position.quantity) * current_position.avg_price + 
                                trade.quantity * trade.price)
                    total_quantity = abs(current_position.quantity) + trade.quantity
                    current_position.quantity = -total_quantity
                    current_position.avg_price = total_cost / total_quantity
            
            self.cash += trade.quantity * trade.price - trade.commission
            
        elif trade.action == "BUY_TO_COVER":
            if current_position and current_position.quantity < 0:
                # 平空头
                cover_quantity = min(trade.quantity, abs(current_position.quantity))
                
                # 计算已实现盈亏
                realized_pnl = (current_position.avg_price - trade.price) * cover_quantity - trade.commission
                current_position.realized_pnl += realized_pnl
                
                # 更新持仓
                current_position.quantity += cover_quantity
                if current_position.quantity == 0:
                    del self.positions[ticker]
                    if realized_pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                
                # 如果还有剩余数量，开多
                remaining = trade.quantity - cover_quantity
                if remaining > 0:
                    self.positions[ticker] = Position(
                        ticker=ticker,
                        quantity=remaining,
                        avg_price=trade.price,
                        direction="long",
                        entry_time=trade.timestamp,
                        current_price=trade.price,
                        entry_event_score=trade.event_score,
                        entry_event_impact=trade.event_impact,
                        entry_confidence=trade.confidence,
                        entry_reason=trade.reason,
                        technical_confirmed=trade.technical_confirmed,
                        news_heat_level=trade.news_heat_level
                    )
            
            self.cash -= trade.cost
    
    def process_signals(self, signals_dict: Dict[str, Any], 
                       prices_dict: Dict[str, float],
                       timestamp: Optional[datetime] = None) -> List[Trade]:
        """
        批量处理信号（FinGPT优化版）
        
        Args:
            signals_dict: 股票信号映射（优先支持TradeSignal对象）
            prices_dict: 股票价格映射
            timestamp: 当前时间
            
        Returns:
            执行的交易列表
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        trades = []
        
        # 首先检查止损止盈
        risk_trades = self.check_stop_loss_take_profit(timestamp)
        trades.extend(risk_trades)
        
        # 统计信号
        signal_stats = {
            'BUY': 0, 'SELL': 0, 'HOLD': 0,
            'technical_confirmed': 0, 'high_confidence': 0,
            'high_event_score': 0, 'extreme_event': 0
        }
        
        # 按事件评分排序处理（优先处理高事件评分）
        sorted_signals = []
        for ticker, signal_data in signals_dict.items():
            if isinstance(signal_data, TradeSignal):
                sorted_signals.append((signal_data.event_score, ticker, signal_data))
            else:
                # 兼容旧格式，使用默认事件评分
                sorted_signals.append((1, ticker, signal_data))
        
        # 按事件评分降序排序
        sorted_signals.sort(key=lambda x: x[0], reverse=True)
        
        # 处理排序后的信号
        for event_score, ticker, signal_data in sorted_signals:
            price = prices_dict.get(ticker)
            if price is None or price <= 0:
                logger.warning(f"无法获取{ticker}的有效价格，跳过信号")
                continue
            
            # 更新持仓价格
            if ticker in self.positions:
                self.positions[ticker].update_price(price)
            
            # 统计信号
            if isinstance(signal_data, TradeSignal):
                signal_stats[signal_data.signal.value] += 1
                if signal_data.technical_confirmed:
                    signal_stats['technical_confirmed'] += 1
                if signal_data.confidence > 0.7:
                    signal_stats['high_confidence'] += 1
                if signal_data.event_score >= 4:
                    signal_stats['high_event_score'] += 1
                if signal_data.event_score == 5:
                    signal_stats['extreme_event'] += 1
            
            # 更新持仓
            trade = self.update_position(ticker, signal_data, price, timestamp=timestamp)
            if trade:
                trades.append(trade)
        
        # 批量处理日志
        if trades or signal_stats['BUY'] + signal_stats['SELL'] > 0:
            logger.info(
                f"批量处理完成: 执行{len(trades)}笔交易 | "
                f"信号: BUY={signal_stats['BUY']}, SELL={signal_stats['SELL']}, HOLD={signal_stats['HOLD']} | "
                f"高事件评分={signal_stats['high_event_score']}, 极高事件={signal_stats['extreme_event']} | "
                f"技术确认={signal_stats['technical_confirmed']}, 高置信度={signal_stats['high_confidence']}"
            )
        
        return trades
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """获取投资组合状态（FinGPT增强版）"""
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_equity = self.cash + total_market_value
        
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        
        daily_return = 0.0
        if self.daily_start_equity > 0:
            daily_return = (total_equity - self.daily_start_equity) / self.daily_start_equity
        
        # 持仓详情（包含事件信息）
        positions_list = []
        high_event_positions = 0
        for ticker, pos in self.positions.items():
            pos_info = pos.to_dict()
            pos_info['weight'] = pos.market_value / total_equity if total_equity > 0 else 0
            positions_list.append(pos_info)
            if pos.entry_event_score >= 4 and pos.quantity != 0:
                high_event_positions += 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cash': self.cash,
            'total_market_value': total_market_value,
            'total_equity': total_equity,
            'initial_capital': self.initial_capital,
            'daily_start_equity': self.daily_start_equity,
            'daily_return': daily_return,
            'total_return': (total_equity - self.initial_capital) / self.initial_capital,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'positions': positions_list,
            'position_count': len([p for p in self.positions.values() if p.quantity != 0]),
            'high_event_positions': high_event_positions,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'high_event_trades': self.high_event_trades,
            'technical_confirmed_trades': self.technical_confirmed_trades,
            'trading_stopped': self.trading_stopped,
            'risk_config': {
                'stop_loss_pct': self.risk_config.stop_loss_pct,
                'take_profit_pct': self.risk_config.take_profit_pct,
                'max_daily_loss_pct': self.risk_config.max_daily_loss_pct,
                'max_positions': self.risk_config.max_positions,
                'force_eod_close': self.risk_config.force_eod_close,
                'event_score_position_multiplier': self.risk_config.event_score_position_multiplier
            }
        }
    
    def get_trade_history(self) -> List[Dict]:
        """获取交易历史"""
        return [trade.to_dict() for trade in self.trade_history]
    
    def update_prices(self, prices_dict: Dict[str, float]):
        """更新持仓价格"""
        for ticker, price in prices_dict.items():
            if ticker in self.positions:
                self.positions[ticker].update_price(price)


# 向后兼容的别名
PortfolioManager = FinGPTPortfolioManager
EnhancedPortfolioManager = FinGPTPortfolioManager


# 测试代码
if __name__ == "__main__":
    print("=== FinGPT投资组合管理器测试 ===\n")
    
    # 创建风险配置
    risk_config = RiskConfig(
        stop_loss_pct=0.02,
        high_event_score_stop_loss_pct=0.03,
        event_score_position_multiplier=1.5,
        max_positions=3
    )
    
    # 创建组合管理器
    portfolio = FinGPTPortfolioManager(
        capital=100000,
        max_position_percent=0.2,
        commission=1.0,
        risk_config=risk_config
    )
    
    print("测试1: 高事件评分交易")
    
    # 创建高事件评分信号
    high_event_signal = TradeSignal(
        ticker="AAPL",
        signal=SignalType.BUY,
        confidence=0.85,
        timestamp=datetime.now(),
        factors={'sentiment_score': 0.8, 'novelty': 0.9},
        reason="强正面情绪(0.80); 事件评分=5(extreme); 技术确认",
        technical_confirmed=True,
        news_heat_level="high",
        event_score=5,  # 极高事件评分
        event_impact="extreme"
    )
    
    trade1 = portfolio.update_position("AAPL", high_event_signal, 150.0)
    if trade1:
        print(f"高事件评分交易: {trade1.action} {trade1.quantity} {trade1.ticker}")
        print(f"事件评分: {trade1.event_score} ({trade1.event_impact})")
        print(f"仓位计算受事件评分影响，使用了{risk_config.event_score_position_multiplier}倍乘数")
    
    print("\n测试2: 批量处理不同事件评分的信号")
    
    signals = {
        "MSFT": TradeSignal(
            ticker="MSFT",
            signal=SignalType.BUY,
            confidence=0.75,
            timestamp=datetime.now(),
            factors={},
            reason="正面情绪; 事件评分=4(high)",
            event_score=4,
            event_impact="high",
            technical_confirmed=True,
            news_heat_level="medium"
        ),
        "GOOGL": TradeSignal(
            ticker="GOOGL",
            signal=SignalType.SELL,
            confidence=0.70,
            timestamp=datetime.now(),
            factors={},
            reason="负面情绪; 事件评分=3(medium)",
            event_score=3,
            event_impact="medium",
            technical_confirmed=False,
            news_heat_level="low"
        )
    }
    
    prices = {"MSFT": 380.0, "GOOGL": 140.0}
    
    trades = portfolio.process_signals(signals, prices)
    print(f"\n批量处理: 执行{len(trades)}笔交易")
    print("注意：高事件评分信号会被优先处理")
    
    # 显示组合状态
    status = portfolio.get_portfolio_status()
    print(f"\n组合状态:")
    print(f"总权益: ${status['total_equity']:.2f}")
    print(f"持仓数: {status['position_count']} (高事件评分持仓: {status['high_event_positions']})")
    print(f"高事件评分交易: {status['high_event_trades']}/{status['total_trades']}")
    
    print("\n✅ FinGPT优化完成:")
    print("- 根据事件评分动态调整仓位大小")
    print("- 高事件评分交易使用更宽松的止损")
    print("- 优先处理高事件评分信号")
    print("- 完整记录和利用所有TradeSignal字段")