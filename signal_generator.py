"""
signal_generator.py - 信号生成模块（FinGPT优化版）
根据因子值判断交易信号，完全集成FinGPT的事件打分机制
支持多股票批量处理、冷却机制、技术指标确认
完全兼容 factor_model_optimized.py 的 EnhancedNewsFactor 数据结构
"""

from __future__ import annotations
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List, Union, Any
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/signal_generator.log', encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建必要的目录
os.makedirs('logs', exist_ok=True)


class SignalType(Enum):
    """交易信号类型枚举"""
    BUY = "BUY"      # 做多/买入
    SELL = "SELL"    # 做空/卖出
    HOLD = "HOLD"    # 观望/无操作


@dataclass
class TradeSignal:
    """交易信号数据结构"""
    ticker: str
    signal: SignalType
    confidence: float  # 置信度 [0, 1]
    timestamp: datetime
    factors: Dict[str, float]  # 原始因子值
    reason: str  # 信号触发原因
    cooldown_blocked: bool = False  # 是否被冷却机制阻止
    technical_confirmed: bool = False  # 技术面是否确认
    news_heat_level: str = "low"  # 新闻热度等级 (low/medium/high)
    event_score: int = 1  # 事件评分 (1-5)
    event_impact: str = "minimal"  # 事件影响等级
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
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


class SignalGenerator:
    """信号生成器类 - FinGPT优化版，集成事件打分机制"""
    
    def __init__(self,
                 pos_thresh: float = 0.7,      # 正面情绪阈值
                 neg_thresh: float = -0.7,     # 负面情绪阈值
                 novel_thresh: float = 0.6,    # 新颖度阈值
                 event_score_threshold: int = 4,  # 事件评分阈值（1-5）
                 cooldown: int = 600,          # 冷却时间（秒）
                 confidence_method: str = 'composite',  # 置信度计算方法
                 use_technical: bool = True,   # 是否使用技术指标
                 min_confidence: float = 0.6,  # 最小置信度阈值
                 require_volume_confirmation: bool = True,  # 是否要求成交量确认
                 min_volume_spike: float = 1.2,  # 最小成交量放大倍数
                 min_news_heat_for_boost: float = 0.6,  # 提升置信度的最低新闻热度
                 use_event_score: bool = True):  # 是否使用事件评分
        """
        初始化信号生成器（FinGPT优化版）
        
        Args:
            pos_thresh: 正面情绪阈值
            neg_thresh: 负面情绪阈值
            novel_thresh: 新颖度阈值
            event_score_threshold: 事件评分阈值，≥此值才考虑交易
            cooldown: 冷却时间（秒）
            confidence_method: 置信度计算方法 ('composite'使用EnhancedNewsFactor的综合置信度)
            use_technical: 是否考虑技术指标
            min_confidence: 最小置信度阈值
            require_volume_confirmation: 是否要求成交量确认
            min_volume_spike: 最小成交量放大倍数
            min_news_heat_for_boost: 提升置信度的最低新闻热度
            use_event_score: 是否使用事件评分作为必要条件
        """
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.novel_thresh = novel_thresh
        self.event_score_threshold = event_score_threshold
        self.cooldown = cooldown
        self.confidence_method = confidence_method
        self.use_technical = use_technical
        self.min_confidence = min_confidence
        self.require_volume_confirmation = require_volume_confirmation
        self.min_volume_spike = min_volume_spike
        self.min_news_heat_for_boost = min_news_heat_for_boost
        self.use_event_score = use_event_score
        
        # 记录最近信号触发时间
        self.last_signal_time: Dict[str, datetime] = {}
        
        # 统计信息
        self.signal_count = defaultdict(int)
        self.cooldown_count = defaultdict(int)
        self.technical_filtered_count = defaultdict(int)
        self.volume_filtered_count = defaultdict(int)
        self.event_filtered_count = defaultdict(int)  # 事件评分过滤统计
        self.confidence_filtered_count = defaultdict(int)
        
        logger.info(
            f"SignalGenerator FinGPT优化版初始化: "
            f"正阈值={pos_thresh}, 负阈值={neg_thresh}, "
            f"事件阈值={event_score_threshold}, 新颖度阈值={novel_thresh}, "
            f"冷却时间={cooldown}秒, 最小置信度={min_confidence}"
        )
    
    def generate_signal_from_enhanced_factor(self, enhanced_factor, 
                                           current_time: Optional[datetime] = None) -> TradeSignal:
        """
        从 EnhancedNewsFactor 对象生成交易信号
        这是与 factor_model_optimized.py 的主要接口
        
        Args:
            enhanced_factor: EnhancedNewsFactor 实例
            current_time: 当前时间，用于冷却机制
            
        Returns:
            TradeSignal对象
        """
        if current_time is None:
            current_time = datetime.now()
        
        ticker = enhanced_factor.ticker
        
        # 检查冷却时间
        cooldown_blocked = self._check_cooldown(ticker, current_time)
        
        # 初始化信号和原因
        signal = SignalType.HOLD
        confidence = 0.0
        reason_parts = []
        technical_confirmed = False
        
        # 提取关键字段
        sentiment_score = enhanced_factor.sentiment_score
        sentiment_label = enhanced_factor.sentiment_label
        event_score = enhanced_factor.event_score
        event_impact = enhanced_factor.event_impact
        novelty = enhanced_factor.novelty
        news_heat = enhanced_factor.news_heat
        is_valid_signal = enhanced_factor.is_valid_signal
        confidence_composite = enhanced_factor.confidence_composite
        
        # 如果因子已标记为无效信号，直接返回HOLD
        if not is_valid_signal:
            reason = f"因子预筛选无效: {enhanced_factor.rationale[:50]}..."
            return TradeSignal(
                ticker=ticker,
                signal=SignalType.HOLD,
                confidence=0.0,
                timestamp=current_time,
                factors=self._extract_factors_dict(enhanced_factor),
                reason=reason,
                cooldown_blocked=False,
                technical_confirmed=False,
                news_heat_level=self._classify_news_heat(news_heat),
                event_score=event_score,
                event_impact=event_impact
            )
        
        if not cooldown_blocked:
            # 1. 事件评分过滤（FinGPT的核心特性）
            if self.use_event_score and event_score < self.event_score_threshold:
                signal = SignalType.HOLD
                reason_parts.append(f"事件评分不足({event_score}<{self.event_score_threshold})")
                self.event_filtered_count[ticker] += 1
            
            # 2. 新颖度过滤
            elif novelty < self.novel_thresh:
                signal = SignalType.HOLD
                reason_parts.append(f"新颖度不足({novelty:.2f}<{self.novel_thresh})")
            
            # 3. 新闻热度过滤
            elif news_heat < 0.2:
                signal = SignalType.HOLD
                reason_parts.append(f"新闻热度过低({news_heat:.2f})")
            
            else:
                # 4. 情绪判断（结合事件重要性）
                if sentiment_score >= self.pos_thresh:
                    signal = SignalType.BUY
                    reason_parts.append(f"强正面情绪({sentiment_score:.2f})")
                    reason_parts.append(f"事件评分={event_score}({event_impact})")
                    
                elif sentiment_score <= self.neg_thresh:
                    signal = SignalType.SELL
                    reason_parts.append(f"强负面情绪({sentiment_score:.2f})")
                    reason_parts.append(f"事件评分={event_score}({event_impact})")
                    
                else:
                    # 对于中等情绪但高事件评分的情况，也可能触发信号
                    if event_score >= 5 and abs(sentiment_score) > 0.5:
                        signal = SignalType.BUY if sentiment_score > 0 else SignalType.SELL
                        reason_parts.append(f"极高事件评分({event_score})触发")
                        reason_parts.append(f"中等{sentiment_label}情绪({sentiment_score:.2f})")
                    else:
                        signal = SignalType.HOLD
                        reason_parts.append(f"情绪强度不足({sentiment_score:.2f})")
                
                # 5. 技术面确认
                if signal != SignalType.HOLD and self.use_technical:
                    tech_confirmed, tech_reason = self._check_technical_confirmation(
                        signal, enhanced_factor
                    )
                    if not tech_confirmed:
                        signal = SignalType.HOLD
                        reason_parts.append(f"技术不确认: {tech_reason}")
                        self.technical_filtered_count[ticker] += 1
                    else:
                        technical_confirmed = True
                        reason_parts.append(f"技术确认: {tech_reason}")
                
                # 6. 成交量确认
                if signal != SignalType.HOLD and self.require_volume_confirmation:
                    volume_spike = enhanced_factor.volume_spike
                    if volume_spike < self.min_volume_spike:
                        signal = SignalType.HOLD
                        reason_parts.append(f"成交量不足({volume_spike:.2f}<{self.min_volume_spike})")
                        self.volume_filtered_count[ticker] += 1
                    else:
                        reason_parts.append(f"成交量确认({volume_spike:.2f}x)")
                
                # 7. 计算最终置信度
                if signal != SignalType.HOLD:
                    if self.confidence_method == 'composite':
                        # 使用EnhancedNewsFactor的综合置信度
                        confidence = confidence_composite
                    else:
                        # 自定义计算
                        confidence = self._calculate_confidence(enhanced_factor, technical_confirmed)
                    
                    # 根据事件评分调整置信度
                    if event_score >= 5:
                        confidence *= 1.2  # 极高事件评分提升20%置信度
                    elif event_score == 4:
                        confidence *= 1.1  # 高事件评分提升10%置信度
                    
                    confidence = np.clip(confidence, 0.0, 1.0)
                    
                    # 8. 最小置信度检查
                    if confidence < self.min_confidence:
                        signal = SignalType.HOLD
                        reason_parts.append(f"置信度不足({confidence:.2f}<{self.min_confidence})")
                        self.confidence_filtered_count[ticker] += 1
                        confidence = 0.0
                    else:
                        # 更新最后信号时间
                        self.last_signal_time[ticker] = current_time
                        self.signal_count[signal.value] += 1
                        reason_parts.append(f"置信度={confidence:.2f}")
        else:
            # 被冷却机制阻止
            reason_parts.append(f"冷却期中(剩余{self._get_cooldown_remaining(ticker, current_time)}秒)")
            self.cooldown_count[ticker] += 1
        
        # 构建原因字符串
        reason = "; ".join(reason_parts) if reason_parts else "无明确信号"
        
        # 构建信号对象
        trade_signal = TradeSignal(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            timestamp=current_time,
            factors=self._extract_factors_dict(enhanced_factor),
            reason=reason,
            cooldown_blocked=cooldown_blocked,
            technical_confirmed=technical_confirmed,
            news_heat_level=self._classify_news_heat(news_heat),
            event_score=event_score,
            event_impact=event_impact
        )
        
        logger.info(
            f"FinGPT信号: {ticker} | 事件={event_score}({event_impact}), "
            f"情绪={sentiment_score:.2f}({sentiment_label}), "
            f"信号={signal.value}(置信度={confidence:.2f}) | {reason}"
        )
        
        return trade_signal
    
    def _extract_factors_dict(self, enhanced_factor) -> Dict[str, Any]:
        """从EnhancedNewsFactor提取因子字典"""
        return {
            'ticker': enhanced_factor.ticker,
            'sentiment_score': enhanced_factor.sentiment_score,
            'sentiment_label': enhanced_factor.sentiment_label,
            'event_score': enhanced_factor.event_score,
            'event_impact': enhanced_factor.event_impact,
            'novelty': enhanced_factor.novelty,
            'news_heat': enhanced_factor.news_heat,
            'uncertainty': enhanced_factor.uncertainty,
            'price_change_5min': enhanced_factor.price_change_5min,
            'price_change_15min': enhanced_factor.price_change_15min,
            'volume_spike': enhanced_factor.volume_spike,
            'volatility_5min': enhanced_factor.volatility_5min,
            'macd_signal': enhanced_factor.macd_signal,
            'rsi_value': enhanced_factor.rsi_value,
            'rsi_signal': enhanced_factor.rsi_signal,
            'bollinger_signal': enhanced_factor.bollinger_signal,
            'confidence_composite': enhanced_factor.confidence_composite,
            'time_decay': enhanced_factor.time_decay
        }
    
    def _check_technical_confirmation(self, signal: SignalType, enhanced_factor) -> Tuple[bool, str]:
        """
        基于EnhancedNewsFactor的技术面确认
        使用factor_model_optimized提供的技术指标
        """
        reasons = []
        confirmed = True
        
        # 提取技术指标
        macd_signal = enhanced_factor.macd_signal
        rsi_value = enhanced_factor.rsi_value
        rsi_signal = enhanced_factor.rsi_signal
        bollinger_signal = enhanced_factor.bollinger_signal
        price_change_5min = enhanced_factor.price_change_5min
        price_change_15min = enhanced_factor.price_change_15min
        volume_spike = enhanced_factor.volume_spike
        
        if signal == SignalType.BUY:
            # 买入信号的技术面确认
            
            # MACD确认
            if macd_signal == 'bearish':
                confirmed = False
                reasons.append("MACD看跌")
            elif macd_signal == 'bullish':
                reasons.append("MACD看涨")
            
            # RSI确认
            if rsi_signal == 'overbought':
                confirmed = False
                reasons.append(f"RSI超买({rsi_value:.1f})")
            elif rsi_signal == 'oversold':
                reasons.append(f"RSI超卖({rsi_value:.1f})适合买入")
            
            # 布林带确认
            if bollinger_signal == 'above_upper':
                reasons.append("突破布林带上轨(谨慎)")
            elif bollinger_signal == 'below_lower':
                reasons.append("触及布林带下轨(反弹机会)")
            
            # 价格动量确认
            if price_change_15min < -0.01:
                confirmed = False
                reasons.append(f"15分钟趋势向下({price_change_15min:.3f})")
            elif price_change_15min > 0.005:
                reasons.append("15分钟趋势向上")
            
            # 特殊情况：高事件评分可以覆盖部分技术不确认
            if not confirmed and enhanced_factor.event_score >= 5:
                confirmed = True
                reasons.append("极高事件评分覆盖技术不确认")
                
        elif signal == SignalType.SELL:
            # 卖出信号的技术面确认
            
            # MACD确认
            if macd_signal == 'bullish':
                confirmed = False
                reasons.append("MACD看涨")
            elif macd_signal == 'bearish':
                reasons.append("MACD看跌")
            
            # RSI确认
            if rsi_signal == 'oversold':
                confirmed = False
                reasons.append(f"RSI超卖({rsi_value:.1f})")
            elif rsi_signal == 'overbought':
                reasons.append(f"RSI超买({rsi_value:.1f})适合卖出")
            
            # 布林带确认
            if bollinger_signal == 'below_lower':
                reasons.append("跌破布林带下轨(谨慎)")
            elif bollinger_signal == 'above_upper':
                reasons.append("触及布林带上轨(回调机会)")
            
            # 价格动量确认
            if price_change_15min > 0.01:
                confirmed = False
                reasons.append(f"15分钟趋势向上({price_change_15min:.3f})")
            elif price_change_15min < -0.005:
                reasons.append("15分钟趋势向下")
            
            # 特殊情况：高事件评分可以覆盖部分技术不确认
            if not confirmed and enhanced_factor.event_score >= 5:
                confirmed = True
                reasons.append("极高事件评分覆盖技术不确认")
        
        reason_text = "; ".join(reasons) if reasons else "技术面中性"
        
        return confirmed, reason_text
    
    def _calculate_confidence(self, enhanced_factor, technical_confirmed: bool) -> float:
        """
        计算置信度（考虑事件评分的影响）
        """
        # 基础因子
        sentiment_score = abs(enhanced_factor.sentiment_score)
        event_score = enhanced_factor.event_score
        novelty = enhanced_factor.novelty
        news_heat = enhanced_factor.news_heat
        uncertainty = enhanced_factor.uncertainty
        time_decay = enhanced_factor.time_decay
        
        # 事件评分权重（1-5分映射到0.4-1.0）
        event_weight = 0.4 + (event_score - 1) * 0.15
        
        # 各组件贡献
        sentiment_contrib = sentiment_score * 0.3
        event_contrib = event_weight * 0.3
        novelty_contrib = novelty * 0.2
        news_heat_contrib = news_heat * 0.1
        technical_contrib = 0.8 if technical_confirmed else 0.5
        technical_contrib *= 0.1
        
        # 综合置信度
        base_confidence = (sentiment_contrib + event_contrib + 
                          novelty_contrib + news_heat_contrib + technical_contrib)
        
        # 不确定性惩罚
        base_confidence *= (1.0 - uncertainty * 0.3)
        
        # 时间衰减
        base_confidence *= (0.8 + 0.2 * time_decay)
        
        return float(np.clip(base_confidence, 0.0, 1.0))
    
    def _classify_news_heat(self, news_heat: float) -> str:
        """分类新闻热度等级"""
        if news_heat >= 0.8:
            return "high"
        elif news_heat >= 0.4:
            return "medium"
        else:
            return "low"
    
    def generate_signal(self, 
                       factors: Union[Dict[str, Any], 'EnhancedNewsFactor'],
                       current_time: Optional[datetime] = None) -> TradeSignal:
        """
        生成交易信号（统一接口）
        
        Args:
            factors: 因子数据，可以是Dict或EnhancedNewsFactor对象
            current_time: 当前时间
            
        Returns:
            TradeSignal对象
        """
        # 检查输入类型
        if hasattr(factors, 'sentiment_score') and hasattr(factors, 'event_score'):
            # EnhancedNewsFactor对象
            return self.generate_signal_from_enhanced_factor(factors, current_time)
        elif isinstance(factors, dict):
            # 字典格式，创建简化的信号
            return self._generate_signal_from_dict(factors, current_time)
        else:
            logger.error(f"不支持的因子类型: {type(factors)}")
            return TradeSignal(
                ticker='UNKNOWN',
                signal=SignalType.HOLD,
                confidence=0.0,
                timestamp=current_time or datetime.now(),
                factors={},
                reason="不支持的因子类型",
                cooldown_blocked=False,
                technical_confirmed=False,
                news_heat_level="low",
                event_score=1,
                event_impact="minimal"
            )
    
    def _generate_signal_from_dict(self, factors: Dict[str, Any], 
                                  current_time: Optional[datetime]) -> TradeSignal:
        """从字典格式生成信号（简化版，主要用于向后兼容）"""
        if current_time is None:
            current_time = datetime.now()
        
        ticker = factors.get('ticker', 'UNKNOWN')
        sentiment_score = factors.get('sentiment_score', 0.0)
        event_score = factors.get('event_score', 1)
        
        # 简单判断
        if event_score >= self.event_score_threshold and sentiment_score >= self.pos_thresh:
            signal = SignalType.BUY
            confidence = 0.7
            reason = f"字典模式: 事件={event_score}, 情绪={sentiment_score:.2f}"
        elif event_score >= self.event_score_threshold and sentiment_score <= self.neg_thresh:
            signal = SignalType.SELL
            confidence = 0.7
            reason = f"字典模式: 事件={event_score}, 情绪={sentiment_score:.2f}"
        else:
            signal = SignalType.HOLD
            confidence = 0.0
            reason = "字典模式: 条件不满足"
        
        return TradeSignal(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            timestamp=current_time,
            factors=factors,
            reason=reason,
            cooldown_blocked=False,
            technical_confirmed=False,
            news_heat_level="medium",
            event_score=event_score,
            event_impact="medium"
        )
    
    def generate_signals_for_universe(self,
                                    factors_list: List['EnhancedNewsFactor'],
                                    current_time: Optional[datetime] = None) -> Dict[str, TradeSignal]:
        """
        为多个股票批量生成信号
        
        Args:
            factors_list: EnhancedNewsFactor对象列表
            current_time: 当前时间
            
        Returns:
            {ticker: TradeSignal} 字典
        """
        if current_time is None:
            current_time = datetime.now()
        
        signals = {}
        
        logger.info(f"批量生成{len(factors_list)}个因子的信号")
        
        for enhanced_factor in factors_list:
            if hasattr(enhanced_factor, 'ticker'):
                signal = self.generate_signal_from_enhanced_factor(enhanced_factor, current_time)
                signals[enhanced_factor.ticker] = signal
        
        # 统计信号分布
        self._log_batch_statistics(signals)
        
        return signals
    
    def _log_batch_statistics(self, signals: Dict[str, TradeSignal]):
        """记录批量处理统计信息"""
        signal_dist = defaultdict(int)
        technical_confirmed_count = 0
        high_event_count = 0
        high_confidence_count = 0
        
        for signal in signals.values():
            signal_dist[signal.signal.value] += 1
            if signal.technical_confirmed:
                technical_confirmed_count += 1
            if signal.event_score >= 4:
                high_event_count += 1
            if signal.confidence >= 0.7:
                high_confidence_count += 1
        
        logger.info(
            f"批量信号统计: "
            f"BUY={signal_dist['BUY']}, "
            f"SELL={signal_dist['SELL']}, "
            f"HOLD={signal_dist['HOLD']}, "
            f"技术确认={technical_confirmed_count}, "
            f"高事件评分={high_event_count}, "
            f"高置信度={high_confidence_count}"
        )
    
    def _check_cooldown(self, ticker: str, current_time: datetime) -> bool:
        """检查是否在冷却期"""
        if ticker not in self.last_signal_time:
            return False
        
        last_time = self.last_signal_time[ticker]
        elapsed = (current_time - last_time).total_seconds()
        
        return elapsed < self.cooldown
    
    def _get_cooldown_remaining(self, ticker: str, current_time: datetime) -> int:
        """获取剩余冷却时间（秒）"""
        if ticker not in self.last_signal_time:
            return 0
        
        last_time = self.last_signal_time[ticker]
        elapsed = (current_time - last_time).total_seconds()
        remaining = max(0, self.cooldown - elapsed)
        
        return int(remaining)
    
    def update_thresholds(self,
                         pos_thresh: Optional[float] = None,
                         neg_thresh: Optional[float] = None,
                         event_score_threshold: Optional[int] = None,
                         min_confidence: Optional[float] = None):
        """动态更新阈值参数"""
        if pos_thresh is not None:
            self.pos_thresh = pos_thresh
            logger.info(f"更新正面情绪阈值: {pos_thresh}")
        
        if neg_thresh is not None:
            self.neg_thresh = neg_thresh
            logger.info(f"更新负面情绪阈值: {neg_thresh}")
        
        if event_score_threshold is not None:
            self.event_score_threshold = event_score_threshold
            logger.info(f"更新事件评分阈值: {event_score_threshold}")
            
        if min_confidence is not None:
            self.min_confidence = min_confidence
            logger.info(f"更新最小置信度阈值: {min_confidence}")
    
    def reset_cooldown(self, ticker: Optional[str] = None):
        """重置冷却时间"""
        if ticker:
            if ticker in self.last_signal_time:
                del self.last_signal_time[ticker]
                logger.info(f"重置{ticker}的冷却时间")
        else:
            self.last_signal_time.clear()
            logger.info("重置所有股票的冷却时间")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取信号生成统计信息"""
        return {
            'signal_counts': dict(self.signal_count),
            'cooldown_blocks': dict(self.cooldown_count),
            'event_filtered': dict(self.event_filtered_count),
            'technical_filtered': dict(self.technical_filtered_count),
            'volume_filtered': dict(self.volume_filtered_count),
            'confidence_filtered': dict(self.confidence_filtered_count),
            'active_cooldowns': len(self.last_signal_time),
            'thresholds': {
                'positive': self.pos_thresh,
                'negative': self.neg_thresh,
                'event_score': self.event_score_threshold,
                'novelty': self.novel_thresh,
                'min_confidence': self.min_confidence
            }
        }


# 测试代码
if __name__ == "__main__":
    print("=== FinGPT信号生成模块测试 ===\n")
    
    # 创建信号生成器
    generator = SignalGenerator(
        pos_thresh=0.7,
        neg_thresh=-0.7,
        event_score_threshold=4,  # 事件评分≥4才考虑交易
        novel_thresh=0.6,
        cooldown=60,  # 测试用短冷却
        min_confidence=0.6,
        use_event_score=True
    )
    
    # 模拟EnhancedNewsFactor对象用于测试
    class MockEnhancedFactor:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    print("测试1: 高事件评分 + 强正面情绪")
    factor1 = MockEnhancedFactor(
        ticker='AAPL',
        sentiment_score=0.85,
        sentiment_label='positive',
        event_score=5,  # 极高事件评分
        event_impact='extreme',
        novelty=0.9,
        news_heat=0.8,
        is_valid_signal=True,
        confidence_composite=0.88,
        uncertainty=0.1,
        time_decay=0.95,
        # 技术指标
        macd_signal='bullish',
        rsi_value=55.0,
        rsi_signal='neutral',
        bollinger_signal='in_band',
        price_change_5min=0.008,
        price_change_15min=0.015,
        volume_spike=2.5,
        volatility_5min=0.02,
        rationale='Major breakthrough announcement'
    )
    
    signal1 = generator.generate_signal_from_enhanced_factor(factor1)
    print(f"结果: {signal1.signal.value}")
    print(f"置信度: {signal1.confidence:.3f}")
    print(f"事件评分: {signal1.event_score} ({signal1.event_impact})")
    print(f"原因: {signal1.reason}\n")
    
    print("测试2: 低事件评分（被过滤）")
    factor2 = MockEnhancedFactor(
        ticker='MSFT',
        sentiment_score=0.75,
        sentiment_label='positive',
        event_score=2,  # 低事件评分
        event_impact='low',
        novelty=0.8,
        news_heat=0.6,
        is_valid_signal=True,
        confidence_composite=0.65,
        uncertainty=0.2,
        time_decay=0.9,
        macd_signal='neutral',
        rsi_value=60.0,
        rsi_signal='neutral',
        bollinger_signal='in_band',
        price_change_5min=0.002,
        price_change_15min=0.005,
        volume_spike=1.1,
        volatility_5min=0.01,
        rationale='Regular earnings update'
    )
    
    signal2 = generator.generate_signal_from_enhanced_factor(factor2)
    print(f"结果: {signal2.signal.value}")
    print(f"置信度: {signal2.confidence:.3f}")
    print(f"原因: {signal2.reason}\n")
    
    print("测试3: 高事件评分 + 强负面情绪 + 技术确认")
    factor3 = MockEnhancedFactor(
        ticker='GOOGL',
        sentiment_score=-0.82,
        sentiment_label='negative',
        event_score=4,
        event_impact='high',
        novelty=0.85,
        news_heat=0.9,
        is_valid_signal=True,
        confidence_composite=0.84,
        uncertainty=0.12,
        time_decay=0.98,
        macd_signal='bearish',
        rsi_value=72.0,
        rsi_signal='overbought',
        bollinger_signal='above_upper',
        price_change_5min=-0.01,
        price_change_15min=-0.02,
        volume_spike=3.0,
        volatility_5min=0.03,
        rationale='Major regulatory investigation announced'
    )
    
    signal3 = generator.generate_signal_from_enhanced_factor(factor3)
    print(f"结果: {signal3.signal.value}")
    print(f"置信度: {signal3.confidence:.3f}")
    print(f"技术确认: {signal3.technical_confirmed}")
    print(f"原因: {signal3.reason}\n")
    
    # 批量测试
    print("测试4: 批量信号生成")
    factors = [factor1, factor2, factor3]
    
    batch_signals = generator.generate_signals_for_universe(factors)
    print("\n批量结果:")
    for ticker, signal in batch_signals.items():
        print(f"{ticker}: {signal.signal.value} "
              f"(事件={signal.event_score}, 置信度={signal.confidence:.3f})")
    
    # 显示统计信息
    print("\n=== 统计信息 ===")
    stats = generator.get_statistics()
    print(f"信号统计: {stats['signal_counts']}")
    print(f"事件过滤: {stats['event_filtered']}")
    print(f"技术过滤: {stats['technical_filtered']}")
    print(f"置信度过滤: {stats['confidence_filtered']}")
    print(f"当前阈值: {stats['thresholds']}")
    
    print("\n测试完成!")
    print("✅ 完全集成FinGPT事件打分机制")
    print("✅ 事件评分作为核心过滤条件")
    print("✅ 技术指标从EnhancedNewsFactor直接获取")
    print("✅ 支持极高事件评分覆盖技术不确认")