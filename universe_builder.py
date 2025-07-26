"""
universe_builder.py - 股票池筛选模块（FinGPT增强版）
支持新闻事件驱动策略，提供ticker到公司名称的映射
"""

from __future__ import annotations
import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import asyncio
import aiohttp
from dotenv import load_dotenv
from types import SimpleNamespace
from collections import defaultdict

# 加载环境变量
load_dotenv()

# 导入配置模块
try:
    from universe_config import (
        UniverseConfig, SYMBOL_MAPPINGS, DELISTED_SYMBOLS, 
        HARD_TO_BORROW_SYMBOLS, get_sp500_symbols, get_nasdaq100_symbols,
        get_tiingo_symbol, is_hard_to_borrow, get_ticker_company_map,
        get_ticker_keywords_map, get_company_to_ticker_map, UNIVERSE_WITH_COMPANIES
    )
    CONFIG_AVAILABLE = True
except ImportError:
    # 如果配置文件不存在，使用内置默认值
    CONFIG_AVAILABLE = False
    UniverseConfig = None
    SYMBOL_MAPPINGS = {}
    DELISTED_SYMBOLS = set()
    HARD_TO_BORROW_SYMBOLS = set()
    get_tiingo_symbol = lambda t: t.replace('-', '.')
    is_hard_to_borrow = lambda t: t in HARD_TO_BORROW_SYMBOLS
    
    # 简化的标的列表
    UNIVERSE_WITH_COMPANIES = [
        {"ticker": "AAPL", "company": "Apple", "keywords": ["Apple Inc", "iPhone"]},
        {"ticker": "MSFT", "company": "Microsoft", "keywords": ["Microsoft", "Windows"]},
        {"ticker": "GOOGL", "company": "Alphabet", "keywords": ["Google", "Alphabet"]},
        {"ticker": "AMZN", "company": "Amazon", "keywords": ["Amazon", "AWS"]},
        {"ticker": "META", "company": "Meta Platforms", "keywords": ["Meta", "Facebook"]},
        {"ticker": "TSLA", "company": "Tesla", "keywords": ["Tesla", "Electric Vehicle"]},
        {"ticker": "NVDA", "company": "NVIDIA", "keywords": ["NVIDIA", "GPU"]},
    ]

# 创建必要的目录
os.makedirs('logs', exist_ok=True)
os.makedirs('cache', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('config', exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/universe_builder.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class StockMetrics:
    """股票筛选指标（增强版，包含新闻覆盖）"""
    ticker: str
    company: str  # 新增：公司名称
    keywords: List[str]  # 新增：新闻关键词
    avg_dollar_volume_20d: float  # 20日平均成交额
    typical_volatility: float  # 典型日内波动率
    easy_to_borrow: bool  # 是否容易借券
    has_recent_halts: bool  # 近期是否有熔断
    premarket_gap: float  # 盘前跳空幅度
    has_earnings_soon: bool  # 近期是否有财报
    has_news_coverage: bool  # 新增：是否有新闻覆盖
    news_source_count: int  # 新增：新闻源数量
    is_tradeable: bool  # 综合判断是否可交易
    reason: str = ""  # 如果不可交易，说明原因
    activity_score: float = 0.0  # 活跃度评分


class UniverseBuilder:
    """股票池构建器（FinGPT增强版）"""
    
    def __init__(self, use_configured_universe: bool = True):
        """
        初始化Universe Builder
        
        Args:
            use_configured_universe: 是否使用配置的标的池（True用于FinGPT策略）
        """
        self.use_configured_universe = use_configured_universe
        
        # API密钥
        self.tiingo_api_key = os.getenv('TIINGO_API_KEY')
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY_ID')
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
        
        if not all([self.tiingo_api_key, self.alpaca_api_key, self.alpaca_secret]):
            logger.warning("缺少部分API密钥，某些功能可能受限")
        
        # 初始化Alpaca客户端（用于获取借券信息）
        try:
            import alpaca_trade_api as tradeapi
            self.alpaca = tradeapi.REST(
                self.alpaca_api_key,
                self.alpaca_secret,
                base_url='https://paper-api.alpaca.markets',
                api_version='v2'
            )
        except ImportError:
            logger.warning("未安装alpaca-trade-api，将跳过借券状态检查")
            self.alpaca = None
        
        # 缓存
        self.cache_file = 'cache/universe_cache.json'
        self._load_cache()
        
        # 加载配置
        if CONFIG_AVAILABLE and UniverseConfig:
            try:
                self.config = UniverseConfig.from_file()
                logger.info("加载配置文件成功")
            except:
                self.config = UniverseConfig()
                logger.info("使用默认配置类")
        else:
            # 使用简化配置
            self.config = SimpleNamespace(
                max_concurrent_requests=25,
                api_retry_count=3,
                api_retry_delay=1.0,
                rolling_window=20,
                min_valid_days=10,
                min_stock_price=1.0,
                min_avg_dollar_volume_percentile=60,
                max_volatility_percentile=90,
                max_premarket_gap=0.08,
                halt_lookback_days=5,
                earnings_lookback_days=2,
                top_active_stocks=60,
                optimal_volatility=0.015,
                volume_score_weight=0.7,
                volatility_score_weight=0.3,
                require_news_coverage=True,
                min_news_sources=1,
                universe_stocks=UNIVERSE_WITH_COMPANIES,
                fallback_min_dollar_volume=50_000_000,
                fallback_max_volatility=0.03,
                fallback_sample_threshold=50,
            )
            logger.info("使用内置默认配置")
        
        # 加载标的映射
        self._load_ticker_mappings()
    
    def _load_ticker_mappings(self):
        """加载ticker到公司名称和关键词的映射"""
        if CONFIG_AVAILABLE:
            self.ticker_company_map = get_ticker_company_map()
            self.ticker_keywords_map = get_ticker_keywords_map()
            self.company_ticker_map = get_company_to_ticker_map()
        else:
            # 从配置生成映射
            self.ticker_company_map = {item['ticker']: item['company'] 
                                     for item in self.config.universe_stocks}
            self.ticker_keywords_map = {item['ticker']: item.get('keywords', [item['company']]) 
                                      for item in self.config.universe_stocks}
            self.company_ticker_map = {}
            for item in self.config.universe_stocks:
                self.company_ticker_map[item['company'].lower()] = item['ticker']
                for keyword in item.get('keywords', []):
                    self.company_ticker_map[keyword.lower()] = item['ticker']
        
        logger.info(f"加载映射完成: {len(self.ticker_company_map)} 个ticker映射")
    
    def _load_cache(self):
        """加载缓存数据"""
        self.cache = {}
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
    
    def _save_cache(self):
        """保存缓存数据"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def get_universe_symbols(self) -> Set[str]:
        """获取候选股票列表"""
        if self.use_configured_universe:
            # 使用配置的标的池（FinGPT模式）
            symbols = {item['ticker'] for item in self.config.universe_stocks}
            logger.info(f"使用配置的标的池: {len(symbols)} 只股票")
            return symbols
        else:
            # 原有逻辑：从S&P500和纳斯达克100筛选
            if CONFIG_AVAILABLE:
                all_symbols = set(get_sp500_symbols() + get_nasdaq100_symbols())
            else:
                all_symbols = {item['ticker'] for item in UNIVERSE_WITH_COMPANIES}
            
            # 移除已退市的股票
            clean_symbols = all_symbols - DELISTED_SYMBOLS
            logger.info(f"候选股票池包含 {len(clean_symbols)} 只股票")
            return clean_symbols
    
    async def check_liquidity_async(self, session: aiohttp.ClientSession, ticker: str) -> Dict:
        """异步检查流动性指标（带重试机制）"""
        if not self.tiingo_api_key:
            return {'ticker': ticker, 'success': False}
            
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        # Tiingo 需要点号格式
        tiingo_ticker = get_tiingo_symbol(ticker)
        
        url = f"https://api.tiingo.com/tiingo/daily/{tiingo_ticker}/prices"
        params = {
            'startDate': start_date.isoformat(),
            'endDate': end_date.isoformat(),
            'token': self.tiingo_api_key
        }
        
        # 重试机制
        for attempt in range(self.config.api_retry_count):
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if len(data) >= 20:
                            df = pd.DataFrame(data)
                            
                            # 过滤脏数据
                            df = df[(df['high'] > 0) & (df['low'] > 0) & (df['volume'] > 0)]
                            df = df[(df['high'] >= df['low'])]
                            df = df[df['close'] > self.config.min_stock_price]
                            
                            if len(df) < self.config.min_valid_days:
                                logger.warning(f"{ticker}: 有效数据不足（仅{len(df)}天）")
                                return {'ticker': ticker, 'success': False}
                            
                            # 计算20日平均成交额
                            df['dollarVolume'] = df['volume'] * (df['high'] + df['low']) / 2
                            avg_dollar_volume = df['dollarVolume'].tail(self.config.rolling_window).mean()
                            
                            # 计算真实波动率
                            tr1 = (df['high'] - df['low']) / df['close'].shift(1)
                            tr2 = ((df['high'] - df['close'].shift(1)).abs()) / df['close'].shift(1)
                            tr3 = ((df['low']  - df['close'].shift(1)).abs()) / df['close'].shift(1)

                            df['trueRange'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                            
                            typical_volatility = (
                                df['trueRange']
                                .rolling(self.config.rolling_window, min_periods=self.config.min_valid_days)
                                .median()
                                .iloc[-1]
                            )
                            
                            if pd.isna(typical_volatility):
                                simple_vol = ((df['high'] - df['low']) / df['close']).tail(self.config.rolling_window).median()
                                typical_volatility = simple_vol if not pd.isna(simple_vol) else 0.02
                            
                            return {
                                'ticker': ticker,
                                'avg_dollar_volume_20d': avg_dollar_volume,
                                'typical_volatility': typical_volatility,
                                'success': True
                            }
                    elif response.status == 404:
                        logger.warning(f"{ticker}: 股票不存在或已退市")
                        return {'ticker': ticker, 'success': False}
                    else:
                        if attempt < self.config.api_retry_count - 1:
                            await asyncio.sleep(self.config.api_retry_delay)
                            continue
                        else:
                            logger.error(f"获取{ticker}数据失败，状态码: {response.status}")
                            return {'ticker': ticker, 'success': False}
            except Exception as e:
                if attempt < self.config.api_retry_count - 1:
                    logger.warning(f"获取{ticker}数据失败 (尝试{attempt+1}/{self.config.api_retry_count}): {e}")
                    await asyncio.sleep(self.config.api_retry_delay)
                    continue
                else:
                    logger.error(f"获取{ticker}流动性数据失败: {e}")
        
        return {'ticker': ticker, 'success': False}
    
    def check_borrow_status(self, ticker: str) -> Tuple[bool, Optional[float]]:
        """检查股票借券状态"""
        if is_hard_to_borrow(ticker):
            logger.info(f"{ticker} 在禁止做空列表中，标记为不可借券")
            return False, None
        
        if not self.alpaca:
            return True, None
        
        # 尝试两种格式：原始格式（横杠）和点号格式
        for symbol in (ticker, ticker.replace('-', '.')):
            try:
                asset = self.alpaca.get_asset(symbol)
                easy_to_borrow = asset.easy_to_borrow
                
                shortable_shares = None
                try:
                    if hasattr(asset, 'shortable_shares'):
                        shortable_shares = asset.shortable_shares
                except:
                    pass
                
                if symbol != ticker:
                    logger.debug(f"{ticker} -> {symbol} (Alpaca格式)")
                
                return easy_to_borrow, shortable_shares
                
            except Exception:
                continue
        
        logger.warning(f"{ticker} 没找到借券信息")
        return False, None
    
    async def check_news_coverage_async(self, session: aiohttp.ClientSession, ticker: str) -> Tuple[bool, int]:
        """
        异步检查新闻覆盖情况
        
        Returns:
            (has_coverage, source_count)
        """
        # 这里简化处理，实际应调用新闻API检查
        # 对于FinGPT策略，假设配置的标的都有新闻覆盖
        if ticker in self.ticker_company_map:
            # 基于关键词数量估算新闻源
            keywords = self.ticker_keywords_map.get(ticker, [])
            source_count = max(1, min(len(keywords), 5))  # 假设每个关键词对应一个新闻源，最多5个
            return True, source_count
        return False, 0
    
    async def check_halt_history_async(self, session: aiohttp.ClientSession, ticker: str) -> bool:
        """异步检查近期熔断历史"""
        # 简化处理
        return False
    
    async def check_premarket_gap_async(self, session: aiohttp.ClientSession, ticker: str) -> float:
        """异步检查盘前跳空"""
        # 简化处理
        return 0.0
    
    async def check_earnings_calendar_async(self, session: aiohttp.ClientSession, ticker: str) -> bool:
        """异步检查财报日程"""
        # 简化处理
        return False
    
    async def evaluate_stock_async(self, session: aiohttp.ClientSession, ticker: str) -> StockMetrics:
        """异步评估单只股票（增强版）"""
        # 获取公司信息
        company = self.ticker_company_map.get(ticker, ticker)
        keywords = self.ticker_keywords_map.get(ticker, [company])
        
        # 并发获取各项指标
        tasks = [
            self.check_liquidity_async(session, ticker),
            self.check_halt_history_async(session, ticker),
            self.check_premarket_gap_async(session, ticker),
            self.check_earnings_calendar_async(session, ticker),
            self.check_news_coverage_async(session, ticker)  # 新增
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 解析结果
        liquidity_data = results[0] if not isinstance(results[0], Exception) else {'success': False}
        has_halts = results[1] if not isinstance(results[1], Exception) else False
        premarket_gap = results[2] if not isinstance(results[2], Exception) else 0.0
        has_earnings = results[3] if not isinstance(results[3], Exception) else False
        news_coverage_result = results[4] if not isinstance(results[4], Exception) else (False, 0)
        
        has_news_coverage, news_source_count = news_coverage_result
        
        # 同步检查借券状态
        easy_to_borrow, shortable_shares = self.check_borrow_status(ticker)
        
        # 构建指标对象
        metrics = StockMetrics(
            ticker=ticker,
            company=company,
            keywords=keywords,
            avg_dollar_volume_20d=liquidity_data.get('avg_dollar_volume_20d', 0),
            typical_volatility=liquidity_data.get('typical_volatility', 1.0),
            easy_to_borrow=easy_to_borrow,
            has_recent_halts=has_halts,
            premarket_gap=abs(premarket_gap),
            has_earnings_soon=has_earnings,
            has_news_coverage=has_news_coverage,
            news_source_count=news_source_count,
            is_tradeable=False,  # 将在主函数中判断
            reason=""
        )
        
        if shortable_shares is not None:
            metrics.activity_score = shortable_shares
        
        return metrics
    
    async def _calculate_thresholds(self, metrics: List[StockMetrics]) -> Dict[str, float]:
        """计算筛选阈值"""
        valid_metrics = [m for m in metrics if m.avg_dollar_volume_20d > 0 and m.typical_volatility < 1.0]
        
        # 检查样本数量是否充足
        if len(valid_metrics) < self.config.fallback_sample_threshold:
            # 样本不足，使用固定阈值
            volume_threshold = self.config.fallback_min_dollar_volume
            volatility_threshold = self.config.fallback_max_volatility
            logger.warning(
                f"样本数量不足 ({len(valid_metrics)} < {self.config.fallback_sample_threshold})，"
                f"使用固定阈值: 成交额 ${volume_threshold/1e6:.1f}M, "
                f"波动率 {volatility_threshold:.1%}"
            )
        else:
            # 样本充足，使用百分位数
            volumes = [m.avg_dollar_volume_20d for m in valid_metrics]
            volatilities = [m.typical_volatility for m in valid_metrics]
            
            volume_threshold = np.percentile(volumes, 100 - self.config.min_avg_dollar_volume_percentile)
            volatility_threshold = np.percentile(volatilities, self.config.max_volatility_percentile)
            
            logger.info(
                f"使用百分位阈值 (样本数: {len(valid_metrics)}): "
                f"成交额 ${volume_threshold/1e6:.1f}M ({self.config.min_avg_dollar_volume_percentile}%分位), "
                f"波动率 {volatility_threshold:.1%} ({self.config.max_volatility_percentile}%分位)"
            )
        
        return {
            'min_dollar_volume': volume_threshold,
            'max_volatility': volatility_threshold
        }
    
    async def build_universe_async(self) -> Tuple[List[str], List[StockMetrics], Dict[str, str]]:
        """
        异步构建交易股票池（增强版）
        
        Returns:
            (tradeable_symbols, all_metrics, company_map)
        """
        logger.info("开始构建交易股票池（FinGPT增强版）...")
        
        # 获取候选股票
        candidates = self.get_universe_symbols()
        
        # 创建异步会话
        async with aiohttp.ClientSession() as session:
            # 动态调整并发数
            max_concurrent = min(self.config.max_concurrent_requests, max(5, len(candidates) // 10))
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def evaluate_with_limit(ticker):
                async with semaphore:
                    return await self.evaluate_stock_async(session, ticker)
            
            # 并发评估所有股票
            logger.info(f"开始并发评估 {len(candidates)} 只股票（最大并发数: {max_concurrent}）")
            tasks = [evaluate_with_limit(ticker) for ticker in candidates]
            all_metrics = await asyncio.gather(*tasks)
        
        # 计算动态阈值
        thresholds = await self._calculate_thresholds(all_metrics)
        volume_threshold = thresholds['min_dollar_volume']
        volatility_threshold = thresholds['max_volatility']
        
        # 重新评估股票（应用动态阈值）
        tradeable_stocks = []
        all_results = []
        
        # 统计各种排除原因
        exclude_reasons = defaultdict(int)
        
        for metrics in all_metrics:
            # 重新判断是否可交易
            reasons = []
            
            # 流动性检查
            if not metrics.avg_dollar_volume_20d > 0 or not metrics.typical_volatility < 1.0:
                reasons.append("无法获取流动性数据")
                exclude_reasons['no_data'] += 1
            else:
                if self.use_configured_universe:
                    # FinGPT模式下放宽流动性要求
                    if metrics.avg_dollar_volume_20d < volume_threshold * 0.5:  # 放宽50%
                        reasons.append(f"成交额不足({metrics.avg_dollar_volume_20d/1e6:.1f}M<{volume_threshold*0.5/1e6:.1f}M)")
                        exclude_reasons['low_volume'] += 1
                else:
                    # 原有严格要求
                    if metrics.avg_dollar_volume_20d < volume_threshold:
                        reasons.append(f"成交额不足({metrics.avg_dollar_volume_20d/1e6:.1f}M<{volume_threshold/1e6:.1f}M)")
                        exclude_reasons['low_volume'] += 1
                
                if metrics.typical_volatility > volatility_threshold:
                    reasons.append(f"日内波动过大({metrics.typical_volatility:.1%}>{volatility_threshold:.1%})")
                    exclude_reasons['high_volatility'] += 1
            
            # 新闻覆盖检查（FinGPT模式必需）
            if self.config.require_news_coverage and self.use_configured_universe:
                if not metrics.has_news_coverage:
                    reasons.append("无新闻覆盖")
                    exclude_reasons['no_news'] += 1
                elif metrics.news_source_count < self.config.min_news_sources:
                    reasons.append(f"新闻源不足({metrics.news_source_count}<{self.config.min_news_sources})")
                    exclude_reasons['insufficient_news'] += 1
            
            # 借券检查
            if not metrics.easy_to_borrow:
                if is_hard_to_borrow(metrics.ticker):
                    reasons.append("禁止做空（策略限制）")
                    exclude_reasons['short_restricted'] += 1
                else:
                    reasons.append("难以借券")
                    exclude_reasons['hard_to_borrow'] += 1
            
            # 其他检查
            if metrics.has_recent_halts:
                reasons.append("近期有熔断")
                exclude_reasons['recent_halts'] += 1
            
            if metrics.premarket_gap > self.config.max_premarket_gap:
                reasons.append(f"盘前跳空过大({metrics.premarket_gap:.1%}>{self.config.max_premarket_gap:.1%})")
                exclude_reasons['large_gap'] += 1
            
            if metrics.has_earnings_soon:
                reasons.append("即将发布财报")
                exclude_reasons['earnings_soon'] += 1
            
            metrics.is_tradeable = len(reasons) == 0
            metrics.reason = "; ".join(reasons) if reasons else "通过所有检查"
            
            all_results.append(metrics)
            if metrics.is_tradeable:
                tradeable_stocks.append(metrics)
        
        # 记录排除原因统计
        if exclude_reasons:
            logger.info("股票排除原因统计:")
            for reason, count in sorted(exclude_reasons.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {reason}: {count} 只")
        
        # 如果可交易股票过多，按综合评分排序选取TOP N
        if len(tradeable_stocks) > self.config.top_active_stocks:
            logger.info(f"可交易股票数({len(tradeable_stocks)})超过目标，进行活跃度排序...")
            
            # 计算综合评分
            for stock in tradeable_stocks:
                volume_score = np.log(stock.avg_dollar_volume_20d / 1e6) if stock.avg_dollar_volume_20d > 0 else 0
                optimal_vol = self.config.optimal_volatility
                volatility_score = 1.0 - abs(stock.typical_volatility - optimal_vol) / optimal_vol
                # FinGPT模式下增加新闻覆盖度权重
                news_score = min(stock.news_source_count / 5.0, 1.0) if self.use_configured_universe else 0
                
                stock.activity_score = (
                    volume_score * self.config.volume_score_weight + 
                    volatility_score * self.config.volatility_score_weight +
                    news_score * 0.2  # 新闻覆盖占20%权重
                )
            
            # 按活跃度评分排序，选取前N只
            tradeable_stocks.sort(key=lambda x: x.activity_score, reverse=True)
            selected_stocks = tradeable_stocks[:self.config.top_active_stocks]
            
            # 更新结果
            selected_tickers = {s.ticker for s in selected_stocks}
            for metrics in all_results:
                if metrics.is_tradeable and metrics.ticker not in selected_tickers:
                    metrics.is_tradeable = False
                    metrics.reason = f"活跃度排名未进前{self.config.top_active_stocks}"
            
            tradeable_symbols = [s.ticker for s in selected_stocks]
        else:
            tradeable_symbols = [s.ticker for s in tradeable_stocks]
        
        # 构建company_map（关键输出）
        company_map = {
            ticker: self.ticker_company_map.get(ticker, ticker)
            for ticker in tradeable_symbols
        }
        
        logger.info(f"筛选完成: {len(tradeable_symbols)}/{len(candidates)} 只股票可交易")
        
        # 特别提示禁止做空的股票
        restricted_in_universe = [s for s in tradeable_symbols if is_hard_to_borrow(s)]
        if restricted_in_universe:
            logger.info(f"注意: 以下股票仅可做多，禁止做空: {', '.join(restricted_in_universe)}")
        
        return tradeable_symbols, all_results, company_map
    
    def build_universe(self) -> Tuple[List[str], pd.DataFrame, Dict[str, str]]:
        """
        构建交易股票池（同步接口，FinGPT增强版）
        
        Returns:
            (tradeable_symbols, results_df, company_map)
        """
        # 运行异步方法
        tradeable_symbols, all_results, company_map = asyncio.run(self.build_universe_async())
        
        # 转换为DataFrame便于分析
        df_results = pd.DataFrame([asdict(m) for m in all_results])
        
        # 保存结果（包含映射信息）
        self.save_results(tradeable_symbols, df_results, company_map)
        
        return tradeable_symbols, df_results, company_map
    
    def save_results(self, symbols: List[str], df_results: pd.DataFrame, company_map: Dict[str, str]):
        """保存筛选结果（增强版）"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建输出目录
        os.makedirs('output', exist_ok=True)
        
        # 标记仅可做多的股票
        symbols_with_restrictions = []
        for symbol in symbols:
            if is_hard_to_borrow(symbol):
                symbols_with_restrictions.append(f"{symbol}*")
            else:
                symbols_with_restrictions.append(symbol)
        
        # 保存可交易股票列表（包含公司映射）
        output_file = f'output/universe_today.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'symbols': symbols,
                'symbols_with_restrictions': symbols_with_restrictions,
                'count': len(symbols),
                'short_restricted': list(HARD_TO_BORROW_SYMBOLS & set(symbols)),
                'company_map': company_map,  # 新增：保存公司映射
                'keywords_map': {ticker: self.ticker_keywords_map.get(ticker, []) 
                               for ticker in symbols}  # 新增：保存关键词映射
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"已保存交易股票列表到: {output_file}")
        
        # 保存详细结果
        detail_file = f'output/universe_detail_{timestamp}.csv'
        df_results.to_csv(detail_file, index=False, encoding='utf-8-sig')
        logger.info(f"已保存详细筛选结果到: {detail_file}")
        
        # 更新缓存
        self.cache['last_update'] = timestamp
        self.cache['symbols'] = symbols
        self.cache['company_map'] = company_map
        self._save_cache()
    
    def load_today_universe(self) -> Tuple[List[str], Dict[str, str]]:
        """
        加载今日交易股票池（增强版）
        
        Returns:
            (symbols, company_map)
        """
        output_file = 'output/universe_today.json'
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"加载交易股票池: {data['count']} 只股票 (更新时间: {data['timestamp']})")
                if 'short_restricted' in data and data['short_restricted']:
                    logger.info(f"禁止做空的股票: {', '.join(data['short_restricted'])}")
                
                # 返回symbols和company_map
                symbols = data['symbols']
                company_map = data.get('company_map', {})
                
                # 如果旧文件没有company_map，从配置重建
                if not company_map:
                    company_map = {ticker: self.ticker_company_map.get(ticker, ticker) 
                                 for ticker in symbols}
                
                return symbols, company_map
        else:
            logger.warning("未找到今日交易股票池文件，将使用默认股票列表")
            default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
            default_map = {ticker: self.ticker_company_map.get(ticker, ticker) 
                         for ticker in default_symbols}
            return default_symbols, default_map
    
    def get_news_keywords_for_ticker(self, ticker: str) -> List[str]:
        """获取指定ticker的新闻关键词"""
        return self.ticker_keywords_map.get(ticker, [self.ticker_company_map.get(ticker, ticker)])
    
    def match_news_to_ticker(self, news_text: str) -> Optional[str]:
        """
        根据新闻内容匹配到对应的ticker
        
        Args:
            news_text: 新闻文本（标题或内容）
            
        Returns:
            匹配到的ticker，如果没有匹配返回None
        """
        news_lower = news_text.lower()
        
        # 在company_ticker_map中查找匹配
        for keyword, ticker in self.company_ticker_map.items():
            if keyword in news_lower:
                return ticker
        
        return None


def main():
    """主函数：构建今日交易股票池（FinGPT增强版）"""
    try:
        # 构建股票池（使用配置的标的）
        builder = UniverseBuilder(use_configured_universe=True)
        symbols, results_df, company_map = builder.build_universe()
        
        # 打印统计信息
        print("\n=== 股票池筛选完成（FinGPT增强版） ===")
        print(f"可交易股票数: {len(symbols)}")
        print(f"总评估股票数: {len(results_df)}")
        print(f"通过率: {len(symbols)/len(results_df)*100:.1f}%")
        
        # 显示公司映射
        print("\n=== Ticker到公司映射 ===")
        for ticker, company in list(company_map.items())[:10]:
            keywords = builder.get_news_keywords_for_ticker(ticker)
            print(f"{ticker}: {company}")
            print(f"  关键词: {', '.join(keywords[:3])}")
        if len(company_map) > 10:
            print(f"... 还有 {len(company_map) - 10} 个映射")
        
        # 统计禁止做空的股票
        short_restricted = [s for s in symbols if is_hard_to_borrow(s)]
        if short_restricted:
            print(f"\n禁止做空的股票 ({len(short_restricted)}只): {', '.join(short_restricted)}")
        
        # 打印不可交易原因统计
        if len(results_df) > len(symbols):
            print("\n=== 排除原因统计 ===")
            excluded_df = results_df[~results_df['is_tradeable']]
            for reason in excluded_df['reason'].value_counts().head(10).items():
                print(f"{reason[0]}: {reason[1]} 只")
        
        # 测试新闻匹配功能
        print("\n=== 新闻匹配测试 ===")
        test_news = [
            "Apple announces new iPhone with AI features",
            "Tesla reports record deliveries in Q4",
            "Microsoft Azure revenue grows 30%"
        ]
        for news in test_news:
            matched_ticker = builder.match_news_to_ticker(news)
            print(f"'{news[:50]}...' -> {matched_ticker}")
        
        return symbols, company_map
        
    except Exception as e:
        logger.error(f"构建股票池失败: {e}")
        raise


if __name__ == "__main__":
    # 运行主程序
    symbols, company_map = main()
    print(f"\n今日可交易股票: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
    print(f"Company mappings available for FinGPT news analysis")