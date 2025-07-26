"""
data_collector.py - 数据采集模块（FinGPT增强版）
提供历史行情数据、财务数据获取，以及实时新闻订阅功能
增强：支持universe_builder的映射，优化新闻收集以支持FinGPT分析
"""

from __future__ import annotations
import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any, Union, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import deque, defaultdict
import requests
from dotenv import load_dotenv
import hashlib

# 尝试导入API客户端
try:
    from tiingo import TiingoClient
except ImportError:
    raise ImportError("请安装tiingo库: pip install tiingo")

try:
    import finnhub
except ImportError:
    raise ImportError("请安装finnhub-python库: pip install finnhub-python")

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/data_collector.log', encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建必要的目录
os.makedirs('logs', exist_ok=True)
os.makedirs('cache', exist_ok=True)


@dataclass
class NewsArticle:
    """新闻文章数据结构（增强版）"""
    ticker: str
    datetime: datetime
    headline: str
    summary: str
    source: str
    url: str
    id: Optional[str] = None
    sentiment: Optional[float] = None
    # 新增字段支持FinGPT
    company: Optional[str] = None  # 公司名称
    keywords_matched: Optional[List[str]] = None  # 匹配的关键词
    relevance_score: Optional[float] = None  # 相关性评分
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'ticker': self.ticker,
            'datetime': self.datetime.isoformat() if isinstance(self.datetime, datetime) else self.datetime,
            'headline': self.headline,
            'summary': self.summary,
            'source': self.source,
            'url': self.url,
            'id': self.id,
            'sentiment': self.sentiment,
            'company': self.company,
            'keywords_matched': self.keywords_matched,
            'relevance_score': self.relevance_score
        }


class DataCollector:
    """数据采集器类 - 负责获取历史数据和实时新闻（FinGPT增强版）"""
    
    def __init__(self, company_map: Optional[Dict[str, str]] = None,
                 keywords_map: Optional[Dict[str, List[str]]] = None):
        """
        初始化数据采集器
        
        Args:
            company_map: ticker到公司名称的映射
            keywords_map: ticker到关键词的映射
        """
        # 获取API密钥
        self.tiingo_api_key = os.getenv('TIINGO_API_KEY')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        
        if not self.tiingo_api_key:
            raise ValueError("缺少TIINGO_API_KEY环境变量")
        if not self.finnhub_api_key:
            raise ValueError("缺少FINNHUB_API_KEY环境变量")
        
        # 检查是否是Tiingo付费用户（通过环境变量配置）
        self.tiingo_plan = os.getenv('TIINGO_PLAN', 'free').lower()
        self.is_tiingo_paid = self.tiingo_plan in ['power', 'enterprise', 'paid']
        
        # 初始化Tiingo客户端 - 需要传入配置字典
        tiingo_config = {
            'api_key': self.tiingo_api_key,
            'session': True  # 使用session以提高性能
        }
        self.tiingo_client = TiingoClient(tiingo_config)
        logger.info(f"Tiingo客户端初始化成功 (计划: {self.tiingo_plan})")
        
        # 初始化Finnhub客户端
        self.finnhub_client = finnhub.Client(api_key=self.finnhub_api_key)
        logger.info("Finnhub客户端初始化成功")
        
        # 保存映射（支持FinGPT）
        self.company_map = company_map or {}
        self.keywords_map = keywords_map or {}
        
        # 构建反向映射（公司/关键词 -> ticker）
        self.reverse_map = self._build_reverse_map()
        
        # 缓存配置
        self.cache = {}
        self.cache_ttl = 3600  # 缓存1小时
        
        # 新闻订阅相关
        self.news_thread = None
        self.news_stop_event = threading.Event()
        self.processed_news_ids = deque(maxlen=2000)  # 增大去重缓存
        self.news_polling_interval = 60  # 默认60秒轮询一次
        
        # 新闻统计（用于FinGPT）
        self.news_stats = defaultdict(lambda: {
            'total_count': 0,
            'last_hour_count': 0,
            'sources': set(),
            'last_update': datetime.now()
        })
        
        # 从universe_config导入符号映射（如果可用）
        try:
            from universe_config import get_tiingo_symbol
            self.get_tiingo_symbol = get_tiingo_symbol
        except ImportError:
            # 使用默认映射
            self.get_tiingo_symbol = lambda ticker: ticker.replace('-', '.')
        
        logger.info(f"DataCollector初始化完成 (映射{len(self.company_map)}个公司)")
    
    def _build_reverse_map(self) -> Dict[str, str]:
        """构建公司/关键词到ticker的反向映射"""
        reverse_map = {}
        
        # 从company_map构建
        for ticker, company in self.company_map.items():
            reverse_map[company.lower()] = ticker
        
        # 从keywords_map构建
        for ticker, keywords in self.keywords_map.items():
            for keyword in keywords:
                reverse_map[keyword.lower()] = ticker
        
        return reverse_map
    
    def update_mappings(self, company_map: Dict[str, str], keywords_map: Dict[str, List[str]]):
        """更新映射（用于动态更新标的池）"""
        self.company_map = company_map
        self.keywords_map = keywords_map
        self.reverse_map = self._build_reverse_map()
        logger.info(f"映射已更新: {len(self.company_map)}个公司")
    
    def match_news_to_ticker(self, headline: str, summary: str = "") -> Tuple[Optional[str], List[str], float]:
        """
        将新闻匹配到ticker（增强版）
        
        Args:
            headline: 新闻标题
            summary: 新闻摘要
            
        Returns:
            (ticker, matched_keywords, relevance_score)
        """
        text = f"{headline} {summary}".lower()
        
        # 记录匹配的关键词和分数
        matches = defaultdict(lambda: {'keywords': [], 'score': 0.0})
        
        # 检查每个可能的匹配
        for keyword, ticker in self.reverse_map.items():
            if keyword in text:
                # 计算匹配分数
                # 标题中的匹配权重更高
                if keyword in headline.lower():
                    score = 2.0
                else:
                    score = 1.0
                
                # 完整词匹配得分更高
                if f" {keyword} " in f" {text} ":
                    score *= 1.5
                
                matches[ticker]['keywords'].append(keyword)
                matches[ticker]['score'] += score
        
        if not matches:
            return None, [], 0.0
        
        # 选择得分最高的ticker
        best_ticker = max(matches.keys(), key=lambda t: matches[t]['score'])
        best_match = matches[best_ticker]
        
        # 归一化相关性分数
        relevance_score = min(1.0, best_match['score'] / 5.0)
        
        return best_ticker, best_match['keywords'], relevance_score
    
    def get_historical_data(self, ticker: str, start_date: str, end_date: str, 
                          frequency: str = 'daily') -> pd.DataFrame:
        """
        获取历史行情数据
        
        Args:
            ticker: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            frequency: 数据频率 ('daily', '1Min', '5Min', '15Min', '30Min', '1Hour')
        
        Returns:
            包含OHLCV数据的DataFrame
        """
        # 检查缓存
        cache_key = f"{ticker}_{start_date}_{end_date}_{frequency}"
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                logger.debug(f"使用缓存数据: {cache_key}")
                return cached_data
        
        # 转换为Tiingo格式
        tiingo_ticker = self.get_tiingo_symbol(ticker)
        
        try:
            logger.info(f"获取{ticker}历史数据: {start_date} 至 {end_date}, 频率: {frequency}")
            
            if frequency == 'daily':
                # 日线数据继续使用Tiingo（工作正常）
                data = self.tiingo_client.get_dataframe(
                    tiingo_ticker,
                    startDate=start_date,
                    endDate=end_date
                )
            else:
                # 分钟级数据改用Finnhub（包含成交量）
                logger.info(f"使用Finnhub获取{ticker}的{frequency}数据")
                
                # Finnhub使用resolution参数：1, 5, 15, 30, 60, D, W, M
                resolution_map = {
                    '1Min': '1',
                    '5Min': '5', 
                    '15Min': '15',
                    '30Min': '30',
                    '1Hour': '60',
                    '60Min': '60'
                }
                resolution = resolution_map.get(frequency, '5')
                
                # 转换日期为时间戳
                start_ts = int(pd.Timestamp(start_date).timestamp())
                end_ts = int(pd.Timestamp(end_date).timestamp())
                
                try:
                    # 调用Finnhub获取K线数据
                    candles = self.finnhub_client.stock_candles(
                        ticker,  # Finnhub不需要转换符号格式
                        resolution,
                        start_ts,
                        end_ts
                    )
                    
                    if candles['s'] == 'ok' and len(candles['t']) > 0:
                        # 转换为DataFrame
                        data = pd.DataFrame({
                            'open': candles['o'],
                            'high': candles['h'],
                            'low': candles['l'],
                            'close': candles['c'],
                            'volume': candles['v'],
                            'timestamp': candles['t']
                        })
                        
                        # 转换时间戳为datetime并设置为索引
                        data['date'] = pd.to_datetime(data['timestamp'], unit='s')
                        data.set_index('date', inplace=True)
                        data.drop('timestamp', axis=1, inplace=True)
                        
                        # 按时间排序
                        data.sort_index(inplace=True)
                        
                        logger.info(f"Finnhub返回{len(data)}条{frequency}数据，成交量范围: {data['volume'].min()}-{data['volume'].max()}")
                    else:
                        logger.warning(f"Finnhub返回空数据或错误: {candles.get('s', 'unknown')}")
                        data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                        
                except Exception as e:
                    logger.error(f"Finnhub获取分钟数据失败: {e}")
                    # 如果Finnhub失败，返回空DataFrame
                    data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            
            # 标准化列名
            data.columns = [col.lower() for col in data.columns]
            
            # 确保有基本的OHLCV列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in data.columns:
                    logger.warning(f"缺少列 {col}，填充为0")
                    data[col] = 0
            
            # 缓存数据
            self.cache[cache_key] = (data, time.time())
            
            logger.info(f"成功获取{ticker}的{len(data)}条历史数据")
            return data
            
        except Exception as e:
            logger.error(f"获取{ticker}历史数据失败: {e}")
            # 返回空DataFrame但包含正确的列
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """
        获取股票基本面数据
        
        Args:
            ticker: 股票代码
            
        Returns:
            包含基本面指标的字典
        """
        # 检查缓存
        cache_key = f"fundamentals_{ticker}"
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl * 24:  # 基本面数据缓存更久
                logger.debug(f"使用缓存的基本面数据: {ticker}")
                return cached_data
        
        # 转换为Tiingo格式
        tiingo_ticker = self.get_tiingo_symbol(ticker)
        
        try:
            logger.info(f"获取{ticker}的基本面数据")
            
            # 获取最新的daily fundamentals
            fundamentals = {}
            
            # 获取每日指标（市值、PE等）
            try:
                daily_data = self.tiingo_client.get_fundamentals_daily(
                    tiingo_ticker,
                    startDate=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    endDate=datetime.now().strftime('%Y-%m-%d')
                )
                
                if daily_data and len(daily_data) > 0:
                    # 取最新的一条数据
                    latest = daily_data[-1]
                    fundamentals.update({
                        'marketCap': latest.get('marketCap'),
                        'peRatio': latest.get('peRatio'),
                        'pbRatio': latest.get('pbRatio'),
                        'psRatio': latest.get('psRatio'),
                        'evToEbitda': latest.get('evToEbitda'),
                        'dividendYield': latest.get('dividendYield'),
                        'date': latest.get('date')
                    })
            except Exception as e:
                logger.warning(f"获取{ticker}每日指标失败: {e}")
            
            # 获取财报数据
            try:
                statements = self.tiingo_client.get_fundamentals_statements(
                    tiingo_ticker,
                    startDate=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                    endDate=datetime.now().strftime('%Y-%m-%d')
                )
                
                if statements and len(statements) > 0:
                    # 取最新的财报
                    latest_statement = statements[-1]
                    fundamentals['latestQuarter'] = {
                        'revenue': latest_statement.get('revenues'),
                        'netIncome': latest_statement.get('netIncome'),
                        'eps': latest_statement.get('eps'),
                        'date': latest_statement.get('date')
                    }
            except Exception as e:
                logger.warning(f"获取{ticker}财报数据失败: {e}")
            
            # 缓存数据
            self.cache[cache_key] = (fundamentals, time.time())
            
            logger.info(f"成功获取{ticker}的基本面数据")
            return fundamentals
            
        except Exception as e:
            logger.error(f"获取{ticker}基本面数据失败: {e}")
            return {}
    
    def get_enhanced_news(self, ticker: str, days_back: int = 1) -> List[NewsArticle]:
        """
        获取增强的新闻数据（为FinGPT准备）
        
        Args:
            ticker: 股票代码
            days_back: 获取多少天前的新闻
            
        Returns:
            增强的NewsArticle列表
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        enhanced_news = []
        
        try:
            # 获取公司新闻
            news_list = self.finnhub_client.company_news(
                ticker,
                _from=start_date,
                to=end_date
            )
            
            for news_item in news_list:
                # 生成新闻ID
                news_id = self._generate_news_id(ticker, news_item)
                
                # 匹配关键词和计算相关性
                matched_ticker, keywords, relevance = self.match_news_to_ticker(
                    news_item.get('headline', ''),
                    news_item.get('summary', '')
                )
                
                # 创建增强的NewsArticle
                article = NewsArticle(
                    ticker=ticker,
                    datetime=datetime.fromtimestamp(news_item.get('datetime', 0)),
                    headline=news_item.get('headline', ''),
                    summary=news_item.get('summary', ''),
                    source=news_item.get('source', ''),
                    url=news_item.get('url', ''),
                    id=news_id,
                    company=self.company_map.get(ticker, ticker),
                    keywords_matched=keywords,
                    relevance_score=relevance
                )
                
                enhanced_news.append(article)
            
            # 更新统计
            self._update_news_stats(ticker, len(news_list))
            
            logger.info(f"获取{ticker}的{len(enhanced_news)}条增强新闻")
            
        except Exception as e:
            logger.error(f"获取{ticker}增强新闻失败: {e}")
        
        return enhanced_news
    
    def subscribe_realtime_news(self, ticker_list: List[str], callback: Callable[[NewsArticle], None],
                              use_keywords: bool = True):
        """
        订阅实时新闻（轮询方式，增强版）
        
        Args:
            ticker_list: 股票代码列表
            callback: 新闻回调函数
            use_keywords: 是否使用关键词扩展搜索
        """
        if self.news_thread and self.news_thread.is_alive():
            logger.warning("新闻订阅线程已在运行，先停止现有线程")
            self.stop_news_subscription()
        
        logger.info(f"开始订阅新闻: {', '.join(ticker_list)} (关键词搜索: {use_keywords})")
        
        # 启动新闻轮询线程
        self.news_stop_event.clear()
        self.news_thread = threading.Thread(
            target=self._news_polling_worker_enhanced,
            args=(ticker_list, callback, use_keywords),
            daemon=True
        )
        self.news_thread.start()
        logger.info("新闻订阅线程已启动")
    
    def _news_polling_worker_enhanced(self, ticker_list: List[str], 
                                    callback: Callable, use_keywords: bool):
        """增强的新闻轮询工作线程"""
        while not self.news_stop_event.is_set():
            try:
                # 获取当前时间和一天前的时间
                today = datetime.now().strftime('%Y-%m-%d')
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                
                new_articles_count = 0
                
                # 如果使用关键词，也获取市场新闻
                if use_keywords:
                    try:
                        market_news = self.finnhub_client.general_news('general')
                        
                        for news_item in market_news[:50]:  # 限制数量避免过多
                            # 尝试匹配到ticker
                            headline = news_item.get('headline', '')
                            summary = news_item.get('summary', '')
                            
                            matched_ticker, keywords, relevance = self.match_news_to_ticker(
                                headline, summary
                            )
                            
                            if matched_ticker and matched_ticker in ticker_list:
                                # 生成新闻ID
                                news_id = self._generate_news_id(matched_ticker, news_item)
                                
                                # 检查是否已处理
                                if news_id in self.processed_news_ids:
                                    continue
                                
                                # 标记为已处理
                                self.processed_news_ids.append(news_id)
                                
                                # 创建增强的NewsArticle
                                article = NewsArticle(
                                    ticker=matched_ticker,
                                    datetime=datetime.fromtimestamp(news_item.get('datetime', 0)),
                                    headline=headline,
                                    summary=summary,
                                    source=news_item.get('source', ''),
                                    url=news_item.get('url', ''),
                                    id=news_id,
                                    company=self.company_map.get(matched_ticker, matched_ticker),
                                    keywords_matched=keywords,
                                    relevance_score=relevance
                                )
                                
                                # 调用回调函数
                                try:
                                    callback(article)
                                    new_articles_count += 1
                                except Exception as e:
                                    logger.error(f"新闻回调函数执行失败: {e}")
                        
                    except Exception as e:
                        logger.error(f"获取市场新闻失败: {e}")
                
                # 获取公司特定新闻
                for ticker in ticker_list:
                    try:
                        # 调用Finnhub获取公司新闻
                        news_list = self.finnhub_client.company_news(
                            ticker, 
                            _from=yesterday, 
                            to=today
                        )
                        
                        for news_item in news_list:
                            # 生成新闻ID
                            news_id = self._generate_news_id(ticker, news_item)
                            
                            # 检查是否已处理
                            if news_id in self.processed_news_ids:
                                continue
                            
                            # 标记为已处理
                            self.processed_news_ids.append(news_id)
                            
                            # 创建增强的NewsArticle
                            article = NewsArticle(
                                ticker=ticker,
                                datetime=datetime.fromtimestamp(news_item.get('datetime', 0)),
                                headline=news_item.get('headline', ''),
                                summary=news_item.get('summary', ''),
                                source=news_item.get('source', ''),
                                url=news_item.get('url', ''),
                                id=news_id,
                                company=self.company_map.get(ticker, ticker),
                                keywords_matched=[self.company_map.get(ticker, ticker)],
                                relevance_score=1.0  # 公司特定新闻相关性最高
                            )
                            
                            # 调用回调函数
                            try:
                                callback(article)
                                new_articles_count += 1
                            except Exception as e:
                                logger.error(f"新闻回调函数执行失败: {e}")
                        
                        # 更新统计
                        self._update_news_stats(ticker, len(news_list))
                        
                        # 避免API限频
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"获取{ticker}新闻失败: {e}")
                
                if new_articles_count > 0:
                    logger.info(f"本轮获取到{new_articles_count}条新新闻")
                
            except Exception as e:
                logger.error(f"新闻轮询出错: {e}")
            
            # 等待下一次轮询
            self.news_stop_event.wait(self.news_polling_interval)
    
    def _generate_news_id(self, ticker: str, news_item: Dict) -> str:
        """生成唯一的新闻ID"""
        # 使用URL作为主要标识，如果没有则使用标题的hash
        url = news_item.get('url', '')
        if url:
            return f"{ticker}_{hashlib.md5(url.encode()).hexdigest()[:8]}"
        else:
            headline = news_item.get('headline', '')
            return f"{ticker}_{hashlib.md5(headline.encode()).hexdigest()[:8]}"
    
    def _update_news_stats(self, ticker: str, count: int):
        """更新新闻统计"""
        now = datetime.now()
        stats = self.news_stats[ticker]
        
        # 更新总数
        stats['total_count'] += count
        
        # 清理过期的小时统计
        if (now - stats['last_update']).total_seconds() > 3600:
            stats['last_hour_count'] = count
        else:
            stats['last_hour_count'] += count
        
        stats['last_update'] = now
    
    def get_news_stats(self, ticker: str) -> Dict[str, Any]:
        """获取新闻统计信息"""
        return dict(self.news_stats.get(ticker, {
            'total_count': 0,
            'last_hour_count': 0,
            'sources': set(),
            'last_update': None
        }))
    
    def stop_news_subscription(self):
        """停止新闻订阅"""
        if self.news_thread and self.news_thread.is_alive():
            logger.info("停止新闻订阅...")
            self.news_stop_event.set()
            self.news_thread.join(timeout=5)
            logger.info("新闻订阅已停止")
    
    def get_news_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        获取股票新闻情绪统计
        
        注意：Finnhub免费计划不支持news_sentiment端点
        这里提供一个简化版本，返回最近新闻的统计信息
        真正的情绪分析将在阶段2使用FinGPT实现
        
        Args:
            ticker: 股票代码
            
        Returns:
            包含情绪指标的字典
        """
        try:
            logger.info(f"获取{ticker}的新闻情绪（使用company_news替代）")
            
            # 获取最近2天的新闻
            today = datetime.now().strftime('%Y-%m-%d')
            two_days_ago = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
            
            news_list = self.finnhub_client.company_news(
                ticker,
                _from=two_days_ago,
                to=today
            )
            
            # 计算新闻统计
            news_count = len(news_list)
            
            # 简单的情绪估计（基于新闻数量和时间）
            # 这只是占位符，真实情绪分析在FinGPT实现
            recent_news = [n for n in news_list if 
                          datetime.fromtimestamp(n.get('datetime', 0)) > 
                          datetime.now() - timedelta(hours=24)]
            
            # 返回简化的情绪数据
            result = {
                'ticker': ticker,
                'buzz': {
                    'articlesInLastWeek': news_count,
                    'weeklyAverage': news_count / 7,
                    'articlesInLast24h': len(recent_news)
                },
                'companyNewsScore': 0.5,  # 中性占位符
                'sentimentScore': 0,  # 将在FinGPT计算真实值
                'sentiment': {
                    'bullishPercent': 50,  # 占位符
                    'bearishPercent': 50   # 占位符
                },
                'note': 'Finnhub免费计划不支持news_sentiment，真实情绪分析将在FinGPT实现'
            }
            
            logger.info(f"{ticker}最近48小时有{news_count}条新闻，24小时内{len(recent_news)}条")
            
            return result
            
        except Exception as e:
            logger.error(f"获取{ticker}新闻失败: {e}")
            return {
                'ticker': ticker,
                'sentimentScore': 0,
                'buzz': {'articlesInLastWeek': 0},
                'error': str(e)
            }
    
    def get_market_news(self, category: str = 'general') -> List[Dict]:
        """
        获取市场新闻
        
        Args:
            category: 新闻类别 (general, forex, crypto, merger)
            
        Returns:
            新闻列表
        """
        try:
            news_list = self.finnhub_client.general_news(category)
            return news_list
        except Exception as e:
            logger.error(f"获取市场新闻失败: {e}")
            return []
    
    def get_earnings_calendar(self, from_date: str, to_date: str, ticker: Optional[str] = None) -> List[Dict]:
        """
        获取财报日历
        
        Args:
            from_date: 开始日期
            to_date: 结束日期
            ticker: 可选的特定股票
            
        Returns:
            财报事件列表
        """
        try:
            earnings = self.finnhub_client.earnings_calendar(
                _from=from_date,
                to=to_date,
                symbol=ticker
            )
            return earnings.get('earningsCalendar', [])
        except Exception as e:
            logger.error(f"获取财报日历失败: {e}")
            return []
    
    def get_multiple_stocks_data(self, tickers: List[str], start_date: str, end_date: str,
                               frequency: str = 'daily', max_workers: int = 10) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票的历史数据
        
        Args:
            tickers: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            max_workers: 最大并发数
            
        Returns:
            {ticker: DataFrame} 字典
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        logger.info(f"批量获取{len(tickers)}只股票的{frequency}数据")
        
        results = {}
        failed = []
        
        def fetch_single(ticker):
            try:
                return ticker, self.get_historical_data(ticker, start_date, end_date, frequency)
            except Exception as e:
                logger.error(f"获取{ticker}失败: {e}")
                return ticker, None
        
        # 使用线程池并发获取
        with ThreadPoolExecutor(max_workers=min(max_workers, len(tickers))) as executor:
            futures = [executor.submit(fetch_single, ticker) for ticker in tickers]
            
            for future in as_completed(futures):
                ticker, data = future.result()
                if data is not None and not data.empty:
                    results[ticker] = data
                else:
                    failed.append(ticker)
        
        success_rate = len(results) / len(tickers) * 100 if tickers else 0
        logger.info(f"批量获取完成: 成功{len(results)}/{len(tickers)} ({success_rate:.1f}%)")
        
        if failed:
            logger.warning(f"失败的股票: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")
        
        return results
    
    def get_universe_news_batch(self, tickers: List[str], days_back: int = 1) -> Dict[str, List[NewsArticle]]:
        """
        批量获取多只股票的新闻（增强版）
        
        Args:
            tickers: 股票列表
            days_back: 获取多少天前的新闻
            
        Returns:
            {ticker: [NewsArticle]} 字典
        """
        news_by_ticker = defaultdict(list)
        
        for ticker in tickers:
            enhanced_news = self.get_enhanced_news(ticker, days_back)
            if enhanced_news:
                news_by_ticker[ticker] = enhanced_news
        
        total_news = sum(len(news) for news in news_by_ticker.values())
        logger.info(f"获取{len(news_by_ticker)}只股票共{total_news}条增强新闻")
        
        return dict(news_by_ticker)
    
    def prepare_news_for_fingpt(self, news_articles: List[NewsArticle]) -> List[Dict[str, Any]]:
        """
        准备新闻数据供FinGPT分析
        
        Args:
            news_articles: NewsArticle列表
            
        Returns:
            格式化的新闻数据列表
        """
        fingpt_data = []
        
        for article in news_articles:
            # 构建FinGPT需要的格式
            fingpt_item = {
                'id': article.id,
                'ticker': article.ticker,
                'company': article.company or self.company_map.get(article.ticker, article.ticker),
                'timestamp': article.datetime.isoformat(),
                'title': article.headline,
                'content': f"{article.headline}. {article.summary}",
                'source': article.source,
                'url': article.url,
                'relevance_score': article.relevance_score or 1.0,
                'keywords': article.keywords_matched or [],
                'metadata': {
                    'news_stats': self.get_news_stats(article.ticker)
                }
            }
            
            fingpt_data.append(fingpt_item)
        
        return fingpt_data


# 示例和测试代码
if __name__ == "__main__":
    # 获取universe映射
    try:
        from universe_builder import UniverseBuilder
        builder = UniverseBuilder(use_configured_universe=True)
        _, _, company_map = builder.build_universe()
        keywords_map = {ticker: builder.get_news_keywords_for_ticker(ticker) 
                       for ticker in company_map.keys()}
    except:
        # 使用默认映射
        company_map = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Alphabet",
            "AMZN": "Amazon",
            "META": "Meta Platforms"
        }
        keywords_map = {
            "AAPL": ["Apple", "iPhone", "Tim Cook"],
            "MSFT": ["Microsoft", "Windows", "Azure"],
            "GOOGL": ["Google", "Alphabet", "Search"],
            "AMZN": ["Amazon", "AWS", "E-commerce"],
            "META": ["Meta", "Facebook", "Instagram"]
        }
    
    # 创建数据采集器实例（带映射）
    collector = DataCollector(company_map=company_map, keywords_map=keywords_map)
    
    # 测试1: 获取历史日线数据
    print("\n=== 测试历史日线数据 ===")
    df = collector.get_historical_data(
        "AAPL",
        "2024-01-01",
        "2024-01-31",
        "daily"
    )
    print(f"获取到{len(df)}条数据")
    if not df.empty:
        print(df.head())
    
    # 测试2: 获取分钟级数据
    print("\n=== 测试分钟级数据 ===")
    df_5min = collector.get_historical_data(
        "AAPL",
        "2024-01-29",
        "2024-01-30",
        "5Min"
    )
    print(f"获取到{len(df_5min)}条5分钟数据")
    if not df_5min.empty:
        print(df_5min.head())
    
    # 测试3: 新闻匹配功能
    print("\n=== 测试新闻匹配功能 ===")
    test_headlines = [
        "Apple announces new AI features for iPhone",
        "Microsoft Azure revenue grows 30% in Q4",
        "Tesla reports record deliveries",
        "Federal Reserve hints at rate cut"
    ]
    
    for headline in test_headlines:
        ticker, keywords, score = collector.match_news_to_ticker(headline)
        print(f"'{headline[:50]}...'")
        print(f"  -> Ticker: {ticker}, Keywords: {keywords}, Score: {score:.2f}")
    
    # 测试4: 获取增强新闻
    print("\n=== 测试增强新闻数据 ===")
    enhanced_news = collector.get_enhanced_news("AAPL", days_back=1)
    print(f"获取到{len(enhanced_news)}条增强新闻")
    if enhanced_news:
        news = enhanced_news[0]
        print(f"示例新闻:")
        print(f"  标题: {news.headline[:80]}...")
        print(f"  公司: {news.company}")
        print(f"  关键词: {news.keywords_matched}")
        print(f"  相关性: {news.relevance_score:.2f}")
    
    # 测试5: 准备FinGPT数据
    print("\n=== 测试FinGPT数据准备 ===")
    if enhanced_news:
        fingpt_data = collector.prepare_news_for_fingpt(enhanced_news[:2])
        print(f"准备了{len(fingpt_data)}条FinGPT数据")
        if fingpt_data:
            print("示例FinGPT数据:")
            print(json.dumps(fingpt_data[0], indent=2, default=str))
    
    # 测试6: 订阅实时新闻（增强版）
    print("\n=== 测试实时新闻订阅（增强版） ===")
    
    def enhanced_news_callback(article: NewsArticle):
        """增强的新闻回调函数"""
        print(f"[新闻] {article.ticker} ({article.company}) {article.datetime}")
        print(f"  标题: {article.headline[:80]}...")
        print(f"  相关性: {article.relevance_score:.2f}, 关键词: {article.keywords_matched}")
    
    # 订阅前5只股票
    universe = list(company_map.keys())[:5]
    print(f"订阅股票: {', '.join(universe)}")
    collector.news_polling_interval = 10  # 测试时缩短轮询间隔
    collector.subscribe_realtime_news(universe, enhanced_news_callback, use_keywords=True)
    
    # 等待一段时间观察新闻
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        print("\n用户中断")
    
    # 停止订阅
    collector.stop_news_subscription()
    
    # 显示新闻统计
    print("\n=== 新闻统计 ===")
    for ticker in universe[:3]:
        stats = collector.get_news_stats(ticker)
        if stats['last_update']:
            print(f"{ticker}: 总计{stats['total_count']}条, 最近一小时{stats['last_hour_count']}条")
    
    print("\n测试完成 - DataCollector已准备好支持FinGPT策略!")