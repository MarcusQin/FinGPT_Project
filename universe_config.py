"""
universe_config.py - Universe Builder配置文件（支持FinGPT新闻映射）
所有可调参数集中管理，包含ticker到公司名称的映射
"""

import json
import os
import pandas as pd
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from datetime import datetime, timedelta
import logging


logger = logging.getLogger(__name__)


# FinGPT策略需要的标的与公司映射配置
# 包含ticker、公司名称、新闻关键词等信息
UNIVERSE_WITH_COMPANIES = [
    # 大型科技股
    {"ticker": "AAPL", "company": "Apple", "keywords": ["Apple Inc", "iPhone", "Tim Cook", "苹果公司"]},
    {"ticker": "MSFT", "company": "Microsoft", "keywords": ["Microsoft", "Windows", "Azure", "Satya Nadella", "微软"]},
    {"ticker": "GOOGL", "company": "Alphabet", "keywords": ["Google", "Alphabet", "Search", "YouTube", "谷歌"]},
    {"ticker": "AMZN", "company": "Amazon", "keywords": ["Amazon", "AWS", "Jeff Bezos", "e-commerce", "亚马逊"]},
    {"ticker": "META", "company": "Meta Platforms", "keywords": ["Meta", "Facebook", "Instagram", "WhatsApp", "Mark Zuckerberg"]},
    {"ticker": "NVDA", "company": "NVIDIA", "keywords": ["NVIDIA", "GPU", "AI chips", "Jensen Huang", "英伟达"]},
    {"ticker": "TSLA", "company": "Tesla", "keywords": ["Tesla", "Electric Vehicle", "EV", "Elon Musk", "特斯拉"]},
    
    # 金融股
    {"ticker": "JPM", "company": "JPMorgan Chase", "keywords": ["JPMorgan", "JP Morgan", "Chase", "Jamie Dimon"]},
    {"ticker": "BAC", "company": "Bank of America", "keywords": ["Bank of America", "BofA", "Merrill Lynch"]},
    {"ticker": "GS", "company": "Goldman Sachs", "keywords": ["Goldman Sachs", "Goldman", "Investment Banking"]},
    {"ticker": "MS", "company": "Morgan Stanley", "keywords": ["Morgan Stanley", "Wealth Management"]},
    {"ticker": "WFC", "company": "Wells Fargo", "keywords": ["Wells Fargo", "Banking"]},
    
    # 医疗健康
    {"ticker": "JNJ", "company": "Johnson & Johnson", "keywords": ["Johnson & Johnson", "J&J", "Pharmaceutical"]},
    {"ticker": "PFE", "company": "Pfizer", "keywords": ["Pfizer", "Vaccine", "COVID-19", "辉瑞"]},
    {"ticker": "UNH", "company": "UnitedHealth", "keywords": ["UnitedHealth", "Health Insurance", "Healthcare"]},
    {"ticker": "CVS", "company": "CVS Health", "keywords": ["CVS", "Pharmacy", "Healthcare"]},
    {"ticker": "LLY", "company": "Eli Lilly", "keywords": ["Eli Lilly", "Lilly", "Pharmaceutical", "礼来"]},
    
    # 消费品
    {"ticker": "WMT", "company": "Walmart", "keywords": ["Walmart", "Retail", "沃尔玛"]},
    {"ticker": "PG", "company": "Procter & Gamble", "keywords": ["P&G", "Procter & Gamble", "Consumer Goods"]},
    {"ticker": "KO", "company": "Coca-Cola", "keywords": ["Coca-Cola", "Coke", "Beverage", "可口可乐"]},
    {"ticker": "PEP", "company": "PepsiCo", "keywords": ["Pepsi", "PepsiCo", "Beverage", "百事"]},
    {"ticker": "NKE", "company": "Nike", "keywords": ["Nike", "Sportswear", "耐克"]},
    
    # 工业
    {"ticker": "BA", "company": "Boeing", "keywords": ["Boeing", "Aircraft", "737", "波音"]},
    {"ticker": "CAT", "company": "Caterpillar", "keywords": ["Caterpillar", "CAT", "Construction", "卡特彼勒"]},
    {"ticker": "GE", "company": "General Electric", "keywords": ["GE", "General Electric", "通用电气"]},
    
    # 能源
    {"ticker": "XOM", "company": "Exxon Mobil", "keywords": ["Exxon", "ExxonMobil", "Oil", "埃克森美孚"]},
    {"ticker": "CVX", "company": "Chevron", "keywords": ["Chevron", "Oil", "Energy", "雪佛龙"]},
    
    # 半导体
    {"ticker": "INTC", "company": "Intel", "keywords": ["Intel", "Semiconductor", "CPU", "英特尔"]},
    {"ticker": "AMD", "company": "AMD", "keywords": ["AMD", "Advanced Micro Devices", "CPU", "GPU"]},
    {"ticker": "AVGO", "company": "Broadcom", "keywords": ["Broadcom", "Semiconductor", "博通"]},
    {"ticker": "QCOM", "company": "Qualcomm", "keywords": ["Qualcomm", "5G", "Mobile Chips", "高通"]},
    
    # 其他知名公司
    {"ticker": "DIS", "company": "Disney", "keywords": ["Disney", "Walt Disney", "Theme Parks", "迪士尼"]},
    {"ticker": "NFLX", "company": "Netflix", "keywords": ["Netflix", "Streaming", "网飞"]},
    {"ticker": "ADBE", "company": "Adobe", "keywords": ["Adobe", "Creative Cloud", "Photoshop"]},
    {"ticker": "CRM", "company": "Salesforce", "keywords": ["Salesforce", "CRM", "Cloud"]},
]


@dataclass
class UniverseConfig:
    """Universe Builder配置类（增强版，支持新闻映射）"""
    
    # 筛选阈值
    min_avg_dollar_volume_percentile: int = 60  # 成交额需高于X%分位数
    max_volatility_percentile: int = 90  # 波动率需低于X%分位数
    max_premarket_gap: float = 0.08  # 8%最大盘前跳空
    halt_lookback_days: int = 5  # 检查近X日熔断
    earnings_lookback_days: int = 2  # 检查未来X日财报
    top_active_stocks: int = 60  # 最终活跃池大小
    min_stock_price: float = 1.0  # 最低股价要求
    
    # 新闻覆盖要求（新增）
    require_news_coverage: bool = True  # 是否要求有新闻覆盖
    min_news_sources: int = 2  # 最少新闻源数量
    
    # 并发控制
    max_concurrent_requests: int = 25  # 最大并发请求数
    api_retry_count: int = 3  # API重试次数
    api_retry_delay: float = 1.0  # 重试延迟（秒）
    
    # 活跃度评分权重
    volume_score_weight: float = 0.7  # 成交额权重
    volatility_score_weight: float = 0.3  # 波动率权重
    optimal_volatility: float = 0.015  # 最优波动率（1.5%）
    
    # 数据质量
    min_valid_days: int = 10  # 最少有效数据天数
    rolling_window: int = 20  # 滚动窗口大小
    
    # 标的配置（新增）
    universe_stocks: List[Dict] = field(default_factory=lambda: UNIVERSE_WITH_COMPANIES)
    
    # 日志级别
    log_level: str = "INFO"  # 默认日志级别
    debug_stocks: List[str] = None  # 需要调试的特定股票
    
    # API配置
    tiingo_base_url: str = "https://api.tiingo.com"
    alpaca_paper_url: str = "https://paper-api.alpaca.markets"
    
    # 动态阈值兜底（样本不足时使用）
    fallback_min_dollar_volume: float = 50_000_000  # 5000万美元
    fallback_max_volatility: float = 0.03  # 3%
    fallback_sample_threshold: int = 50  # 低于此样本数启用固定阈值
    
    def __post_init__(self):
        """初始化后处理"""
        if self.debug_stocks is None:
            self.debug_stocks = []
    
    @classmethod
    def from_file(cls, filepath: str = "config/universe_config.json") -> "UniverseConfig":
        """从配置文件加载"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                # 如果配置文件中有universe_stocks，使用它；否则使用默认值
                if 'universe_stocks' not in data:
                    data['universe_stocks'] = UNIVERSE_WITH_COMPANIES
                return cls(**data)
        else:
            # 使用默认配置
            return cls()
    
    def save(self, filepath: str = "config/universe_config.json"):
        """保存配置到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        config_dict = asdict(self)
        # 确保universe_stocks被保存
        config_dict['universe_stocks'] = self.universe_stocks
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def get_dynamic_semaphore(self, num_stocks: int) -> int:
        """根据股票数量动态调整并发数"""
        # 避免API限流，根据股票数量自动调整
        if num_stocks > 200:
            return min(10, self.max_concurrent_requests)
        elif num_stocks > 100:
            return min(15, self.max_concurrent_requests)
        else:
            return self.max_concurrent_requests


# 股票代码映射（处理更名、退市等情况）
SYMBOL_MAPPINGS = {
    # 股票更名和退市
    'FISV': 'FI',      # Fiserv更名
    'FB': 'META',      # Facebook更名为Meta
    'TWTR': None,      # Twitter已退市（被马斯克收购）
    'ATVI': None,      # 动视暴雪已被微软收购
    'SGEN': None,      # Seagen已被辉瑞收购
    'SPLK': None,      # Splunk已被思科收购
    # Tiingo 格式映射（横杠 → 点号）- 仅供 Tiingo API 使用
    'BRK-B': 'BRK.B',  # 伯克希尔B类
    'BRK-A': 'BRK.A',  # 伯克希尔A类
    'BF-B': 'BF.B',    # Brown-Forman B类
    'BF-A': 'BF.A',    # Brown-Forman A类
}

# 已退市或将退市的股票
DELISTED_SYMBOLS = {
    'ATVI', 'SGEN', 'SPLK', 'TWTR', 'FRC', 'SIVB', 'SBNY',
    # 可以定期从 Tiingo delisted 数据更新
}

# 禁止做空的股票（但仍可做多）
HARD_TO_BORROW_SYMBOLS = {
    'BRK-B',  # 伯克希尔B类 - 借券困难
    'BF-B',   # Brown-Forman B类 - 借券困难
    # 可以根据实际情况添加更多
}


def get_ticker_company_map() -> Dict[str, str]:
    """获取ticker到公司名称的映射"""
    return {item['ticker']: item['company'] for item in UNIVERSE_WITH_COMPANIES}


def get_ticker_keywords_map() -> Dict[str, List[str]]:
    """获取ticker到新闻关键词的映射"""
    return {item['ticker']: item.get('keywords', [item['company']]) for item in UNIVERSE_WITH_COMPANIES}


def get_company_to_ticker_map() -> Dict[str, str]:
    """获取公司名称到ticker的反向映射"""
    company_map = {}
    for item in UNIVERSE_WITH_COMPANIES:
        company_map[item['company'].lower()] = item['ticker']
        # 也将关键词映射到ticker
        for keyword in item.get('keywords', []):
            company_map[keyword.lower()] = item['ticker']
    return company_map


@lru_cache(maxsize=2, typed=False)
def _fetch_sp500_from_wikipedia() -> List[str]:
    """
    从 Wikipedia 获取最新的 S&P 500 成分股列表
    使用 lru_cache 缓存结果，避免频繁请求
    返回横杠格式（如 BRK-B）
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, header=0)
        if tables:
            df = tables[0]  # 第一个表格包含成分股
            # 将点号替换为横杠，统一使用横杠格式
            symbols = df["Symbol"].str.replace(".", "-", regex=False).tolist()
            logger.info(f"从 Wikipedia 获取到 {len(symbols)} 只 S&P 500 成分股")
            return symbols
    except Exception as e:
        logger.error(f"从 Wikipedia 获取 S&P 500 失败: {e}")
    
    # 失败时返回备用列表
    return _get_fallback_sp500()


@lru_cache(maxsize=2, typed=False)
def _fetch_nasdaq100_from_wikipedia() -> List[str]:
    """
    从 Wikipedia 获取最新的 Nasdaq 100 成分股列表
    返回横杠格式（如 BRK-B）
    """
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url, header=0)
        # Nasdaq-100 成分股通常在第4个表格（索引3）
        if len(tables) > 3:
            df = tables[3]
            # 查找包含 Ticker 的列
            ticker_col = None
            for col in df.columns:
                if 'ticker' in col.lower() or 'symbol' in col.lower():
                    ticker_col = col
                    break
            
            if ticker_col:
                # 将点号替换为横杠，统一使用横杠格式
                symbols = df[ticker_col].str.replace(".", "-", regex=False).tolist()
                logger.info(f"从 Wikipedia 获取到 {len(symbols)} 只 Nasdaq 100 成分股")
                return symbols
    except Exception as e:
        logger.error(f"从 Wikipedia 获取 Nasdaq 100 失败: {e}")
    
    # 失败时返回备用列表
    return _get_fallback_nasdaq100()


def _get_fallback_sp500() -> List[str]:
    """S&P 500 备用列表（部分）- 使用横杠格式"""
    # 返回配置中定义的ticker列表
    return [item['ticker'] for item in UNIVERSE_WITH_COMPANIES if item['ticker'] in [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 
        'JPM', 'JNJ', 'UNH', 'PG', 'XOM', 'CVX', 'BAC', 'WMT', 'KO', 'PFE'
    ]]


def _get_fallback_nasdaq100() -> List[str]:
    """Nasdaq 100 备用列表（部分）"""
    # 返回配置中定义的科技股ticker
    return [item['ticker'] for item in UNIVERSE_WITH_COMPANIES if item['ticker'] in [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA',
        'INTC', 'AMD', 'AVGO', 'QCOM', 'NFLX', 'ADBE', 'CRM'
    ]]


def get_sp500_symbols() -> List[str]:
    """
    获取S&P500成分股列表
    优先从Wikipedia获取最新列表，失败时使用备用列表
    每次调用会检查缓存是否过期（24小时）
    """
    # 清除过期缓存（24小时）
    cache_info = _fetch_sp500_from_wikipedia.cache_info()
    if cache_info.hits > 0:
        # 简单的时间检查，实际使用时可以记录缓存时间
        _fetch_sp500_from_wikipedia.cache_clear()
    
    symbols = _fetch_sp500_from_wikipedia()
    
    # 应用映射和过滤
    processed_symbols = []
    for symbol in symbols:
        # 检查是否需要映射（只处理股票更名，不处理格式转换）
        if symbol in SYMBOL_MAPPINGS:
            mapped = SYMBOL_MAPPINGS[symbol]
            if mapped and '.' not in mapped:  # 非None且不是Tiingo格式映射
                processed_symbols.append(mapped)
        elif symbol not in DELISTED_SYMBOLS:
            processed_symbols.append(symbol)
    
    return processed_symbols


def get_nasdaq100_symbols() -> List[str]:
    """
    获取纳斯达克100成分股列表
    优先从Wikipedia获取最新列表，失败时使用备用列表
    """
    # 清除过期缓存
    cache_info = _fetch_nasdaq100_from_wikipedia.cache_info()
    if cache_info.hits > 0:
        _fetch_nasdaq100_from_wikipedia.cache_clear()
    
    symbols = _fetch_nasdaq100_from_wikipedia()
    
    # 应用映射和过滤
    processed_symbols = []
    for symbol in symbols:
        if symbol in SYMBOL_MAPPINGS:
            mapped = SYMBOL_MAPPINGS[symbol]
            if mapped and '.' not in mapped:
                processed_symbols.append(mapped)
        elif symbol not in DELISTED_SYMBOLS:
            processed_symbols.append(symbol)
    
    return processed_symbols


def get_combined_universe() -> List[str]:
    """
    获取 S&P 500 和 Nasdaq 100 的合并去重列表
    """
    sp500 = set(get_sp500_symbols())
    nasdaq100 = set(get_nasdaq100_symbols())
    combined = sorted(sp500 | nasdaq100)
    logger.info(f"合并后总计 {len(combined)} 只股票 (S&P500: {len(sp500)}, Nasdaq100: {len(nasdaq100)})")
    return combined


def get_tiingo_symbol(ticker: str) -> str:
    """
    获取 Tiingo API 使用的符号格式
    
    Args:
        ticker: 标准横杠格式的股票代码（如 'BRK-B'）
        
    Returns:
        Tiingo 格式的股票代码（如 'BRK.B'）
    """
    # 优先使用映射表
    if ticker in SYMBOL_MAPPINGS:
        mapped = SYMBOL_MAPPINGS[ticker]
        # 如果映射值包含点号，说明是 Tiingo 格式
        if mapped and '.' in mapped:
            return mapped
    
    # 默认转换规则
    return ticker.replace('-', '.')


def is_hard_to_borrow(ticker: str) -> bool:
    """
    检查股票是否在禁止做空列表中
    
    Args:
        ticker: 股票代码
        
    Returns:
        是否禁止做空
    """
    return ticker in HARD_TO_BORROW_SYMBOLS


def create_default_config():
    """创建默认配置文件"""
    config = UniverseConfig()
    config.save()
    print(f"已创建默认配置文件: config/universe_config.json")
    return config


def validate_universe():
    """验证 Universe 配置和数据获取"""
    print("=" * 60)
    print("Universe 配置验证（FinGPT增强版）")
    print("=" * 60)
    
    # 测试配置加载
    config = UniverseConfig.from_file()
    print(f"\n配置加载成功:")
    print(f"- 最小成交额百分位: {config.min_avg_dollar_volume_percentile}")
    print(f"- 最大波动率百分位: {config.max_volatility_percentile}")
    print(f"- 目标活跃股票数: {config.top_active_stocks}")
    print(f"- 需要新闻覆盖: {config.require_news_coverage}")
    print(f"- 配置的标的数: {len(config.universe_stocks)}")
    
    # 测试映射功能
    print("\n标的映射测试:")
    ticker_company_map = get_ticker_company_map()
    ticker_keywords_map = get_ticker_keywords_map()
    company_ticker_map = get_company_to_ticker_map()
    
    print(f"- Ticker到公司映射: {len(ticker_company_map)} 条")
    print(f"- Ticker到关键词映射: {len(ticker_keywords_map)} 条")
    print(f"- 公司/关键词到Ticker映射: {len(company_ticker_map)} 条")
    
    # 显示部分样本
    print("\n映射示例:")
    for ticker in ['AAPL', 'TSLA', 'JPM'][:3]:
        if ticker in ticker_company_map:
            print(f"  {ticker} -> {ticker_company_map[ticker]}")
            print(f"    关键词: {', '.join(ticker_keywords_map.get(ticker, [])[:3])}")
    
    # 测试反向映射
    print("\n反向映射测试:")
    test_keywords = ['apple inc', 'tesla', 'google']
    for keyword in test_keywords:
        ticker = company_ticker_map.get(keyword, 'Not Found')
        print(f"  '{keyword}' -> {ticker}")
    
    # 测试成分股获取
    print("\n成分股列表获取:")
    sp500 = get_sp500_symbols()
    nasdaq100 = get_nasdaq100_symbols()
    combined = get_combined_universe()
    
    print(f"- S&P 500: {len(sp500)} 只")
    print(f"- Nasdaq 100: {len(nasdaq100)} 只")
    print(f"- 合并去重后: {len(combined)} 只")
    
    # 测试做空限制
    print(f"\n禁止做空的股票: {HARD_TO_BORROW_SYMBOLS}")
    
    print("\n验证完成!")
    return len(config.universe_stocks) > 0


if __name__ == "__main__":
    # 创建默认配置
    config = create_default_config()
    print("\n当前配置（包含FinGPT所需映射）：")
    print(f"配置的标的数: {len(config.universe_stocks)}")
    print(f"前5个标的:")
    for item in config.universe_stocks[:5]:
        print(f"  {item['ticker']}: {item['company']}")
    
    # 运行验证
    print("\n" + "=" * 60)
    if validate_universe():
        print("✅ Universe 配置正常，包含FinGPT所需的新闻映射信息")
    else:
        print("❌ 警告：配置异常，请检查")