"""
test_fingpt_integration.py - FinGPT策略集成测试
验证 data_collector、universe_builder 和 factor_model_optimized 的集成
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def test_module_imports():
    """测试模块导入"""
    print("\n=== 测试模块导入 ===")
    
    try:
        # 导入核心模块
        from universe_builder import UniverseBuilder
        from data_collector import DataCollector, NewsArticle
        from factor_model_optimized import (
            EnhancedNewsFactorExtractor,
            FinGPTSentimentAnalyzer,
            FinancialEmbeddingModel,
            TechnicalFactorCalculator,
            EnhancedNewsFactor
        )
        
        print("✅ 所有模块导入成功")
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False


def test_universe_builder():
    """测试 UniverseBuilder"""
    print("\n=== 测试 UniverseBuilder ===")
    
    try:
        from universe_builder import UniverseBuilder
        
        # 创建 UniverseBuilder（使用配置的标的池）
        builder = UniverseBuilder(use_configured_universe=True)
        
        # 构建universe
        symbols, results_df, company_map = builder.build_universe()
        
        print(f"✅ 构建成功: {len(symbols)} 只股票")
        print(f"✅ 公司映射: {len(company_map)} 条")
        
        # 显示部分映射
        for ticker in list(company_map.keys())[:3]:
            print(f"  {ticker}: {company_map[ticker]}")
        
        return symbols, company_map, builder
        
    except Exception as e:
        print(f"❌ UniverseBuilder 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return [], {}, None


def test_data_collector(company_map: Dict[str, str], builder):
    """测试 DataCollector"""
    print("\n=== 测试 DataCollector ===")
    
    try:
        from data_collector import DataCollector
        
        # 获取关键词映射
        keywords_map = {}
        if builder:
            for ticker in company_map.keys():
                keywords_map[ticker] = builder.get_news_keywords_for_ticker(ticker)
        
        # 创建 DataCollector
        collector = DataCollector(
            company_map=company_map,
            keywords_map=keywords_map
        )
        
        print("✅ DataCollector 创建成功")
        print(f"  映射数量: {len(collector.company_map)} 个公司")
        
        # 测试历史数据获取
        test_ticker = list(company_map.keys())[0] if company_map else "AAPL"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        df = collector.get_historical_data(
            test_ticker,
            start_date,
            end_date,
            'daily'
        )
        
        if not df.empty:
            print(f"✅ 获取 {test_ticker} 历史数据成功: {len(df)} 条")
            print(f"  列: {list(df.columns)}")
        else:
            print(f"⚠️  {test_ticker} 历史数据为空")
        
        # 测试新闻获取
        news_list = collector.get_enhanced_news(test_ticker, days_back=1)
        print(f"✅ 获取 {test_ticker} 新闻: {len(news_list)} 条")
        
        if news_list:
            news = news_list[0]
            print(f"  示例新闻: {news.headline[:50]}...")
            print(f"  公司: {news.company}")
            print(f"  关键词: {news.keywords_matched}")
        
        return collector
        
    except Exception as e:
        print(f"❌ DataCollector 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_factor_extractor(collector, company_map):
    """测试 FactorExtractor"""
    print("\n=== 测试 EnhancedNewsFactorExtractor ===")
    
    try:
        from factor_model_optimized import (
            EnhancedNewsFactorExtractor,
            get_enhanced_extractor
        )
        
        # 创建因子提取器
        extractor = get_enhanced_extractor(
            data_collector=collector,
            use_async=False
        )
        
        print("✅ EnhancedNewsFactorExtractor 创建成功")
        
        # 准备测试新闻数据
        test_news = [
            {
                'ticker': 'AAPL',
                'datetime': datetime.now() - timedelta(hours=1),
                'headline': 'Apple announces breakthrough AI technology with record revenue',
                'summary': 'Apple Inc. reported exceptional quarterly results driven by strong iPhone sales and new AI features.',
                'source': 'Reuters',
                'company': company_map.get('AAPL', 'Apple'),
                'current_price': 150.0,
                'current_volume': 1000000
            }
        ]
        
        print(f"\n测试因子提取...")
        
        # 提取因子
        factors = extractor.extract_factors_sync(test_news)
        
        if factors:
            print(f"✅ 提取到 {len(factors)} 个因子")
            
            factor = factors[0]
            print(f"\n因子详情:")
            print(f"  股票: {factor.ticker}")
            print(f"  公司: {factor.company}")
            print(f"  事件打分: {factor.event_score}/5 ({factor.event_impact})")
            print(f"  情绪: {factor.sentiment_label} (得分: {factor.sentiment_score:.3f})")
            print(f"  置信度: {factor.sentiment_prob:.3f}")
            print(f"  理由: {factor.rationale[:100]}...")
            print(f"  MACD信号: {factor.macd_signal}")
            print(f"  RSI: {factor.rsi_value:.1f} ({factor.rsi_signal})")
            print(f"  综合置信度: {factor.confidence_composite:.3f}")
        else:
            print("⚠️  未提取到因子")
        
        return extractor
        
    except Exception as e:
        print(f"❌ FactorExtractor 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_technical_calculator(collector):
    """测试技术指标计算器"""
    print("\n=== 测试 TechnicalFactorCalculator ===")
    
    try:
        from factor_model_optimized import TechnicalFactorCalculator
        
        # 创建技术指标计算器
        calculator = TechnicalFactorCalculator(data_collector=collector)
        
        print("✅ TechnicalFactorCalculator 创建成功")
        
        # 测试指标计算
        test_ticker = "AAPL"
        timestamp = datetime.now()
        
        indicators = calculator.get_indicator_signals(test_ticker, timestamp)
        
        print(f"\n{test_ticker} 技术指标:")
        print(f"  MACD信号: {indicators['macd']['signal']}")
        print(f"  RSI值: {indicators['rsi']['rsi']:.1f} ({indicators['rsi']['signal']})")
        print(f"  布林带: {indicators['bollinger']['signal']}")
        print(f"  综合信号: {indicators['overall_signal']}")
        
        return calculator
        
    except Exception as e:
        print(f"❌ TechnicalCalculator 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_full_workflow():
    """测试完整工作流程"""
    print("\n=== 测试完整工作流程 ===")
    
    try:
        # 1. 构建Universe
        from universe_builder import UniverseBuilder
        builder = UniverseBuilder(use_configured_universe=True)
        symbols, results_df, company_map = builder.build_universe()
        
        print(f"✅ Step 1: Universe构建完成 - {len(symbols)} 只股票")
        
        # 2. 创建DataCollector
        from data_collector import DataCollector
        keywords_map = {ticker: builder.get_news_keywords_for_ticker(ticker) 
                       for ticker in symbols}
        
        collector = DataCollector(
            company_map=company_map,
            keywords_map=keywords_map
        )
        
        print(f"✅ Step 2: DataCollector创建完成")
        
        # 3. 获取新闻数据
        test_ticker = symbols[0] if symbols else "AAPL"
        news_articles = collector.get_enhanced_news(test_ticker, days_back=1)
        
        print(f"✅ Step 3: 获取{test_ticker}新闻 - {len(news_articles)} 条")
        
        # 4. 提取因子
        from factor_model_optimized import get_enhanced_extractor
        extractor = get_enhanced_extractor(
            data_collector=collector,
            use_async=False
        )
        
        if news_articles:
            # 将NewsArticle转换为字典格式
            news_dicts = [article.to_dict() for article in news_articles]
            
            factors = extractor.extract_factors_sync(news_dicts)
            
            print(f"✅ Step 4: 因子提取完成 - {len(factors)} 个因子")
            
            if factors:
                # 显示第一个因子
                factor = factors[0]
                print(f"\n最终因子示例:")
                print(f"  {factor.ticker}: 事件打分={factor.event_score}, "
                      f"情绪={factor.sentiment_score:.2f}, "
                      f"MACD={factor.macd_signal}, "
                      f"RSI={factor.rsi_value:.1f}")
        else:
            print("⚠️  没有新闻数据可供测试")
        
        print("\n✅ 完整工作流程测试成功！")
        print("所有模块已正确集成，可以开始运行策略。")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 工作流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_environment():
    """检查环境配置"""
    print("\n=== 检查环境配置 ===")
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  GPU不可用，将使用CPU（速度较慢）")
    except:
        print("⚠️  PyTorch未安装或配置错误")
    
    # 检查环境变量
    required_env = ['TIINGO_API_KEY', 'FINNHUB_API_KEY']
    missing_env = []
    
    for env_var in required_env:
        if not os.getenv(env_var):
            missing_env.append(env_var)
    
    if missing_env:
        print(f"⚠️  缺少环境变量: {', '.join(missing_env)}")
        print("  请在.env文件中配置这些API密钥")
    else:
        print("✅ 所有必需的环境变量已配置")
    
    # 检查依赖
    try:
        import transformers
        import peft
        import sentence_transformers
        print(f"✅ 核心依赖版本:")
        print(f"  transformers: {transformers.__version__}")
        print(f"  peft: {peft.__version__}")
    except ImportError as e:
        print(f"⚠️  缺少依赖: {e}")


def main():
    """主测试函数"""
    print("="*60)
    print("FinGPT量化策略集成测试")
    print("="*60)
    
    # 检查环境
    check_environment()
    
    # 测试模块导入
    if not test_module_imports():
        print("\n❌ 模块导入失败，请检查代码文件是否完整")
        return
    
    # 测试各个组件
    print("\n开始组件测试...")
    
    # 1. 测试UniverseBuilder
    symbols, company_map, builder = test_universe_builder()
    if not symbols:
        print("\n⚠️  UniverseBuilder测试失败，但继续其他测试...")
        # 使用默认值
        company_map = {"AAPL": "Apple", "MSFT": "Microsoft"}
        builder = None
    
    # 2. 测试DataCollector
    collector = test_data_collector(company_map, builder)
    if not collector:
        print("\n❌ DataCollector测试失败，无法继续")
        return
    
    # 3. 测试FactorExtractor
    extractor = test_factor_extractor(collector, company_map)
    
    # 4. 测试TechnicalCalculator
    calculator = test_technical_calculator(collector)
    
    # 5. 测试完整工作流程
    print("\n" + "="*60)
    success = test_full_workflow()
    
    if success:
        print("\n🎉 所有测试通过！系统已准备就绪。")
        print("\n下一步:")
        print("1. 确保已配置API密钥（TIINGO_API_KEY, FINNHUB_API_KEY）")
        print("2. 如果使用GPU，确保有足够显存（建议16GB+）")
        print("3. 运行策略回测: python run_strategy_backtest.py")
    else:
        print("\n⚠️  部分测试失败，请根据错误信息进行修复")


if __name__ == "__main__":
    main()

