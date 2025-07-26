"""
factor_model_optimized.py - 增强版因子建模模块（FinGPT Llama2-13B LoRA版本）
使用FinGPT fingpt-sentiment_llama2-13b_lora进行情绪分析和事件打分（1-5分）
部署在AWS上，直接加载模型，无需API调用
新增：MACD、布林带、RSI等技术指标计算和信号判定
"""

from __future__ import annotations
import os
import logging
import asyncio
import time
import torch
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizerFast,
    BitsAndBytesConfig
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from scipy.special import expit as sigmoid
import hashlib
import warnings
import re
import json
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/factor_model_fingpt.log', encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建必要的目录
os.makedirs('logs', exist_ok=True)
os.makedirs('cache', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 检查CUDA可用性
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {DEVICE}")
if DEVICE.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


@dataclass
class EnhancedNewsFactor:
    """增强版新闻因子数据结构 - 包含事件打分、技术指标和新闻热度"""
    ticker: str
    datetime: datetime
    headline: str
    
    # 情绪因子（由FinGPT生成）
    sentiment_score: float      # 情绪得分 [-1, 1]
    sentiment_label: str        # 情绪标签
    sentiment_prob: float       # 情绪置信度
    uncertainty: float          # 不确定性 [0, 1]
    rationale: str             # 解释文本
    
    # 突发事件打分（1-5分）
    event_score: int           # 突发事件重要性评分 [1, 5]
    event_impact: str          # 事件影响等级：minimal/low/medium/high/extreme
    
    # 技术指标因子
    price_change_5min: float    # 新闻前后5分钟价格变动 [-1, 1]
    price_change_15min: float   # 新闻前后15分钟价格变动 [-1, 1]
    volume_spike: float         # 成交量异常倍数 [0.1, 10+]
    volatility_5min: float      # 5分钟价格波动率
    
    # 经典技术指标
    macd_signal: str           # MACD信号：bullish/bearish/neutral
    rsi_value: float           # RSI值 [0, 100]
    rsi_signal: str            # RSI信号：overbought/oversold/neutral
    bollinger_signal: str      # 布林带信号：above_upper/below_lower/in_band
    
    # 新闻因子
    similarity_avg: float       # 平均相似度
    novelty: float             # 新颖度 [0, 1]
    cluster_distance: float     # 到最近聚类中心的距离
    similar_count: int          # 相似新闻计数
    news_heat: float           # 新闻热度指数 [0, 1]
    time_decay: float          # 时间衰减因子 [0, 1]
    
    # 元数据
    event_hash: str            # 事件哈希（去重用）
    is_valid_signal: bool      # 是否为有效信号源
    confidence_composite: float # 综合置信度
    
    # 技术指标详细数据
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    
    # FinGPT原始响应
    fingpt_response: Optional[str] = None
    
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'ticker': self.ticker,
            'datetime': self.datetime.isoformat() if isinstance(self.datetime, datetime) else self.datetime,
            'headline': self.headline,
            
            # 情绪因子
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
            'sentiment_prob': self.sentiment_prob,
            'uncertainty': self.uncertainty,
            'rationale': self.rationale[:100] + '...' if len(self.rationale) > 100 else self.rationale,
            
            # 事件打分
            'event_score': self.event_score,
            'event_impact': self.event_impact,
            
            # 技术指标因子
            'price_change_5min': self.price_change_5min,
            'price_change_15min': self.price_change_15min,
            'volume_spike': self.volume_spike,
            'volatility_5min': self.volatility_5min,
            
            # 经典技术指标
            'macd_signal': self.macd_signal,
            'rsi_value': self.rsi_value,
            'rsi_signal': self.rsi_signal,
            'bollinger_signal': self.bollinger_signal,
            
            # 新闻因子
            'similarity_avg': self.similarity_avg,
            'novelty': self.novelty,
            'cluster_distance': self.cluster_distance,
            'similar_count': self.similar_count,
            'news_heat': self.news_heat,
            'time_decay': self.time_decay,
            
            # 元数据
            'event_hash': self.event_hash,
            'is_valid_signal': self.is_valid_signal,
            'confidence_composite': self.confidence_composite,
            
            # 技术指标详细数据
            'technical_indicators': self.technical_indicators
        }


class FinGPTSentimentAnalyzer:
    """FinGPT情绪分析器 - 使用fingpt-sentiment_llama2-13b_lora模型"""
    
    def __init__(self, 
                 base_model: str = "NousResearch/Llama-2-13b-hf",
                 peft_model: str = "FinGPT/fingpt-sentiment_llama2-13b_lora",
                 use_8bit: bool = True,
                 max_new_tokens: int = 50,
                 temperature: float = 0.7,
                 local_model_path: Optional[str] = None):
        """
        初始化FinGPT情绪分析模型
        
        Args:
            base_model: 基础Llama2模型名称
            peft_model: FinGPT LoRA权重名称  
            use_8bit: 是否使用8bit量化
            max_new_tokens: 最大生成token数
            temperature: 生成温度
            local_model_path: 本地模型路径（如果已下载）
        """
        self.base_model = base_model
        self.peft_model = peft_model
        self.device = DEVICE
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_8bit = use_8bit
        
        logger.info(f"加载FinGPT情绪分析模型: {peft_model}")
        logger.info(f"基础模型: {base_model}")
        logger.info(f"量化设置: 8bit={use_8bit}")
        
        try:
            # 使用本地路径或从HuggingFace下载
            base_path = local_model_path or base_model
            
            # 加载Tokenizer
            self.tokenizer = LlamaTokenizerFast.from_pretrained(
                base_path,
                trust_remote_code=True
            )
            
            # 设置padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 配置模型加载选项
            if use_8bit and self.device.type == 'cuda':
                # 8bit量化配置
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
                
                self.model = LlamaForCausalLM.from_pretrained(
                    base_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                logger.info("使用8bit量化加载基础模型")
            else:
                # 标准加载
                self.model = LlamaForCausalLM.from_pretrained(
                    base_path,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    trust_remote_code=True
                ).to(self.device)
                logger.info(f"标准加载基础模型到 {self.device}")
            
            # 加载LoRA权重
            logger.info(f"加载LoRA权重: {peft_model}")
            self.model = PeftModel.from_pretrained(self.model, peft_model)
            self.model.eval()
            
            # 预定义的prompt模板
            self.sentiment_prompt_template = """Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}
Input: {news_text}
Answer:"""

            self.event_score_prompt_template = """Instruction: Analyze the following financial news. Determine the sentiment (positive, neutral, or negative) and rate the significance of any new event mentioned on a scale of 1 to 5 (5 = very significant). Provide your answer in format: Sentiment=<pos/neu/neg>; EventScore=<1-5>; Reason=<brief explanation>
Input: {news_text}
Answer:"""
            
            # 统计计数器
            self.analysis_stats = {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'avg_processing_time': 0.0
            }
            
            logger.info("FinGPT情绪分析模型加载成功")
            
            # 测试推理
            self._test_inference()
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            logger.error("请确保:")
            logger.error("1. AWS实例有足够的GPU显存（建议至少16GB）")
            logger.error("2. 已安装必要的依赖: pip install transformers==4.32.0 peft==0.5.0 bitsandbytes")
            logger.error("3. 模型文件可访问（从HuggingFace下载）")
            raise
    
    def _test_inference(self):
        """测试模型推理能力"""
        try:
            test_text = "Apple announces record iPhone sales, beating analyst expectations by 20%."
            logger.info("测试模型推理...")
            
            result = self.analyze_news(test_text)
            logger.info(f"测试成功: {result}")
            
        except Exception as e:
            logger.error(f"模型推理测试失败: {e}")
            raise
    
    def analyze_news(self, news_text: str, company: Optional[str] = None) -> Dict[str, Any]:
        """
        使用FinGPT分析新闻情绪和事件重要性
        
        Args:
            news_text: 新闻文本
            company: 公司名称（可选）
            
        Returns:
            包含情绪和事件打分的分析结果字典
        """
        start_time = time.time()
        self.analysis_stats['total'] += 1
        
        try:
            # 构建prompt
            if company:
                news_text = f"[{company}] {news_text}"
            
            # 使用综合分析的prompt
            prompt = self.event_score_prompt_template.format(news_text=news_text[:800])
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024  # 为生成留出空间
            ).to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码响应
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的部分
            if "Answer:" in response:
                analysis_text = response.split("Answer:")[-1].strip()
            else:
                analysis_text = response[len(prompt):].strip()
            
            # 解析响应
            result = self._parse_fingpt_response(analysis_text, news_text)
            result['raw_response'] = analysis_text
            
            # 更新统计
            self.analysis_stats['successful'] += 1
            processing_time = time.time() - start_time
            self.analysis_stats['avg_processing_time'] = (
                (self.analysis_stats['avg_processing_time'] * (self.analysis_stats['successful'] - 1) + 
                 processing_time) / self.analysis_stats['successful']
            )
            
            logger.debug(f"FinGPT分析完成: {news_text[:50]}... -> {result['sentiment']} (事件评分={result['event_score']})")
            
            return result
            
        except Exception as e:
            logger.error(f"FinGPT分析失败: {e}")
            self.analysis_stats['failed'] += 1
            
            # 返回默认值
            return {
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'event_score': 1,
                'confidence': 0.5,
                'rationale': f'Analysis failed: {str(e)}',
                'raw_response': ''
            }
    
    def _parse_fingpt_response(self, response: str, original_text: str = "") -> Dict[str, Any]:
        """解析FinGPT响应"""
        result = {
            'sentiment': 'neutral',
            'sentiment_score': 0.0,
            'event_score': 1,
            'confidence': 0.8,  # FinGPT默认置信度
            'rationale': ''
        }
        
        try:
            # 方法1: 尝试解析结构化格式 "Sentiment=pos; EventScore=4; Reason=..."
            sentiment_match = re.search(r'Sentiment\s*=\s*(\w+)', response, re.IGNORECASE)
            event_match = re.search(r'EventScore\s*=\s*([1-5])', response, re.IGNORECASE)
            reason_match = re.search(r'Reason\s*=\s*(.+?)(?:;|$)', response, re.IGNORECASE)
            
            if sentiment_match:
                sentiment = sentiment_match.group(1).lower()
                if sentiment.startswith('pos'):
                    result['sentiment'] = 'positive'
                    result['sentiment_score'] = 0.8
                elif sentiment.startswith('neg'):
                    result['sentiment'] = 'negative'
                    result['sentiment_score'] = -0.8
                else:
                    result['sentiment'] = 'neutral'
                    result['sentiment_score'] = 0.0
            else:
                # 方法2: 直接搜索情绪词
                if re.search(r'\b(positive|bullish|good|great|strong|up|gain|rise|increase)\b', response, re.IGNORECASE):
                    result['sentiment'] = 'positive'
                    result['sentiment_score'] = 0.8
                elif re.search(r'\b(negative|bearish|bad|poor|weak|down|loss|fall|decline|drop)\b', response, re.IGNORECASE):
                    result['sentiment'] = 'negative'
                    result['sentiment_score'] = -0.8
                else:
                    result['sentiment'] = 'neutral'
                    result['sentiment_score'] = 0.0
            
            # 解析事件评分
            if event_match:
                result['event_score'] = int(event_match.group(1))
            else:
                # 根据新闻内容和情绪强度推测事件重要性
                text_lower = original_text.lower()
                if any(word in text_lower for word in ['record', 'breakthrough', 'acquisition', 'merger', 'bankruptcy', 'lawsuit']):
                    result['event_score'] = 4
                elif any(word in text_lower for word in ['earnings', 'revenue', 'profit', 'sales', 'partnership', 'contract']):
                    result['event_score'] = 3
                elif any(word in text_lower for word in ['announce', 'launch', 'release', 'update']):
                    result['event_score'] = 2
                else:
                    result['event_score'] = 1
            
            # 解析理由
            if reason_match:
                result['rationale'] = reason_match.group(1).strip()
            elif response:
                result['rationale'] = response[:200].strip()
            else:
                result['rationale'] = "No specific reason provided"
            
        except Exception as e:
            logger.error(f"解析FinGPT响应失败: {e}")
            result['rationale'] = f"Parse error: {str(e)}"
        
        return result
    
    def batch_analyze(self, news_list: List[Tuple[str, Optional[str]]]) -> List[Dict[str, Any]]:
        """
        批量分析新闻
        
        Args:
            news_list: [(news_text, company_name), ...] 列表
            
        Returns:
            分析结果列表
        """
        results = []
        
        for news_text, company in news_list:
            result = self.analyze_news(news_text, company)
            results.append(result)
            
            # 避免过热，给GPU一点休息时间
            if len(news_list) > 1:
                time.sleep(0.1)
        
        return results
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """获取分析统计信息"""
        return {
            'total_analyses': self.analysis_stats['total'],
            'successful': self.analysis_stats['successful'],
            'failed': self.analysis_stats['failed'],
            'success_rate': self.analysis_stats['successful'] / max(1, self.analysis_stats['total']),
            'avg_processing_time': self.analysis_stats['avg_processing_time']
        }


class FinancialEmbeddingModel:
    """金融文本嵌入模型"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.device = DEVICE
        
        logger.info(f"加载嵌入模型: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name, device=str(self.device))
            
            if self.device.type == 'cuda':
                self.model.half()
            
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"嵌入模型加载成功，维度: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            raise
    
    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入向量"""
        if not text or not text.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        try:
            text_with_context = f"Financial news: {text[:500]}"
            
            embedding = self.model.encode(
                text_with_context,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"获取嵌入失败: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def batch_get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """批量获取嵌入"""
        if not texts:
            return []
        
        texts_with_context = [f"Financial news: {t[:500]}" if t else " " for t in texts]
        
        try:
            embeddings = self.model.encode(
                texts_with_context,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False
            )
            
            return [emb.astype(np.float32) for emb in embeddings]
            
        except Exception as e:
            logger.error(f"批量获取嵌入失败: {e}")
            return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]
    
    def compute_similarity(self, embed1: np.ndarray, embed2: np.ndarray) -> float:
        """计算余弦相似度"""
        try:
            similarity = np.dot(embed1, embed2)
            return float(np.clip(similarity, -1.0, 1.0))
        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
            return 0.0


class EnhancedNoveltyDetector:
    """增强版新颖度和新闻热度检测器"""
    
    def __init__(self, n_clusters: int = 8, update_interval: int = 1800, max_heat_window: int = 60):
        self.n_clusters = n_clusters
        self.update_interval = update_interval
        self.max_heat_window = max_heat_window
        self.last_update = datetime.now()
        
        # 每个股票维护独立的聚类器和统计
        self.clusterers = {}
        self.embeddings_buffer = defaultdict(list)
        self.event_hashes = defaultdict(set)
        self.cluster_stats = defaultdict(dict)
        
        # 新闻热度统计
        self.news_timeline = defaultdict(list)
        self.heat_stats = defaultdict(dict)
        
        logger.info(f"增强版新颖度检测器初始化: K={n_clusters}, 热度窗口={max_heat_window}分钟")
    
    def update_clusters(self, ticker: str, embeddings: List[np.ndarray]):
        """更新聚类中心"""
        if len(embeddings) < self.n_clusters:
            return
        
        if ticker not in self.clusterers:
            self.clusterers[ticker] = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                batch_size=32,
                n_init=3
            )
        
        X = np.array(embeddings)
        self.clusterers[ticker].partial_fit(X)
        
        labels = self.clusterers[ticker].predict(X)
        cluster_counts = np.bincount(labels, minlength=self.n_clusters)
        
        for i in range(self.n_clusters):
            cluster_size = cluster_counts[i]
            self.cluster_stats[ticker][i] = {
                'size': int(cluster_size),
                'density': cluster_size / len(embeddings) if len(embeddings) > 0 else 0
            }
        
        logger.debug(f"更新{ticker}的聚类中心，样本数: {len(embeddings)}")
    
    def calculate_novelty_and_heat(self, ticker: str, embedding: np.ndarray, 
                                  headline: str, timestamp: datetime,
                                  source: str = "") -> Tuple[float, float, str, int, float, float]:
        """
        计算新颖度和新闻热度
        
        Returns:
            (novelty_score, cluster_distance, event_hash, similar_count, news_heat, time_decay)
        """
        # 计算事件哈希
        event_str = f"{headline.lower().strip()}{source.lower().strip()}"
        event_hash = hashlib.md5(event_str.encode()).hexdigest()[:16]
        
        # 检查是否已见过（完全重复）
        if event_hash in self.event_hashes[ticker]:
            logger.debug(f"发现重复新闻: {headline[:50]}...")
            return 0.0, 0.0, event_hash, 0, 0.0, 0.0
        
        # 添加到缓冲区和时间线
        self.embeddings_buffer[ticker].append(embedding)
        self.event_hashes[ticker].add(event_hash)
        self.news_timeline[ticker].append({
            'timestamp': timestamp,
            'hash': event_hash,
            'headline': headline
        })
        
        # 限制缓冲区大小
        max_buffer_size = 1000
        if len(self.embeddings_buffer[ticker]) > max_buffer_size:
            self.embeddings_buffer[ticker] = self.embeddings_buffer[ticker][-max_buffer_size//2:]
        
        if len(self.event_hashes[ticker]) > max_buffer_size:
            self.event_hashes[ticker] = set(list(self.event_hashes[ticker])[-max_buffer_size//2:])
        
        # 清理过期的时间线记录
        cutoff_time = timestamp - timedelta(minutes=self.max_heat_window * 2)
        self.news_timeline[ticker] = [
            item for item in self.news_timeline[ticker] 
            if item['timestamp'] > cutoff_time
        ]
        
        # 检查是否需要更新聚类
        current_time = datetime.now()
        if (current_time - self.last_update).seconds > self.update_interval:
            for t in self.embeddings_buffer:
                if len(self.embeddings_buffer[t]) >= self.n_clusters:
                    self.update_clusters(t, self.embeddings_buffer[t])
            self.last_update = current_time
        
        # 计算新颖度和聚类信息
        similar_count = 1
        cluster_distance = 1.0
        novelty = 1.0
        
        if ticker in self.clusterers and hasattr(self.clusterers[ticker], 'cluster_centers_'):
            centers = self.clusterers[ticker].cluster_centers_
            distances = np.linalg.norm(centers - embedding.reshape(1, -1), axis=1)
            cluster_distance = np.min(distances)
            closest_cluster = np.argmin(distances)
            
            cluster_info = self.cluster_stats[ticker].get(closest_cluster, {})
            similar_count = cluster_info.get('size', 1)
            cluster_density = cluster_info.get('density', 0.1)
            
            density_penalty = 1.0 / (1.0 + 2.0 * cluster_density)
            raw_novelty = sigmoid(3.0 * (cluster_distance - 0.3))
            novelty = raw_novelty * density_penalty
        
        # 计算新闻热度
        news_heat, time_decay = self._calculate_news_heat(ticker, timestamp)
        
        return float(novelty), float(cluster_distance), event_hash, similar_count, news_heat, time_decay
    
    def _calculate_news_heat(self, ticker: str, timestamp: datetime) -> Tuple[float, float]:
        """计算新闻热度和时间衰减因子"""
        cutoff_time = timestamp - timedelta(minutes=self.max_heat_window)
        recent_news = [
            item for item in self.news_timeline[ticker]
            if item['timestamp'] > cutoff_time
        ]
        
        news_count = len(recent_news)
        
        if news_count <= 1:
            news_heat = 0.1
        elif news_count <= 3:
            news_heat = 0.3
        elif news_count <= 6:
            news_heat = 0.6
        elif news_count <= 10:
            news_heat = 0.8
        else:
            news_heat = 1.0
        
        now = datetime.now()
        time_diff_hours = (now - timestamp).total_seconds() / 3600
        time_decay = np.exp(-time_diff_hours / 0.5)
        time_decay = float(np.clip(time_decay, 0.0, 1.0))
        
        self.heat_stats[ticker] = {
            'recent_news_count': news_count,
            'current_heat': news_heat,
            'last_update': timestamp
        }
        
        return news_heat, time_decay


class TechnicalFactorCalculator:
    """技术指标因子计算器"""
    
    def __init__(self, data_collector=None):
        self.data_collector = data_collector
        self.price_cache = defaultdict(dict)
        self.volume_cache = defaultdict(dict)
        self.indicator_cache = defaultdict(dict)
        self.cache_ttl = 900  # 15分钟缓存
        
        # 技术指标参数
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.rsi_period = 14
        self.bollinger_period = 20
        self.bollinger_std = 2
        
        if data_collector is not None:
            logger.info("技术指标计算器初始化完成 - 已连接DataCollector")
        else:
            logger.info("技术指标计算器初始化完成 - 独立模式")
    
    def compute_macd(self, prices: pd.Series) -> Dict[str, Any]:
        """计算MACD指标"""
        if len(prices) < self.macd_slow:
            return {
                'macd_line': 0.0,
                'signal_line': 0.0,
                'histogram': 0.0,
                'signal': 'neutral'
            }
        
        try:
            ema_fast = prices.ewm(span=self.macd_fast, adjust=False).mean()
            ema_slow = prices.ewm(span=self.macd_slow, adjust=False).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
            histogram = macd_line - signal_line
            
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_histogram = histogram.iloc[-1]
            
            if len(macd_line) >= 2:
                prev_macd = macd_line.iloc[-2]
                prev_signal = signal_line.iloc[-2]
                
                if prev_macd <= prev_signal and current_macd > current_signal:
                    signal = 'bullish'
                elif prev_macd >= prev_signal and current_macd < current_signal:
                    signal = 'bearish'
                elif current_macd > current_signal:
                    signal = 'bullish'
                elif current_macd < current_signal:
                    signal = 'bearish'
                else:
                    signal = 'neutral'
            else:
                signal = 'neutral'
            
            return {
                'macd_line': float(current_macd),
                'signal_line': float(current_signal),
                'histogram': float(current_histogram),
                'signal': signal
            }
            
        except Exception as e:
            logger.error(f"计算MACD失败: {e}")
            return {
                'macd_line': 0.0,
                'signal_line': 0.0,
                'histogram': 0.0,
                'signal': 'neutral'
            }
    
    def compute_rsi(self, prices: pd.Series) -> Dict[str, Any]:
        """计算RSI指标"""
        if len(prices) < self.rsi_period + 1:
            return {
                'rsi': 50.0,
                'signal': 'neutral'
            }
        
        try:
            delta = prices.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
            avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)
            
            current_rsi = float(rsi.iloc[-1])
            
            if current_rsi > 70:
                signal = 'overbought'
            elif current_rsi < 30:
                signal = 'oversold'
            elif current_rsi > 50:
                signal = 'bullish'
            elif current_rsi < 50:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            return {
                'rsi': current_rsi,
                'signal': signal
            }
            
        except Exception as e:
            logger.error(f"计算RSI失败: {e}")
            return {
                'rsi': 50.0,
                'signal': 'neutral'
            }
    
    def compute_bollinger(self, prices: pd.Series) -> Dict[str, Any]:
        """计算布林带指标"""
        if len(prices) < self.bollinger_period:
            return {
                'upper': 0.0,
                'middle': 0.0,
                'lower': 0.0,
                'width': 0.0,
                'signal': 'in_band'
            }
        
        try:
            middle = prices.rolling(window=self.bollinger_period).mean()
            std = prices.rolling(window=self.bollinger_period).std()
            
            upper = middle + (self.bollinger_std * std)
            lower = middle - (self.bollinger_std * std)
            
            current_price = float(prices.iloc[-1])
            current_upper = float(upper.iloc[-1])
            current_middle = float(middle.iloc[-1])
            current_lower = float(lower.iloc[-1])
            current_width = current_upper - current_lower
            
            if current_price > current_upper:
                signal = 'above_upper'
            elif current_price < current_lower:
                signal = 'below_lower'
            elif current_price > current_middle:
                signal = 'above_middle'
            elif current_price < current_middle:
                signal = 'below_middle'
            else:
                signal = 'in_band'
            
            return {
                'upper': current_upper,
                'middle': current_middle,
                'lower': current_lower,
                'width': current_width,
                'signal': signal
            }
            
        except Exception as e:
            logger.error(f"计算布林带失败: {e}")
            return {
                'upper': 0.0,
                'middle': 0.0,
                'lower': 0.0,
                'width': 0.0,
                'signal': 'in_band'
            }
    
    def get_indicator_signals(self, ticker: str, timestamp: datetime) -> Dict[str, Any]:
        """获取所有技术指标信号"""
        try:
            end_time = timestamp
            start_time = timestamp - timedelta(days=30)
            
            cache_key = f"{ticker}_indicators_{int(timestamp.timestamp())}"
            if cache_key in self.indicator_cache:
                cached_data, cache_time = self.indicator_cache[cache_key]
                if time.time() - cache_time < 60:
                    return cached_data
            
            price_data = self._get_price_series(ticker, start_time, end_time)
            
            if price_data is None or len(price_data) < 20:
                logger.warning(f"价格数据不足以计算技术指标: {ticker}")
                return self._get_default_signals()
            
            macd_result = self.compute_macd(price_data)
            rsi_result = self.compute_rsi(price_data)
            bollinger_result = self.compute_bollinger(price_data)
            
            bullish_count = 0
            bearish_count = 0
            
            if macd_result['signal'] == 'bullish':
                bullish_count += 1
            elif macd_result['signal'] == 'bearish':
                bearish_count += 1
            
            if rsi_result['signal'] in ['oversold', 'bullish']:
                bullish_count += 1
            elif rsi_result['signal'] in ['overbought', 'bearish']:
                bearish_count += 1
            
            if bollinger_result['signal'] == 'below_lower':
                bullish_count += 1
            elif bollinger_result['signal'] == 'above_upper':
                bearish_count += 1
            
            if bullish_count >= 2:
                overall_signal = 'bullish'
            elif bearish_count >= 2:
                overall_signal = 'bearish'
            else:
                overall_signal = 'neutral'
            
            result = {
                'macd': macd_result,
                'rsi': rsi_result,
                'bollinger': bollinger_result,
                'overall_signal': overall_signal,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count
            }
            
            self.indicator_cache[cache_key] = (result, time.time())
            
            return result
            
        except Exception as e:
            logger.error(f"获取技术指标信号失败: {ticker}, {e}")
            return self._get_default_signals()
    
    def _get_price_series(self, ticker: str, start_time: datetime, end_time: datetime) -> Optional[pd.Series]:
        """获取价格序列"""
        if self.data_collector is None:
            return None
        
        try:
            df = self.data_collector.get_historical_data(
                ticker,
                start_time.strftime('%Y-%m-%d'),
                end_time.strftime('%Y-%m-%d'),
                'daily'
            )
            
            if df.empty or 'close' not in df.columns:
                return None
            
            return df['close']
            
        except Exception as e:
            logger.error(f"获取价格序列失败: {ticker}, {e}")
            return None
    
    def _get_default_signals(self) -> Dict[str, Any]:
        """获取默认信号值"""
        return {
            'macd': {
                'macd_line': 0.0,
                'signal_line': 0.0,
                'histogram': 0.0,
                'signal': 'neutral'
            },
            'rsi': {
                'rsi': 50.0,
                'signal': 'neutral'
            },
            'bollinger': {
                'upper': 0.0,
                'middle': 0.0,
                'lower': 0.0,
                'width': 0.0,
                'signal': 'in_band'
            },
            'overall_signal': 'neutral',
            'bullish_count': 0,
            'bearish_count': 0
        }
    
    def calculate_technical_factors(self, ticker: str, timestamp: datetime, 
                                  current_price: Optional[float] = None,
                                  current_volume: Optional[float] = None) -> Dict[str, float]:
        """计算技术指标因子"""
        try:
            end_time = timestamp
            start_time = timestamp - timedelta(minutes=30)
            
            price_data = self._get_price_data(ticker, start_time, end_time)
            volume_data = self._get_volume_data(ticker, start_time, end_time)
            
            factors = {
                'price_change_5min': self._calculate_price_change(price_data, current_price, 5),
                'price_change_15min': self._calculate_price_change(price_data, current_price, 15),
                'volume_spike': self._calculate_volume_spike(volume_data, current_volume),
                'volatility_5min': self._calculate_volatility(price_data, 5)
            }
            
            indicator_signals = self.get_indicator_signals(ticker, timestamp)
            
            factors['macd_signal'] = indicator_signals['macd']['signal']
            factors['rsi_value'] = indicator_signals['rsi']['rsi']
            factors['rsi_signal'] = indicator_signals['rsi']['signal']
            factors['bollinger_signal'] = indicator_signals['bollinger']['signal']
            factors['technical_overall'] = indicator_signals['overall_signal']
            
            return factors
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {ticker} - {e}")
            return {
                'price_change_5min': 0.0,
                'price_change_15min': 0.0,
                'volume_spike': 1.0,
                'volatility_5min': 0.0,
                'macd_signal': 'neutral',
                'rsi_value': 50.0,
                'rsi_signal': 'neutral',
                'bollinger_signal': 'in_band',
                'technical_overall': 'neutral'
            }
    
    def _get_price_data(self, ticker: str, start_time: datetime, end_time: datetime) -> List[Tuple[datetime, float]]:
        """获取价格数据"""
        if self.data_collector is None:
            return []
        
        try:
            df = self.data_collector.get_historical_data(
                ticker,
                start_time.strftime('%Y-%m-%d'),
                end_time.strftime('%Y-%m-%d'),
                '5Min'
            )
            
            if df.empty or 'close' not in df.columns:
                return []
            
            price_data = []
            for idx, row in df.iterrows():
                if isinstance(idx, datetime) and start_time <= idx <= end_time:
                    close_price = row['close']
                    if pd.notna(close_price) and close_price > 0:
                        price_data.append((idx, float(close_price)))
            
            return sorted(price_data, key=lambda x: x[0])
            
        except Exception as e:
            logger.error(f"获取价格数据失败: {ticker}, {e}")
            return []
    
    def _get_volume_data(self, ticker: str, start_time: datetime, end_time: datetime) -> List[Tuple[datetime, float]]:
        """获取成交量数据"""
        if self.data_collector is None:
            return []
        
        try:
            df = self.data_collector.get_historical_data(
                ticker,
                start_time.strftime('%Y-%m-%d'),
                end_time.strftime('%Y-%m-%d'),
                '5Min'
            )
            
            if df.empty or 'volume' not in df.columns:
                return []
            
            volume_data = []
            for idx, row in df.iterrows():
                if isinstance(idx, datetime) and start_time <= idx <= end_time:
                    volume = row['volume']
                    if pd.notna(volume) and volume >= 0:
                        volume_data.append((idx, float(volume)))
            
            return sorted(volume_data, key=lambda x: x[0])
            
        except Exception as e:
            logger.error(f"获取成交量数据失败: {ticker}, {e}")
            return []
    
    def _calculate_price_change(self, price_data: List[Tuple[datetime, float]], 
                              current_price: Optional[float], 
                              minutes: int) -> float:
        """计算价格变动百分比"""
        if not price_data or current_price is None:
            return 0.0
        
        try:
            target_time = price_data[-1][0] - timedelta(minutes=minutes)
            
            prev_price = None
            for timestamp, price in reversed(price_data):
                if timestamp <= target_time:
                    prev_price = price
                    break
            
            if prev_price is None or prev_price == 0:
                return 0.0
            
            price_change = (current_price - prev_price) / prev_price
            return float(np.clip(price_change, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"计算价格变动失败: {e}")
            return 0.0
    
    def _calculate_volume_spike(self, volume_data: List[Tuple[datetime, float]], 
                               current_volume: Optional[float]) -> float:
        """计算成交量异常倍数"""
        if not volume_data or current_volume is None:
            return 1.0
        
        try:
            volumes = [vol for _, vol in volume_data if vol > 0]
            if not volumes:
                return 1.0
            
            avg_volume = np.mean(volumes)
            if avg_volume == 0:
                return 1.0
            
            volume_spike = current_volume / avg_volume
            return float(np.clip(volume_spike, 0.1, 10.0))
            
        except Exception as e:
            logger.error(f"计算成交量异常失败: {e}")
            return 1.0
    
    def _calculate_volatility(self, price_data: List[Tuple[datetime, float]], 
                            minutes: int) -> float:
        """计算指定时间窗口的价格波动率"""
        if len(price_data) < 2:
            return 0.0
        
        try:
            cutoff_time = price_data[-1][0] - timedelta(minutes=minutes)
            recent_prices = [price for timestamp, price in price_data if timestamp > cutoff_time]
            
            if len(recent_prices) < 2:
                return 0.0
            
            price_returns = []
            for i in range(1, len(recent_prices)):
                if recent_prices[i-1] > 0:
                    ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                    price_returns.append(ret)
            
            if not price_returns:
                return 0.0
            
            volatility = np.std(price_returns)
            return float(np.clip(volatility, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"计算波动率失败: {e}")
            return 0.0


class EnhancedNewsFactorExtractor:
    """增强版新闻因子提取器 - 使用FinGPT Llama2-13B LoRA"""
    
    def __init__(self,
                 fingpt_analyzer: Optional[FinGPTSentimentAnalyzer] = None,
                 embedding_model: Optional[FinancialEmbeddingModel] = None,
                 data_collector=None,
                 use_async: bool = False,  # FinGPT模型较大，默认同步处理
                 batch_size: int = 4):     # 减小批次大小
        """初始化增强版因子提取器"""
        # 模型
        self.fingpt_analyzer = fingpt_analyzer or FinGPTSentimentAnalyzer()
        self.embedding_model = embedding_model or FinancialEmbeddingModel()
        
        # 新颖度和热度检测器
        self.novelty_detector = EnhancedNoveltyDetector()
        
        # 技术指标计算器
        self.technical_calculator = TechnicalFactorCalculator(data_collector)
        
        # 异步处理器（暂时禁用，FinGPT模型较大，批处理复杂）
        self.use_async = False  # 强制同步
        
        # 历史缓存
        self.similarity_cache = defaultdict(lambda: deque(maxlen=10))
        
        # 因子存储（用于SignalGenerator查询）
        self.sentiment_scores = {}  # {(ticker, time): score}
        
        logger.info("增强版因子提取器初始化完成（使用FinGPT Llama2-13B LoRA）")
    
    def update_sentiment_factor(self, ticker: str, timestamp: datetime, score: int):
        """更新情绪因子（事件打分）"""
        key = (ticker, timestamp.replace(second=0, microsecond=0))
        self.sentiment_scores[key] = score
        logger.debug(f"更新情绪因子: {ticker} @ {timestamp} = {score}")
    
    def get_sentiment_score(self, ticker: str, timestamp: datetime) -> Optional[int]:
        """获取情绪因子（事件打分）"""
        key = (ticker, timestamp.replace(second=0, microsecond=0))
        return self.sentiment_scores.get(key)
    
    def extract_factors_from_news_articles(self, news_articles: List) -> List[EnhancedNewsFactor]:
        """从NewsArticle对象列表提取因子"""
        news_dicts = []
        for article in news_articles:
            if hasattr(article, 'to_dict'):
                news_dict = article.to_dict()
            elif isinstance(article, dict):
                news_dict = article.copy()
            else:
                logger.warning(f"未知的新闻数据格式: {type(article)}")
                continue
            
            # 确保包含必要字段
            required_fields = ['ticker', 'datetime', 'headline', 'summary', 'source']
            for field in required_fields:
                if field not in news_dict:
                    news_dict[field] = ''
            
            # 确保datetime字段为datetime对象
            if isinstance(news_dict['datetime'], datetime):
                pass
            elif isinstance(news_dict['datetime'], str):
                try:
                    news_dict['datetime'] = datetime.fromisoformat(news_dict['datetime'].replace('Z', '+00:00'))
                except:
                    news_dict['datetime'] = datetime.now()
            else:
                news_dict['datetime'] = datetime.now()
            
            news_dicts.append(news_dict)
        
        if not news_dicts:
            logger.warning("没有有效的新闻数据用于因子提取")
            return []
        
        logger.info(f"从{len(news_articles)}条新闻文章中提取因子")
        return self.extract_factors_sync(news_dicts)
    
    def extract_factors_sync(self, news_list: List[Dict[str, Any]]) -> List[EnhancedNewsFactor]:
        """同步提取增强因子"""
        factors = []
        
        # 按ticker分组以提高处理效率
        news_by_ticker = defaultdict(list)
        for news in news_list:
            ticker = news.get('ticker', 'UNKNOWN')
            news_by_ticker[ticker].append(news)
        
        for ticker, ticker_news in news_by_ticker.items():
            try:
                # 准备文本和公司信息
                texts = [self._prepare_text(news) for news in ticker_news]
                companies = [news.get('company') for news in ticker_news]
                
                # 批量FinGPT分析（使用新的LoRA模型）
                fingpt_results = self.fingpt_analyzer.batch_analyze(
                    list(zip(texts, companies))
                )
                
                # 批量嵌入
                embeddings = self.embedding_model.batch_get_embeddings(texts)
                
                # 处理每条新闻
                for i, news in enumerate(ticker_news):
                    fingpt_result = fingpt_results[i]
                    embedding = embeddings[i]
                    timestamp = self._parse_datetime(news.get('datetime'))
                    
                    # 提取FinGPT分析结果
                    sentiment_label = fingpt_result['sentiment']
                    sentiment_score = fingpt_result['sentiment_score']
                    event_score = fingpt_result['event_score']
                    confidence = fingpt_result['confidence']
                    rationale = fingpt_result['rationale']
                    
                    # 新颖度和热度检测
                    novelty, cluster_dist, event_hash, similar_count, news_heat, time_decay = \
                        self.novelty_detector.calculate_novelty_and_heat(
                            ticker, embedding, news.get('headline', ''), timestamp, news.get('source', '')
                        )
                    
                    # 确定事件影响等级
                    impact_map = {
                        5: "extreme",
                        4: "high", 
                        3: "medium",
                        2: "low",
                        1: "minimal"
                    }
                    event_impact = impact_map.get(event_score, "minimal")
                    
                    # 更新情绪因子存储
                    self.update_sentiment_factor(ticker, timestamp, event_score)
                    
                    # 获取技术指标
                    technical_factors = self.technical_calculator.calculate_technical_factors(
                        ticker, timestamp, 
                        news.get('current_price'),
                        news.get('current_volume')
                    )
                    
                    # 获取技术指标信号
                    indicator_signals = self.technical_calculator.get_indicator_signals(ticker, timestamp)
                    
                    # 相似度计算
                    similarity_avg = self._calculate_similarity(ticker, embedding)
                    
                    # 综合有效性判断
                    is_valid_signal = self._evaluate_signal_validity(
                        sentiment_score, confidence, novelty, news_heat, technical_factors, event_score
                    )
                    
                    # 计算综合置信度
                    confidence_composite = self._calculate_composite_confidence(
                        sentiment_score, confidence, novelty, news_heat, time_decay, technical_factors, event_score
                    )
                    
                    # 创建增强因子
                    factor = EnhancedNewsFactor(
                        ticker=ticker,
                        datetime=timestamp,
                        headline=news.get('headline', ''),
                        
                        # 情绪因子（来自FinGPT）
                        sentiment_score=sentiment_score,
                        sentiment_label=sentiment_label,
                        sentiment_prob=confidence,
                        uncertainty=1.0 - confidence,
                        rationale=rationale,
                        
                        # 事件打分
                        event_score=event_score,
                        event_impact=event_impact,
                        
                        # 技术指标因子
                        price_change_5min=technical_factors['price_change_5min'],
                        price_change_15min=technical_factors['price_change_15min'],
                        volume_spike=technical_factors['volume_spike'],
                        volatility_5min=technical_factors['volatility_5min'],
                        
                        # 经典技术指标
                        macd_signal=technical_factors.get('macd_signal', 'neutral'),
                        rsi_value=technical_factors.get('rsi_value', 50.0),
                        rsi_signal=technical_factors.get('rsi_signal', 'neutral'),
                        bollinger_signal=technical_factors.get('bollinger_signal', 'in_band'),
                        
                        # 新闻因子
                        similarity_avg=similarity_avg,
                        novelty=novelty,
                        cluster_distance=cluster_dist,
                        similar_count=similar_count,
                        news_heat=news_heat,
                        time_decay=time_decay,
                        
                        # 元数据
                        event_hash=event_hash,
                        is_valid_signal=is_valid_signal,
                        confidence_composite=confidence_composite,
                        
                        # 技术指标详细数据
                        technical_indicators=indicator_signals,
                        
                        # FinGPT原始响应
                        fingpt_response=fingpt_result.get('raw_response', ''),
                        
                        embedding=embedding
                    )
                    
                    factors.append(factor)
                    
                    # 更新缓存
                    self.similarity_cache[ticker].append(embedding)
                    
                    logger.info(
                        f"FinGPT因子提取: {ticker} | "
                        f"事件打分={event_score}({event_impact}), "
                        f"情绪={sentiment_score:.3f}({sentiment_label}) "
                        f"置信度={confidence:.3f}, "
                        f"新颖度={novelty:.3f}, 热度={news_heat:.3f}, "
                        f"MACD={technical_factors.get('macd_signal', 'NA')}, "
                        f"RSI={technical_factors.get('rsi_value', 'NA'):.1f}, "
                        f"综合置信度={confidence_composite:.3f}"
                    )
                
            except Exception as e:
                logger.error(f"处理{ticker}新闻失败: {e}")
                continue
        
        logger.info(f"成功提取 {len(factors)} 个增强因子")
        
        # 输出统计信息
        if hasattr(self.fingpt_analyzer, 'get_analysis_stats'):
            stats = self.fingpt_analyzer.get_analysis_stats()
            logger.info(f"FinGPT分析统计: {stats}")
        
        return factors
    
    def _prepare_text(self, news: Dict) -> str:
        """准备输入文本"""
        headline = news.get('headline', '')
        summary = news.get('summary', '')
        
        # 组合标题和摘要，限制长度
        if summary and len(summary) > 20:
            return f"{headline}. {summary}"[:800]
        return headline[:800]
    
    def _parse_datetime(self, dt) -> datetime:
        """解析时间"""
        if isinstance(dt, datetime):
            return dt
        elif isinstance(dt, str):
            try:
                return datetime.fromisoformat(dt.replace('Z', '+00:00'))
            except:
                return datetime.now()
        elif isinstance(dt, (int, float)):
            return datetime.fromtimestamp(dt)
        else:
            return datetime.now()
    
    def _calculate_similarity(self, ticker: str, embedding: np.ndarray) -> float:
        """计算与历史新闻的相似度"""
        cache = list(self.similarity_cache[ticker])
        if not cache:
            return 0.0
        
        recent_embeddings = cache[-5:]
        similarities = []
        
        for cached_emb in recent_embeddings:
            sim = self.embedding_model.compute_similarity(embedding, cached_emb)
            similarities.append(sim)
        
        return float(np.mean(similarities))
    
    def _evaluate_signal_validity(self, sentiment_score: float, confidence: float,
                                 novelty: float, news_heat: float, 
                                 technical_factors: Dict, event_score: int) -> bool:
        """评估信号有效性（考虑FinGPT事件打分）"""
        # 基础过滤：置信度
        if confidence < 0.5:
            return False
        
        # 事件打分过滤：分数太低的事件无效
        if event_score < 2:
            return False
        
        # 新颖度过滤：过于相似的新闻无效
        if novelty < 0.1:
            return False
        
        # 技术指标确认（对于高分事件可以放宽）
        technical_overall = technical_factors.get('technical_overall', 'neutral')
        if event_score >= 4:
            # 高分事件，技术面不反对即可
            return technical_overall != 'bearish' if sentiment_score > 0 else technical_overall != 'bullish'
        else:
            # 低分事件，需要技术面支持
            sentiment_direction = 'bullish' if sentiment_score > 0 else 'bearish'
            return technical_overall == sentiment_direction or technical_overall == 'neutral'
    
    def _calculate_composite_confidence(self, sentiment_score: float, confidence: float,
                                      novelty: float, news_heat: float, time_decay: float,
                                      technical_factors: Dict, event_score: int) -> float:
        """计算综合置信度（考虑FinGPT事件打分）"""
        try:
            # 各组件权重（事件打分影响权重分配）
            if event_score >= 4:
                # 高分事件，情绪和事件权重更高
                sentiment_weight = 0.35
                event_weight = 0.35
                technical_weight = 0.15
                novelty_weight = 0.1
                heat_weight = 0.05
            else:
                # 低分事件，技术面权重更高
                sentiment_weight = 0.2
                event_weight = 0.2
                technical_weight = 0.35
                novelty_weight = 0.15
                heat_weight = 0.1
            
            # FinGPT置信度
            fingpt_conf = confidence
            
            # 事件置信度（基于事件打分）
            event_conf = event_score / 5.0
            
            # 技术指标置信度
            technical_overall = technical_factors.get('technical_overall', 'neutral')
            if technical_overall == 'bullish' or technical_overall == 'bearish':
                technical_conf = 0.8
            else:
                technical_conf = 0.3
            
            # 综合置信度
            composite = (
                fingpt_conf * sentiment_weight +
                event_conf * event_weight +
                technical_conf * technical_weight +
                novelty * novelty_weight +
                news_heat * heat_weight
            )
            
            # 时间衰减调整
            composite *= (0.7 + 0.3 * time_decay)
            
            return float(np.clip(composite, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"计算综合置信度失败: {e}")
            return 0.5


# 单例管理
_fingpt_analyzer = None
_embedding_model = None

def get_fingpt_models(local_model_path: Optional[str] = None):
    """获取FinGPT模型实例"""
    global _fingpt_analyzer, _embedding_model
    
    if _fingpt_analyzer is None:
        _fingpt_analyzer = FinGPTSentimentAnalyzer(
            local_model_path=local_model_path
        )
    if _embedding_model is None:
        _embedding_model = FinancialEmbeddingModel()
    
    return _fingpt_analyzer, _embedding_model

def get_enhanced_extractor(data_collector=None, **kwargs) -> EnhancedNewsFactorExtractor:
    """获取增强因子提取器"""
    fingpt_analyzer, embedding_model = get_fingpt_models()
    return EnhancedNewsFactorExtractor(
        fingpt_analyzer=fingpt_analyzer,
        embedding_model=embedding_model,
        data_collector=data_collector,
        **kwargs
    )


# AWS部署提示
def print_aws_deployment_guide():
    """打印AWS部署指南"""
    print("""
    === AWS部署FinGPT Llama2-13B LoRA指南 ===
    
    1. EC2实例选择：
       - 推荐: g5.2xlarge (A10G 24GB) 或 p3.2xlarge (V100 16GB)
       - 最低: g4dn.xlarge (T4 16GB) - 需要8bit量化
       - 操作系统: Deep Learning AMI (Ubuntu 20.04)
    
    2. 环境准备：
       ```bash
       # 更新系统
       sudo apt update && sudo apt upgrade -y
       
       # 安装Python依赖
       pip install transformers==4.32.0 peft==0.5.0
       pip install sentencepiece accelerate bitsandbytes
       pip install sentence-transformers scikit-learn
       pip install torch  # 确保CUDA版本
       
       # 登录HuggingFace（如需要）
       huggingface-cli login
       ```
    
    3. 模型下载：
       模型会在首次运行时自动下载，包括：
       - 基础模型: NousResearch/Llama-2-13b-hf (~26GB)
       - LoRA权重: FinGPT/fingpt-sentiment_llama2-13b_lora (~1GB)
    
    4. 优化建议：
       - 使用8bit量化可将显存需求降至~13GB
       - 使用S3存储模型文件，避免重复下载
       - 配置CloudWatch监控GPU使用率
       - 使用Spot实例可节省70%成本
    """)


# 测试代码
if __name__ == "__main__":
    print("\n=== FinGPT Llama2-13B LoRA因子建模模块测试 ===\n")
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告: 未检测到GPU，FinGPT将在CPU上运行（速度很慢）")
    
    # 打印部署指南
    print_aws_deployment_guide()
    
    # 测试数据
    test_news = [
        {
            'ticker': 'AAPL',
            'datetime': datetime.now() - timedelta(hours=2),
            'headline': 'Apple announces revolutionary breakthrough in AI technology with record-breaking revenue growth',
            'summary': 'Major acquisition completed, earnings beat expectations by 50%, stock price surges',
            'source': 'Reuters',
            'company': 'Apple Inc.',
            'current_price': 150.0,
            'current_volume': 2000000
        },
        {
            'ticker': 'MSFT',
            'datetime': datetime.now() - timedelta(hours=1),
            'headline': 'Microsoft faces major antitrust investigation, potential billion-dollar fine looms',
            'summary': 'Regulatory pressure mounts, massive layoffs announced, revenue misses by 30%',
            'source': 'Bloomberg',
            'company': 'Microsoft Corporation',
            'current_price': 280.0,
            'current_volume': 1500000
        }
    ]
    
    try:
        # 创建增强版提取器（使用FinGPT LoRA）
        print("\n初始化FinGPT Llama2-13B LoRA因子提取器...")
        extractor = get_enhanced_extractor(data_collector=None, use_async=False)
        
        # 提取因子
        print("\n使用FinGPT LoRA提取因子...")
        factors = extractor.extract_factors_sync(test_news)
        
        print(f"\n提取到 {len(factors)} 个增强因子")
        print("=" * 120)
        
        for factor in factors:
            print(f"\n股票: {factor.ticker}")
            print(f"时间: {factor.datetime.strftime('%Y-%m-%d %H:%M')}")
            print(f"标题: {factor.headline[:80]}...")
            
            # FinGPT LoRA分析结果
            print(f"\n[FinGPT LoRA分析]")
            print(f"事件打分: {factor.event_score}/5 ({factor.event_impact})")
            print(f"情绪: {factor.sentiment_label} (得分: {factor.sentiment_score:.3f})")
            print(f"置信度: {factor.sentiment_prob:.3f}")
            print(f"理由: {factor.rationale}")
            
            # 技术指标
            print(f"\n[技术指标]")
            print(f"MACD信号: {factor.macd_signal}")
            print(f"RSI: {factor.rsi_value:.1f} ({factor.rsi_signal})")
            print(f"布林带: {factor.bollinger_signal}")
            
            # 综合评估
            print(f"\n[综合评估]")
            print(f"有效信号: {factor.is_valid_signal}")
            print(f"综合置信度: {factor.confidence_composite:.3f}")
            print(f"新颖度: {factor.novelty:.3f}")
            print(f"新闻热度: {factor.news_heat:.3f}")
            print("-" * 120)
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保:")
        print("1. 有足够的GPU显存（建议16GB以上）")
        print("2. 已安装必要的依赖包: transformers==4.32.0 peft==0.5.0 bitsandbytes")
        print("3. 可以访问HuggingFace模型（网络连接正常）")
        print("4. 如需要，请先登录HuggingFace: huggingface-cli login")
    
    print("\n测试完成!")
    print("\n主要特性:")
    print("✅ 使用FinGPT fingpt-sentiment_llama2-13b_lora专业金融模型")
    print("✅ 自动生成事件打分（1-5分）和情绪分析")
    print("✅ 集成MACD、布林带、RSI等经典技术指标")
    print("✅ 支持AWS GPU实例部署（g5.2xlarge推荐）")
    print("✅ 8bit量化支持，降低显存需求至~13GB")
    print("✅ 完全兼容原有signal_generator和portfolio_manager接口")