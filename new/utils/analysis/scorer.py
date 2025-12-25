"""
币对评分模块
"""
from typing import List
from api import BinanceAPI
from utils import Config, Logger

class PairScorer:
    """币对评分器 - 选择最佳交易对"""
    
    def __init__(self, api: BinanceAPI, config: Config, logger: Logger):
        self.api = api
        self.config = config
        self.logger = logger
    
    def select_top_pairs(self) -> List[str]:
        """选择评分最高的交易对"""
        try:
            tickers = self.api.get_all_tickers()
            
            # 过滤和评分
            scored_pairs = []
            for ticker in tickers:
                symbol = ticker['symbol']
                volume = ticker.get('volume', 0)
                change = abs(ticker.get('change', 0))
                
                # 过滤条件
                if volume < self.config.MIN_24H_VOLUME:
                    continue
                
                # 计算评分（成交量 + 波动率）
                score = volume * (1 + change / 100)
                scored_pairs.append((symbol, score))
            
            # 按评分排序
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前N个
            top_pairs = [pair[0] for pair in scored_pairs[:self.config.TOP_PAIRS_COUNT]]
            
            self.logger.info(f"选择了 {len(top_pairs)} 个交易对")
            return top_pairs
            
        except Exception as e:
            self.logger.error(f"选择币对失败: {e}")
            return []

