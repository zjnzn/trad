"""
币对评分模块
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
from api.binance_api import BinanceAPI
from ..logger import Logger
from ..config import Config

# ==================== 币对评分系统 ====================
class PairScorer:
    """优化的币对评分器"""

    def __init__(self, api: BinanceAPI, config: Config, logger: Logger):
        self.api = api
        self.config = config
        self.logger = logger

    def score_pair(self, ticker: Dict) -> float:
        """对单个币对评分 - 优化评分算法"""
        score = 0.0

        # 成交量权重 (35%)
        if ticker["volume"] > self.config.MIN_24H_VOLUME:
            volume_score = min(np.log10(ticker["volume"]) / 2, 10)
            score += volume_score * 0.35
        else:
            return 0

        # 波动率权重 (30%) - 理想范围2-10%
        volatility = abs(ticker.get("change", 0))
        if 2 < volatility < 10:
            score += (1 - abs(volatility - 6) / 10) * 3.0
        elif volatility > 15:
            score -= 2.0  # 惩罚过高波动

        # 价格水平 (15%) - 避免过低价格币
        if ticker["price"] > 0.0001:
            score += 1.5

        # 趋势权重 (20%) - 优先上涨趋势
        if ticker.get("change", 0) > 0:
            score += min(ticker["change"] / 5, 2.0)

        return max(0, score)

    def select_top_pairs(self) -> List[str]:
        """选择评分最高的币对 - 并行评分"""
        self.logger.info("开始评分币对...")

        tickers = self.api.get_all_tickers()
        if not tickers:
            return []

        # 快速过滤
        filtered = [t for t in tickers if t["volume"] >= self.config.MIN_24H_VOLUME]

        # 并行评分
        scored_pairs = []
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            future_to_ticker = {
                executor.submit(self.score_pair, t): t for t in filtered
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    score = future.result()
                    if score > 0:
                        scored_pairs.append((ticker["symbol"], score))
                except Exception as e:
                    self.logger.error(f"评分失败 {ticker['symbol']}: {e}")

        # 按评分排序
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = [pair[0] for pair in scored_pairs[: self.config.TOP_PAIRS_COUNT]]

        self.logger.info(f"已选择 {len(top_pairs)} 个高评分币对")
        if top_pairs:
            self.logger.info(f"Top 5: {', '.join(top_pairs[:5])}")

        return top_pairs
    




