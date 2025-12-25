
from typing import Dict
import numpy as np
import pandas as pd
from utils import Config

# ==================== 趋势分析器 ====================
class TrendAnalyzer:
    """高级趋势分析器"""

    def __init__(self, config: Config):
        self.config = config

    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        """全面分析趋势"""
        if len(df) < 200:
            return {"direction": "neutral", "strength": 0, "quality": 0, "valid": False}

        # 1. 趋势方向判断
        direction = self._get_trend_direction(df)

        # 2. 趋势强度 (ADX)
        strength = self._get_trend_strength(df)

        # 3. 趋势质量
        quality = self._get_trend_quality(df)

        # 4. 趋势有效性
        valid = strength >= self.config.TREND_STRENGTH_THRESHOLD and quality > 0.5

        return {
            "direction": direction,
            "strength": strength,
            "quality": quality,
            "valid": valid,
        }

    def _get_trend_direction(self, df: pd.DataFrame) -> str:
        """判断趋势方向"""
        # 多周期均线判断
        ma_fast = df["close"].rolling(window=20).mean().iloc[-1]
        ma_slow = df["close"].rolling(window=50).mean().iloc[-1]
        ma_trend = df["close"].rolling(window=200).mean().iloc[-1]
        current_price = df["close"].iloc[-1]

        # 多头排列
        if current_price > ma_fast > ma_slow > ma_trend:
            return "uptrend"
        # 空头排列
        elif current_price < ma_fast < ma_slow < ma_trend:
            return "downtrend"
        else:
            return "sideways"

    def _get_trend_strength(self, df: pd.DataFrame) -> float:
        """计算趋势强度 (0-1)"""
        # 使用ADX
        if "adx" in df.columns:
            adx = df["adx"].iloc[-1]
            # ADX标准化到0-1
            strength = min(adx / 50, 1.0)  # ADX>50视为强趋势
        else:
            # 备用方法: 价格与均线的距离
            ma_slow = df["close"].rolling(window=50).mean().iloc[-1]
            price_deviation = abs(df["close"].iloc[-1] - ma_slow) / ma_slow
            strength = min(price_deviation * 10, 1.0)

        return strength

    def _get_trend_quality(self, df: pd.DataFrame) -> float:
        """评估趋势质量 (0-1)"""
        scores = []

        # 1. 均线角度 (越陡越好)
        ma_fast = df["close"].rolling(window=20).mean()
        ma_angle = (ma_fast.iloc[-1] - ma_fast.iloc[-5]) / ma_fast.iloc[-5]
        angle_score = min(abs(ma_angle) * 100, 1.0)
        scores.append(angle_score)

        # 2. 回撤控制 (回撤越小越好)
        recent_high = df["high"].tail(20).max()
        recent_low = df["low"].tail(20).min()
        drawdown = (recent_high - df["close"].iloc[-1]) / recent_high
        drawdown_score = max(1 - drawdown * 5, 0)
        scores.append(drawdown_score)

        # 3. 成交量趋势 (放量趋势更可靠)
        vol_trend = df["volume"].tail(20).mean() / df["volume"].tail(50).mean()
        volume_score = min(vol_trend, 1.0)
        scores.append(volume_score)

        return np.mean(scores)

    def is_trend_aligned(self, df: pd.DataFrame, signal: int) -> bool:
        """检查信号是否与趋势一致"""
        trend = self.analyze_trend(df)

        if not trend["valid"]:
            return False  # 无明显趋势,拒绝交易

        # 买入信号必须在上升趋势
        if signal == 1 and trend["direction"] != "uptrend":
            return False

        # 卖出信号在下降趋势更可靠(但也允许在上升趋势中止盈)
        if signal == -1 and trend["direction"] == "uptrend":
            # 上升趋势中的卖出信号要更谨慎
            if trend["strength"] > 0.6:  # 强趋势不轻易卖出
                return False

        return True
