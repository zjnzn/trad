
# ==================== 策略基类 ====================
from typing import Dict, List, Optional
import pandas as pd

from .config import Config
from .analysis.indicators import IndicatorComputer


class Strategy:
    """优化的策略基类"""

    def __init__(self, name: str):
        self.name = name
        self.params = {}

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算交易信号"""
        raise NotImplementedError

    def should_enter(self, df: pd.DataFrame) -> bool:
        """是否应该入场"""
        if len(df) < 2:
            return False
        return df["signal"].iloc[-1] == 1 and df["signal"].iloc[-2] != 1  # 避免重复信号

    def should_exit(self, df: pd.DataFrame, position: Dict) -> bool:
        """是否应该出场"""
        if len(df) < 2:
            return False
        return df["signal"].iloc[-1] == -1

# ==================== RSI策略 ====================
class RSIStrategy(Strategy):
    """优化的RSI策略"""

    def __init__(self):
        super().__init__("RSI_STRATEGY")
        self.params = {
            "period": 14,
            "oversold": 30,
            "overbought": 70,
            "smooth": 3,  # 信号平滑
        }

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算RSI信号"""
        if len(df) < self.params["period"] + 1:
            df["signal"] = 0
            return df

        period = self.params["period"]
        smooth = self.params["smooth"]

        # 计算RSI
        IndicatorComputer.compute_rsi(period, df)

        # 信号平滑
        IndicatorComputer.compute_rsi_smooth(smooth, df)

        # 生成信号
        df["signal"] = 0
        df.loc[df["rsi_smooth"] < self.params["oversold"], "signal"] = 1
        df.loc[df["rsi_smooth"] > self.params["overbought"], "signal"] = -1

        return df


# ==================== MACD策略 ====================
class MACDStrategy(Strategy):
    """优化的MACD策略"""

    def __init__(self):
        super().__init__("MACD_STRATEGY")
        self.params = {"fast": 12, "slow": 26, "signal": 9}

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD信号"""
        if len(df) < self.params["slow"] + 1:
            df["signal"] = 0
            return df

        fast = self.params["fast"]
        slow = self.params["slow"]
        signal = self.params["signal"]
        # 计算MACD
        IndicatorComputer.compute_macd(fast, slow, df)
        IndicatorComputer.compute_signal_line(signal, df)
        IndicatorComputer.compute_histogram(df)

        # 生成信号 - 添加趋势过滤
        df["signal"] = 0
        # 金叉且柱状图递增
        df.loc[
            (df["macd"] > df["signal_line"])
            & (df["macd"].shift(1) <= df["signal_line"].shift(1))
            & (df["histogram"] > df["histogram"].shift(1)),
            "signal",
        ] = 1
        # 死叉
        df.loc[
            (df["macd"] < df["signal_line"])
            & (df["macd"].shift(1) >= df["signal_line"].shift(1)),
            "signal",
        ] = -1

        return df


# ==================== 布林带策略 ====================
class BollingerStrategy(Strategy):
    """优化的布林带策略"""

    def __init__(self):
        super().__init__("BOLLINGER_STRATEGY")
        self.params = {
            "period": 20,
            "std_dev": 2,
            "squeeze_threshold": 0.02,  # 布林带收窄阈值
        }

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带信号"""
        if len(df) < self.params["period"] + 1:
            df["signal"] = 0
            return df

        period = self.params["period"]
        std_dev = self.params["std_dev"]

        # 计算布林带
        IndicatorComputer.compute_bandwidth(period, std_dev, df)

        # 生成信号 - 突破下轨且带宽不太窄
        df["signal"] = 0
        df.loc[
            (df["close"] < df["band_lower"])
            & (df["band_width"] > self.params["squeeze_threshold"]),
            "signal",
        ] = 1
        df.loc[df["close"] > df["band_upper"], "signal"] = -1

        return df


# ==================== 均线交叉策略 ====================
class MACrossStrategy(Strategy):
    """双均线交叉策略 - 经典趋势跟踪"""

    def __init__(self):
        super().__init__("MA_CROSS_STRATEGY")
        self.params = {
            "fast_period": 10,  # 快线
            "slow_period": 30,  # 慢线
            "trend_filter": 100,  # 趋势过滤线
        }

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算均线交叉信号"""
        if len(df) < self.params["slow_period"] + 1:
            df["signal"] = 0
            return df
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]
        trend_filter = self.params["trend_filter"]
        # 计算均线
        IndicatorComputer.compute_ma(fast_period, slow_period, trend_filter, df)

        # 生成信号 - 金叉且在趋势线上方
        df["signal"] = 0
        df.loc[
            (df["ma_fast"] > df["ma_slow"])
            & (df["ma_fast"].shift(1) <= df["ma_slow"].shift(1))
            & (df["close"] > df["ma_trend"]),
            "signal",
        ] = 1

        # 死叉出场
        df.loc[
            (df["ma_fast"] < df["ma_slow"])
            & (df["ma_fast"].shift(1) >= df["ma_slow"].shift(1)),
            "signal",
        ] = -1

        return df


# ==================== 网格交易策略 ====================
class GridStrategy(Strategy):
    """网格交易策略 - 震荡市场"""

    def __init__(self):
        super().__init__("GRID_STRATEGY")
        self.params = {
            "grid_size": 0.02,  # 网格间距2%
            "lookback": 100,
            "grids": 5,  # 网格数量
        }
        self.grid_levels = []
        self.last_price = None

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算网格信号"""
        if len(df) < self.params["lookback"]:
            df["signal"] = 0
            return df

        # 计算价格范围
        high = df["high"].tail(self.params["lookback"]).max()
        low = df["low"].tail(self.params["lookback"]).min()
        mid = (high + low) / 2

        # 生成网格线
        self.grid_levels = []
        for i in range(-self.params["grids"], self.params["grids"] + 1):
            level = mid * (1 + i * self.params["grid_size"])
            self.grid_levels.append(level)

        # 计算信号
        df["signal"] = 0
        current_price = df["close"].iloc[-1]

        # 价格接近下网格线买入
        for level in self.grid_levels:
            if abs(current_price - level) / level < 0.005:  # 0.5%范围内
                if current_price < mid:
                    df.loc[df.index[-1], "signal"] = 1
                elif current_price > mid:
                    df.loc[df.index[-1], "signal"] = -1
                break

        return df

    def should_exit(self, df: pd.DataFrame, position: Dict) -> bool:
        """网格策略出场 - 到达上网格"""
        if len(df) < 2:
            return False

        current_price = df["close"].iloc[-1]
        entry_price = position["entry_price"]

        # 盈利超过一个网格间距
        profit_pct = (current_price - entry_price) / entry_price
        if profit_pct > self.params["grid_size"]:
            return True

        return df["signal"].iloc[-1] == -1


# ==================== 突破策略 ====================
class BreakoutStrategy(Strategy):
    """通道突破策略 - 抓住趋势启动"""

    def __init__(self):
        super().__init__("BREAKOUT_STRATEGY")
        self.params = {
            "period": 20,
            "atr_period": 14,
            "breakout_multiplier": 1.5,  # 突破倍数
        }

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算突破信号"""
        if len(df) < self.params["period"] + 1:
            df["signal"] = 0
            return df

        period = self.params["period"]
        atr_period = self.params["atr_period"]
        # 计算通道
        IndicatorComputer.compute_channel(period, df)

        # 计算ATR (波动率)
        IndicatorComputer.compute_art(atr_period, df)

        # 计算成交量
        IndicatorComputer.compute_vol_ma(period, df)

        # 生成信号 - 突破上轨且成交量放大
        df["signal"] = 0
        df.loc[
            (df["close"] > df["upper_channel"].shift(1))
            & (df["volume"] > df["vol_ma"] * 1.5)  # 成交量放大
            & (df["atr"] > df["atr"].shift(1)),
            "signal",
        ] = 1  # 波动增加

        # 跌破下轨出场
        df.loc[df["close"] < df["lower_channel"], "signal"] = -1

        return df


# ==================== 均值回归策略 ====================
class MeanReversionStrategy(Strategy):
    """均值回归策略 - 超跌反弹"""

    def __init__(self):
        super().__init__("MEAN_REVERSION_STRATEGY")
        self.params = {
            "period": 20,
            "std_dev": 2.5,
            "rsi_period": 14,
            "rsi_oversold": 25,
        }

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算均值回归信号"""
        if len(df) < self.params["period"] + 1:
            df["signal"] = 0
            return df

        period = self.params["period"]
        std_dev = self.params["std_dev"]
        rsi_period = self.params["rsi_period"]

        # 计算标准差带
        IndicatorComputer.compute_band(period, std_dev, df)

        # 计算RSI
        IndicatorComputer.compute_rsi(rsi_period, df)

        # 计算价格偏离度
        IndicatorComputer.compute_deviation(df)

        # 生成信号 - 超跌+RSI超卖
        df["signal"] = 0
        df.loc[
            (df["close"] < df["lower_band"])
            & (df["rsi"] < self.params["rsi_oversold"])
            & (df["deviation"] < -0.03),
            "signal",
        ] = 1  # 偏离超过3%

        # 回归均值出场
        df.loc[df["close"] > df["mean"], "signal"] = -1

        return df


# ==================== EMA交叉策略 ====================
class EMAStrategy(Strategy):
    """指数移动平均线策略 - 快速反应"""

    def __init__(self):
        super().__init__("EMA_STRATEGY")
        self.params = {"fast": 8, "medium": 21, "slow": 55}

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算EMA信号"""
        if len(df) < self.params["slow"] + 1:
            df["signal"] = 0
            return df

        fast = self.params["fast"]
        medium = self.params["medium"]
        slow = self.params["slow"]

        # 计算EMA
        IndicatorComputer.compute_ema(fast, medium, slow, df)

        # 三线多头排列
        IndicatorComputer.compute_bullish_alignment(df)

        # 生成信号
        df["signal"] = 0
        df.loc[
            (df["ema_fast"] > df["ema_medium"])
            & (df["ema_fast"].shift(1) <= df["ema_medium"].shift(1))
            & df["bullish_alignment"],
            "signal",
        ] = 1

        df.loc[
            (df["ema_fast"] < df["ema_medium"])
            & (df["ema_fast"].shift(1) >= df["ema_medium"].shift(1)),
            "signal",
        ] = -1

        return df


# ==================== 动量突破策略 ====================
class MomentumStrategy(Strategy):
    """动量突破策略 - 强者恒强"""

    def __init__(self):
        super().__init__("MOMENTUM_STRATEGY")
        self.params = {"period": 14, "threshold": 5}  # 动量阈值%

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量信号"""
        if len(df) < self.params["period"] + 1:
            df["signal"] = 0
            return df

        period = self.params["period"]
        # 计算动量
        IndicatorComputer.compute_momentum(period, df)

        # 计算动量变化率
        IndicatorComputer.compute_momentum_change(df)

        # 计算成交量动量
        IndicatorComputer.compute_volume_ratio(df)

        # 生成信号 - 强动量+成交量确认
        df["signal"] = 0
        df.loc[
            (df["momentum"] > self.params["threshold"])
            & (df["momentum_change"] > 0)
            & (df["volume_ratio"] > 1.2),
            "signal",
        ] = 1

        df.loc[
            (df["momentum"] < -self.params["threshold"]) | (df["momentum_change"] < -2),
            "signal",
        ] = -1

        return df
    




# ==================== 策略管理器（工厂模式）====================
class StrategyFactory:
    """策略工厂 - 使用工厂模式创建策略"""
    
    _strategy_classes = {
        "RSI_STRATEGY": RSIStrategy,
        "MACD_STRATEGY": MACDStrategy,
        "BOLLINGER_STRATEGY": BollingerStrategy,
        "MA_CROSS_STRATEGY": MACrossStrategy,
        "GRID_STRATEGY": GridStrategy,
        "BREAKOUT_STRATEGY": BreakoutStrategy,
        "MEAN_REVERSION_STRATEGY": MeanReversionStrategy,
        "EMA_STRATEGY": EMAStrategy,
        "MOMENTUM_STRATEGY": MomentumStrategy,
    }
    
    @classmethod
    def create_strategy(cls, name: str) -> Optional[Strategy]:
        """创建策略实例"""
        strategy_class = cls._strategy_classes.get(name)
        if strategy_class:
            return strategy_class()
        return None
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class):
        """注册新策略 - 提高可扩展性"""
        cls._strategy_classes[name] = strategy_class
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """获取所有可用策略名称"""
        return list(cls._strategy_classes.keys())

class StrategyManager:
    """策略管理器 - 使用工厂模式和管理模式"""

    def __init__(self, config: Config):
        self.config = config
        self.factory = StrategyFactory
        self.strategies = self._init_strategies()

    def _init_strategies(self) -> Dict[str, Strategy]:
        """初始化策略 - 延迟加载"""
        strategies = {}
        for name in self.config.STRATEGIES:
            strategy = self.factory.create_strategy(name)
            if strategy:
                strategies[name] = strategy
        return strategies

    def get_strategy(self, name: str) -> Optional[Strategy]:
        """获取策略 - 支持动态创建"""
        if name in self.strategies:
            return self.strategies[name]
        # 尝试动态创建
        strategy = self.factory.create_strategy(name)
        if strategy:
            self.strategies[name] = strategy
        return strategy

    def get_all_strategies(self) -> List[Strategy]:
        """获取所有激活的策略"""
        return list(self.strategies.values())
    
    def register_strategy(self, name: str, strategy_class):
        """注册新策略"""
        self.factory.register_strategy(name, strategy_class)


