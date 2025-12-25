

# ==================== 多指标分析器 ====================
class MultiIndicatorAnalyzer:
    """多指标共振分析器"""

    def __init__(self, config: Config):
        self.config = config

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        if len(df) < 100:
            return df

        # RSI
        IndicatorComputer.compute_rsi(14, df)

        # MACD
        IndicatorComputer.compute_macd(12, 26, df)
        IndicatorComputer.compute_signal_line(9, df)
        IndicatorComputer.compute_histogram(df)

        # 布林带
        IndicatorComputer.compute_bandwidth(20, 2, df)

        # 均线系统
        IndicatorComputer.compute_ma(20, 50, 200, df)

        # EMA系统
        IndicatorComputer.compute_ema(12, 26, 55, df)

        # ATR (波动率)
        IndicatorComputer.compute_art(14, df)

        # ADX (趋势强度)
        IndicatorComputer.compute_adx(df)

        # 成交量指标
        IndicatorComputer.compute_volume_ma(20, df)
        IndicatorComputer.compute_vol_ratio(df)

        # OBV (能量潮)
        IndicatorComputer.compute_obv(df)

        return df

    def check_indicator_agreement(self, df: pd.DataFrame) -> Dict[str, int]:
        """检查多个指标是否共振 - 返回看涨/看跌信号数量"""
        if len(df) < 2:
            return {"bullish": 0, "bearish": 0}

        bullish_count = 0
        bearish_count = 0

        # 1. RSI信号
        if df["rsi"].iloc[-1] < 30:
            bullish_count += 1
        elif df["rsi"].iloc[-1] > 70:
            bearish_count += 1

        # 2. MACD信号
        if (
            df["macd"].iloc[-1] > df["signal_line"].iloc[-1]
            and df["macd"].iloc[-2] <= df["signal_line"].iloc[-2]
        ):
            bullish_count += 1
        elif (
            df["macd"].iloc[-1] < df["signal_line"].iloc[-1]
            and df["macd"].iloc[-2] >= df["signal_line"].iloc[-2]
        ):
            bearish_count += 1

        # 3. MACD柱状图趋势
        if df["histogram"].iloc[-1] > df["histogram"].iloc[-2]:
            bullish_count += 1
        else:
            bearish_count += 1

        # 4. 布林带信号
        if df["close"].iloc[-1] < df["band_lower"].iloc[-1]:
            bullish_count += 1
        elif df["close"].iloc[-1] > df["band_upper"].iloc[-1]:
            bearish_count += 1

        # 5. 均线排列
        if (
            df["ma_fast"].iloc[-1] > df["ma_slow"].iloc[-1]
            and df["close"].iloc[-1] > df["ma_fast"].iloc[-1]
        ):
            bullish_count += 1
        elif (
            df["ma_fast"].iloc[-1] < df["ma_slow"].iloc[-1]
            and df["close"].iloc[-1] < df["ma_fast"].iloc[-1]
        ):
            bearish_count += 1

        # 6. EMA交叉
        if (
            df["ema_fast"].iloc[-1] > df["ema_medium"].iloc[-1]
            and df["ema_fast"].iloc[-2] <= df["ema_medium"].iloc[-2]
        ):
            bullish_count += 1
        elif (
            df["ema_fast"].iloc[-1] < df["ema_medium"].iloc[-1]
            and df["ema_fast"].iloc[-2] >= df["ema_medium"].iloc[-2]
        ):
            bearish_count += 1

        # 7. 成交量确认
        if df["volume_ratio"].iloc[-1] > self.config.MIN_VOLUME_RATIO:
            if df["close"].iloc[-1] > df["close"].iloc[-2]:
                bullish_count += 1
            else:
                bearish_count += 1

        return {"bullish": bullish_count, "bearish": bearish_count}

