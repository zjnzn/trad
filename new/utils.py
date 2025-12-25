
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
    



class IndicatorComputer:

    # 计算RSI
    @staticmethod
    def compute_rsi(period, df: pd.DataFrame):
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

    # 信号平滑
    @staticmethod
    def compute_rsi_smooth(smooth, df: pd.DataFrame):
        df["rsi_smooth"] = df["rsi"].rolling(window=smooth).mean()

    # 计算MACD
    @staticmethod
    def compute_macd(fast, slow, df: pd.DataFrame):
        exp1 = df["close"].ewm(span=fast, adjust=False).mean()
        exp2 = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd"] = exp1 - exp2

    @staticmethod
    def compute_signal_line(signal_period, df: pd.DataFrame):
        df["signal_line"] = df["macd"].ewm(span=signal_period, adjust=False).mean()

    @staticmethod
    def compute_histogram(df: pd.DataFrame):
        df["histogram"] = df["macd"] - df["signal_line"]

    # 计算布林带
    @staticmethod
    def compute_bandwidth(period, std_dev, df: pd.DataFrame):
        df["sma"] = df["close"].rolling(window=period).mean()
        df["std"] = df["close"].rolling(window=period).std()
        df["band_upper"] = df["sma"] + (df["std"] * std_dev)
        df["band_lower"] = df["sma"] - (df["std"] * std_dev)
        df["band_width"] = (df["band_upper"] - df["band_lower"]) / df["sma"]

    # 计算均线
    @staticmethod
    def compute_ma(fast_period, slow_period, trend_filter, df: pd.DataFrame):
        df["ma_fast"] = df["close"].rolling(window=fast_period).mean()
        df["ma_slow"] = df["close"].rolling(window=slow_period).mean()
        df["ma_trend"] = df["close"].rolling(window=trend_filter).mean()

    # 计算通道
    @staticmethod
    def compute_channel(period, df: pd.DataFrame):
        df["upper_channel"] = df["high"].rolling(window=period).max()
        df["lower_channel"] = df["low"].rolling(window=period).min()
        df["mid_channel"] = (df["upper_channel"] + df["lower_channel"]) / 2

    # 计算ATR (波动率)
    @staticmethod
    def compute_art(atr_period, df: pd.DataFrame):
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = abs(df["high"] - df["close"].shift(1))
        df["tr3"] = abs(df["low"] - df["close"].shift(1))
        df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr"] = df["tr"].rolling(window=atr_period).mean()

    # 计算成交量
    @staticmethod
    def compute_vol_ma(period, df: pd.DataFrame):
        df["vol_ma"] = df["volume"].rolling(window=period).mean()

    # 计算成交量
    @staticmethod
    def compute_vol_ratio(df: pd.DataFrame):
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

    # 计算标准差带
    @staticmethod
    def compute_band(period, std_dev, df: pd.DataFrame):
        df["mean"] = df["close"].rolling(window=period).mean()
        df["std"] = df["close"].rolling(window=period).std()
        df["lower_band"] = df["mean"] - (df["std"] * std_dev)
        df["upper_band"] = df["mean"] + (df["std"] * std_dev)

    # 计算价格偏离度
    @staticmethod
    def compute_deviation(df: pd.DataFrame):
        df["deviation"] = (df["close"] - df["mean"]) / df["mean"]

    # 计算EMA
    @staticmethod
    def compute_ema(fast, medium, slow, df: pd.DataFrame):
        df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
        df["ema_medium"] = df["close"].ewm(span=medium, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()

    # 三线多头排列
    @staticmethod
    def compute_bullish_alignment(df: pd.DataFrame):
        df["bullish_alignment"] = (df["ema_fast"] > df["ema_medium"]) & (
            df["ema_medium"] > df["ema_slow"]
        )

    # 计算动量
    @staticmethod
    def compute_momentum(period, df: pd.DataFrame):
        df["momentum"] = (
            (df["close"] - df["close"].shift(period)) / df["close"].shift(period) * 100
        )

    # 计算动量变化率
    @staticmethod
    def compute_momentum_change(df: pd.DataFrame):
        df["momentum_change"] = df["momentum"].diff()

    # 计算成交量动量
    @staticmethod
    def compute_volume_ratio(df: pd.DataFrame):
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

    # ADX (趋势强度)
    @staticmethod
    def compute_adx(df: pd.DataFrame, period: int = 14):
        """计算ADX (Average Directional Index)"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # +DM, -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # TR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR
        atr = tr.rolling(window=period).mean()

        # +DI, -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # DX, ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df["adx"] = dx.rolling(window=period).mean()

    # OBV (能量潮)
    @staticmethod
    def compute_obv(df: pd.DataFrame):
        """计算OBV (On Balance Volume)"""
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = pd.Series(obv, index=df.index)






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
        IndicatorComputer.compute_vol_ma(20, df)
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


# ==================== 信号过滤器 ====================
class SignalFilter:
    """高级信号过滤系统"""

    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.multi_indicator = MultiIndicatorAnalyzer(config)
        self.trend_analyzer = TrendAnalyzer(config)

    def filter_signal(
        self, df: pd.DataFrame, signal: int, symbol: str, strategy_name: str
    ) -> bool:
        """综合过滤信号 - 返回True表示通过"""
        if signal == 0 or len(df) < 200:
            return False

        # 计算所有指标
        df = self.multi_indicator.calculate_all_indicators(df)

        reasons = []

        # 1. 多指标共振过滤
        if (
            self.config.ENABLE_MULTI_INDICATOR_FILTER and signal == 1
        ):  # 只对买入信号过滤
            agreement = self.multi_indicator.check_indicator_agreement(df)

            if agreement["bullish"] < self.config.MIN_INDICATORS_AGREE:
                reasons.append(
                    f"指标共振不足 ({agreement['bullish']}/{self.config.MIN_INDICATORS_AGREE})"
                )
                self.logger.info(
                    f"[FILTER] {symbol} {strategy_name} 被过滤: {reasons[-1]}"
                )
                return False

            # 如果有反向信号过多,也拒绝
            if agreement["bearish"] > agreement["bullish"]:
                reasons.append(
                    f"反向信号过多 (空:{agreement['bearish']} vs 多:{agreement['bullish']})"
                )
                self.logger.info(
                    f"[FILTER] {symbol} {strategy_name} 被过滤: {reasons[-1]}"
                )
                return False

        # 2. 趋势过滤
        if self.config.ENABLE_TREND_FILTER and signal == 1:  # 只对买入信号过滤
            if not self.trend_analyzer.is_trend_aligned(df, signal):
                trend = self.trend_analyzer.analyze_trend(df)
                reasons.append(
                    f"趋势不一致 (方向:{trend['direction']}, 强度:{trend['strength']:.2f})"
                )
                self.logger.info(
                    f"[FILTER] {symbol} {strategy_name} 被过滤: {reasons[-1]}"
                )
                return False

        # 3. 成交量过滤
        if self.config.VOLUME_FILTER and signal == 1:
            if df["volume_ratio"].iloc[-1] < self.config.MIN_VOLUME_RATIO:
                reasons.append(f"成交量不足 ({df['volume_ratio'].iloc[-1]:.2f}x)")
                self.logger.info(
                    f"[FILTER] {symbol} {strategy_name} 被过滤: {reasons[-1]}"
                )
                return False

        # 4. 波动率过滤 (避免在极低波动时交易)
        if "band_width" in df.columns:
            band_width = df["band_width"].iloc[-1]
            if band_width < 0.02:  # 布林带宽度<2%
                reasons.append(f"波动率过低 ({band_width*100:.1f}%)")
                self.logger.info(
                    f"[FILTER] {symbol} {strategy_name} 被过滤: {reasons[-1]}"
                )
                return False

        # 通过所有过滤
        if self.config.ENABLE_MULTI_INDICATOR_FILTER or self.config.ENABLE_TREND_FILTER:
            agreement = self.multi_indicator.check_indicator_agreement(df)
            trend = self.trend_analyzer.analyze_trend(df)
            self.logger.info(
                f"[PASS] {symbol} {strategy_name} 通过过滤 | "
                f"指标共振:{agreement['bullish']} | "
                f"趋势:{trend['direction']}({trend['strength']:.2f})"
            )

        return True


