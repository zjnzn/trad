

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

