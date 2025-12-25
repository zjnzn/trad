
"""
配置模块
"""
from dataclasses import dataclass, field
from typing import List, Dict


# ==================== 配置类 ====================
@dataclass
class Config:
    """系统配置"""

    API_KEY: str = "YOUR_API_KEY"
    API_SECRET: str = "YOUR_API_SECRET"

    # 交易参数
    MIN_TRADE_AMOUNT: float = 11.0
    MAX_POSITIONS: int = 5
    POSITION_SIZE: float = 100.0

    # 币对筛选
    QUOTE_CURRENCY: str = "USDT"
    MIN_24H_VOLUME: float = 500000
    TOP_PAIRS_COUNT: int = 20
    BLACKLIST_PAIRS: List[str] = None  # 黑名单币对

    # 策略参数
    STRATEGIES: List[str] = None
    STRATEGY_WEIGHTS: Dict[str, float] = None  # 策略权重
    ENABLE_NOTIFICATIONS: bool = False  # 启用通知(未来扩展)

    # 信号过滤参数
    ENABLE_MULTI_INDICATOR_FILTER: bool = True  # 多指标共振过滤
    MIN_INDICATORS_AGREE: int = 2  # 至少2个指标同意才开仓
    ENABLE_TREND_FILTER: bool = True  # 趋势过滤
    TREND_STRENGTH_THRESHOLD: float = 0.3  # 趋势强度阈值
    VOLUME_FILTER: bool = True  # 成交量过滤
    MIN_VOLUME_RATIO: float = 1.2  # 最小成交量比率

    # 风控参数
    MAX_LOSS_PER_TRADE: float = 0.02
    MAX_DAILY_LOSS: float = 0.05
    TRAILING_STOP_PCT: float = 0.015  # 移动止盈1.5%
    TAKE_PROFIT_PCT: float = 0.05  # 止盈5%
    FORCE_EXIT_HOURS: int = 168  # 强制平仓时间(小时) 7天

    # 回测参数
    BACKTEST_DAYS: int = 360
    OPTIMIZE_ITERATIONS: int = 30  # 减少优化次数提升速度

    # 机器学习参数
    ENABLE_ML_OPTIMIZATION: bool = True  # 启用ML优化
    ML_TRAINING_DAYS: int = 60  # ML训练数据天数
    ML_RETRAIN_HOURS: int = 24  # 每24小时重新训练
    ML_MIN_SAMPLES: int = 100  # 最少样本数
    ML_CONFIDENCE_THRESHOLD: float = 0.6  # ML预测置信度阈值

    # 系统参数
    TEST_MODE: bool = True
    LOG_LEVEL: str = "INFO"
    DB_PATH: str = "trading_system.db"
    SCAN_INTERVAL: int = 60  # 扫描间隔(秒)
    MAX_WORKERS: int = 5  # 线程池大小
    CACHE_TTL: int = 300  # 缓存过期时间(秒)

    def __post_init__(self):
        if self.STRATEGIES is None:
            self.STRATEGIES = [
                "RSI_STRATEGY",
                "MACD_STRATEGY",
                "BOLLINGER_STRATEGY",
                "MA_CROSS_STRATEGY",
                "GRID_STRATEGY",
                "BREAKOUT_STRATEGY",
                "MEAN_REVERSION_STRATEGY",
            ]
        if self.BLACKLIST_PAIRS is None:
            self.BLACKLIST_PAIRS = []
        if self.STRATEGY_WEIGHTS is None:
            self.STRATEGY_WEIGHTS = {
                "RSI_STRATEGY": 1.0,
                "MACD_STRATEGY": 1.0,
                "BOLLINGER_STRATEGY": 1.0,
                "MA_CROSS_STRATEGY": 1.2,  # 均线策略权重稍高
                "GRID_STRATEGY": 0.8,
                "BREAKOUT_STRATEGY": 1.0,
                "MEAN_REVERSION_STRATEGY": 0.9,
            }

