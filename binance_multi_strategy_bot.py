"""
币安量化交易系统 - 优化版
警告: 此代码仅供学习使用,实盘交易存在巨大风险,可能导致资金损失
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass, asdict
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import threading
from functools import lru_cache
import signal
import sys

# ML依赖条件导入
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import joblib

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARNING] scikit-learn未安装，ML功能将被禁用")
    print("安装命令: pip install scikit-learn joblib")

warnings.filterwarnings("ignore")


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
    BACKTEST_DAYS: int = 300
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


# ==================== 日志系统 ====================
class Logger:
    """优化的日志管理器"""

    def __init__(self, config: Config):
        self.config = config
        self.setup_logger()
        self._log_buffer = []
        self._buffer_lock = threading.Lock()

    def setup_logger(self):
        """配置日志 - 优化编码处理"""
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # 文件处理器
        file_handler = logging.FileHandler(
            f'trading_{datetime.now().strftime("%Y%m%d")}.log', encoding="utf-8"
        )
        file_handler.setFormatter(formatter)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Windows编码设置
        if sys.platform == "win32":
            import io

            try:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
            except:
                pass

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.LOG_LEVEL))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def critical(self, msg: str):
        """记录严重错误"""
        self.logger.critical(msg)

    def trade(
        self, action: str, symbol: str, price: float, amount: float, strategy: str
    ):
        """交易日志 - 修复格式化问题"""
        price_str = f"{price:.8f}" if price else "MARKET"
        msg = f"[TRADE] {action} {symbol} | Price: {price_str} | Amount: {amount:.8f} | Strategy: {strategy}"
        self.logger.info(msg)


# ==================== 数据库管理 ====================
class Database:
    """优化的线程安全数据库管理"""

    def __init__(self, config: Config):
        self.config = config
        self.db_path = config.DB_PATH
        self._connections = {}
        self._lock = threading.Lock()
        self.create_tables()

    def get_connection(self):
        """获取线程安全的数据库连接"""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id not in self._connections:
                self._connections[thread_id] = sqlite3.connect(
                    self.db_path, check_same_thread=False, timeout=30.0  # 增加超时时间
                )
                self._connections[thread_id].execute(
                    "PRAGMA journal_mode=WAL"
                )  # 提升并发性能
            return self._connections[thread_id]

    def create_tables(self):
        """创建数据表 - 添加索引优化查询"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 持仓表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                amount REAL NOT NULL,
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exit_price REAL,
                exit_time TIMESTAMP,
                pnl REAL,
                status TEXT DEFAULT 'OPEN',
                highest_price REAL,
                stop_loss_price REAL
            )
        """
        )

        # 添加索引
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol, strategy, status)"
        )

        # 交易记录表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                fee REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)"
        )

        # 账户快照表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                balance REAL NOT NULL,
                equity REAL NOT NULL,
                daily_pnl REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # 策略表现表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                win_rate REAL,
                avg_pnl REAL,
                total_trades INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def save_position(
        self,
        symbol: str,
        strategy: str,
        side: str,
        entry_price: float,
        amount: float,
        stop_loss_price: float = None,
    ):
        """保存持仓"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO positions (symbol, strategy, side, entry_price, amount, highest_price, stop_loss_price)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (symbol, strategy, side, entry_price, amount, entry_price, stop_loss_price),
        )
        conn.commit()
        return cursor.lastrowid

    def update_position(self, position_id: int, exit_price: float, pnl: float):
        """更新持仓"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE positions 
            SET exit_price = ?, exit_time = CURRENT_TIMESTAMP, 
                pnl = ?, status = 'CLOSED'
            WHERE id = ?
        """,
            (exit_price, pnl, position_id),
        )
        conn.commit()

    def update_highest_price(self, position_id: int, highest_price: float):
        """更新最高价(用于移动止盈)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE positions SET highest_price = ? WHERE id = ?
        """,
            (highest_price, position_id),
        )
        conn.commit()

    def get_open_positions(self) -> List[Dict]:
        """获取开仓持仓 - 优化查询"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM positions WHERE status = 'OPEN' ORDER BY entry_time DESC
        """
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def has_open_position(self, symbol: str, strategy: str) -> bool:
        """检查是否已有开仓 - 使用索引优化"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 1 FROM positions 
            WHERE symbol = ? AND strategy = ? AND status = 'OPEN' LIMIT 1
        """,
            (symbol, strategy),
        )
        return cursor.fetchone() is not None

    def save_trade(
        self,
        symbol: str,
        strategy: str,
        side: str,
        price: float,
        amount: float,
        fee: float,
    ):
        """保存交易记录"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO trades (symbol, strategy, side, price, amount, fee)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (symbol, strategy, side, price, amount, fee),
        )
        conn.commit()

    def save_account_snapshot(self, balance: float, equity: float, daily_pnl: float):
        """保存账户快照"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO account_snapshots (balance, equity, daily_pnl)
            VALUES (?, ?, ?)
        """,
            (balance, equity, daily_pnl),
        )
        conn.commit()

    def get_daily_pnl(self) -> float:
        """获取当日盈亏 - 优化查询"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COALESCE(SUM(pnl), 0) FROM positions 
            WHERE DATE(exit_time) = DATE('now') AND status = 'CLOSED'
        """
        )
        return cursor.fetchone()[0]

    def get_strategy_stats(self, strategy: str, days: int = 7) -> Dict:
        """获取策略统计"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
                AVG(pnl) as avg_pnl,
                SUM(pnl) as total_pnl
            FROM positions 
            WHERE strategy = ? AND status = 'CLOSED' 
            AND exit_time >= datetime('now', '-' || ? || ' days')
        """,
            (strategy, days),
        )
        row = cursor.fetchone()
        return {
            "total_trades": row[0] or 0,
            "win_rate": row[1] or 0,
            "avg_pnl": row[2] or 0,
            "total_pnl": row[3] or 0,
        }

    def cleanup_old_data(self, days: int = 90):
        """清理旧数据"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM trades WHERE timestamp < datetime('now', '-' || ? || ' days')
        """,
            (days,),
        )
        cursor.execute(
            """
            DELETE FROM account_snapshots WHERE timestamp < datetime('now', '-' || ? || ' days')
        """,
            (days,),
        )
        conn.commit()


# ==================== 交易所接口 ====================
class BinanceAPI:
    """优化的币安API封装"""

    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.exchange = ccxt.binance(
            {
                "apiKey": config.API_KEY,
                "secret": config.API_SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
                "timeout": 30000,
            }
        )

        if config.TEST_MODE:
            self.exchange.set_sandbox_mode(True)
            self.logger.info("[TEST MODE] 测试模式已启用")

        # 价格缓存
        self._price_cache = {}
        self._cache_time = {}
        self._cache_lock = threading.Lock()

    def get_balance(self) -> Dict[str, float]:
        """获取账户余额 - 添加重试"""
        for retry in range(3):
            try:
                balance = self.exchange.fetch_balance()
                return {
                    "free": balance["USDT"]["free"],
                    "used": balance["USDT"]["used"],
                    "total": balance["USDT"]["total"],
                }
            except Exception as e:
                if retry == 2:
                    self.logger.error(f"获取余额失败: {e}")
                    return {"free": 0, "used": 0, "total": 0}
                time.sleep(1)

    def get_all_tickers(self) -> List[Dict]:
        """获取所有交易对行情 - 添加缓存"""
        try:
            tickers = self.exchange.fetch_tickers()
            result = []
            for symbol, ticker in tickers.items():
                if symbol.endswith(f"/{self.config.QUOTE_CURRENCY}"):
                    if (
                        symbol.replace(f"/{self.config.QUOTE_CURRENCY}", "")
                        in self.config.BLACKLIST_PAIRS
                    ):
                        continue
                    result.append(
                        {
                            "symbol": symbol,
                            "price": ticker["last"],
                            "volume": ticker["quoteVolume"],
                            "change": ticker["percentage"],
                        }
                    )
            return result
        except Exception as e:
            self.logger.error(f"获取行情失败: {e}")
            return []

    @lru_cache(maxsize=100)
    def get_ohlcv_cached(
        self, symbol: str, timeframe: str, limit: int, cache_key: int
    ) -> pd.DataFrame:
        """带缓存的K线数据获取"""
        return self._get_ohlcv_internal(symbol, timeframe, limit)

    def get_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> pd.DataFrame:
        """获取K线数据 - 使用缓存并添加重试"""
        cache_key = int(time.time() / self.config.CACHE_TTL)
        try:
            return self.get_ohlcv_cached(symbol, timeframe, limit, cache_key)
        except Exception as e:
            self.logger.error(f"获取K线数据失败 {symbol}: {e}")
            # 清除缓存并重试一次
            self.get_ohlcv_cached.cache_clear()
            try:
                return self._get_ohlcv_internal(symbol, timeframe, limit)
            except:
                return pd.DataFrame()

    def _get_ohlcv_internal(
        self, symbol: str, timeframe: str, limit: int
    ) -> pd.DataFrame:
        """内部K线获取方法 - 增强重试"""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"获取K线失败 {symbol} (重试 {attempt+1}/{max_retries}): {e}"
                    )
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"获取K线数据失败 {symbol}: {e}")
                    return pd.DataFrame()

    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """获取当前价格 - 使用缓存和重试"""
        with self._cache_lock:
            now = time.time()
            if symbol in self._price_cache:
                if now - self._cache_time.get(symbol, 0) < 5:  # 5秒缓存
                    return self._price_cache[symbol]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker["last"]
                with self._cache_lock:
                    self._price_cache[symbol] = price
                    self._cache_time[symbol] = now
                return price
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    self.logger.error(f"获取价格失败 {symbol}: {e}")
                    return None

    def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float = None,
        strategy: str = "MANUAL",
    ) -> Dict:
        """创建订单 - 添加重试"""
        for retry in range(3):
            try:
                order_type = "limit" if price else "market"
                params = {"price": price} if price else {}

                order = self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    params=params,
                )

                self.logger.trade(
                    side.upper(), symbol, price or order["price"], amount, strategy
                )
                return order
            except Exception as e:
                if retry == 2:
                    self.logger.error(f"创建订单失败 {symbol}: {e}")
                    return {}
                time.sleep(1)

    def get_trading_fee(self, symbol: str) -> float:
        """获取交易手续费率"""
        try:
            fees = self.exchange.fetch_trading_fees()
            return fees.get(symbol, {}).get("taker", 0.001)
        except:
            return 0.001


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


# ==================== 策略基类 ====================
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


# ==================== 机器学习优化器 ====================
class MLOptimizer:
    """机器学习参数优化和信号预测"""

    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.models = {}  # 每个策略一个模型
        self.scalers = {}
        self.last_train_time = {}
        self.model_path = "ml_models"

        import os

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """从K线数据提取特征 - 添加异常处理"""
        if len(df) < 50:
            return np.array([])

        features = []

        try:
            # 价格特征
            close_20 = df["close"].iloc[-20] if len(df) >= 20 else df["close"].iloc[0]
            close_5 = df["close"].iloc[-5] if len(df) >= 5 else df["close"].iloc[0]

            features.append(df["close"].iloc[-1] / close_20 - 1 if close_20 > 0 else 0)
            features.append(df["close"].iloc[-1] / close_5 - 1 if close_5 > 0 else 0)
            features.append(
                df["high"].iloc[-1] / df["low"].iloc[-1] - 1
                if df["low"].iloc[-1] > 0
                else 0
            )

            # 技术指标特征
            # RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            features.append(min(rsi.iloc[-1] / 100, 1.0))

            # MACD
            exp1 = df["close"].ewm(span=12, adjust=False).mean()
            exp2 = df["close"].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9, adjust=False).mean()
            macd_val = (
                (macd.iloc[-1] - signal_line.iloc[-1]) / df["close"].iloc[-1]
                if df["close"].iloc[-1] > 0
                else 0
            )
            features.append(np.clip(macd_val, -1, 1))

            # 布林带位置
            sma = df["close"].rolling(window=20).mean()
            std = df["close"].rolling(window=20).std()
            bb_position = (df["close"].iloc[-1] - sma.iloc[-1]) / (std.iloc[-1] + 1e-10)
            features.append(np.clip(bb_position, -3, 3))

            # 均线特征
            ma_fast = df["close"].rolling(window=20).mean().iloc[-1]
            ma_slow = (
                df["close"].rolling(window=50).mean().iloc[-1]
                if len(df) >= 50
                else ma_fast
            )
            features.append(df["close"].iloc[-1] / ma_fast - 1 if ma_fast > 0 else 0)
            features.append(ma_fast / ma_slow - 1 if ma_slow > 0 else 0)

            # 成交量特征
            vol_ma = df["volume"].rolling(window=20).mean()
            vol_ratio = df["volume"].iloc[-1] / (vol_ma.iloc[-1] + 1e-10)
            features.append(min(vol_ratio, 10.0))

            # 波动率特征
            returns = df["close"].pct_change()
            volatility = returns.rolling(window=20).std().iloc[-1]
            features.append(min(volatility * 100, 10.0))

            # 动量特征
            close_14 = df["close"].iloc[-14] if len(df) >= 14 else df["close"].iloc[0]
            momentum = (
                (df["close"].iloc[-1] - close_14) / close_14 if close_14 > 0 else 0
            )
            features.append(np.clip(momentum, -1, 1))

            # 趋势特征 (线性回归斜率)
            x = np.arange(min(20, len(df)))
            y = df["close"].tail(min(20, len(df))).values
            if len(y) >= 2:
                slope = (
                    np.polyfit(x, y, 1)[0] / df["close"].iloc[-1]
                    if df["close"].iloc[-1] > 0
                    else 0
                )
                features.append(np.clip(slope, -0.1, 0.1))
            else:
                features.append(0)

            # 确保所有特征都是有限数值
            features = [0 if not np.isfinite(f) else f for f in features]

            return np.array(features)

        except Exception as e:
            self.logger.error(f"[ML] 特征提取失败: {e}")
            return np.array([])

    def prepare_training_data(
        self, api, symbol: str, strategy
    ) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        # 获取历史数据
        df = api.get_ohlcv(symbol, "1h", self.config.ML_TRAINING_DAYS * 24)
        if len(df) < 200:
            return np.array([]), np.array([])

        X_list = []
        y_list = []

        # 滑动窗口生成训练样本
        for i in range(200, len(df) - 24):  # 预测未来24小时
            # 提取特征
            sample_df = df.iloc[: i + 1].copy()
            features = self.extract_features(sample_df)

            if len(features) == 0:
                continue

            # 计算标签 (未来24小时最高收益)
            future_prices = df["close"].iloc[i + 1 : i + 25].values
            current_price = df["close"].iloc[i]
            max_return = (max(future_prices) - current_price) / current_price

            # 二分类: 盈利>2% 为1, 否则为0
            label = 1 if max_return > 0.02 else 0

            X_list.append(features)
            y_list.append(label)

        if len(X_list) < self.config.ML_MIN_SAMPLES:
            return np.array([]), np.array([])

        return np.array(X_list), np.array(y_list)

    def train_model(self, api, symbol: str, strategy):
        """训练ML模型"""
        try:
            self.logger.info(f"[ML] 开始训练模型: {symbol} - {strategy.name}")

            # 准备数据
            X, y = self.prepare_training_data(api, symbol, strategy)

            if len(X) == 0:
                self.logger.warning(f"[ML] 训练数据不足: {symbol}")
                return False

            # 检查正负样本比例
            positive_ratio = np.sum(y) / len(y)
            self.logger.info(f"[ML] 正样本比例: {positive_ratio:.2%}")

            if positive_ratio < 0.1 or positive_ratio > 0.9:
                self.logger.warning(f"[ML] 样本不平衡: {symbol}")
                return False

            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 特征标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 训练随机森林分类器
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train_scaled, y_train)

            # 评估模型
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)

            self.logger.info(
                f"[ML] 模型训练完成: {symbol} | "
                f"训练准确率:{train_score:.2%} | 测试准确率:{test_score:.2%}"
            )

            # 保存模型
            model_key = f"{symbol}_{strategy.name}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.last_train_time[model_key] = time.time()

            # 持久化
            self._save_model(model_key, model, scaler)

            return True

        except Exception as e:
            self.logger.error(f"[ML] 训练失败 {symbol}: {e}")
            return False

    def predict_signal_quality(
        self, df: pd.DataFrame, symbol: str, strategy_name: str
    ) -> float:
        """预测信号质量 - 返回0-1的置信度"""
        model_key = f"{symbol}_{strategy_name}"

        # 检查是否有模型
        if model_key not in self.models:
            # 尝试从磁盘加载
            if self._load_model(model_key):
                self.logger.info(f"[ML] 已加载模型: {model_key}")
            else:
                # 没有模型,返回中性值（允许交易但不增强信心）
                return 0.5

        # 检查模型是否过期
        last_train = self.last_train_time.get(model_key, 0)
        if time.time() - last_train > self.config.ML_RETRAIN_HOURS * 3600:
            self.logger.info(f"[ML] 模型过期: {model_key}")
            return 0.5

        try:
            # 提取特征
            features = self.extract_features(df)
            if len(features) == 0:
                return 0.5

            # 标准化
            scaler = self.scalers[model_key]
            features_scaled = scaler.transform(features.reshape(1, -1))

            # 预测概率
            model = self.models[model_key]
            proba = model.predict_proba(features_scaled)[0]

            # 返回正类概率
            confidence = proba[1]

            return confidence

        except Exception as e:
            self.logger.error(f"[ML] 预测失败 {symbol}: {e}")
            return 0.5

    def optimize_strategy_params(self, api, symbol: str, strategy) -> Dict:
        """使用ML优化策略参数"""
        try:
            self.logger.info(f"[ML] 使用贝叶斯优化参数: {strategy.name}")

            # 获取历史数据
            df = api.get_ohlcv(symbol, "1h", 500)
            if len(df) < 200:
                return strategy.params

            best_params = strategy.params.copy()
            best_score = -999

            # 定义参数搜索空间
            param_space = self._get_param_space(strategy.name)

            # 贝叶斯优化 (简化版 - 使用随机搜索)
            for _ in range(20):
                # 随机采样参数
                test_params = {}
                for param_name, (min_val, max_val) in param_space.items():
                    if isinstance(min_val, int):
                        test_params[param_name] = np.random.randint(
                            min_val, max_val + 1
                        )
                    else:
                        test_params[param_name] = np.random.uniform(min_val, max_val)

                # 测试参数
                strategy.params = test_params
                score = self._evaluate_params(df, strategy)

                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()

            self.logger.info(f"[ML] 最优参数: {best_params} | 分数: {best_score:.2f}")
            return best_params

        except Exception as e:
            self.logger.error(f"[ML] 参数优化失败: {e}")
            return strategy.params

    def _get_param_space(self, strategy_name: str) -> Dict:
        """获取参数搜索空间"""
        spaces = {
            "RSI_STRATEGY": {
                "period": (10, 20),
                "oversold": (20, 35),
                "overbought": (65, 80),
            },
            "MACD_STRATEGY": {"fast": (8, 16), "slow": (20, 30), "signal": (7, 12)},
            "MA_CROSS_STRATEGY": {
                "fast_period": (5, 15),
                "slow_period": (20, 40),
                "trend_filter": (80, 120),
            },
            "BOLLINGER_STRATEGY": {"period": (15, 25), "std_dev": (1.5, 2.5)},
        }
        return spaces.get(strategy_name, {})

    def _evaluate_params(self, df: pd.DataFrame, strategy) -> float:
        """评估参数质量"""
        try:
            df_copy = df.copy()
            df_copy = strategy.calculate_signals(df_copy)

            # 模拟交易
            capital = 1000
            position = None
            trades = []

            for i in range(len(df_copy)):
                if position is None and df_copy["signal"].iloc[i] == 1:
                    position = {
                        "entry_price": df_copy["close"].iloc[i],
                        "amount": capital * 0.95 / df_copy["close"].iloc[i],
                    }
                elif position and df_copy["signal"].iloc[i] == -1:
                    exit_price = df_copy["close"].iloc[i]
                    pnl = (exit_price - position["entry_price"]) * position["amount"]
                    capital += pnl
                    trades.append(pnl)
                    position = None

            if len(trades) < 3:
                return -999

            # 计算Sharpe比率
            returns = np.array(trades) / 1000
            sharpe = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252)

            return sharpe

        except:
            return -999

    def _save_model(self, model_key: str, model, scaler):
        """保存模型到磁盘"""
        try:
            import os

            model_file = os.path.join(self.model_path, f"{model_key}_model.pkl")
            scaler_file = os.path.join(self.model_path, f"{model_key}_scaler.pkl")

            # 确保父目录存在（支持 model_key 包含子目录，例如 "UNI/.."）
            model_dir = os.path.dirname(model_file)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)

            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)

        except Exception as e:
            self.logger.error(f"[ML] 保存模型失败: {e}")

    def _load_model(self, model_key: str) -> bool:
        """从磁盘加载模型"""
        try:
            model_file = f"{self.model_path}/{model_key}_model.pkl"
            scaler_file = f"{self.model_path}/{model_key}_scaler.pkl"

            import os

            if os.path.exists(model_file) and os.path.exists(scaler_file):
                self.models[model_key] = joblib.load(model_file)
                self.scalers[model_key] = joblib.load(scaler_file)
                self.last_train_time[model_key] = time.time()
                return True

            return False

        except Exception as e:
            self.logger.error(f"[ML] 加载模型失败: {e}")
            return False


# ==================== 策略管理器 ====================
class StrategyManager:
    """策略管理器"""

    def __init__(self, config: Config):
        self.config = config
        self.strategies = self._init_strategies()

    def _init_strategies(self) -> Dict[str, Strategy]:
        """初始化策略"""
        return {
            "RSI_STRATEGY": RSIStrategy(),
            "MACD_STRATEGY": MACDStrategy(),
            "BOLLINGER_STRATEGY": BollingerStrategy(),
            "MA_CROSS_STRATEGY": MACrossStrategy(),
            "GRID_STRATEGY": GridStrategy(),
            "BREAKOUT_STRATEGY": BreakoutStrategy(),
            "MEAN_REVERSION_STRATEGY": MeanReversionStrategy(),
            "EMA_STRATEGY": EMAStrategy(),
            "MOMENTUM_STRATEGY": MomentumStrategy(),
        }

    def get_strategy(self, name: str) -> Strategy:
        """获取策略"""
        return self.strategies.get(name)

    def get_all_strategies(self) -> List[Strategy]:
        """获取所有激活的策略"""
        return [
            self.strategies[name]
            for name in self.config.STRATEGIES
            if name in self.strategies
        ]


# ==================== 回测引擎 ====================
class BacktestEngine:
    """改进的回测引擎 - 解决评分异常问题"""
    def __init__(self, api: BinanceAPI, config: Config, logger: Logger):
        self.api = api
        self.config = config
        self.logger = logger

    def backtest_strategy(self, strategy, symbol: str) -> Dict:
        """改进的回测方法"""
        # 1. 增加回测数据量
        lookback_days = max(self.config.BACKTEST_DAYS, 60)  # 至少60天
        df = self.api.get_ohlcv(symbol, "1h", lookback_days * 24)
        
        if df.empty or len(df) < 100:  # 提高最低数据要求
            self.logger.warning(f"[BACKTEST] {symbol} 数据不足: {len(df)}条")
            return {
                "sharpe": -999,
                "return": 0,
                "trades": 0,
                "win_rate": 0,
                "reason": "数据不足"
            }

        # 2. 计算信号
        try:
            df = strategy.calculate_signals(df)
        except Exception as e:
            self.logger.error(f"[BACKTEST] {symbol} 信号计算失败: {e}")
            return {
                "sharpe": -999,
                "return": 0,
                "trades": 0,
                "win_rate": 0,
                "reason": "信号计算错误"
            }

        # 3. 模拟交易
        initial_capital = 1000
        capital = initial_capital
        position = None
        trades = []
        trade_details = []

        for i in range(len(df)):
            # 开仓逻辑
            if position is None and df["signal"].iloc[i] == 1:
                entry_price = df["close"].iloc[i]
                position = {
                    "entry_price": entry_price,
                    "amount": capital * 0.95 / entry_price,
                    "entry_idx": i,
                    "entry_time": df["timestamp"].iloc[i] if "timestamp" in df.columns else i
                }

            # 平仓逻辑
            elif position and self._should_exit(df, i, position):
                exit_price = df["close"].iloc[i]
                
                # 计算盈亏
                pnl = (exit_price - position["entry_price"]) * position["amount"]
                fee = abs(pnl) * 0.002  # 手续费
                net_pnl = pnl - fee
                
                capital += net_pnl
                trades.append(net_pnl)
                
                # 记录交易详情
                trade_details.append({
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "pnl": net_pnl,
                    "pnl_pct": (exit_price - position["entry_price"]) / position["entry_price"],
                    "hold_bars": i - position["entry_idx"]
                })
                
                position = None

        # 4. 改进的评分计算
        return self._calculate_improved_metrics(
            trades, trade_details, initial_capital, capital, symbol, strategy.name
        )

    def _should_exit(self, df, current_idx, position) -> bool:
        """改进的出场判断"""
        # 信号出场
        if df["signal"].iloc[current_idx] == -1:
            return True
        
        # 时间止损 (持仓超过7天)
        if current_idx - position["entry_idx"] > 168:
            return True
        
        # 固定止损 (亏损超过3%)
        current_price = df["close"].iloc[current_idx]
        loss_pct = (current_price - position["entry_price"]) / position["entry_price"]
        if loss_pct < -0.03:
            return True
        
        # 固定止盈 (盈利超过5%)
        if loss_pct > 0.05:
            return True
        
        return False

    def _calculate_improved_metrics(
        self, trades, trade_details, initial_capital, final_capital, symbol, strategy_name
    ) -> Dict:
        """改进的指标计算"""
        
        # 1. 交易次数检查
        if not trades:
            return {
                "sharpe": -999,
                "return": 0,
                "trades": 0,
                "win_rate": 0,
                "reason": "无交易"
            }
        
        if len(trades) < 5:  # 降低最低交易次数要求
            # 不直接返回-999,而是给一个惩罚分数
            total_return = (final_capital - initial_capital) / initial_capital
            return {
                "sharpe": -10,  # 给一个低分而不是-999
                "return": total_return,
                "trades": len(trades),
                "win_rate": len([t for t in trades if t > 0]) / len(trades),
                "reason": f"交易次数不足({len(trades)}<5)"
            }

        # 2. 基础指标
        returns = np.array(trades) / initial_capital
        total_return = (final_capital - initial_capital) / initial_capital
        win_rate = len([t for t in trades if t > 0]) / len(trades)
        
        # 3. 改进的Sharpe计算
        if np.std(returns) < 1e-10:  # 收益波动太小
            sharpe = 0
        else:
            # 使用调整后的Sharpe (考虑交易次数)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (mean_return / std_return) * np.sqrt(252)  # 年化
            
            # 交易次数惩罚 (交易太少降低可信度)
            trade_penalty = min(len(trades) / 10, 1.0)
            sharpe *= trade_penalty

        # 4. 综合评分 (不只看Sharpe)
        composite_score = self._calculate_composite_score(
            sharpe, total_return, win_rate, len(trades), trade_details
        )

        # 5. 详细日志
        self.logger.info(
            f"[BACKTEST] {symbol}-{strategy_name} | "
            f"Sharpe:{sharpe:.2f} 收益:{total_return*100:+.2f}% "
            f"胜率:{win_rate*100:.1f}% 交易:{len(trades)}次 "
            f"综合分:{composite_score:.2f}"
        )

        return {
            "sharpe": sharpe,
            "return": total_return,
            "trades": len(trades),
            "win_rate": win_rate,
            "composite_score": composite_score,  # 新增综合评分
            "avg_trade": np.mean(trades),
            "max_drawdown": self._calculate_max_drawdown(trades),
            "reason": "正常"
        }

    def _calculate_composite_score(
        self, sharpe, total_return, win_rate, trade_count, trade_details
    ) -> float:
        """计算综合评分 (0-100)"""
        score = 0
        
        # 1. Sharpe占40% (正常化到0-40)
        if sharpe > -10:
            sharpe_score = min(max(sharpe / 3 * 40, 0), 40)
        else:
            sharpe_score = 0
        score += sharpe_score
        
        # 2. 总收益占30%
        if total_return > 0:
            return_score = min(total_return * 100, 30)
        else:
            return_score = max(total_return * 50, -30)  # 亏损惩罚减半
        score += return_score
        
        # 3. 胜率占20%
        win_score = win_rate * 20
        score += win_score
        
        # 4. 稳定性占10%
        if trade_details:
            pnl_pcts = [t["pnl_pct"] for t in trade_details]
            stability = 1 - min(np.std(pnl_pcts), 1.0)  # 波动越小越稳定
            score += stability * 10
        
        return max(score, -50)  # 最低-50分

    def _calculate_max_drawdown(self, trades) -> float:
        """计算最大回撤"""
        if not trades:
            return 0
        
        cumulative = np.cumsum(trades)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1000)  # 避免除零
        
        return abs(np.min(drawdown))



# ==================== 风险管理 ====================
class RiskManager:
    """增强的风险管理器"""

    def __init__(self, config: Config, db: Database, logger: Logger):
        self.config = config
        self.db = db
        self.logger = logger

    def check_daily_loss(self, balance: float) -> bool:
        """检查日损失限制"""
        daily_pnl = self.db.get_daily_pnl()
        max_loss = balance * self.config.MAX_DAILY_LOSS
        if daily_pnl < -max_loss:
            self.logger.warning(
                f"[RISK] 达到日损失限制: {daily_pnl:.2f} / {-max_loss:.2f}"
            )
            return False
        return True

    def check_position_limit(self) -> bool:
        """检查持仓数量限制"""
        open_positions = self.db.get_open_positions()
        if len(open_positions) >= self.config.MAX_POSITIONS:
            return False
        return True

    def calculate_position_size(self, balance: float, price: float) -> float:
        """动态计算仓位大小 - 添加安全检查"""
        if balance <= 0 or price <= 0:
            return 0

        # 根据账户余额动态调整
        max_position = min(
            self.config.POSITION_SIZE,
            balance * 0.15,  # 最多15%
            balance / max(self.config.MAX_POSITIONS, 1),  # 防止除零
        )

        amount = max_position / price

        # 确保仓位不会太小
        if amount * price < self.config.MIN_TRADE_AMOUNT:
            return 0

        return amount

    def check_stop_loss(self, position: Dict, current_price: float) -> Tuple[bool, str]:
        """检查止损 - 包括固定止损和移动止盈"""
        entry_price = position["entry_price"]
        highest_price = position.get("highest_price", entry_price)

        if position["side"] == "buy":
            # 固定止损
            loss_pct = (current_price - entry_price) / entry_price
            if loss_pct < -self.config.MAX_LOSS_PER_TRADE:
                return True, f"固定止损 ({loss_pct*100:.2f}%)"

            # 移动止盈
            if current_price > highest_price:
                self.db.update_highest_price(position["id"], current_price)
                highest_price = current_price

            trailing_loss = (current_price - highest_price) / highest_price
            if (
                trailing_loss < -self.config.TRAILING_STOP_PCT
                and highest_price > entry_price * 1.02
            ):
                return True, f"移动止盈 (最高:{highest_price:.6f})"

            # 固定止盈
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct > self.config.TAKE_PROFIT_PCT:
                return True, f"止盈 ({profit_pct*100:.2f}%)"

        return False, ""

    def should_trade(self, symbol: str, strategy: str) -> bool:
        """综合判断是否应该交易"""
        # 检查策略表现
        stats = self.db.get_strategy_stats(strategy, days=7)
        if stats["total_trades"] > 10 and stats["win_rate"] < 30:
            self.logger.warning(
                f"[RISK] 策略 {strategy} 胜率过低: {stats['win_rate']:.1f}%"
            )
            return False

        return True


# ==================== 主交易引擎 ====================
class TradingEngine:
    """优化的主交易引擎"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger(config)
        self.db = Database(config)
        self.api = BinanceAPI(config, self.logger)
        self.scorer = PairScorer(self.api, config, self.logger)
        self.strategy_manager = StrategyManager(config)
        self.backtest_engine = BacktestEngine(self.api, config, self.logger)
        self.risk_manager = RiskManager(config, self.db, self.logger)

        self.selected_pairs = []
        self.strategy_scores = {}
        self.running = False

        # 信号过滤器
        self.signal_filter = SignalFilter(config, self.logger)

        # ML优化器
        if config.ENABLE_ML_OPTIMIZATION:
            if ML_AVAILABLE:
                self.ml_optimizer = MLOptimizer(config, self.logger)
            else:
                self.logger.warning("[WARNING] ML功能已禁用: scikit-learn未安装")
                self.ml_optimizer = None
        else:
            self.ml_optimizer = None

        # 注册信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """优雅退出 - 改进信号处理"""
        print("\n")  # 换行使输出更清晰
        self.logger.info("[SIGNAL] 收到退出信号,正在安全停止...")
        self.running = False
        # 不要在信号处理器中做太多事情

    def initialize(self) -> bool:
        """初始化系统"""
        self.logger.info("=" * 50)
        self.logger.info("初始化币安量化交易系统 v2.0")
        self.logger.info("=" * 50)

        # 检查账户
        balance = self.api.get_balance()
        self.logger.info(f"账户余额: {balance['total']:.2f} USDT")

        if balance["total"] < self.config.MIN_TRADE_AMOUNT:
            self.logger.error("账户余额不足!")
            return False

        # 选择币对
        self.selected_pairs = self.scorer.select_top_pairs()
        if not self.selected_pairs:
            self.logger.error("未找到合适的交易对!")
            return False

        # 回测优化策略
        self.optimize_strategies()

        # # 初始化ML模型（如果启用）
        # if self.ml_optimizer and self.config.ENABLE_ML_OPTIMIZATION:
        #     self.logger.info("[ML] 开始初始化训练模型...")
        #     trained_count = 0
        #     for symbol in self.selected_pairs[:3]:  # 只训练前3个币对
        #         for strategy in self.strategy_manager.get_all_strategies():
        #             try:
        #                 if self.ml_optimizer.train_model(self.api, symbol, strategy):
        #                     trained_count += 1
        #             except Exception as e:
        #                 self.logger.error(
        #                     f"[ML] 初始化训练失败 {symbol}-{strategy.name}: {e}"
        #                 )

        #     if trained_count > 0:
        #         self.logger.info(f"[ML] 成功训练 {trained_count} 个模型")
        #     else:
        #         self.logger.warning("[ML] 没有成功训练任何模型，ML功能将被禁用")
        #         self.ml_optimizer = None

        # 清理旧数据
        self.db.cleanup_old_data()

        # 运行健康检查
        diagnostics = SystemDiagnostics(self)
        health = diagnostics.run_health_check()

        if health["overall"] != "HEALTHY":
            self.logger.warning("系统存在问题,请检查上述警告")

        return True

    def optimize_strategies(self):
        """优化所有策略 - 使用ML"""
        self.logger.info("开始评估策略...")

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_strategy = {}
            for strategy in self.strategy_manager.get_all_strategies():
                # 在3币对上测试
                future = executor.submit(self._evaluate_strategy, strategy)
                future_to_strategy[future] = strategy

            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    avg_score = future.result()
                    self.strategy_scores[strategy.name] = avg_score
                    self.logger.info(f"{strategy.name} Sharpe: {avg_score:.2f}")
                except Exception as e:
                    self.logger.error(f"评估策略失败 {strategy.name}: {e}")
                    self.strategy_scores[strategy.name] = -999

        # ML参数优化
        if self.ml_optimizer and self.config.ENABLE_ML_OPTIMIZATION:
            self.logger.info("[ML] 开始参数优化...")
            for strategy in self.strategy_manager.get_all_strategies():
                if self.strategy_scores.get(strategy.name, -999) > -5:
                    # 在表现较好的币对上优化
                    symbol = self.selected_pairs[0] if self.selected_pairs else None
                    if symbol:
                        optimized_params = self.ml_optimizer.optimize_strategy_params(
                            self.api, symbol, strategy
                        )
                        strategy.params = optimized_params

        # 按评分排序
        sorted_strategies = sorted(
            self.strategy_scores.items(), key=lambda x: x[1], reverse=True
        )
        if sorted_strategies:
            self.logger.info(f"最佳策略: {sorted_strategies[0][0]}")

    def _evaluate_strategy(self, strategy: Strategy) -> float:
        """评估单个策略"""
        scores = []
        for symbol in self.selected_pairs[:3]:
            result = self.backtest_engine.backtest_strategy(strategy, symbol)
            if result["trades"] > 0:
                scores.append(result["sharpe"])
        return np.mean(scores) if scores else -999

    def scan_signals(self):
        """扫描交易信号 - 优化并发控制和错误处理"""
        balance = self.api.get_balance()

        # 风险检查
        if not self.risk_manager.check_daily_loss(balance["total"]):
            return

        if not self.risk_manager.check_position_limit():
            return

        # 统计任务
        tasks = []
        for symbol in self.selected_pairs:
            for strategy in self.strategy_manager.get_all_strategies():
                weight = self.config.STRATEGY_WEIGHTS.get(strategy.name, 1.0)
                score = self.strategy_scores.get(strategy.name, 0)
                if score > -5 and weight > 0:
                    tasks.append((symbol, strategy))

        # 限制并发任务数量,避免API限流
        # 如果没有待处理任务，直接返回以避免创建 max_workers=0 的线程池
        if not tasks:
            self.logger.info("没有任务需要扫描，跳过本次扫描")
            return

        max_concurrent = min(self.config.MAX_WORKERS, len(tasks))

        # 并行扫描 - 修复超时处理
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(self.check_entry_signal, symbol, strategy): (
                    symbol,
                    strategy,
                )
                for symbol, strategy in tasks
            }

            # 不使用整体超时，而是单个任务超时
            for future in as_completed(futures):
                symbol, strategy = futures[future]
                try:
                    future.result(timeout=10)
                except TimeoutError:
                    self.logger.warning(
                        f"[TIMEOUT] 扫描超时: {symbol} - {strategy.name}"
                    )
                except Exception as e:
                    self.logger.error(f"扫描错误 {symbol}: {e}")

    def check_entry_signal(self, symbol: str, strategy: Strategy):
        """检查入场信号 - 添加信号过滤和ML预测"""
        try:
            # 风险检查
            if not self.risk_manager.should_trade(symbol, strategy.name):
                return

            df = self.api.get_ohlcv(symbol, "1h", 200)  # 增加到200根K线以支持趋势分析
            if df.empty:
                return

            df = strategy.calculate_signals(df)

            if strategy.should_enter(df):
                # 应用信号过滤器
                if not self.signal_filter.filter_signal(df, 1, symbol, strategy.name):
                    return

                # ML预测信号质量
                if self.ml_optimizer and self.config.ENABLE_ML_OPTIMIZATION:
                    confidence = self.ml_optimizer.predict_signal_quality(
                        df, symbol, strategy.name
                    )

                    # 如果置信度低于阈值且不是中性值（0.5表示没有模型）
                    if (
                        confidence < self.config.ML_CONFIDENCE_THRESHOLD
                        and confidence != 0.5
                    ):
                        self.logger.info(
                            f"[ML] 信号质量不足: {symbol} | "
                            f"置信度:{confidence:.2%} < {self.config.ML_CONFIDENCE_THRESHOLD:.2%}"
                        )
                        return

                    # 只有在有模型且置信度高时才记录
                    if confidence != 0.5:
                        self.logger.info(
                            f"[ML] 信号质量良好: {symbol} | 置信度:{confidence:.2%}"
                        )
                    else:
                        self.logger.info(f"[ML] {symbol} 暂无模型，使用基础策略")

                # 通过所有检查,开仓
                self.enter_position(symbol, strategy)

        except Exception as e:
            self.logger.error(f"检查信号错误 {symbol}: {e}")

    def enter_position(self, symbol: str, strategy: Strategy):
        """进入持仓 - 增强安全检查"""
        try:
            # 检查是否已有持仓
            if self.db.has_open_position(symbol, strategy.name):
                return

            # 再次检查持仓限制 (防止并发问题)
            if len(self.db.get_open_positions()) >= self.config.MAX_POSITIONS:
                self.logger.warning(f"[SKIP] 已达持仓上限: {symbol}")
                return

            balance = self.api.get_balance()
            if balance["free"] < self.config.MIN_TRADE_AMOUNT:
                self.logger.warning(f"[SKIP] 余额不足: {balance['free']:.2f} USDT")
                return

            current_price = self.api.get_ticker_price(symbol)
            if not current_price or current_price <= 0:
                self.logger.error(f"[ERROR] 无效价格: {symbol}")
                return

            # 计算仓位
            amount = self.risk_manager.calculate_position_size(
                balance["free"], current_price
            )
            if amount == 0:
                self.logger.warning(f"[SKIP] 仓位计算为0: {symbol}")
                return

            position_value = amount * current_price

            # 检查最小金额
            if position_value < self.config.MIN_TRADE_AMOUNT:
                self.logger.warning(
                    f"[SKIP] 金额不足: {symbol} ({position_value:.2f} < {self.config.MIN_TRADE_AMOUNT})"
                )
                return

            # 检查余额充足
            fee_estimate = position_value * 0.002  # 0.2%手续费估算
            if position_value + fee_estimate > balance["free"]:
                self.logger.warning(f"[SKIP] 余额不足以支付手续费: {symbol}")
                return

            # 计算止损价
            stop_loss_price = current_price * (1 - self.config.MAX_LOSS_PER_TRADE)

            # 创建订单
            order = self.api.create_order(symbol, "buy", amount, strategy=strategy.name)

            if order and "id" in order:
                # 获取实际成交价格
                actual_price = order.get("price", current_price)
                actual_amount = order.get("filled", amount)

                # 保存持仓
                position_id = self.db.save_position(
                    symbol,
                    strategy.name,
                    "buy",
                    actual_price,
                    actual_amount,
                    stop_loss_price,
                )

                # 保存交易
                fee = position_value * self.api.get_trading_fee(symbol)
                self.db.save_trade(
                    symbol, strategy.name, "buy", actual_price, actual_amount, fee
                )

                self.logger.info(
                    f"[SUCCESS] 开仓: {symbol} | 策略:{strategy.name} | "
                    f"价格:{actual_price:.8f} | 金额:{position_value:.2f} USDT | "
                    f"止损:{stop_loss_price:.8f}"
                )
            else:
                self.logger.error(f"[ERROR] 订单创建失败: {symbol}")

        except Exception as e:
            import traceback

            self.logger.error(f"开仓失败 {symbol}: {e}")
            self.logger.error(f"详情: {traceback.format_exc()}")

    def check_exit_signals(self):
        """检查出场信号"""
        open_positions = self.db.get_open_positions()

        for position in open_positions:
            try:
                symbol = position["symbol"]
                strategy = self.strategy_manager.get_strategy(position["strategy"])
                if not strategy:
                    continue

                # 获取当前价格
                current_price = self.api.get_ticker_price(symbol)
                if not current_price:
                    continue

                # 检查强制平仓时间
                entry_time_str = position["entry_time"].replace("Z", "+00:00")
                entry_time = datetime.fromisoformat(entry_time_str)

                if entry_time.tzinfo is not None:
                    now = datetime.now(entry_time.tzinfo)
                else:
                    now = datetime.now()

                hold_hours = (now - entry_time).total_seconds() / 3600
                if hold_hours > self.config.FORCE_EXIT_HOURS:
                    self.exit_position(
                        position, current_price, f"持仓超时({hold_hours:.0f}h)"
                    )
                    continue

                # 检查止损止盈
                should_stop, reason = self.risk_manager.check_stop_loss(
                    position, current_price
                )
                if should_stop:
                    self.exit_position(position, current_price, reason)
                    continue

                # 检查策略信号
                df = self.api.get_ohlcv(symbol, "1h", 100)
                if not df.empty:
                    df = strategy.calculate_signals(df)
                    if strategy.should_exit(df, position):
                        self.exit_position(position, current_price, "策略信号")

            except Exception as e:
                self.logger.error(f"检查出场失败 {position['symbol']}: {e}")

    def exit_position(self, position: Dict, exit_price: float, reason: str):
        """退出持仓 - 增强错误处理"""
        try:
            symbol = position["symbol"]
            amount = position["amount"]

            # 验证数据有效性
            if amount <= 0 or exit_price <= 0:
                self.logger.error(
                    f"[ERROR] 无效的平仓参数: {symbol} amount={amount} price={exit_price}"
                )
                return

            # 创建卖单
            order = self.api.create_order(
                symbol, "sell", amount, strategy=position["strategy"]
            )

            if order and "id" in order:
                # 获取实际成交价格
                actual_exit_price = order.get("price", exit_price)
                actual_amount = order.get("filled", amount)

                # 计算盈亏
                entry_value = position["entry_price"] * position["amount"]
                exit_value = actual_exit_price * actual_amount
                fee = exit_value * self.api.get_trading_fee(symbol)

                # 净盈亏 = 卖出金额 - 买入金额 - 手续费
                pnl = exit_value - entry_value - fee
                pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0

                # 更新持仓
                self.db.update_position(position["id"], actual_exit_price, pnl)

                # 保存交易
                self.db.save_trade(
                    symbol,
                    position["strategy"],
                    "sell",
                    actual_exit_price,
                    actual_amount,
                    fee,
                )

                emoji = "[WIN]" if pnl > 0 else "[LOSS]"
                self.logger.info(
                    f"{emoji} 平仓: {symbol} | {reason} | "
                    f"入场:{position['entry_price']:.8f} 出场:{actual_exit_price:.8f} | "
                    f"盈亏: {pnl:.2f} USDT ({pnl_pct:+.2f}%) | "
                    f"持仓时间: {self._calculate_hold_time(position)}"
                )
            else:
                self.logger.error(f"[ERROR] 平仓订单创建失败: {symbol}")

        except Exception as e:
            import traceback

            self.logger.error(f"平仓失败 {symbol}: {e}")
            self.logger.error(f"详情: {traceback.format_exc()}")

    def _calculate_hold_time(self, position: Dict) -> str:
        """计算持仓时间 - 修复时区问题"""
        try:
            # 处理时区
            entry_time_str = position["entry_time"].replace("Z", "+00:00")
            entry_time = datetime.fromisoformat(entry_time_str)

            # 统一使用UTC时间
            if entry_time.tzinfo is not None:
                now = datetime.now(entry_time.tzinfo)
            else:
                now = datetime.now()

            hold_seconds = (now - entry_time).total_seconds()

            if hold_seconds < 3600:
                return f"{hold_seconds/60:.0f}m"
            elif hold_seconds < 86400:
                return f"{hold_seconds/3600:.1f}h"
            else:
                return f"{hold_seconds/86400:.1f}d"
        except Exception as e:
            self.logger.error(f"计算持仓时间失败: {e}")
            return "N/A"

    def sync_positions(self):
        """同步持仓数据 - 优化差异检测"""
        try:
            balance = self.api.exchange.fetch_balance()
            open_positions = self.db.get_open_positions()

            for position in open_positions:
                symbol_base = position["symbol"].split("/")[0]
                if symbol_base in balance:
                    actual_amount = balance[symbol_base]["total"]
                    db_amount = position["amount"]

                    # 只有差异超过10%才报警
                    if abs(actual_amount - db_amount) / max(db_amount, 0.01) > 0.1:
                        # 排除测试网数据问题
                        if actual_amount > db_amount * 5:
                            self.logger.warning(
                                f"[SYNC] 测试网数据异常: {position['symbol']} "
                                f"忽略此差异"
                            )
                        else:
                            self.logger.warning(
                                f"[SYNC] 持仓差异: {position['symbol']} "
                                f"数据库:{db_amount:.6f} 实际:{actual_amount:.6f}"
                            )
        except Exception as e:
            self.logger.error(f"同步持仓失败: {e}")

    def save_account_state(self):
        """保存账户状态"""
        try:
            balance = self.api.get_balance()
            open_positions = self.db.get_open_positions()
            daily_pnl = self.db.get_daily_pnl()

            # 计算总权益
            equity = balance["total"]
            position_value = 0
            unrealized_pnl = 0

            for position in open_positions:
                try:
                    current_price = self.api.get_ticker_price(position["symbol"])
                    if current_price:
                        pos_value = position["amount"] * current_price
                        position_value += pos_value
                        pnl = (current_price - position["entry_price"]) * position[
                            "amount"
                        ]
                        unrealized_pnl += pnl
                except:
                    pass

            equity += position_value

            self.db.save_account_snapshot(balance["total"], equity, daily_pnl)

            # 打印状态报告
            self.logger.info(f"\n{'='*60}")
            self.logger.info(
                f"账户状态报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            self.logger.info(f"{'='*60}")
            self.logger.info(f"可用余额: {balance['free']:.2f} USDT")
            self.logger.info(f"总余额: {balance['total']:.2f} USDT")
            self.logger.info(f"持仓价值: {position_value:.2f} USDT")
            self.logger.info(f"总权益: {equity:.2f} USDT")
            self.logger.info(f"未实现盈亏: {unrealized_pnl:.2f} USDT")
            self.logger.info(f"今日已实现: {daily_pnl:.2f} USDT")
            self.logger.info(
                f"持仓数: {len(open_positions)}/{self.config.MAX_POSITIONS}"
            )

            if open_positions:
                self.logger.info(f"\n当前持仓:")
                for pos in open_positions:
                    try:
                        current_price = self.api.get_ticker_price(pos["symbol"])
                        if current_price:
                            pnl = (current_price - pos["entry_price"]) * pos["amount"]
                            pnl_pct = (pnl / (pos["entry_price"] * pos["amount"])) * 100

                            # 修复持仓时间计算 - 统一时区
                            entry_time_str = pos["entry_time"].replace("Z", "+00:00")
                            entry_time = datetime.fromisoformat(entry_time_str)

                            if entry_time.tzinfo is not None:
                                now = datetime.now(entry_time.tzinfo)
                            else:
                                now = datetime.now()

                            hold_time = (now - entry_time).total_seconds() / 3600

                            # 格式化持仓时间
                            if hold_time < 1:
                                time_str = f"{hold_time*60:.0f}m"
                            else:
                                time_str = f"{hold_time:.1f}h"

                            self.logger.info(
                                f"  {pos['symbol']:12} | {pos['strategy']:15} | "
                                f"入:{pos['entry_price']:.8f} 现:{current_price:.8f} | "
                                f"盈亏:{pnl:+.2f} ({pnl_pct:+.2f}%) | {time_str}"
                            )
                    except Exception as e:
                        self.logger.error(f"  获取{pos['symbol']}状态失败: {e}")

            # 策略表现
            self.logger.info(f"\n策略表现 (7日):")
            for strategy_name in self.config.STRATEGIES:
                stats = self.db.get_strategy_stats(strategy_name, days=7)
                if stats["total_trades"] > 0:
                    self.logger.info(
                        f"  {strategy_name:15} | "
                        f"交易:{stats['total_trades']:3} | "
                        f"胜率:{stats['win_rate']:.1f}% | "
                        f"总盈亏:{stats['total_pnl']:+.2f}"
                    )
                else:
                    self.logger.info(f"  {strategy_name:15} | 暂无交易记录")

            self.logger.info(f"{'='*60}\n")

        except Exception as e:
            self.logger.error(f"保存账户状态失败: {e}")

    def print_summary(self):
        """打印运行总结"""
        try:
            self.logger.info("\n" + "=" * 60)
            self.logger.info("系统运行总结")
            self.logger.info("=" * 60)

            # 总体统计
            conn = self.db.get_connection()
            cursor = conn.cursor()

            # 总交易次数
            cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'CLOSED'")
            total_trades = cursor.fetchone()[0]

            # 总盈亏
            cursor.execute("SELECT SUM(pnl) FROM positions WHERE status = 'CLOSED'")
            total_pnl = cursor.fetchone()[0] or 0

            # 胜率
            cursor.execute(
                """
                SELECT 
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) 
                FROM positions WHERE status = 'CLOSED'
            """
            )
            win_rate = cursor.fetchone()[0] or 0

            self.logger.info(f"总交易次数: {total_trades}")
            self.logger.info(f"总盈亏: {total_pnl:+.2f} USDT")
            self.logger.info(f"整体胜率: {win_rate:.1f}%")

            # 最佳/最差交易
            cursor.execute(
                """
                SELECT symbol, pnl FROM positions 
                WHERE status = 'CLOSED' ORDER BY pnl DESC LIMIT 1
            """
            )
            best = cursor.fetchone()
            if best:
                self.logger.info(f"最佳交易: {best[0]} (+{best[1]:.2f} USDT)")

            cursor.execute(
                """
                SELECT symbol, pnl FROM positions 
                WHERE status = 'CLOSED' ORDER BY pnl ASC LIMIT 1
            """
            )
            worst = cursor.fetchone()
            if worst:
                self.logger.info(f"最差交易: {worst[0]} ({worst[1]:.2f} USDT)")

            self.logger.info("=" * 60 + "\n")

        except Exception as e:
            self.logger.error(f"生成总结失败: {e}")

    def run(self):
        """运行主循环 - 增强错误恢复"""
        if not self.initialize():
            self.logger.error("初始化失败")
            return

        self.running = True
        self.logger.info("[START] 系统启动成功\n")

        iteration = 0
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.running:
            try:
                iteration += 1
                self.logger.info(f"\n{'='*60}")
                self.logger.info(
                    f"第 {iteration} 次扫描 - {datetime.now().strftime('%H:%M:%S')}"
                )
                self.logger.info(f"{'='*60}")

                # 同步持仓
                if iteration % 10 == 0:
                    self.sync_positions()

                # 扫描入场信号
                self.scan_signals()

                # 检查出场信号
                self.check_exit_signals()

                # 保存账户状态
                if iteration % 6 == 0:
                    self.save_account_state()

                # 每小时重新评分币对
                if iteration % 60 == 0:
                    self.logger.info("[UPDATE] 重新评分币对...")
                    self.selected_pairs = self.scorer.select_top_pairs()

                    # ML模型训练
                    if (
                        self.ml_optimizer
                        and iteration % (self.config.ML_RETRAIN_HOURS * 60) == 0
                    ):
                        self.logger.info("[ML] 开始训练模型...")
                        for symbol in self.selected_pairs[:3]:
                            for strategy in self.strategy_manager.get_all_strategies():
                                try:
                                    self.ml_optimizer.train_model(
                                        self.api, symbol, strategy
                                    )
                                except Exception as e:
                                    self.logger.error(f"[ML] 训练失败 {symbol}: {e}")

                # 显示简要状态
                if iteration % 5 == 0:
                    open_pos = len(self.db.get_open_positions())
                    balance = self.api.get_balance()
                    self.logger.info(
                        f"[STATUS] 持仓:{open_pos}/{self.config.MAX_POSITIONS} | "
                        f"余额:{balance['free']:.2f} USDT | "
                        f"连续错误:{consecutive_errors}"
                    )

                # 重置错误计数
                consecutive_errors = 0

                # 等待下次扫描
                time.sleep(self.config.SCAN_INTERVAL)

            except KeyboardInterrupt:
                self.logger.info("\n[SHUTDOWN] 用户中断")
                break
            except Exception as e:
                consecutive_errors += 1
                import traceback

                error_details = traceback.format_exc()
                self.logger.error(f"主循环错误 (第{consecutive_errors}次): {e}")
                self.logger.error(f"错误详情:\n{error_details}")

                # 连续错误过多,停止运行
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(
                        f"[FATAL] 连续错误{consecutive_errors}次,系统停止"
                    )
                    break

                # 错误后等待更长时间
                time.sleep(self.config.SCAN_INTERVAL * 2)

        # 打印运行总结
        self.print_summary()
        self.logger.info("[STOP] 系统已停止")


# ==================== 系统诊断工具 ====================
class SystemDiagnostics:
    """系统诊断和健康检查"""

    def __init__(self, engine):
        self.engine = engine
        self.logger = engine.logger

    def run_health_check(self) -> Dict:
        """运行系统健康检查"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("系统健康检查")
        self.logger.info("=" * 60)

        health = {
            "api_connection": self._check_api_connection(),
            "database": self._check_database(),
            "balance": self._check_balance(),
            "positions": self._check_positions(),
            "strategies": self._check_strategies(),
            "ml_models": self._check_ml_models(),
            "overall": "HEALTHY",
        }

        # 评估总体健康状态
        issues = []
        for key, status in health.items():
            if key != "overall" and status != "OK":
                issues.append(f"{key}: {status}")

        if issues:
            health["overall"] = "ISSUES_FOUND"
            self.logger.warning(f"发现问题: {', '.join(issues)}")
        else:
            self.logger.info("系统健康状态: 良好")

        self.logger.info("=" * 60 + "\n")
        return health

    def _check_api_connection(self) -> str:
        """检查API连接"""
        try:
            balance = self.engine.api.get_balance()
            if balance["total"] > 0:
                self.logger.info("[OK] API连接正常")
                return "OK"
            else:
                self.logger.warning("[WARN] 余额为0")
                return "WARN"
        except Exception as e:
            self.logger.error(f"[ERROR] API连接失败: {e}")
            return "ERROR"

    def _check_database(self) -> str:
        """检查数据库"""
        try:
            positions = self.engine.db.get_open_positions()
            self.logger.info(f"[OK] 数据库正常 (持仓:{len(positions)})")
            return "OK"
        except Exception as e:
            self.logger.error(f"[ERROR] 数据库错误: {e}")
            return "ERROR"

    def _check_balance(self) -> str:
        """检查账户余额"""
        try:
            balance = self.engine.api.get_balance()
            min_balance = self.engine.config.MIN_TRADE_AMOUNT * 2

            if balance["free"] < min_balance:
                self.logger.warning(
                    f"[WARN] 可用余额不足: {balance['free']:.2f} < {min_balance:.2f}"
                )
                return "WARN"

            self.logger.info(f"[OK] 余额充足: {balance['free']:.2f} USDT")
            return "OK"
        except Exception as e:
            self.logger.error(f"[ERROR] 余额检查失败: {e}")
            return "ERROR"

    def _check_positions(self) -> str:
        """检查持仓状态"""
        try:
            positions = self.engine.db.get_open_positions()

            for pos in positions:
                current_price = self.engine.api.get_ticker_price(pos["symbol"])
                if current_price:
                    pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]

                    # 检查是否有严重亏损
                    if pnl_pct < -0.05:  # 亏损>5%
                        self.logger.warning(
                            f"[WARN] {pos['symbol']} 亏损严重: {pnl_pct*100:.2f}%"
                        )

            self.logger.info(f"[OK] 持仓检查完成: {len(positions)}个")
            return "OK"
        except Exception as e:
            self.logger.error(f"[ERROR] 持仓检查失败: {e}")
            return "ERROR"

    def _check_strategies(self) -> str:
        """检查策略状态"""
        try:
            issues = []
            for strategy_name, score in self.engine.strategy_scores.items():
                if score < -5:
                    issues.append(f"{strategy_name}(Sharpe:{score:.2f})")

            if issues:
                self.logger.warning(f"[WARN] 策略表现差: {', '.join(issues)}")
                return "WARN"

            self.logger.info("[OK] 策略状态正常")
            return "OK"
        except Exception as e:
            self.logger.error(f"[ERROR] 策略检查失败: {e}")
            return "ERROR"

    def _check_ml_models(self) -> str:
        """检查ML模型"""
        if not self.engine.ml_optimizer:
            self.logger.info("[N/A] ML功能未启用")
            return "N/A"

        try:
            model_count = len(self.engine.ml_optimizer.models)
            if model_count == 0:
                self.logger.info("[INFO] 首次运行，ML模型将在后台训练")
                return "OK"  # 首次运行没有模型是正常的

            self.logger.info(f"[OK] ML模型: {model_count}个")
            return "OK"
        except Exception as e:
            self.logger.error(f"[ERROR] ML检查失败: {e}")
            return "ERROR"


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置参数
    config = Config(
        API_KEY="GoEdbkGG3QELC5hD8aaX9rn9aA6XQA5Ss1C8I9lIkqc466yupg1sbYJJpFa4umCi",
        API_SECRET="ZbCqTI13N85IMwoRF6CC2fULLBW3EytXZnBoAgJ6PTCo8rM06sFIkVFoaKFyt7Fj",
        TEST_MODE=True,
        MIN_TRADE_AMOUNT=11.0,
        POSITION_SIZE=100.0,
        MAX_POSITIONS=5,
        TOP_PAIRS_COUNT=20,
        # 选择要使用的策略
        STRATEGIES=[
            "RSI_STRATEGY",  # RSI超买超卖
            # "MACD_STRATEGY",  # MACD趋势
            "MA_CROSS_STRATEGY",  # 均线交叉 ⭐推荐
            # "BOLLINGER_STRATEGY",  # 布林带
            # "GRID_STRATEGY",  # 网格交易
            "BREAKOUT_STRATEGY",  # 突破策略
            "MEAN_REVERSION_STRATEGY",  # 均值回归
            # "EMA_STRATEGY",  # EMA快速反应
            # "MOMENTUM_STRATEGY",  # 动量策略
        ],
        # 策略权重 (可选)
        STRATEGY_WEIGHTS={
            "MA_CROSS_STRATEGY": 1.5,  # 主力
            "RSI_STRATEGY": 1.0,
            "BREAKOUT_STRATEGY": 1.2,
            "MEAN_REVERSION_STRATEGY": 0.8,
        },
        # 信号过滤配置 ⭐新增
        ENABLE_MULTI_INDICATOR_FILTER=True,  # 启用多指标共振
        MIN_INDICATORS_AGREE=2,  # 至少2个指标同意
        ENABLE_TREND_FILTER=True,  # 启用趋势过滤
        TREND_STRENGTH_THRESHOLD=0.3,  # 趋势强度阈值
        VOLUME_FILTER=False,  # 成交量过滤
        MIN_VOLUME_RATIO=1.2,  # 最小成交量比率
        # 机器学习配置 ⭐新增
        ENABLE_ML_OPTIMIZATION=False,  # 启用ML优化
        ML_TRAINING_DAYS=60,  # 训练数据天数
        ML_RETRAIN_HOURS=24,  # 每24小时重新训练
        ML_MIN_SAMPLES=100,  # 最少样本数
        ML_CONFIDENCE_THRESHOLD=0.5,  # 置信度阈值60%
        BLACKLIST_PAIRS=["USDC", "FDUSD", "TUSD"],  # 稳定币黑名单
        SCAN_INTERVAL=60,
        MAX_WORKERS=5,
    )

    # 风险声明
    # print("\n" + "=" * 60)
    # print("[WARNING] 重要风险提示")
    # print("=" * 60)
    # print("1. 此代码仅供学习研究,不构成投资建议")
    # print("2. 加密货币交易风险极高,可能导致全部本金损失")
    # print("3. 请务必先在测试网测试,确保理解所有功能")
    # print("4. 任何策略都无法保证盈利")
    # print("5. 请根据自身风险承受能力谨慎投资")
    # print("6. ML模型需要足够的历史数据,初次运行可能需要较长时间")
    # if not ML_AVAILABLE:
    #     print("7. [警告] scikit-learn未安装，ML优化功能已禁用")
    # print("=" * 60)

    # confirm = input("\n我已知晓并同意承担所有风险 (输入 YES 继续): ")
    # if confirm != "YES":
    #     print("已取消运行")
    #     exit()

    # 创建并运行交易引擎
    engine = TradingEngine(config)
    engine.run()
