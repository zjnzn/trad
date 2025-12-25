"""
技术指标计算模块
"""
import pandas as pd
import numpy as np

class IndicatorComputer:
    """技术指标计算器 - 增加数据验证"""
    
    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, min_length: int = 2) -> bool:
        """验证DataFrame有效性"""
        if df is None or df.empty:
            return False
        if len(df) < min_length:
            return False
        required_cols = ['close', 'high', 'low', 'open', 'volume']
        return all(col in df.columns for col in required_cols)
    
    @staticmethod
    def compute_rsi(period: int, df: pd.DataFrame):
        """计算RSI - 增加验证"""
        if not IndicatorComputer._validate_dataframe(df, period + 1):
            df['rsi'] = 50.0  # 默认值
            return
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 填充NaN
        df['rsi'].fillna(50, inplace=True)
    
    @staticmethod
    def compute_rsi_smooth(smooth: int, df: pd.DataFrame):
        """RSI平滑"""
        if 'rsi' in df.columns:
            df['rsi_smooth'] = df['rsi'].rolling(window=smooth).mean()
            df['rsi_smooth'].fillna(df['rsi'], inplace=True)
    
    @staticmethod
    def compute_macd(fast: int, slow: int, df: pd.DataFrame):
        """计算MACD"""
        if not IndicatorComputer._validate_dataframe(df, slow + 1):
            df['macd'] = 0.0
            return
        
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = exp1 - exp2
    
    @staticmethod
    def compute_signal_line(signal_period: int, df: pd.DataFrame):
        """MACD信号线"""
        if 'macd' in df.columns:
            df['signal_line'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
    
    @staticmethod
    def compute_histogram(df: pd.DataFrame):
        """MACD柱状图"""
        if 'macd' in df.columns and 'signal_line' in df.columns:
            df['histogram'] = df['macd'] - df['signal_line']
    
    @staticmethod
    def compute_bandwidth(period: int, std_dev: float, df: pd.DataFrame):
        """布林带"""
        if not IndicatorComputer._validate_dataframe(df, period + 1):
            return
        
        df['sma'] = df['close'].rolling(window=period).mean()
        df['std'] = df['close'].rolling(window=period).std()
        df['band_upper'] = df['sma'] + (df['std'] * std_dev)
        df['band_lower'] = df['sma'] - (df['std'] * std_dev)
        df['band_width'] = ValidationUtils.safe_divide(
            df['band_upper'] - df['band_lower'],
            df['sma']
        )
    
    @staticmethod
    def compute_ma(fast_period: int, slow_period: int, trend_filter: int, df: pd.DataFrame):
        """移动平均线"""
        if not IndicatorComputer._validate_dataframe(df, max(fast_period, slow_period, trend_filter)):
            return
        
        df['ma_fast'] = df['close'].rolling(window=fast_period).mean()
        df['ma_slow'] = df['close'].rolling(window=slow_period).mean()
        df['ma_trend'] = df['close'].rolling(window=trend_filter).mean()
    
    @staticmethod
    def compute_channel(period: int, df: pd.DataFrame):
        """价格通道"""
        if not IndicatorComputer._validate_dataframe(df, period):
            return
        
        df['upper_channel'] = df['high'].rolling(window=period).max()
        df['lower_channel'] = df['low'].rolling(window=period).min()
        df['mid_channel'] = (df['upper_channel'] + df['lower_channel']) / 2
    
    @staticmethod
    def compute_art(atr_period: int, df: pd.DataFrame):
        """ATR波动率"""
        if not IndicatorComputer._validate_dataframe(df, atr_period + 1):
            return
        
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=atr_period).mean()
    
    @staticmethod
    def compute_volume_ma(period: int, df: pd.DataFrame):
        """成交量均线"""
        if not IndicatorComputer._validate_dataframe(df, period):
            return
        
        df['volume_ma'] = df['volume'].rolling(window=period).mean()
    
    @staticmethod
    def compute_vol_ratio(df: pd.DataFrame):
        """成交量比率"""
        if 'volume' in df.columns and 'volume_ma' in df.columns:
            df['volume_ratio'] = ValidationUtils.safe_divide(
                df['volume'],
                df['volume_ma'],
                default=1.0
            )
    
    @staticmethod
    def compute_band(period: int, std_dev: float, df: pd.DataFrame):
        """标准差带"""
        if not IndicatorComputer._validate_dataframe(df, period):
            return
        
        df['mean'] = df['close'].rolling(window=period).mean()
        df['std'] = df['close'].rolling(window=period).std()
        df['lower_band'] = df['mean'] - (df['std'] * std_dev)
        df['upper_band'] = df['mean'] + (df['std'] * std_dev)
    
    @staticmethod
    def compute_deviation(df: pd.DataFrame):
        """价格偏离度"""
        if 'close' in df.columns and 'mean' in df.columns:
            df['deviation'] = ValidationUtils.safe_divide(
                df['close'] - df['mean'],
                df['mean']
            )
    
    @staticmethod
    def compute_ema(fast: int, medium: int, slow: int, df: pd.DataFrame):
        """指数移动平均"""
        if not IndicatorComputer._validate_dataframe(df, max(fast, medium, slow)):
            return
        
        df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
        df['ema_medium'] = df['close'].ewm(span=medium, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    
    @staticmethod
    def compute_bullish_alignment(df: pd.DataFrame):
        """多头排列"""
        if all(col in df.columns for col in ['ema_fast', 'ema_medium', 'ema_slow']):
            df['bullish_alignment'] = (
                (df['ema_fast'] > df['ema_medium']) & 
                (df['ema_medium'] > df['ema_slow'])
            )
    
    @staticmethod
    def compute_momentum(period: int, df: pd.DataFrame):
        """动量"""
        if not IndicatorComputer._validate_dataframe(df, period + 1):
            return
        
        df['momentum'] = ValidationUtils.safe_divide(
            df['close'] - df['close'].shift(period),
            df['close'].shift(period)
        ) * 100
    
    @staticmethod
    def compute_momentum_change(df: pd.DataFrame):
        """动量变化"""
        if 'momentum' in df.columns:
            df['momentum_change'] = df['momentum'].diff()
    
    @staticmethod
    def compute_volume_ratio(df: pd.DataFrame):
        """成交量比率"""
        if 'volume' in df.columns:
            vol_ma = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = ValidationUtils.safe_divide(df['volume'], vol_ma, 1.0)
    
    @staticmethod
    def compute_adx(df: pd.DataFrame, period: int = 14):
        """ADX趋势强度"""
        if not IndicatorComputer._validate_dataframe(df, period + 1):
            df['adx'] = 25.0  # 默认中性值
            return
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(window=period).mean()
        df['adx'].fillna(25.0, inplace=True)
    
    @staticmethod
    def compute_obv(df: pd.DataFrame):
        """OBV能量潮"""
        if not IndicatorComputer._validate_dataframe(df):
            df['obv'] = 0
            return
        
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = pd.Series(obv, index=df.index)