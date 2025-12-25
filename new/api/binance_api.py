import time
from typing import Dict, List, Optional

import pandas as pd
from utils import Config,Logger,CacheManager,RetryDecorator,ValidationUtils


# ==================== 交易所接口（优化缓存和重试）====================
import ccxt


class BinanceAPI:
    """优化的币安API封装"""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.cache_manager = CacheManager(config.CACHE_TTL)
        
        self.exchange = ccxt.binance({
            'apiKey': config.API_KEY,
            'secret': config.API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'timeout': 30000,
        })
        
        if config.TEST_MODE:
            self.exchange.set_sandbox_mode(True)
            self.logger.info("[TEST MODE] 测试模式已启用")
    
    @RetryDecorator.retry(max_attempts=3, delay=1.0)
    def get_balance(self) -> Dict[str, float]:
        """获取账户余额"""
        balance = self.exchange.fetch_balance()
        return {
            'free': balance['USDT']['free'],
            'used': balance['USDT']['used'],
            'total': balance['USDT']['total']
        }
    
    def get_all_tickers(self) -> List[Dict]:
        """获取所有交易对行情"""
        cache_key = 'all_tickers'
        cached = self.cache_manager.get(cache_key, ttl=60)
        if cached:
            return cached
        
        try:
            tickers = self.exchange.fetch_tickers()
            result = []
            for symbol, ticker in tickers.items():
                if symbol.endswith(f"/{self.config.QUOTE_CURRENCY}"):
                    base = symbol.replace(f"/{self.config.QUOTE_CURRENCY}", "")
                    if base in self.config.BLACKLIST_PAIRS:
                        continue
                    result.append({
                        'symbol': symbol,
                        'price': ticker['last'],
                        'volume': ticker['quoteVolume'],
                        'change': ticker['percentage']
                    })
            
            self.cache_manager.set(cache_key, result)
            return result
        except Exception as e:
            self.logger.error(f"获取行情失败: {e}")
            return []
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """获取K线数据 - 使用缓存"""
        cache_key = f"ohlcv_{symbol}_{timeframe}_{limit}_{int(time.time() / self.config.CACHE_TTL)}"
        cached = self.cache_manager.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            df = self._fetch_ohlcv_internal(symbol, timeframe, limit)
            if not df.empty:
                self.cache_manager.set(cache_key, df)
            return df
        except Exception as e:
            self.logger.error(f"获取K线数据失败 {symbol}: {e}")
            return pd.DataFrame()
    
    @RetryDecorator.retry(max_attempts=3, delay=2.0)
    def _fetch_ohlcv_internal(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """内部K线获取方法"""
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """获取当前价格 - 使用缓存"""
        cache_key = f"price_{symbol}"
        cached = self.cache_manager.get(cache_key, ttl=5)
        if cached is not None:
            return cached
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']
            if ValidationUtils.validate_price(price):
                self.cache_manager.set(cache_key, price)
                return price
            return None
        except Exception as e:
            self.logger.error(f"获取价格失败 {symbol}: {e}")
            return None
    
    @RetryDecorator.retry(max_attempts=3, delay=1.0)
    def create_order(self, symbol: str, side: str, amount: float, 
                    price: float = None, strategy: str = "MANUAL") -> Dict:
        """创建订单"""
        order_type = 'limit' if price else 'market'
        params = {'price': price} if price else {}
        
        order = self.exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            params=params
        )
        
        self.logger.trade(side.upper(), symbol, price or order.get('price', 0), amount, strategy)
        return order
    
    def get_trading_fee(self, symbol: str) -> float:
        """获取交易手续费率"""
        try:
            fees = self.exchange.fetch_trading_fees()
            return fees.get(symbol, {}).get('taker', 0.001)
        except:
            return 0.001