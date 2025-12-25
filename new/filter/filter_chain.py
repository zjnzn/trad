

# ==================== 过滤器链模式 =====================

from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class Filter(ABC):
    """过滤器基类 - 责任链模式"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.next_filter = None
    
    def set_next(self, filter_obj):
        """设置下一个过滤器"""
        self.next_filter = filter_obj
        return filter_obj
    
    def filter(self, df: pd.DataFrame, signal: int, symbol: str, strategy_name: str) -> tuple[bool, List[str]]:
        """
        执行过滤
        返回: (是否通过, 拒绝原因列表)
        """
        passed, reason = self._do_filter(df, signal, symbol, strategy_name)
        
        if not passed:
            return False, [reason] if reason else []
        
        # 传递给下一个过滤器
        if self.next_filter:
            return self.next_filter.filter(df, signal, symbol, strategy_name)
        
        return True, []
    
    @abstractmethod
    def _do_filter(self, df: pd.DataFrame, signal: int, symbol: str, strategy_name: str) -> tuple[bool, str]:
        """执行具体过滤逻辑"""
        pass

class MultiIndicatorFilter(Filter):
    """多指标共振过滤器"""
    
    def __init__(self, config, logger, multi_indicator):
        super().__init__(config, logger)
        self.multi_indicator = multi_indicator
    
    def _do_filter(self, df: pd.DataFrame, signal: int, symbol: str, strategy_name: str) -> tuple[bool, str]:
        if not self.config.ENABLE_MULTI_INDICATOR_FILTER or signal != 1:
            return True, ""
        
        agreement = self.multi_indicator.check_indicator_agreement(df)
        
        if agreement["bullish"] < self.config.MIN_INDICATORS_AGREE:
            return False, f"指标共振不足 ({agreement['bullish']}/{self.config.MIN_INDICATORS_AGREE})"
        
        if agreement["bearish"] > agreement["bullish"]:
            return False, f"反向信号过多 (空:{agreement['bearish']} vs 多:{agreement['bullish']})"
        
        return True, ""

class TrendFilter(Filter):
    """趋势过滤器"""
    
    def __init__(self, config, logger, trend_analyzer):
        super().__init__(config, logger)
        self.trend_analyzer = trend_analyzer
    
    def _do_filter(self, df: pd.DataFrame, signal: int, symbol: str, strategy_name: str) -> tuple[bool, str]:
        if not self.config.ENABLE_TREND_FILTER or signal != 1:
            return True, ""
        
        if not self.trend_analyzer.is_trend_aligned(df, signal):
            trend = self.trend_analyzer.analyze_trend(df)
            return False, f"趋势不一致 (方向:{trend['direction']}, 强度:{trend['strength']:.2f})"
        
        return True, ""

class VolumeFilter(Filter):
    """成交量过滤器"""
    
    def _do_filter(self, df: pd.DataFrame, signal: int, symbol: str, strategy_name: str) -> tuple[bool, str]:
        if not self.config.VOLUME_FILTER or signal != 1:
            return True, ""
        
        if "volume_ratio" not in df.columns or len(df) == 0:
            return True, ""
        
        volume_ratio = df["volume_ratio"].iloc[-1]
        if volume_ratio < self.config.MIN_VOLUME_RATIO:
            return False, f"成交量不足 ({volume_ratio:.2f}x)"
        
        return True, ""

class VolatilityFilter(Filter):
    """波动率过滤器"""
    
    def _do_filter(self, df: pd.DataFrame, signal: int, symbol: str, strategy_name: str) -> tuple[bool, str]:
        if signal != 1 or "band_width" not in df.columns or len(df) == 0:
            return True, ""
        
        band_width = df["band_width"].iloc[-1]
        if band_width < 0.02:  # 布林带宽度<2%
            return False, f"波动率过低 ({band_width*100:.1f}%)"
        
        return True, ""

class FilterChain:
    """过滤器链管理器"""
    
    def __init__(self, config, logger, multi_indicator, trend_analyzer):
        self.config = config
        self.logger = logger
        self.chain = self._build_chain(multi_indicator, trend_analyzer)
    
    def _build_chain(self, multi_indicator, trend_analyzer):
        """构建过滤器链"""
        filters = []
        
        if self.config.ENABLE_MULTI_INDICATOR_FILTER:
            filters.append(MultiIndicatorFilter(self.config, self.logger, multi_indicator))
        
        if self.config.ENABLE_TREND_FILTER:
            filters.append(TrendFilter(self.config, self.logger, trend_analyzer))
        
        if self.config.VOLUME_FILTER:
            filters.append(VolumeFilter(self.config, self.logger))
        
        filters.append(VolatilityFilter(self.config, self.logger))
        
        # 链接过滤器
        if filters:
            for i in range(len(filters) - 1):
                filters[i].set_next(filters[i + 1])
            return filters[0]
        
        return None
    
    def filter(self, df: pd.DataFrame, signal: int, symbol: str, strategy_name: str) -> bool:
        """执行过滤链"""
        if self.chain is None:
            return True
        
        if signal == 0 or len(df) < 200:
            return False
        
        passed, reasons = self.chain.filter(df, signal, symbol, strategy_name)
        
        if not passed:
            for reason in reasons:
                self.logger.info(f"[FILTER] {symbol} {strategy_name} 被过滤: {reason}")
            return False
        
        # 记录通过信息
        if self.config.ENABLE_MULTI_INDICATOR_FILTER or self.config.ENABLE_TREND_FILTER:
            # 从链中获取分析器
            multi_indicator = None
            trend_analyzer = None
            current = self.chain
            while current:
                if isinstance(current, MultiIndicatorFilter):
                    multi_indicator = current.multi_indicator
                elif isinstance(current, TrendFilter):
                    trend_analyzer = current.trend_analyzer
                current = current.next_filter
            
            agreement = multi_indicator.check_indicator_agreement(df) if multi_indicator else {}
            trend = trend_analyzer.analyze_trend(df) if trend_analyzer else {}
            self.logger.info(
                f"[PASS] {symbol} {strategy_name} 通过过滤 | "
                f"指标共振:{agreement.get('bullish', 0)} | "
                f"趋势:{trend.get('direction', 'N/A')}({trend.get('strength', 0):.2f})"
            )
        
        return True

