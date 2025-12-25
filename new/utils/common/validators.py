"""
数据验证工具模块
"""
import numpy as np

# ==================== 工具类 ====================
class ValidationUtils:
    """数据验证工具类"""
    
    @staticmethod
    def validate_price(price: float, name: str = "price") -> bool:
        """验证价格有效性"""
        if price is None or not isinstance(price, (int, float)):
            return False
        return price > 0 and np.isfinite(price)
    
    @staticmethod
    def validate_amount(amount: float, name: str = "amount") -> bool:
        """验证数量有效性"""
        if amount is None or not isinstance(amount, (int, float)):
            return False
        return amount > 0 and np.isfinite(amount)
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """安全除法"""
        if denominator == 0 or not np.isfinite(denominator):
            return default
        result = numerator / denominator
        return result if np.isfinite(result) else default
    
    @staticmethod
    def clip_value(value: float, min_val: float, max_val: float) -> float:
        """限制数值范围"""
        if not np.isfinite(value):
            return min_val
        return max(min_val, min(max_val, value))