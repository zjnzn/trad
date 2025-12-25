"""
仓位计算工具类
"""
from typing import Dict, Optional
from utils.common import ValidationUtils

class PositionCalculator:
    """仓位计算工具类 - 提取重复的仓位计算逻辑"""
    
    @staticmethod
    def calculate_entry_position(
        balance: float,
        price: float,
        config,
        risk_manager
    ) -> Dict[str, Optional[float]]:
        """
        计算入场仓位
        返回: {
            'amount': 数量,
            'position_value': 仓位价值,
            'fee_estimate': 手续费估算,
            'error': 错误信息（如果有）
        }
        """
        result = {
            'amount': None,
            'position_value': None,
            'fee_estimate': None,
            'error': None
        }
        
        # 验证价格
        if not ValidationUtils.validate_price(price):
            result['error'] = "无效价格"
            return result
        
        if balance <= 0:
            result['error'] = "余额不足"
            return result
        
        # 计算仓位
        amount = risk_manager.calculate_position_size(balance, price)
        if amount == 0:
            result['error'] = "仓位计算为0"
            return result
        
        position_value = amount * price
        
        # 检查最小金额
        if position_value < config.MIN_TRADE_AMOUNT:
            result['error'] = f"金额不足: {position_value:.2f} < {config.MIN_TRADE_AMOUNT}"
            return result
        
        # 估算手续费
        fee_estimate = position_value * 0.002  # 0.2%
        
        # 检查余额充足
        if position_value + fee_estimate > balance:
            result['error'] = "余额不足以支付手续费"
            return result
        
        result['amount'] = amount
        result['position_value'] = position_value
        result['fee_estimate'] = fee_estimate
        return result
    
    @staticmethod
    def calculate_pnl(
        entry_price: float,
        entry_amount: float,
        exit_price: float,
        exit_amount: float,
        fee: float
    ) -> Dict[str, float]:
        """
        计算盈亏
        返回: {
            'entry_value': 入场价值,
            'exit_value': 出场价值,
            'pnl': 净盈亏,
            'pnl_pct': 盈亏百分比
        }
        """
        entry_value = entry_price * entry_amount
        exit_value = exit_price * exit_amount
        pnl = exit_value - entry_value - fee
        pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
        
        return {
            'entry_value': entry_value,
            'exit_value': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        }

