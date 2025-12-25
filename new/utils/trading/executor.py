"""
交易执行器模块 - 模板方法模式
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import pandas as pd

# ==================== 交易执行器基类（模板方法模式）====================

class TradeExecutor(ABC):
    """交易执行器基类 - 使用模板方法模式统一交易流程"""
    
    def __init__(self, api, db, risk_manager, logger, config):
        self.api = api
        self.db = db
        self.risk_manager = risk_manager
        self.logger = logger
        self.config = config
    
    def execute_entry(self, symbol: str, strategy, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        执行入场交易 - 模板方法
        返回: (是否成功, 错误信息)
        """
        # 1. 前置检查
        if not self._pre_entry_checks(symbol, strategy):
            return False, "前置检查失败"
        
        # 2. 计算仓位
        balance = self.api.get_balance()
        current_price = self.api.get_ticker_price(symbol)
        if not current_price or current_price <= 0:
            return False, "无效价格"
        
        amount = self.risk_manager.calculate_position_size(balance["free"], current_price)
        if amount == 0:
            return False, "仓位计算为0"
        
        # 3. 验证金额
        position_value = amount * current_price
        if position_value < self.config.MIN_TRADE_AMOUNT:
            return False, f"金额不足: {position_value:.2f} < {self.config.MIN_TRADE_AMOUNT}"
        
        # 4. 检查余额（含手续费）
        fee_estimate = position_value * 0.002
        if position_value + fee_estimate > balance["free"]:
            return False, "余额不足以支付手续费"
        
        # 5. 执行交易
        return self._execute_order(symbol, strategy, amount, current_price)
    
    def execute_exit(self, position: Dict, exit_price: float, reason: str) -> Tuple[bool, Optional[str]]:
        """
        执行出场交易 - 模板方法
        返回: (是否成功, 错误信息)
        """
        # 1. 验证数据
        if not self._validate_exit_data(position, exit_price):
            return False, "无效的平仓参数"
        
        # 2. 执行订单
        symbol = position["symbol"]
        amount = position["amount"]
        order = self.api.create_order(symbol, "sell", amount, strategy=position["strategy"])
        
        if not order or "id" not in order:
            return False, "订单创建失败"
        
        # 3. 计算盈亏
        actual_exit_price = order.get("price", exit_price)
        actual_amount = order.get("filled", amount)
        
        entry_value = position["entry_price"] * position["amount"]
        exit_value = actual_exit_price * actual_amount
        fee = exit_value * self.api.get_trading_fee(symbol)
        pnl = exit_value - entry_value - fee
        
        # 4. 更新数据库
        self.db.update_position(position["id"], actual_exit_price, pnl)
        self.db.save_trade(
            symbol, position["strategy"], "sell",
            actual_exit_price, actual_amount, fee
        )
        
        return True, None
    
    def _pre_entry_checks(self, symbol: str, strategy) -> bool:
        """前置检查 - 可被子类重写"""
        # 检查是否已有持仓
        if self.db.has_open_position(symbol, strategy.name):
            return False
        
        # 检查持仓限制
        if len(self.db.get_open_positions()) >= self.config.MAX_POSITIONS:
            self.logger.warning(f"[SKIP] 已达持仓上限: {symbol}")
            return False
        
        # 检查余额
        balance = self.api.get_balance()
        if balance["free"] < self.config.MIN_TRADE_AMOUNT:
            self.logger.warning(f"[SKIP] 余额不足: {balance['free']:.2f} USDT")
            return False
        
        return True
    
    def _validate_exit_data(self, position: Dict, exit_price: float) -> bool:
        """验证出场数据"""
        amount = position.get("amount", 0)
        return amount > 0 and exit_price > 0
    
    @abstractmethod
    def _execute_order(self, symbol: str, strategy, amount: float, price: float) -> Tuple[bool, Optional[str]]:
        """执行订单 - 子类实现"""
        pass

