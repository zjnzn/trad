


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
