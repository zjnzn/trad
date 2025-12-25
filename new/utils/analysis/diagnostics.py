
"""
系统诊断模块
"""
from typing import Dict

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
