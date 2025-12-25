
"""
回测引擎模块
"""
import numpy as np
from typing import Dict

from api.binance_api import BinanceAPI
from ..logger import Logger
from ..config import Config


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
