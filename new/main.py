
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
