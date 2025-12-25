
# ==================== 机器学习优化器 ====================
class MLOptimizer:
    """机器学习参数优化和信号预测"""

    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.models = {}  # 每个策略一个模型
        self.scalers = {}
        self.last_train_time = {}
        self.model_path = "ml_models"

        import os

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """从K线数据提取特征 - 添加异常处理"""
        if len(df) < 50:
            return np.array([])

        features = []

        try:
            # 价格特征
            close_20 = df["close"].iloc[-20] if len(df) >= 20 else df["close"].iloc[0]
            close_5 = df["close"].iloc[-5] if len(df) >= 5 else df["close"].iloc[0]

            features.append(df["close"].iloc[-1] / close_20 - 1 if close_20 > 0 else 0)
            features.append(df["close"].iloc[-1] / close_5 - 1 if close_5 > 0 else 0)
            features.append(
                df["high"].iloc[-1] / df["low"].iloc[-1] - 1
                if df["low"].iloc[-1] > 0
                else 0
            )

            # 技术指标特征
            # RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            features.append(min(rsi.iloc[-1] / 100, 1.0))

            # MACD
            exp1 = df["close"].ewm(span=12, adjust=False).mean()
            exp2 = df["close"].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9, adjust=False).mean()
            macd_val = (
                (macd.iloc[-1] - signal_line.iloc[-1]) / df["close"].iloc[-1]
                if df["close"].iloc[-1] > 0
                else 0
            )
            features.append(np.clip(macd_val, -1, 1))

            # 布林带位置
            sma = df["close"].rolling(window=20).mean()
            std = df["close"].rolling(window=20).std()
            bb_position = (df["close"].iloc[-1] - sma.iloc[-1]) / (std.iloc[-1] + 1e-10)
            features.append(np.clip(bb_position, -3, 3))

            # 均线特征
            ma_fast = df["close"].rolling(window=20).mean().iloc[-1]
            ma_slow = (
                df["close"].rolling(window=50).mean().iloc[-1]
                if len(df) >= 50
                else ma_fast
            )
            features.append(df["close"].iloc[-1] / ma_fast - 1 if ma_fast > 0 else 0)
            features.append(ma_fast / ma_slow - 1 if ma_slow > 0 else 0)

            # 成交量特征
            vol_ma = df["volume"].rolling(window=20).mean()
            vol_ratio = df["volume"].iloc[-1] / (vol_ma.iloc[-1] + 1e-10)
            features.append(min(vol_ratio, 10.0))

            # 波动率特征
            returns = df["close"].pct_change()
            volatility = returns.rolling(window=20).std().iloc[-1]
            features.append(min(volatility * 100, 10.0))

            # 动量特征
            close_14 = df["close"].iloc[-14] if len(df) >= 14 else df["close"].iloc[0]
            momentum = (
                (df["close"].iloc[-1] - close_14) / close_14 if close_14 > 0 else 0
            )
            features.append(np.clip(momentum, -1, 1))

            # 趋势特征 (线性回归斜率)
            x = np.arange(min(20, len(df)))
            y = df["close"].tail(min(20, len(df))).values
            if len(y) >= 2:
                slope = (
                    np.polyfit(x, y, 1)[0] / df["close"].iloc[-1]
                    if df["close"].iloc[-1] > 0
                    else 0
                )
                features.append(np.clip(slope, -0.1, 0.1))
            else:
                features.append(0)

            # 确保所有特征都是有限数值
            features = [0 if not np.isfinite(f) else f for f in features]

            return np.array(features)

        except Exception as e:
            self.logger.error(f"[ML] 特征提取失败: {e}")
            return np.array([])

    def prepare_training_data(
        self, api, symbol: str, strategy
    ) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        # 获取历史数据
        df = api.get_ohlcv(symbol, "1h", self.config.ML_TRAINING_DAYS * 24)
        if len(df) < 200:
            return np.array([]), np.array([])

        X_list = []
        y_list = []

        # 滑动窗口生成训练样本
        for i in range(200, len(df) - 24):  # 预测未来24小时
            # 提取特征
            sample_df = df.iloc[: i + 1].copy()
            features = self.extract_features(sample_df)

            if len(features) == 0:
                continue

            # 计算标签 (未来24小时最高收益)
            future_prices = df["close"].iloc[i + 1 : i + 25].values
            current_price = df["close"].iloc[i]
            max_return = (max(future_prices) - current_price) / current_price

            # 二分类: 盈利>2% 为1, 否则为0
            label = 1 if max_return > 0.02 else 0

            X_list.append(features)
            y_list.append(label)

        if len(X_list) < self.config.ML_MIN_SAMPLES:
            return np.array([]), np.array([])

        return np.array(X_list), np.array(y_list)

    def train_model(self, api, symbol: str, strategy):
        """训练ML模型"""
        try:
            self.logger.info(f"[ML] 开始训练模型: {symbol} - {strategy.name}")

            # 准备数据
            X, y = self.prepare_training_data(api, symbol, strategy)

            if len(X) == 0:
                self.logger.warning(f"[ML] 训练数据不足: {symbol}")
                return False

            # 检查正负样本比例
            positive_ratio = np.sum(y) / len(y)
            self.logger.info(f"[ML] 正样本比例: {positive_ratio:.2%}")

            if positive_ratio < 0.1 or positive_ratio > 0.9:
                self.logger.warning(f"[ML] 样本不平衡: {symbol}")
                return False

            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 特征标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 训练随机森林分类器
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train_scaled, y_train)

            # 评估模型
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)

            self.logger.info(
                f"[ML] 模型训练完成: {symbol} | "
                f"训练准确率:{train_score:.2%} | 测试准确率:{test_score:.2%}"
            )

            # 保存模型
            model_key = f"{symbol}_{strategy.name}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.last_train_time[model_key] = time.time()

            # 持久化
            self._save_model(model_key, model, scaler)

            return True

        except Exception as e:
            self.logger.error(f"[ML] 训练失败 {symbol}: {e}")
            return False

    def predict_signal_quality(
        self, df: pd.DataFrame, symbol: str, strategy_name: str
    ) -> float:
        """预测信号质量 - 返回0-1的置信度"""
        model_key = f"{symbol}_{strategy_name}"

        # 检查是否有模型
        if model_key not in self.models:
            # 尝试从磁盘加载
            if self._load_model(model_key):
                self.logger.info(f"[ML] 已加载模型: {model_key}")
            else:
                # 没有模型,返回中性值（允许交易但不增强信心）
                return 0.5

        # 检查模型是否过期
        last_train = self.last_train_time.get(model_key, 0)
        if time.time() - last_train > self.config.ML_RETRAIN_HOURS * 3600:
            self.logger.info(f"[ML] 模型过期: {model_key}")
            return 0.5

        try:
            # 提取特征
            features = self.extract_features(df)
            if len(features) == 0:
                return 0.5

            # 标准化
            scaler = self.scalers[model_key]
            features_scaled = scaler.transform(features.reshape(1, -1))

            # 预测概率
            model = self.models[model_key]
            proba = model.predict_proba(features_scaled)[0]

            # 返回正类概率
            confidence = proba[1]

            return confidence

        except Exception as e:
            self.logger.error(f"[ML] 预测失败 {symbol}: {e}")
            return 0.5

    def optimize_strategy_params(self, api, symbol: str, strategy) -> Dict:
        """使用ML优化策略参数"""
        try:
            self.logger.info(f"[ML] 使用贝叶斯优化参数: {strategy.name}")

            # 获取历史数据
            df = api.get_ohlcv(symbol, "1h", 500)
            if len(df) < 200:
                return strategy.params

            best_params = strategy.params.copy()
            best_score = -999

            # 定义参数搜索空间
            param_space = self._get_param_space(strategy.name)

            # 贝叶斯优化 (简化版 - 使用随机搜索)
            for _ in range(20):
                # 随机采样参数
                test_params = {}
                for param_name, (min_val, max_val) in param_space.items():
                    if isinstance(min_val, int):
                        test_params[param_name] = np.random.randint(
                            min_val, max_val + 1
                        )
                    else:
                        test_params[param_name] = np.random.uniform(min_val, max_val)

                # 测试参数
                strategy.params = test_params
                score = self._evaluate_params(df, strategy)

                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()

            self.logger.info(f"[ML] 最优参数: {best_params} | 分数: {best_score:.2f}")
            return best_params

        except Exception as e:
            self.logger.error(f"[ML] 参数优化失败: {e}")
            return strategy.params

    def _get_param_space(self, strategy_name: str) -> Dict:
        """获取参数搜索空间"""
        spaces = {
            "RSI_STRATEGY": {
                "period": (10, 20),
                "oversold": (20, 35),
                "overbought": (65, 80),
            },
            "MACD_STRATEGY": {"fast": (8, 16), "slow": (20, 30), "signal": (7, 12)},
            "MA_CROSS_STRATEGY": {
                "fast_period": (5, 15),
                "slow_period": (20, 40),
                "trend_filter": (80, 120),
            },
            "BOLLINGER_STRATEGY": {"period": (15, 25), "std_dev": (1.5, 2.5)},
        }
        return spaces.get(strategy_name, {})

    def _evaluate_params(self, df: pd.DataFrame, strategy) -> float:
        """评估参数质量"""
        try:
            df_copy = df.copy()
            df_copy = strategy.calculate_signals(df_copy)

            # 模拟交易
            capital = 1000
            position = None
            trades = []

            for i in range(len(df_copy)):
                if position is None and df_copy["signal"].iloc[i] == 1:
                    position = {
                        "entry_price": df_copy["close"].iloc[i],
                        "amount": capital * 0.95 / df_copy["close"].iloc[i],
                    }
                elif position and df_copy["signal"].iloc[i] == -1:
                    exit_price = df_copy["close"].iloc[i]
                    pnl = (exit_price - position["entry_price"]) * position["amount"]
                    capital += pnl
                    trades.append(pnl)
                    position = None

            if len(trades) < 3:
                return -999

            # 计算Sharpe比率
            returns = np.array(trades) / 1000
            sharpe = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252)

            return sharpe

        except:
            return -999

    def _save_model(self, model_key: str, model, scaler):
        """保存模型到磁盘"""
        try:
            import os

            model_file = os.path.join(self.model_path, f"{model_key}_model.pkl")
            scaler_file = os.path.join(self.model_path, f"{model_key}_scaler.pkl")

            # 确保父目录存在（支持 model_key 包含子目录，例如 "UNI/.."）
            model_dir = os.path.dirname(model_file)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)

            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)

        except Exception as e:
            self.logger.error(f"[ML] 保存模型失败: {e}")

    def _load_model(self, model_key: str) -> bool:
        """从磁盘加载模型"""
        try:
            model_file = f"{self.model_path}/{model_key}_model.pkl"
            scaler_file = f"{self.model_path}/{model_key}_scaler.pkl"

            import os

            if os.path.exists(model_file) and os.path.exists(scaler_file):
                self.models[model_key] = joblib.load(model_file)
                self.scalers[model_key] = joblib.load(scaler_file)
                self.last_train_time[model_key] = time.time()
                return True

            return False

        except Exception as e:
            self.logger.error(f"[ML] 加载模型失败: {e}")
            return False
