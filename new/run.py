from utils import Config

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置参数
    config = Config(
        API_KEY="GoEdbkGG3QELC5hD8aaX9rn9aA6XQA5Ss1C8I9lIkqc466yupg1sbYJJpFa4umCi",
        API_SECRET="ZbCqTI13N85IMwoRF6CC2fULLBW3EytXZnBoAgJ6PTCo8rM06sFIkVFoaKFyt7Fj",
        TEST_MODE=True,
        MIN_TRADE_AMOUNT=11.0,
        POSITION_SIZE=100.0,
        MAX_POSITIONS=5,
        TOP_PAIRS_COUNT=20,
        # 选择要使用的策略
        STRATEGIES=[
            "RSI_STRATEGY",  # RSI超买超卖
            # "MACD_STRATEGY",  # MACD趋势
            "MA_CROSS_STRATEGY",  # 均线交叉 ⭐推荐
            # "BOLLINGER_STRATEGY",  # 布林带
            # "GRID_STRATEGY",  # 网格交易
            "BREAKOUT_STRATEGY",  # 突破策略
            "MEAN_REVERSION_STRATEGY",  # 均值回归
            # "EMA_STRATEGY",  # EMA快速反应
            # "MOMENTUM_STRATEGY",  # 动量策略
        ],
        # 策略权重 (可选)
        STRATEGY_WEIGHTS={
            "MA_CROSS_STRATEGY": 1.5,  # 主力
            "RSI_STRATEGY": 1.0,
            "BREAKOUT_STRATEGY": 1.2,
            "MEAN_REVERSION_STRATEGY": 0.8,
        },
        # 信号过滤配置 ⭐新增
        ENABLE_MULTI_INDICATOR_FILTER=True,  # 启用多指标共振
        MIN_INDICATORS_AGREE=2,  # 至少2个指标同意
        ENABLE_TREND_FILTER=True,  # 启用趋势过滤
        TREND_STRENGTH_THRESHOLD=0.3,  # 趋势强度阈值
        VOLUME_FILTER=False,  # 成交量过滤
        MIN_VOLUME_RATIO=1.2,  # 最小成交量比率
        # 机器学习配置 ⭐新增
        ENABLE_ML_OPTIMIZATION=False,  # 启用ML优化
        ML_TRAINING_DAYS=60,  # 训练数据天数
        ML_RETRAIN_HOURS=24,  # 每24小时重新训练
        ML_MIN_SAMPLES=100,  # 最少样本数
        ML_CONFIDENCE_THRESHOLD=0.5,  # 置信度阈值60%
        BLACKLIST_PAIRS=["USDC", "FDUSD", "TUSD"],  # 稳定币黑名单
        SCAN_INTERVAL=60,
        MAX_WORKERS=5,
    )

    # 风险声明
    # print("\n" + "=" * 60)
    # print("[WARNING] 重要风险提示")
    # print("=" * 60)
    # print("1. 此代码仅供学习研究,不构成投资建议")
    # print("2. 加密货币交易风险极高,可能导致全部本金损失")
    # print("3. 请务必先在测试网测试,确保理解所有功能")
    # print("4. 任何策略都无法保证盈利")
    # print("5. 请根据自身风险承受能力谨慎投资")
    # print("6. ML模型需要足够的历史数据,初次运行可能需要较长时间")
    # if not ML_AVAILABLE:
    #     print("7. [警告] scikit-learn未安装，ML优化功能已禁用")
    # print("=" * 60)

    # confirm = input("\n我已知晓并同意承担所有风险 (输入 YES 继续): ")
    # if confirm != "YES":
    #     print("已取消运行")
    #     exit()

    # 创建并运行交易引擎
    engine = TradingEngine(config)
    engine.run()
