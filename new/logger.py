
# ==================== 日志系统 ====================
class Logger:
    """优化的日志管理器"""

    def __init__(self, config: Config):
        self.config = config
        self.setup_logger()
        self._log_buffer = []
        self._buffer_lock = threading.Lock()

    def setup_logger(self):
        """配置日志 - 优化编码处理"""
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # 文件处理器
        file_handler = logging.FileHandler(
            f'trading_{datetime.now().strftime("%Y%m%d")}.log', encoding="utf-8"
        )
        file_handler.setFormatter(formatter)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Windows编码设置
        if sys.platform == "win32":
            import io

            try:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
            except:
                pass

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.LOG_LEVEL))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def critical(self, msg: str):
        """记录严重错误"""
        self.logger.critical(msg)

    def trade(
        self, action: str, symbol: str, price: float, amount: float, strategy: str
    ):
        """交易日志 - 修复格式化问题"""
        price_str = f"{price:.8f}" if price else "MARKET"
        msg = f"[TRADE] {action} {symbol} | Price: {price_str} | Amount: {amount:.8f} | Strategy: {strategy}"
        self.logger.info(msg)

