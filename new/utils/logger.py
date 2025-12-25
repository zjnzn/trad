# ==================== 日志系统（优化编码处理）====================
from datetime import datetime
import logging
import sys

from .config import Config
from .common.validators import ValidationUtils

class Logger:
    """优化的日志管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_logger()
    
    def setup_logger(self):
        """配置日志 - 优化Windows编码"""
        # Windows编码设置
        if sys.platform == "win32":
            import io
            try:
                if hasattr(sys.stdout, 'buffer'):
                    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
                if hasattr(sys.stderr, 'buffer'):
                    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
            except Exception:
                pass
        
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(
            f'trading_{datetime.now().strftime("%Y%m%d")}.log',
            encoding='utf-8',
            errors='replace'
        )
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.LOG_LEVEL))
        self.logger.handlers.clear()  # 清除已有处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, msg: str):
        try:
            self.logger.info(msg)
        except Exception:
            self.logger.info(msg.encode('utf-8', errors='replace').decode('utf-8'))
    
    def warning(self, msg: str):
        try:
            self.logger.warning(msg)
        except Exception:
            self.logger.warning(msg.encode('utf-8', errors='replace').decode('utf-8'))
    
    def error(self, msg: str):
        try:
            self.logger.error(msg)
        except Exception:
            self.logger.error(msg.encode('utf-8', errors='replace').decode('utf-8'))
    
    def critical(self, msg: str):
        try:
            self.logger.critical(msg)
        except Exception:
            self.logger.critical(msg.encode('utf-8', errors='replace').decode('utf-8'))
    
    def trade(self, action: str, symbol: str, price: float, amount: float, strategy: str):
        """交易日志"""
        price_str = f"{price:.8f}" if ValidationUtils.validate_price(price) else "MARKET"
        msg = f"[TRADE] {action} {symbol} | Price: {price_str} | Amount: {amount:.8f} | Strategy: {strategy}"
        self.info(msg)