# ==================== 统一异常处理 =====================
import traceback
from typing import Callable, Any, Optional
from functools import wraps

class TradingException(Exception):
    """交易系统自定义异常"""
    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        super().__init__(message)
        self.error_code = error_code
        self.original_error = original_error

class ExceptionHandler:
    """统一异常处理工具类"""
    
    @staticmethod
    def safe_execute(
        func: Callable,
        logger=None,
        default_return: Any = None,
        error_message: str = None,
        reraise: bool = False
    ) -> Any:
        """
        安全执行函数，统一异常处理
        """
        try:
            return func()
        except TradingException as e:
            if logger:
                logger.error(f"[TRADING_ERROR] {e.error_code}: {e}")
            if reraise:
                raise
            return default_return
        except Exception as e:
            error_msg = error_message or f"执行失败: {e}"
            if logger:
                logger.error(f"[ERROR] {error_msg}")
                logger.error(f"详情: {traceback.format_exc()}")
            if reraise:
                raise
            return default_return
    
    @staticmethod
    def handle_api_error(func: Callable, logger=None, retry_count: int = 0):
        """处理API错误"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if logger:
                    logger.error(f"[API_ERROR] {func.__name__}: {e}")
                if retry_count > 0:
                    # 可以在这里实现重试逻辑
                    pass
                raise TradingException(
                    f"API调用失败: {func.__name__}",
                    error_code="API_ERROR",
                    original_error=e
                )
        return wrapper
    
    @staticmethod
    def handle_database_error(func: Callable, logger=None):
        """处理数据库错误"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if logger:
                    logger.error(f"[DB_ERROR] {func.__name__}: {e}")
                raise TradingException(
                    f"数据库操作失败: {func.__name__}",
                    error_code="DB_ERROR",
                    original_error=e
                )
        return wrapper

