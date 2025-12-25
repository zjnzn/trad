"""
重试装饰器模块
"""
import time
from functools import wraps

class RetryDecorator:
    """重试装饰器"""
    
    @staticmethod
    def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """
        重试装饰器
        :param max_attempts: 最大尝试次数
        :param delay: 初始延迟
        :param backoff: 延迟倍数
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                attempt = 0
                current_delay = delay
                
                while attempt < max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        attempt += 1
                        if attempt >= max_attempts:
                            raise
                        time.sleep(current_delay)
                        current_delay *= backoff
                
            return wrapper
        return decorator