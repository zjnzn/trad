"""
缓存管理模块
"""
import time
import threading
from typing import Optional, Any

class CacheManager:
    """统一的缓存管理器"""
    
    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.Lock()
        self.default_ttl = default_ttl
    
    def get(self, key: str, ttl: int = None) -> Optional[Any]:
        """获取缓存"""
        ttl = ttl or self.default_ttl
        with self.lock:
            if key not in self.cache:
                return None
            
            if time.time() - self.timestamps.get(key, 0) > ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            return self.cache[key]
    
    def set(self, key: str, value: Any):
        """设置缓存"""
        with self.lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()