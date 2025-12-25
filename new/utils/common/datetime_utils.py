from datetime import datetime, timezone

class DateTimeUtils:
    """时间处理工具类 - 统一时区处理"""
    
    @staticmethod
    def now_utc() -> datetime:
        """获取当前UTC时间"""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def parse_datetime(dt_str: str) -> datetime:
        """解析时间字符串，统一转为UTC"""
        if not dt_str:
            return DateTimeUtils.now_utc()
        
        # 处理SQLite的时间格式
        dt_str = dt_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(dt_str)
        
        # 确保有时区信息
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        return dt
    
    @staticmethod
    def calculate_duration(start: datetime, end: datetime = None) -> float:
        """计算时间差（小时）"""
        if end is None:
            end = DateTimeUtils.now_utc()
        
        # 确保都有时区
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        
        return (end - start).total_seconds() / 3600
    
    @staticmethod
    def format_duration(hours: float) -> str:
        """格式化持续时间"""
        if hours < 1:
            return f"{hours*60:.0f}m"
        elif hours < 24:
            return f"{hours:.1f}h"
        else:
            return f"{hours/24:.1f}d"
