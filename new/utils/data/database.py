"""
数据库管理模块
"""
import sqlite3
import threading
from contextlib import contextmanager
from typing import List, Dict

from utils import Config

# ==================== 数据库管理（优化连接池）====================
class Database:
    """优化的线程安全数据库管理"""
    
    def __init__(self, config: Config):
        self.config = config
        self.db_path = config.DB_PATH
        self._local = threading.local()
        self._lock = threading.Lock()
        self.create_tables()
    
    @property
    def connection(self):
        """获取线程本地连接"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
                isolation_level='DEFERRED'  # 优化并发
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")  # 提升性能
        return self._local.conn
    
    @contextmanager
    def transaction(self):
        """事务上下文管理器"""
        conn = self.connection
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def create_tables(self):
        """创建数据表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 持仓表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                amount REAL NOT NULL,
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exit_price REAL,
                exit_time TIMESTAMP,
                pnl REAL,
                status TEXT DEFAULT 'OPEN',
                highest_price REAL,
                stop_loss_price REAL
            )
        """)
        
        # 索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol_strategy ON positions(symbol, strategy, status)")
        
        # 交易记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                fee REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
        
        # 账户快照表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                balance REAL NOT NULL,
                equity REAL NOT NULL,
                daily_pnl REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 策略表现表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                win_rate REAL,
                avg_pnl REAL,
                total_trades INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_position(self, symbol: str, strategy: str, side: str, entry_price: float,
                     amount: float, stop_loss_price: float = None) -> int:
        """保存持仓"""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO positions 
                (symbol, strategy, side, entry_price, amount, highest_price, stop_loss_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, strategy, side, entry_price, amount, entry_price, stop_loss_price))
            return cursor.lastrowid
    
    def update_position(self, position_id: int, exit_price: float, pnl: float):
        """更新持仓"""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE positions 
                SET exit_price = ?, exit_time = CURRENT_TIMESTAMP, 
                    pnl = ?, status = 'CLOSED'
                WHERE id = ?
            """, (exit_price, pnl, position_id))
    
    def update_highest_price(self, position_id: int, highest_price: float):
        """更新最高价"""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE positions SET highest_price = ? WHERE id = ?
            """, (highest_price, position_id))
    
    def get_open_positions(self) -> List[Dict]:
        """获取开仓持仓 - 线程安全"""
        with self._lock:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM positions WHERE status = 'OPEN' ORDER BY entry_time DESC
            """)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def has_open_position(self, symbol: str, strategy: str) -> bool:
        """检查是否已有开仓 - 线程安全"""
        with self._lock:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT 1 FROM positions 
                WHERE symbol = ? AND strategy = ? AND status = 'OPEN' LIMIT 1
            """, (symbol, strategy))
            return cursor.fetchone() is not None
    
    def save_trade(self, symbol: str, strategy: str, side: str, price: float,
                   amount: float, fee: float):
        """保存交易记录"""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (symbol, strategy, side, price, amount, fee)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, strategy, side, price, amount, fee))
    
    def save_account_snapshot(self, balance: float, equity: float, daily_pnl: float):
        """保存账户快照"""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO account_snapshots (balance, equity, daily_pnl)
                VALUES (?, ?, ?)
            """, (balance, equity, daily_pnl))
    
    def get_daily_pnl(self) -> float:
        """获取当日盈亏"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT COALESCE(SUM(pnl), 0) FROM positions 
            WHERE DATE(exit_time) = DATE('now') AND status = 'CLOSED'
        """)
        return cursor.fetchone()[0]
    
    def get_strategy_stats(self, strategy: str, days: int = 7) -> Dict:
        """获取策略统计"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0) as win_rate,
                AVG(pnl) as avg_pnl,
                SUM(pnl) as total_pnl
            FROM positions 
            WHERE strategy = ? AND status = 'CLOSED' 
            AND exit_time >= datetime('now', '-' || ? || ' days')
        """, (strategy, days))
        row = cursor.fetchone()
        return {
            'total_trades': row[0] or 0,
            'win_rate': row[1] or 0,
            'avg_pnl': row[2] or 0,
            'total_pnl': row[3] or 0
        }
    
    def cleanup_old_data(self, days: int = 90):
        """清理旧数据"""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM trades WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            cursor.execute("""
                DELETE FROM account_snapshots WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))