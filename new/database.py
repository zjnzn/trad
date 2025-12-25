
# ==================== 数据库管理 ====================
class Database:
    """优化的线程安全数据库管理"""

    def __init__(self, config: Config):
        self.config = config
        self.db_path = config.DB_PATH
        self._connections = {}
        self._lock = threading.Lock()
        self.create_tables()

    def get_connection(self):
        """获取线程安全的数据库连接"""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id not in self._connections:
                self._connections[thread_id] = sqlite3.connect(
                    self.db_path, check_same_thread=False, timeout=30.0  # 增加超时时间
                )
                self._connections[thread_id].execute(
                    "PRAGMA journal_mode=WAL"
                )  # 提升并发性能
            return self._connections[thread_id]

    def create_tables(self):
        """创建数据表 - 添加索引优化查询"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 持仓表
        cursor.execute(
            """
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
        """
        )

        # 添加索引
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol, strategy, status)"
        )

        # 交易记录表
        cursor.execute(
            """
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
        """
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)"
        )

        # 账户快照表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                balance REAL NOT NULL,
                equity REAL NOT NULL,
                daily_pnl REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # 策略表现表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                win_rate REAL,
                avg_pnl REAL,
                total_trades INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def save_position(
        self,
        symbol: str,
        strategy: str,
        side: str,
        entry_price: float,
        amount: float,
        stop_loss_price: float = None,
    ):
        """保存持仓"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO positions (symbol, strategy, side, entry_price, amount, highest_price, stop_loss_price)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (symbol, strategy, side, entry_price, amount, entry_price, stop_loss_price),
        )
        conn.commit()
        return cursor.lastrowid

    def update_position(self, position_id: int, exit_price: float, pnl: float):
        """更新持仓"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE positions 
            SET exit_price = ?, exit_time = CURRENT_TIMESTAMP, 
                pnl = ?, status = 'CLOSED'
            WHERE id = ?
        """,
            (exit_price, pnl, position_id),
        )
        conn.commit()

    def update_highest_price(self, position_id: int, highest_price: float):
        """更新最高价(用于移动止盈)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE positions SET highest_price = ? WHERE id = ?
        """,
            (highest_price, position_id),
        )
        conn.commit()

    def get_open_positions(self) -> List[Dict]:
        """获取开仓持仓 - 优化查询"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM positions WHERE status = 'OPEN' ORDER BY entry_time DESC
        """
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def has_open_position(self, symbol: str, strategy: str) -> bool:
        """检查是否已有开仓 - 使用索引优化"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 1 FROM positions 
            WHERE symbol = ? AND strategy = ? AND status = 'OPEN' LIMIT 1
        """,
            (symbol, strategy),
        )
        return cursor.fetchone() is not None

    def save_trade(
        self,
        symbol: str,
        strategy: str,
        side: str,
        price: float,
        amount: float,
        fee: float,
    ):
        """保存交易记录"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO trades (symbol, strategy, side, price, amount, fee)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (symbol, strategy, side, price, amount, fee),
        )
        conn.commit()

    def save_account_snapshot(self, balance: float, equity: float, daily_pnl: float):
        """保存账户快照"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO account_snapshots (balance, equity, daily_pnl)
            VALUES (?, ?, ?)
        """,
            (balance, equity, daily_pnl),
        )
        conn.commit()

    def get_daily_pnl(self) -> float:
        """获取当日盈亏 - 优化查询"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COALESCE(SUM(pnl), 0) FROM positions 
            WHERE DATE(exit_time) = DATE('now') AND status = 'CLOSED'
        """
        )
        return cursor.fetchone()[0]

    def get_strategy_stats(self, strategy: str, days: int = 7) -> Dict:
        """获取策略统计"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
                AVG(pnl) as avg_pnl,
                SUM(pnl) as total_pnl
            FROM positions 
            WHERE strategy = ? AND status = 'CLOSED' 
            AND exit_time >= datetime('now', '-' || ? || ' days')
        """,
            (strategy, days),
        )
        row = cursor.fetchone()
        return {
            "total_trades": row[0] or 0,
            "win_rate": row[1] or 0,
            "avg_pnl": row[2] or 0,
            "total_pnl": row[3] or 0,
        }

    def cleanup_old_data(self, days: int = 90):
        """清理旧数据"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM trades WHERE timestamp < datetime('now', '-' || ? || ' days')
        """,
            (days,),
        )
        cursor.execute(
            """
            DELETE FROM account_snapshots WHERE timestamp < datetime('now', '-' || ? || ' days')
        """,
            (days,),
        )
        conn.commit()