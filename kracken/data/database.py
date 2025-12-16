"""PostgreSQL integration."""
import psycopg2
from psycopg2 import sql
from typing import Dict, List, Optional
from utils.config import load_config
from utils.logger import logger

class DatabaseManager:
    """Database manager for PostgreSQL."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = load_config(config_path)
        self.connection = None
        self.connect()
    
    def connect(self):
        """Connect to PostgreSQL database."""
        try:
            db_config = self.config.get("database", {})
            self.connection = psycopg2.connect(
                host=db_config.get("host", "localhost"),
                database=db_config.get("name", "kracken"),
                user=db_config.get("user", "kracken"),
                password=db_config.get("password", ""),
                port=db_config.get("port", 5432)
            )
            logger.info("✅ Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            self.connection = None
    
    def disconnect(self):
        """Disconnect from database."""
        if self.connection:
            self.connection.close()
            logger.info("🔌 Disconnected from PostgreSQL database")
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        """Execute a SQL query."""
        if not self.connection:
            self.connect()
            if not self.connection:
                return None
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                self.connection.commit()
                return True
        except Exception as e:
            logger.error(f"⚠️ Error executing query: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def create_tables(self):
        """Create necessary tables if they don't exist."""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                strategy_name VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                action VARCHAR(10) NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                entry_price DECIMAL(20, 8) NOT NULL,
                exit_price DECIMAL(20, 8),
                size DECIMAL(20, 8) NOT NULL,
                pnl DECIMAL(20, 2),
                pnl_pct DECIMAL(10, 2),
                balance DECIMAL(20, 2) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS performance (
                id SERIAL PRIMARY KEY,
                strategy_name VARCHAR(100) NOT NULL,
                period_start TIMESTAMP NOT NULL,
                period_end TIMESTAMP NOT NULL,
                total_trades INTEGER NOT NULL,
                return_pct DECIMAL(10, 2) NOT NULL,
                sharpe_ratio DECIMAL(10, 2) NOT NULL,
                max_drawdown_pct DECIMAL(10, 2) NOT NULL,
                win_rate_pct DECIMAL(10, 2) NOT NULL,
                profit_factor DECIMAL(10, 2) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS signals (
                id SERIAL PRIMARY KEY,
                strategy_name VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                action VARCHAR(10) NOT NULL,
                price DECIMAL(20, 8) NOT NULL,
                confidence DECIMAL(10, 2) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for query in queries:
            self.execute_query(query)
        logger.info("✅ Database tables created/verified")
    
    def log_trade(self, trade: Dict) -> bool:
        """Log a trade to the database."""
        query = """
        INSERT INTO trades (
            strategy_name, symbol, action, entry_time, exit_time,
            entry_price, exit_price, size, pnl, pnl_pct, balance
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            trade.get("strategy_name"),
            trade.get("symbol"),
            trade.get("action"),
            trade.get("entry_time"),
            trade.get("exit_time"),
            trade.get("entry_price"),
            trade.get("exit_price"),
            trade.get("size"),
            trade.get("pnl"),
            trade.get("pnl_pct"),
            trade.get("balance")
        )
        return self.execute_query(query, params) is not None
    
    def log_performance(self, strategy_name: str, period_start, period_end: str, metrics: Dict) -> bool:
        """Log performance metrics to the database."""
        query = """
        INSERT INTO performance (
            strategy_name, period_start, period_end, total_trades,
            return_pct, sharpe_ratio, max_drawdown_pct, win_rate_pct, profit_factor
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            strategy_name,
            period_start,
            period_end,
            metrics.get("Total Trades", 0),
            metrics.get("Return [%]", 0),
            metrics.get("Sharpe Ratio", 0),
            metrics.get("Max Drawdown [%]", 0),
            metrics.get("Win Rate [%]", 0),
            metrics.get("Profit Factor", 0)
        )
        return self.execute_query(query, params) is not None
    
    def log_signal(self, strategy_name: str, symbol: str, action: str, 
                  price: float, confidence: float, timestamp) -> bool:
        """Log a trading signal to the database."""
        query = """
        INSERT INTO signals (
            strategy_name, symbol, action, price, confidence, timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """
        params = (
            strategy_name,
            symbol,
            action,
            price,
            confidence,
            timestamp
        )
        return self.execute_query(query, params) is not None
    
    def get_trades(self, strategy_name: str = None, limit: int = 100) -> List[Dict]:
        """Get trades from the database."""
        query = "SELECT * FROM trades"
        params = ()
        
        if strategy_name:
            query += " WHERE strategy_name = %s"
            params = (strategy_name,)
        
        query += " ORDER BY entry_time DESC LIMIT %s"
        params = params + (limit,)
        
        results = self.execute_query(query, params, fetch=True)
        if not results:
            return []
        
        columns = [
            "id", "strategy_name", "symbol", "action", "entry_time", "exit_time",
            "entry_price", "exit_price", "size", "pnl", "pnl_pct", "balance", "created_at"
        ]
        return [dict(zip(columns, row)) for row in results]
    
    def get_performance(self, strategy_name: str = None, limit: int = 10) -> List[Dict]:
        """Get performance metrics from the database."""
        query = "SELECT * FROM performance"
        params = ()
        
        if strategy_name:
            query += " WHERE strategy_name = %s"
            params = (strategy_name,)
        
        query += " ORDER BY period_end DESC LIMIT %s"
        params = params + (limit,)
        
        results = self.execute_query(query, params, fetch=True)
        if not results:
            return []
        
        columns = [
            "id", "strategy_name", "period_start", "period_end", "total_trades",
            "return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct",
            "profit_factor", "created_at"
        ]
        return [dict(zip(columns, row)) for row in results]

