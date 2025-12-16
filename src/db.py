"""SQLite trade ledger."""
import sqlite3
import datetime as dt
from pathlib import Path
from .config import settings

# Ensure data directory exists
Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(settings.database_path, check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS orders(
    txid TEXT PRIMARY KEY,
    side TEXT,
    vol REAL,
    price REAL,
    ts TEXT
)""")
conn.commit()

def insert_order(txid, side, vol, price):
    """Insert order into database."""
    conn.execute(
        "INSERT INTO orders VALUES(?,?,?,?,?)",
        (txid, side, vol, price, dt.datetime.utcnow().isoformat())
    )
    conn.commit()

def open_positions_count():
    """Count open buy positions."""
    cur = conn.execute("SELECT COUNT(*) FROM orders WHERE side='buy'")
    return cur.fetchone()[0]

def get_all_orders():
    """Get all orders."""
    cur = conn.execute("SELECT * FROM orders ORDER BY ts DESC")
    return cur.fetchall()

