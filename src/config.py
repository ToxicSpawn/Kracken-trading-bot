"""100% env-driven configuration, no secrets in code."""
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings from environment variables."""
    kraken_api_key: str
    kraken_private_key: str
    kraken_pair: str = "XBTUSD"
    order_type: str = "limit"
    spread_pct: float = 0.1
    quote_amount: float
    max_open_orders: int = 3
    check_interval: int = 60
    database_path: str = "data/bot.sqlite"
    metrics_port: int = 8000
    log_level: str = "INFO"
    dry_run: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

