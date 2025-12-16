"""Configuration management."""
import json
import os
from typing import Dict, Any
from utils.logger import logger

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                logger.info(f"✅ Loaded configuration from {config_path}")
                return config
        else:
            logger.warning(f"⚠️ Config file {config_path} not found, using defaults")
            return get_default_config()
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        "exchanges": {},
        "strategies": {
            "ml": {"enabled": False},
            "arbitrage": {"enabled": False},
            "quantum": {"enabled": False}
        },
        "risk": {
            "position_sizing": "kelly",
            "kelly": {
                "win_rate": 0.55,
                "risk_reward_ratio": 2,
                "max_position_pct": 0.02
            }
        },
        "backtesting": {
            "initial_balance": 10000,
            "commission": 0.001,
            "slippage": 0.0005
        },
        "database": {
            "host": "localhost",
            "name": "kracken",
            "user": "kracken",
            "password": "",
            "port": 5432
        },
        "telegram": {
            "enabled": False,
            "token": "",
            "chat_id": ""
        }
    }

