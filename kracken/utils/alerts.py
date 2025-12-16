"""Telegram/Discord alerts."""
import requests
from typing import Optional, Dict
from utils.config import load_config
from utils.logger import logger

class AlertSystem:
    """Alert system for notifications."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = load_config(config_path)
        self.telegram_token = self.config.get("telegram", {}).get("token", "")
        self.chat_id = self.config.get("telegram", {}).get("chat_id", "")
        self.enabled = self.config.get("telegram", {}).get("enabled", False)
    
    def send_telegram_alert(self, message: str) -> bool:
        """Send Telegram alert."""
        if not self.enabled or not self.telegram_token or not self.chat_id:
            return False
        
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        try:
            response = requests.post(url, data=payload, timeout=5)
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Failed to send Telegram alert: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
            return False
    
    def send_trade_alert(self, strategy_name: str, symbol: str, action: str, 
                        price: float, size: float, pnl: Optional[float] = None) -> bool:
        """Send trade execution alert."""
        if pnl is not None:
            message = f"📊 <b>{strategy_name}</b>\n" \
                      f"🔹 Symbol: {symbol}\n" \
                      f"🔹 Action: {action.upper()}\n" \
                      f"🔹 Price: {price:.8f}\n" \
                      f"🔹 Size: {size:.4f}\n" \
                      f"🔹 PnL: ${pnl:.2f}"
        else:
            message = f"📊 <b>{strategy_name}</b>\n" \
                      f"🔹 Symbol: {symbol}\n" \
                      f"🔹 Action: {action.upper()}\n" \
                      f"🔹 Price: {price:.8f}\n" \
                      f"🔹 Size: {size:.4f}"
        
        return self.send_telegram_alert(message)
    
    def send_error_alert(self, error_message: str) -> bool:
        """Send error alert."""
        message = f"❌ <b>ERROR</b>\n" \
                  f"🔹 {error_message}"
        return self.send_telegram_alert(message)
    
    def send_performance_alert(self, strategy_name: str, metrics: Dict) -> bool:
        """Send performance metrics alert."""
        message = f"📈 <b>Performance Update - {strategy_name}</b>\n" \
                  f"🔹 Return: {metrics.get('Return [%]', 0):.2f}%\n" \
                  f"🔹 Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}\n" \
                  f"🔹 Max Drawdown: {metrics.get('Max Drawdown [%]', 0):.2f}%\n" \
                  f"🔹 Win Rate: {metrics.get('Win Rate [%]', 0):.2f}%\n" \
                  f"🔹 Profit Factor: {metrics.get('Profit Factor', 0):.2f}"
        
        return self.send_telegram_alert(message)

def send_telegram_alert(message: str) -> bool:
    """Convenience function to send Telegram alert."""
    alert_system = AlertSystem()
    return alert_system.send_telegram_alert(message)

