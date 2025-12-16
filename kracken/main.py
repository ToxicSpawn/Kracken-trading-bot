"""Main entry point for the Kracken trading bot."""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine import TradingEngine
from utils.logger import logger

async def main():
    """Main entry point."""
    try:
        logger.info("🚀 Starting Kracken Trading Bot")
        engine = TradingEngine()
        
        # Start the trading engine
        await engine.start()
    except KeyboardInterrupt:
        logger.info("👋 Received keyboard interrupt, shutting down...")
        await engine.stop()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())

