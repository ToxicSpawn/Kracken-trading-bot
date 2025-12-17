"""Main event loop."""
import time
import signal
import sys
import argparse
from .strategy import place_dca
from .logger import log
from .metrics import start_metrics
from .config import settings

running = True

def _exit(signum, frame):
    """Handle shutdown signals."""
    global running
    running = False
    log.info("shutdown signal received")

signal.signal(signal.SIGINT, _exit)
signal.signal(signal.SIGTERM, _exit)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Kraken DCA Bot")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    args = parser.parse_args()
    
    # Override dry_run if specified via command line
    if args.dry_run:
        settings.dry_run = True
    
    log.info("kraken-bot starting (dry_run=%s)", settings.dry_run)
    start_metrics()
    
    while running:
        try:
            place_dca()
        except Exception as e:
            log.exception("loop error")
        time.sleep(settings.check_interval)
    
    log.info("graceful exit")
    sys.exit(0)

if __name__ == "__main__":
    main()

