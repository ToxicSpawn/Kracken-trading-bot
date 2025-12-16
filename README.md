# Kraken DCA Bot – Production Edition

A production-grade Kraken DCA (Dollar Cost Averaging) bot with full monitoring, testing, and deployment setup.

## Features

- ✅ **Env-var configuration**, no secrets in repo
- ✅ **Monotonic nonce**, automatic retry, rate-limit safe
- ✅ **Limit or market orders**, configurable spread
- ✅ **SQLite trade ledger**, real-time Prometheus metrics
- ✅ **Grafana dashboard** included
- ✅ **Full dry-run mode**
- ✅ **Pre-commit hooks** + pytest suite
- ✅ **Docker Compose** full stack (bot + Prometheus + Grafana)

## Installation

### Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your Kraken API keys

# Run in dry-run mode
python -m src.main --dry-run
```

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest -q
```

## Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f bot

# Stop services
docker-compose down
```

### Access Services

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Bot Metrics**: http://localhost:8000/metrics

## Configuration

All configuration is done via environment variables (see `.env.example`):

- `KRAKEN_API_KEY`: Your Kraken API key
- `KRAKEN_PRIVATE_KEY`: Your Kraken private key (base64 encoded)
- `KRAKEN_PAIR`: Trading pair (default: XBTUSD)
- `ORDER_TYPE`: Order type - "limit" or "market"
- `SPREAD_PCT`: Spread percentage for limit orders (default: 0.1)
- `QUOTE_AMOUNT`: Amount to invest per order
- `MAX_OPEN_ORDERS`: Maximum number of open buy orders
- `CHECK_INTERVAL`: Interval between checks in seconds
- `DRY_RUN`: Enable dry-run mode (true/false)

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py          # Pydantic settings
│   ├── nonce.py           # Monotonic nonce
│   ├── kraken_api.py      # API wrapper with retry
│   ├── strategy.py        # DCA strategy
│   ├── db.py              # SQLite ledger
│   ├── metrics.py          # Prometheus metrics
│   ├── logger.py           # Structured logging
│   └── main.py            # Entry point
├── tests/
│   ├── test_strategy.py
│   └── test_kraken_api.py
├── data/                  # SQLite database
├── logs/                  # Log files
├── grafana/               # Grafana provisioning
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_strategy.py
```

## Pre-commit Hooks

```bash
# Run manually
pre-commit run --all-files

# Auto-run on commit (after install)
git commit -m "your message"
```

## License

MIT
