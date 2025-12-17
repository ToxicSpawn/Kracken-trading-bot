# Quick Start Guide

## 1. Install & Run Dry-Run

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp env.example .env
# Edit .env with your settings (set DRY_RUN=true for testing)

# Run in dry-run mode
python -m src.main --dry-run
```

## 2. Run Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest -q
```

## 3. Docker Full Stack

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f bot

# Access services
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Bot Metrics: http://localhost:8000/metrics
```

## Configuration

1. Copy `env.example` to `.env`
2. Add your Kraken API credentials:
   - `KRAKEN_API_KEY`: Your API key
   - `KRAKEN_PRIVATE_KEY`: Your private key (base64 encoded)
3. Configure trading parameters:
   - `QUOTE_AMOUNT`: Amount per order
   - `SPREAD_PCT`: Spread for limit orders
   - `MAX_OPEN_ORDERS`: Maximum open positions
4. Set `DRY_RUN=true` for testing

## What You Get

✅ **pytest passes** - Full test coverage  
✅ **pre-commit run --all passes** - Code quality checks  
✅ **docker-compose up** - Full stack with Prometheus & Grafana  
✅ **Production-ready** - No secrets in code, proper error handling  
✅ **MIT License** - Ready for GitHub

