# Kracken Trading Bot

A professional-grade cryptocurrency trading bot with advanced strategies, risk management, and backtesting capabilities.

## Features

- **Async WebSocket Core**: Real-time market data via WebSocket connections
- **Advanced Strategies**: 
  - Machine Learning (LSTM, Random Forest)
  - Arbitrage (Triangular, Market Making)
  - Quantum-ready optimizations
- **Risk Management**:
  - Kelly Criterion position sizing
  - Dynamic stop-loss with trailing
  - Black swan protection
- **Backtesting**: 
  - Full backtesting engine
  - Monte Carlo simulations
  - Walk-forward optimization
- **Monitoring**: 
  - PostgreSQL database integration
  - Telegram alerts
  - Performance tracking

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the bot:
   - Update `config.json` with your exchange API keys
   - Set up Telegram bot (optional)
   - Configure database settings (optional)

## Usage

### Running the Bot

```bash
python main.py
```

### Docker Deployment

```bash
cd docker
docker-compose up --build
```

## Configuration

Edit `config.json` to:
- Add exchange API keys
- Enable/disable strategies
- Configure risk parameters
- Set up database and alerts

## Project Structure

```
kracken/
├── core/                  # Core trading engine
├── strategies/            # Trading strategies
├── data/                  # Data handling
├── ml/                    # Machine learning
├── risk/                  # Risk management
├── backtesting/           # Backtesting & validation
├── utils/                 # Utilities
└── docker/                # Docker & Kubernetes
```

## License

MIT

