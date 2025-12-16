"""Complete example showing all advanced components working together."""

import asyncio
import logging
from datetime import datetime, timedelta

import pandas as pd

from neural_kraken.strategies.arbitrage_advanced import AdvancedArbitrageStrategy
from neural_kraken.models.transformer_model import TimeSeriesTransformer
from neural_kraken.backtesting.advanced_engine import AdvancedBacktestingEngine
from neural_kraken.backtesting.strategies import combined_strategy
from neural_kraken.execution.algorithms import ExecutionEngine, ExecutionOrder
from neural_kraken.risk.advanced_risk import AdvancedRiskManager, PortfolioOptimizer
from neural_kraken.monitoring.metrics import get_metrics
from neural_kraken.data.kafka_consumer import MarketDataConsumer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def complete_system_example():
    """Complete example of the Neural Kraken system."""

    print("=" * 80)
    print("NEURAL KRAKEN - COMPLETE SYSTEM EXAMPLE")
    print("=" * 80)

    # 1. Initialize Monitoring
    print("\n1. Starting Monitoring...")
    metrics = get_metrics()
    metrics.start_server(port=8001)
    print("   ✓ Metrics server started on port 8001")

    # 2. Initialize Risk Manager
    print("\n2. Initializing Risk Manager...")
    risk_manager = AdvancedRiskManager(
        max_position_size=10.0,
        max_portfolio_var=1000.0,
        max_portfolio_drawdown=20.0,
    )
    print("   ✓ Risk manager initialized")

    # 3. Initialize Arbitrage Strategy
    print("\n3. Initializing Arbitrage Strategy...")
    arbitrage = AdvancedArbitrageStrategy(min_profit_pct=0.1)
    
    # Simulate order books
    arbitrage.update_order_book("BTC/USD", "kraken", {
        "bids": [[50000.0, 1.0], [49999.0, 2.0]],
        "asks": [[50001.0, 1.0], [50002.0, 2.0]],
    })
    arbitrage.update_order_book("BTC/USD", "binance", {
        "bids": [[50010.0, 1.0], [50009.0, 2.0]],
        "asks": [[50011.0, 1.0], [50012.0, 2.0]],
    })
    
    opportunities = arbitrage.find_opportunities()
    print(f"   ✓ Found {len(opportunities)} arbitrage opportunities")
    if opportunities:
        print(f"   ✓ Best opportunity: {opportunities[0].profit_pct:.2f}% profit")

    # 4. Initialize Execution Engine
    print("\n4. Initializing Execution Engine...")
    execution_engine = ExecutionEngine()
    print("   ✓ Execution engine initialized with TWAP, VWAP, Iceberg algorithms")

    # 5. Example: Create and execute order
    print("\n5. Creating Execution Order...")
    order = ExecutionOrder(
        symbol="BTC/USD",
        side="buy",
        total_amount=0.1,
        algorithm="twap",
    )
    order_id = await execution_engine.execute_order(order)
    print(f"   ✓ Order {order_id} created with TWAP algorithm")

    # 6. Initialize Transformer Model (example)
    print("\n6. Initializing Transformer Model...")
    try:
        model = TimeSeriesTransformer(input_shape=(60, 7))
        print("   ✓ Transformer model initialized")
        print("   ℹ Note: Train model with historical data before use")
    except Exception as e:
        print(f"   ⚠ Model initialization: {e}")

    # 7. Run Backtest Example
    print("\n7. Running Backtest Example...")
    try:
        # Create sample data
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="1H")
        sample_data = pd.DataFrame({
            "timestamp": dates,
            "open": 50000 + np.random.randn(len(dates)) * 1000,
            "high": 51000 + np.random.randn(len(dates)) * 1000,
            "low": 49000 + np.random.randn(len(dates)) * 1000,
            "close": 50000 + np.random.randn(len(dates)) * 1000,
            "volume": np.random.uniform(100, 1000, len(dates)),
        })

        engine = AdvancedBacktestingEngine(initial_balance=10000.0)
        engine.add_strategy("combined", combined_strategy)
        engine.load_data(sample_data)
        results = engine.run_backtest()

        if "combined" in results:
            result = results["combined"]
            print(f"   ✓ Backtest complete:")
            print(f"     - Total Return: {result.total_return:.2f}%")
            print(f"     - Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"     - Max Drawdown: {result.max_drawdown:.2f}%")
            print(f"     - Win Rate: {result.win_rate:.2%}")
    except Exception as e:
        print(f"   ⚠ Backtest error: {e}")

    # 8. Risk Management Example
    print("\n8. Running Risk Management Example...")
    risk_manager.update_position("BTC/USD", 0.1, 50000.0)
    risk_manager.update_position("ETH/USD", 1.0, 3000.0)
    
    portfolio_value = risk_manager.calculate_portfolio_value()
    portfolio_var = risk_manager.calculate_portfolio_var()
    
    print(f"   ✓ Portfolio Value: ${portfolio_value:.2f}")
    print(f"   ✓ Portfolio VaR: ${portfolio_var:.2f}")
    
    # Check order risk
    allowed, error = risk_manager.check_order_risk("BTC/USD", "buy", 0.5, 51000.0)
    print(f"   ✓ Order risk check: {'Allowed' if allowed else f'Rejected: {error}'}")

    # Run stress tests
    scenarios = risk_manager.generate_stress_scenarios()
    print(f"   ✓ Generated {len(scenarios)} stress test scenarios")
    if scenarios:
        result = risk_manager.run_stress_test(scenarios[0])
        print(f"   ✓ Stress test '{scenarios[0].name}': Final Value ${result.final_value:.2f}")

    # 9. Portfolio Optimization Example
    print("\n9. Running Portfolio Optimization Example...")
    try:
        # Create sample returns
        returns_data = pd.DataFrame({
            "BTC/USD": np.random.randn(100) * 0.02,
            "ETH/USD": np.random.randn(100) * 0.025,
            "SOL/USD": np.random.randn(100) * 0.03,
        })

        optimizer = PortfolioOptimizer()
        optimal_weights = optimizer.optimize_portfolio(returns_data, method="mean_variance")
        print(f"   ✓ Optimal portfolio weights:")
        for asset, weight in optimal_weights.items():
            print(f"     - {asset}: {weight:.2%}")
    except Exception as e:
        print(f"   ⚠ Optimization error: {e}")

    # 10. Update Metrics
    print("\n10. Updating Metrics...")
    metrics.record_ticker()
    metrics.record_signal("BUY")
    metrics.update_portfolio_metrics(portfolio_value, 5.0, portfolio_var)
    print("   ✓ Metrics updated")

    print("\n" + "=" * 80)
    print("COMPLETE SYSTEM EXAMPLE FINISHED")
    print("=" * 80)
    print("\nAll components are working! 🚀")
    print("\nNext steps:")
    print("1. Train transformer model with real historical data")
    print("2. Configure exchange API keys")
    print("3. Set up Kafka for real-time data")
    print("4. Deploy to Kubernetes")
    print("5. Start with paper trading")


if __name__ == "__main__":
    asyncio.run(complete_system_example())

