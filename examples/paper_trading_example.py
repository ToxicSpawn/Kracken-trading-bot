"""Example: Using paper trading simulator."""

import asyncio
from utils.paper_trading import PaperTradingSimulator


async def run_paper_trading_example():
    """Run paper trading example."""
    # Create simulator with $10,000 starting balance
    simulator = PaperTradingSimulator(
        initial_balance=10000.0,
        base_currency="USD",
        commission_rate=0.001,  # 0.1%
        slippage_rate=0.0005,  # 0.05%
    )

    print("=== Paper Trading Simulator ===\n")
    print(f"Starting balance: ${simulator.initial_balance:,.2f}\n")

    # Simulate some trades
    current_price = 50000.0  # BTC price

    # Buy 0.1 BTC
    print("1. Placing buy order: 0.1 BTC @ market price")
    order1 = simulator.create_order(
        symbol="BTC/USD",
        side="buy",
        amount=0.1,
        order_type="market",
        current_price=current_price,
    )
    print(f"   Order ID: {order1.order_id}")
    print(f"   Status: {order1.status}")
    print(f"   Filled Price: ${order1.filled_price:.2f}\n")

    # Check balance
    balance = simulator.get_balance("BTC/USD")
    print(f"Balance after buy:")
    print(f"   USD: ${balance.base_balance:.2f}")
    print(f"   BTC: {balance.quote_balance:.6f}\n")

    # Price goes up
    new_price = 52000.0
    print(f"2. Price moves to ${new_price:.2f}")

    # Check PnL
    pnl = simulator.get_pnl("BTC/USD", new_price)
    total_value = simulator.get_total_value("BTC/USD", new_price)
    print(f"   Total Value: ${total_value:.2f}")
    print(f"   PnL: ${pnl:.2f} ({pnl/simulator.initial_balance*100:.2f}%)\n")

    # Sell half
    print("3. Placing sell order: 0.05 BTC @ market price")
    order2 = simulator.create_order(
        symbol="BTC/USD",
        side="sell",
        amount=0.05,
        order_type="market",
        current_price=new_price,
    )
    print(f"   Order ID: {order2.order_id}")
    print(f"   Status: {order2.status}")
    print(f"   Filled Price: ${order2.filled_price:.2f}\n")

    # Final balance
    balance = simulator.get_balance("BTC/USD")
    final_value = simulator.get_total_value("BTC/USD", new_price)
    final_pnl = simulator.get_pnl("BTC/USD", new_price)

    print("=== Final Results ===")
    print(f"USD Balance: ${balance.base_balance:.2f}")
    print(f"BTC Balance: {balance.quote_balance:.6f}")
    print(f"Total Value: ${final_value:.2f}")
    print(f"Total PnL: ${final_pnl:.2f} ({final_pnl/simulator.initial_balance*100:.2f}%)")


if __name__ == "__main__":
    asyncio.run(run_paper_trading_example())

