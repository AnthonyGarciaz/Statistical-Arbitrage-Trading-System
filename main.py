import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import coint
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


class StatArbSystem:
    def __init__(self, lookback_period: int = 252, z_threshold: float = 2.0,
                 stop_loss: float = 0.05, take_profit: float = 0.1):
        """Initialize the Statistical Arbitrage Trading System."""
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.pairs_data = {}

    # ... (previous methods remain the same until backtest_pair) ...

    def backtest_pair(self, price1: pd.Series, price2: pd.Series,
                      initial_capital: float = 100000) -> Dict:
        """Backtest a pairs trading strategy."""
        # Calculate hedge ratio and spread
        hedge_ratio, _ = self.calculate_hedge_ratio(price1, price2)
        spread = self.calculate_spread(price1, price2, hedge_ratio)

        # Generate signals
        signals = self.generate_signals(spread)

        # Initialize position and portfolio variables
        position = 0
        portfolio_value = pd.Series(index=price1.index, dtype=float)
        portfolio_value.iloc[0] = initial_capital
        returns = pd.Series(index=price1.index, dtype=float)
        returns.iloc[0] = 0.0

        # Track trades
        trades = []
        current_trade = None

        for i in range(1, len(signals.index)):
            if signals.iloc[i] != 0 and position == 0:
                # Enter new position
                position = signals.iloc[i]
                entry_spread = spread.iloc[i]
                current_trade = {
                    'entry_date': spread.index[i],
                    'entry_spread': entry_spread,
                    'position': position
                }

            elif position != 0:
                # Calculate return
                spread_return = (spread.iloc[i] - spread.iloc[i - 1]) / abs(spread.iloc[i - 1])
                returns.iloc[i] = position * spread_return

                # Check stop loss and take profit
                total_return = (spread.iloc[i] - current_trade['entry_spread']) / \
                               abs(current_trade['entry_spread'])

                if (position * total_return < -self.stop_loss) or \
                        (position * total_return > self.take_profit) or \
                        (signals.iloc[i] == -position):
                    # Close position
                    current_trade['exit_date'] = spread.index[i]
                    current_trade['exit_spread'] = spread.iloc[i]
                    current_trade['return'] = position * total_return
                    trades.append(current_trade)

                    position = 0
                    current_trade = None

            # Update portfolio value
            portfolio_value.iloc[i] = portfolio_value.iloc[i - 1] * (1 + returns.iloc[i])

        # Calculate performance metrics
        total_return = (portfolio_value.iloc[-1] - initial_capital) / initial_capital
        daily_returns = portfolio_value.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        max_drawdown = (portfolio_value / portfolio_value.cummax() - 1).min()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'portfolio_value': portfolio_value,
            'returns': returns
        }

    def plot_pair_analysis(self, price1: pd.Series, price2: pd.Series,
                           results: Dict) -> None:
        """Plot comprehensive pair trading analysis."""
        # Set the style
        plt.style.use('default')  # Use default style instead of seaborn
        sns.set_style("whitegrid")  # Apply seaborn's grid style

        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 2, figure=fig)

        # Plot 1: Price Series
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(price1.index, price1 / price1.iloc[0], label=price1.name)
        ax1.plot(price2.index, price2 / price2.iloc[0], label=price2.name)
        ax1.set_title('Normalized Price Series')
        ax1.legend()

        # Plot 2: Spread Z-Score
        ax2 = fig.add_subplot(gs[1, 0])
        hedge_ratio, _ = self.calculate_hedge_ratio(price1, price2)
        spread = self.calculate_spread(price1, price2, hedge_ratio)
        z_score = (spread - spread.rolling(window=self.lookback_period).mean()) / \
                  spread.rolling(window=self.lookback_period).std()
        ax2.plot(z_score)
        ax2.axhline(y=self.z_threshold, color='r', linestyle='--')
        ax2.axhline(y=-self.z_threshold, color='r', linestyle='--')
        ax2.set_title('Spread Z-Score')

        # Plot 3: Portfolio Value
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(results['portfolio_value'])
        ax3.set_title('Portfolio Value')

        # Plot 4: Returns Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        sns.histplot(data=results['returns'].dropna(), kde=True, ax=ax4)
        ax4.set_title('Returns Distribution')

        # Plot 5: Drawdown
        ax5 = fig.add_subplot(gs[2, 1])
        drawdown = results['portfolio_value'] / results['portfolio_value'].cummax() - 1
        ax5.plot(drawdown)
        ax5.set_title('Drawdown')

        plt.tight_layout()
        plt.show()


def main():
    # Initialize the system
    stat_arb = StatArbSystem(
        lookback_period=252,  # One year of trading days
        z_threshold=2.0,  # Entry/exit threshold
        stop_loss=0.05,  # 5% stop loss
        take_profit=0.10  # 10% take profit
    )

    # Example usage
    tickers = ['XLF', 'BAC', 'JPM', 'WFC', 'C', 'GS']
    start_date = '2020-01-01'
    end_date = '2023-12-31'

    # Fetch data
    data = stat_arb.fetch_data(tickers, start_date, end_date)

    # Find cointegrated pairs
    pairs = stat_arb.find_cointegrated_pairs(data)
    print("\nCointegrated Pairs:")
    for pair in pairs:
        print(f"{pair[0]} - {pair[1]}: p-value = {pair[2]:.4f}")

    if pairs:
        # Analyze first pair
        stock1, stock2, _ = pairs[0]
        results = stat_arb.backtest_pair(data[stock1], data[stock2])

        print(f"\nBacktest Results for {stock1}-{stock2}:")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")

        # Plot analysis
        stat_arb.plot_pair_analysis(data[stock1], data[stock2], results)


if __name__ == "__main__":
    main()