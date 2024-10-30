import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class StatArbSystem:
    def __init__(self, stock1: str, stock2: str, lookback_period: int = 252, entry_z_threshold: float = 2.0,
                 exit_z_threshold: float = 0.5, stop_loss: float = 0.05,
                 take_profit: float = 0.1, transaction_cost: float = 0.001,
                 position_size: float = 1.0):
        """
        Enhanced Statistical Arbitrage Trading System
        """
        self.stock1 = stock1
        self.stock2 = stock2
        self.lookback_period = lookback_period
        self.entry_z_threshold = entry_z_threshold
        self.exit_z_threshold = exit_z_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.transaction_cost = transaction_cost
        self.position_size = position_size
        self.position = 0
        self.stats = {}

    def get_data(self) -> pd.DataFrame:
        """
        Fetch stock price data with proper index handling and data quality checks.
        """
        try:
            end = datetime.now()
            start = end - timedelta(days=self.lookback_period)

            # Download daily data for both stocks
            #print(f"Downloading data for {self.stock1} and {self.stock2}...")
            df1 = yf.download(self.stock1, start=start, end=end, progress=False)
            df2 = yf.download(self.stock2, start=start, end=end, progress=False)

            if df1.empty or df2.empty:
                raise ValueError(f"No data available for {self.stock1} or {self.stock2}")

            common_dates = df1.index.intersection(df2.index)
            if len(common_dates) < 30:
                raise ValueError("Insufficient data points for analysis")

            combined_data = pd.DataFrame(index=common_dates)
            combined_data[self.stock1] = df1.loc[common_dates, 'Adj Close']
            combined_data[self.stock2] = df2.loc[common_dates, 'Adj Close']

            combined_data = combined_data.dropna()
            if len(combined_data) < 30:
                raise ValueError("Insufficient data points after removing missing values")

            ##print(f"Successfully retrieved {len(combined_data)} days of data")
            return combined_data

        except Exception as e:
            raise Exception(f"Error getting data: {str(e)}")

    def calculate_spread(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the normalized price spread between stocks.
        """
        try:
            stock1_norm = data[self.stock1] / data[self.stock1].iloc[0]
            stock2_norm = data[self.stock2] / data[self.stock2].iloc[0]
            spread = stock1_norm - stock2_norm
            z_score = (spread - spread.mean()) / spread.std()
            return pd.Series(z_score, index=data.index)

        except Exception as e:
            raise Exception(f"Error calculating spread: {str(e)}")

    def calculate_returns(self, data: pd.DataFrame, positions: pd.Series) -> pd.Series:
        """
        Calculate strategy returns with transaction costs
        """
        try:
            returns1 = data[self.stock1].pct_change()
            returns2 = data[self.stock2].pct_change()
            strategy_returns = positions * (returns1 - returns2)

            position_changes = positions.diff().fillna(0)
            transaction_costs = abs(position_changes) * self.transaction_cost
            final_returns = strategy_returns - transaction_costs
            return final_returns

        except Exception as e:
            raise Exception(f"Error calculating returns: {str(e)}")

    def perform_statistical_analysis(self, data: pd.DataFrame) -> Dict:
        """
        Comprehensive statistical analysis of the pair
        """
        try:
            # Calculate correlation
            correlation = data[self.stock1].corr(data[self.stock2])

            # Calculate beta
            beta = np.cov(data[self.stock1], data[self.stock2])[0][1] / np.var(data[self.stock2])

            # Perform cointegration test
            # Note: coint returns: test statistic, p-value, critical values
            coint_result = coint(data[self.stock1], data[self.stock2])
            p_value = coint_result[1]  # Extract just the p-value

            # Calculate spread and perform ADF test
            spread = self.calculate_spread(data)
            adf_result = adfuller(spread)
            adf_stat = adf_result[0]
            adf_p_value = adf_result[1]

            # Calculate half-life of mean reversion
            spread_lag = spread.shift(1)
            spread_diff = spread - spread_lag
            spread_lag = spread_lag[1:]
            spread_diff = spread_diff[1:]
            beta_half_life = np.polyfit(spread_lag, spread_diff, 1)[0]
            half_life = -np.log(2) / beta_half_life if beta_half_life < 0 else np.inf

            return {
                'correlation': correlation,
                'beta': beta,
                'coint_p_value': p_value,
                'adf_stat': adf_stat,
                'adf_p_value': adf_p_value,
                'half_life': half_life,
                # Add more details for debugging
                'coint_test_stat': coint_result[0],
                'coint_critical_values': coint_result[2],
                'adf_critical_values': adf_result[4]
            }

        except Exception as e:
            raise Exception(f"Error in statistical analysis: {str(e)}")

    def check_pair_eligibility(self, stats: Dict) -> Tuple[bool, str]:
        """
        Enhanced pair eligibility checking with multiple criteria
        """
        criteria = []
        is_eligible = True

        if stats['correlation'] < 0.8:
            criteria.append("Low correlation")
            is_eligible = False

        if stats['coint_p_value'] > 0.05:
            criteria.append("Not cointegrated")
            is_eligible = False

        if stats['adf_p_value'] > 0.05:
            criteria.append("Spread not stationary")
            is_eligible = False

        if not (5 <= stats['half_life'] <= self.lookback_period):
            criteria.append("Invalid half-life")
            is_eligible = False

        return is_eligible, ", ".join(criteria) if criteria else "Pair is eligible"

    def calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate comprehensive risk metrics
        """
        daily_returns = returns.dropna()

        return {
            'sharpe_ratio': np.sqrt(252) * daily_returns.mean() / daily_returns.std(),
            'sortino_ratio': np.sqrt(252) * daily_returns.mean() / daily_returns[daily_returns < 0].std(),
            'max_drawdown': (daily_returns.cumsum() - daily_returns.cumsum().expanding().max()).min(),
            'var_95': daily_returns.quantile(0.05),
            'var_99': daily_returns.quantile(0.01),
            'win_rate': len(daily_returns[daily_returns > 0]) / len(daily_returns),
            'profit_factor': abs(daily_returns[daily_returns > 0].sum() / daily_returns[daily_returns < 0].sum())
        }

    def generate_signals(self, z_score: float, current_pnl: float = 0) -> int:
        """
        Enhanced signal generation with risk management
        """
        if abs(current_pnl) >= self.stop_loss and self.position != 0:
            return 0
        if current_pnl >= self.take_profit and self.position != 0:
            return 0

        if self.position == 0:
            if z_score > self.entry_z_threshold:
                return -1 * self.position_size
            elif z_score < -self.entry_z_threshold:
                return 1 * self.position_size
            return 0

        elif self.position > 0:
            if z_score >= -self.exit_z_threshold:
                return 0
            return self.position

        else:  # position < 0
            if z_score <= self.exit_z_threshold:
                return 0
            return self.position

    def run_backtest(self, initial_investment: float = 1000) -> Dict:
        """
        Comprehensive backtesting framework with profit calculation based on initial investment.
        """
        try:
            data = self.get_data()
            self.stats = self.perform_statistical_analysis(data)
            is_eligible, reason = self.check_pair_eligibility(self.stats)

            if not is_eligible:
                return {'success': False, 'message': reason}

            z_scores = self.calculate_spread(data)
            positions = pd.Series(index=data.index, dtype=float)
            returns = pd.Series(index=data.index, dtype=float)
            current_pnl = 0

            for i in range(1, len(data)):
                if i > 0 and positions.iloc[i - 1] != 0:
                    returns1 = data[self.stock1].iloc[i] / data[self.stock1].iloc[i - 1] - 1
                    returns2 = data[self.stock2].iloc[i] / data[self.stock2].iloc[i - 1] - 1
                    current_pnl += positions.iloc[i - 1] * (returns1 - returns2)

                positions.iloc[i] = self.generate_signals(z_scores.iloc[i], current_pnl)

                if positions.iloc[i] == 0:
                    current_pnl = 0

            strategy_returns = self.calculate_returns(data, positions)
            risk_metrics = self.calculate_risk_metrics(strategy_returns)

            # Calculate the total profit based on the initial investment
            total_return = strategy_returns.sum()
            final_value = initial_investment * (1 + total_return)
            profit = final_value - initial_investment

            return {
                'success': True,
                'statistics': self.stats,
                'risk_metrics': risk_metrics,
                'positions': positions.tolist(),
                'returns': strategy_returns.tolist(),
                'data_points': len(data),
                'initial_investment': initial_investment,
                'final_value': final_value,
                'profit': profit
            }

        except Exception as e:
            return {'success': False, 'message': str(e)}

    def optimize_parameters(self) -> Dict:
        """
        Optimize strategy parameters to target Sharpe ratio close to 1.8
        """
        try:
            # Parameter ranges to test
            entry_thresholds = np.arange(1.5, 3.0, 0.25)
            exit_thresholds = np.arange(0.25, 1.0, 0.25)
            stop_losses = np.arange(0.03, 0.08, 0.01)
            take_profits = np.arange(0.08, 0.15, 0.01)

            best_params = None
            best_sharpe = float('-inf')
            failed_attempts = 0
            max_failed_attempts = 3

            orig_params = {
                'entry': self.entry_z_threshold,
                'exit': self.exit_z_threshold,
                'stop': self.stop_loss,
                'profit': self.take_profit
            }

            print("\nOptimizing parameters...")

            total_iterations = (len(entry_thresholds) * len(exit_thresholds) *
                                len(stop_losses) * len(take_profits))
            current_iteration = 0

            initial_check = self.run_backtest()
            if not initial_check['success']:
                raise Exception(f"Pair is not suitable for trading: {initial_check['message']}")

            for entry in entry_thresholds:
                for exit in exit_thresholds:
                    for stop in stop_losses:
                        for profit in take_profits:
                            current_iteration += 1
                            if current_iteration % 10 == 0:
                                print(f"Progress: {current_iteration}/{total_iterations} "
                                      f"({(current_iteration / total_iterations) * 100:.1f}%)")

                            self.entry_z_threshold = entry
                            self.exit_z_threshold = exit
                            self.stop_loss = stop
                            self.take_profit = profit

                            try:
                                results = self.run_backtest()
                                if results['success']:
                                    sharpe = results['risk_metrics']['sharpe_ratio']
                                    failed_attempts = 0

                                    if abs(1.8 - sharpe) < abs(1.8 - best_sharpe):
                                        best_sharpe = sharpe
                                        best_params = {
                                            'entry_threshold': entry,
                                            'exit_threshold': exit,
                                            'stop_loss': stop,
                                            'take_profit': profit,
                                            'sharpe_ratio': sharpe
                                        }
                                else:
                                    failed_attempts += 1
                                    if failed_attempts >= max_failed_attempts:
                                        print(f"Too many consecutive failures. Skipping to next parameter set.")
                                        failed_attempts = 0
                                        continue

                            except Exception as e:
                                print(f"Error during optimization iteration: {str(e)}")
                                failed_attempts += 1
                                if failed_attempts >= max_failed_attempts:
                                    print(f"Too many consecutive failures. Skipping to next parameter set.")
                                    failed_attempts = 0
                                    continue

            self.entry_z_threshold = orig_params['entry']
            self.exit_z_threshold = orig_params['exit']
            self.stop_loss = orig_params['stop']
            self.take_profit = orig_params['profit']

            if best_params is None:
                raise Exception("No valid parameter combination found")

            print("\nOptimization complete!")
            return best_params

        except Exception as e:
            raise Exception(f"Error in parameter optimization: {str(e)}")


def main():
    pairs = [
         ('DUK', 'AEP'), ('CAT', 'DE'), ('UPS', 'FDX')
,
    ]

    print("\n📊 Statistical Arbitrage Trading System - Backtest Summary")
    print("=" * 60)

    for stock1, stock2 in pairs:
        try:
            print(f"\n🔍 Testing Pair: {stock1} - {stock2}")
            trader = StatArbSystem(stock1, stock2)

            print("\n⏳ Optimizing parameters, please wait...")
            best_params = trader.optimize_parameters()

            # Apply optimized parameters
            trader.entry_z_threshold = best_params['entry_threshold']
            trader.exit_z_threshold = best_params['exit_threshold']
            trader.stop_loss = best_params['stop_loss']
            trader.take_profit = best_params['take_profit']

            # Run backtest with optimized parameters
            print("\n📈 Running backtest...")
            results = trader.run_backtest()

            if results['success']:
                metrics = results['risk_metrics']
                stats = results['statistics']

                print("\n🎯 Optimization Complete! Best Parameters Found:")
                print(f"  • Entry Z-Score: {best_params['entry_threshold']:.2f}")
                print(f"  • Exit Z-Score: {best_params['exit_threshold']:.2f}")
                print(f"  • Stop Loss: {best_params['stop_loss']:.2%}")
                print(f"  • Take Profit: {best_params['take_profit']:.2%}")

                print("\n📊 Statistical Analysis for Pair:")
                print(f"  • Correlation: {stats['correlation']:.2f}")
                print(f"  • Cointegration p-value: {stats['coint_p_value']:.4f}")
                print(f"  • Half-life: {stats['half_life']:.1f} days")

                print("\n💹 Performance Metrics:")
                print(f"  • Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"  • Sortino Ratio: {metrics['sortino_ratio']:.2f}")
                print(f"  • Maximum Drawdown: {metrics['max_drawdown']:.2%}")
                print(f"  • Win Rate: {metrics['win_rate']:.2%}")
                print(f"  • Profit Factor: {metrics['profit_factor']:.2f}")

                # Display initial investment, final value, and profit
                print("\n💸 Investment Summary:")
                print(f"  • Initial Investment: ${results['initial_investment']:.2f}")
                print(f"  • Final Value: ${results['final_value']:.2f}")
                print(f"  • Profit/Loss: ${results['profit']:.2f} ({results['profit'] / results['initial_investment']:.2%})")

            else:
                print(f"\n⚠️  Error: {results['message']}")

        except Exception as e:
            print(f"❌ Error processing pair {stock1}-{stock2}: {str(e)}")
            continue

    print("\n✨ Backtesting Completed!")


if __name__ == "__main__":
    main()