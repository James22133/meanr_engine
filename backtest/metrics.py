import numpy as np
import pandas as pd
from scipy import stats

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio using downside deviation."""
    excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.inf
    downside_std = np.sqrt(np.mean(downside_returns**2))
    if downside_std == 0:
        return np.inf
    return np.sqrt(252) * excess_returns.mean() / downside_std

def calculate_probabilistic_sharpe(returns: pd.Series, risk_free_rate: float = 0.0, 
                                 benchmark_sharpe: float = 0.0) -> float:
    """Calculate Probabilistic Sharpe Ratio (PSR) from López de Prado's book."""
    excess_returns = returns - risk_free_rate/252
    sr = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    skew = stats.skew(excess_returns)
    kurt = stats.kurtosis(excess_returns)
    n = len(returns)
    
    # Calculate PSR
    sr_std = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt-1)/4 * sr**2) / (n-1))
    psr = stats.norm.cdf((sr - benchmark_sharpe) / sr_std)
    return psr

def calculate_metrics(returns: pd.Series, equity_curve: pd.Series, 
                     trades: list, risk_free_rate: float = 0.0) -> dict:
    """Calculate comprehensive performance metrics including risk-adjusted ratios."""
    metrics = {}
    
    # Basic metrics
    metrics['annualized_return'] = returns.mean() * 252
    metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
    metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_volatility'] if metrics['annualized_volatility'] > 0 else 0
    
    # Risk-adjusted metrics
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns, risk_free_rate)
    metrics['probabilistic_sharpe'] = calculate_probabilistic_sharpe(returns, risk_free_rate)
    
    # Drawdown metrics
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    metrics['max_drawdown'] = drawdown.min()
    metrics['avg_drawdown'] = drawdown[drawdown < 0].mean()
    
    # Trade metrics
    closed_trades = [t for t in trades if t.exit_date is not None]
    winning_trades = [t for t in closed_trades if t.pnl and t.pnl > 0]
    metrics['win_rate'] = len(winning_trades) / len(closed_trades) if closed_trades else 0
    metrics['profit_factor'] = (sum(t.pnl for t in winning_trades) / 
                              abs(sum(t.pnl for t in closed_trades if t.pnl < 0)) 
                              if sum(t.pnl for t in closed_trades if t.pnl < 0) != 0 else np.inf)
    
    # Holding period metrics
    holding_periods = [(t.exit_date - t.entry_date).days for t in closed_trades]
    metrics['avg_holding_period'] = np.mean(holding_periods) if holding_periods else 0
    
    # Trade counts
    metrics['total_trades'] = len(closed_trades)
    metrics['winning_trades'] = len(winning_trades)
    
    return metrics 