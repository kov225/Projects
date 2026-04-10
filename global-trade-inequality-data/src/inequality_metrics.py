import numpy as np
import pandas as pd
import logging

# Researcher-grade statistical utilities for economic analysis
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def calculate_gini(array):
    """
    Computes the Gini Coefficient as a measure of trade inequality.
    
    A Gini of 0 represents perfect equality, while 1 represents 
    maximal inequality (one country holds all trade value).
    """
    # Based on the formula: G = (sum_{i=1}^n sum_{j=1}^n |x_i - x_j|) / (2 * n^2 * mean(x))
    array = array.flatten()
    if np.any(array < 0):
        raise ValueError("Gini coefficient requires non-negative values.")
    
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def calculate_hhi(array):
    """
    Computes the Herfindahl-Hirschman Index (HHI) for market concentration.
    
    HHI = sum(market_share^2). Values range from 1/n to 1.
    High HHI indicates a monopolistic or highly concentrated trade block.
    """
    shares = array / np.sum(array)
    return np.sum(shares**2)

def analyze_trade_concentration(df, value_col='trade_value', group_col='country'):
    """
    Performs a distributional analysis of trade concentration across groups.
    """
    logger.info(f"Analyzing concentration for: {group_col}")
    
    grouped = df.groupby(group_col)[value_col].sum().values
    
    gini = calculate_gini(grouped)
    hhi = calculate_hhi(grouped)
    
    print("\n" + "="*40)
    print(f"TRADE CONCENTRATION REPORT: {group_col}")
    print("="*40)
    print(f"Gini Coefficient:  {gini:.4f}")
    print(f"HHI Index:         {hhi:.4f}")
    print(f"Concentration:     {'High' if gini > 0.5 else 'Moderate' if gini > 0.3 else 'Low'}")
    print("="*40)
    
    return {"gini": gini, "hhi": hhi}

if __name__ == "__main__":
    # Simulated demonstration data representing trade values across 10 regions
    simulated_trade = pd.DataFrame({
        'country': [f'Region_{i}' for i in range(10)],
        'trade_value': [10, 20, 15, 5, 800, 12, 18, 5, 7, 10] # Highly skewed
    })
    
    analyze_trade_concentration(simulated_trade)
