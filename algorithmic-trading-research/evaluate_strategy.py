import pandas as pd
from strategy import full_strategy_pipeline

if __name__ == "__main__":
    print("Loading data...")
    try:
        apple = pd.read_csv("e:/Projects/Projects/algorithmic-trading-research/cleaned_apple.csv")
        
        params = {
            "df": apple,
            "VALID_WEEKS": 52,
            "depth_grid": [2, 3],
            "leaf_grid": [4, 5],
            "n_subsets": 10,
            "n_trajectories": 1000,
            "n_weeks": 50,
        }
        
        results = full_strategy_pipeline(params)
        print("Backtest completed successfully.")
    except Exception as e:
        print(f"Error during execution: {e}")
