"""
generate_plot.py : Generates the feature correlation heatmap for the Apple trading strategy.
Run from inside the algorithmic-trading-research/ directory.
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_path = os.path.join(os.path.dirname(__file__), 'cleaned_apple.csv')

try:
    df = pd.read_csv(data_path)
    df['Daily_Return'] = df['CLOSE'].pct_change()
    df['Rolling_Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
    df['Rolling_Mean_20'] = df['CLOSE'].rolling(window=20).mean()
    df['Price_to_Mean_Ratio'] = df['CLOSE'] / df['Rolling_Mean_20']
    df = df.dropna()

    features = ['OPEN', 'CLOSE', 'VOL', 'Daily_Return', 'Rolling_Volatility_20', 'Price_to_Mean_Ratio']
    available = [f for f in features if f in df.columns]

    corr = df[available].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f",
                vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 11})
    plt.title('Correlation Matrix of Engineered AAPL Market Features', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), 'feature_correlation_heatmap.png')
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to: {out_path}")
except Exception as e:
    print(f"Error: {e}")
