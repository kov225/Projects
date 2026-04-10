import pandas as pd
import numpy as np
import logging

# Engineer-grade QoS (Quality of Service) Analytics for Video Streaming
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class QualityOfServiceAnalyzer:
    """
    Calculates Quality of Service (QoS) metrics across device categories.
    
    In video streaming pipelines, analyzing the distribution of buffering 
    events and throughput efficiency is critical for optimizing CDN selection 
    and encoding ladders.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def calculate_buffering_ratio(self):
        """
        Computes Buffering Ratio: (Total Buffer Time) / (Total Playback Time).
        A high buffering ratio (>1%) is a leading indicator of churn.
        """
        self.data['buffering_ratio'] = self.data['buffer_ms'] / (self.data['playback_ms'] + 1e-6)
        return self.data.groupby('device_category')['buffering_ratio'].mean().sort_values()

    def bootstrap_buffering_diff(self, group_a='Smart TV', group_b='Mobile', n_iterations=1000):
        """
        Performs bootstrap resampling to determine if the difference 
        in buffering ratios between two device categories is statistically significant.
        """
        logger.info(f"Running Bootstrap Analysis: {group_a} vs {group_b}")
        
        a_vals = self.data[self.data['device_category'] == group_a]['buffering_ratio'].values
        b_vals = self.data[self.data['device_category'] == group_b]['buffering_ratio'].values
        
        diffs = []
        for _ in range(n_iterations):
            a_sample = np.random.choice(a_vals, size=len(a_vals), replace=True)
            b_sample = np.random.choice(b_vals, size=len(b_vals), replace=True)
            diffs.append(np.mean(a_sample) - np.mean(b_sample))
            
        ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
        
        print("\n" + "="*40)
        print("BOOTSTRAP QoS SIGNIFICANCE REPORT")
        print("="*40)
        print(f"Mean Difference:    {np.mean(diffs):.4f}")
        print(f"95% Confidence Int: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"Significant:        {'Yes' if ci_lower > 0 or ci_upper < 0 else 'No'}")
        print("="*40)
        
        return ci_lower, ci_upper

if __name__ == "__main__":
    # Simulated QoS telemetry data
    np.random.seed(42)
    devices = ['Smart TV', 'Mobile', 'Web', 'Game Console']
    data = pd.DataFrame({
        'device_category': np.random.choice(devices, 1000),
        'playback_ms': np.random.uniform(300000, 3600000, 1000), # 5 to 60 mins
        'buffer_ms': np.random.gamma(shape=2, scale=500, size=1000) # skewed buffering
    })
    
    analyzer = QualityOfServiceAnalyzer(data)
    ratios = analyzer.calculate_buffering_ratio()
    print("Buffering Ratios by Device:")
    print(ratios)
    
    analyzer.bootstrap_buffering_diff()
