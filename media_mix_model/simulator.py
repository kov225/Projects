import numpy as np
import pandas as pd
import json
import os
from scipy.stats import norm

class MMMSimulator:
    """
    High-fidelity simulator for a Direct-to-Consumer (DTC) brand's marketing environment.
    Mimics real-world dynamics like adstock, saturation, seasonality, and multicollinearity.
    """
    
    def __init__(self, n_weeks=156, seed=42):
        self.n_weeks = n_weeks
        self.seed = seed
        np.random.seed(seed)
        
        # Channel configurations
        self.channels = {
            "linear_tv": {
                "adstock_alpha": 0.55,
                "saturation_k": 80000,
                "saturation_s": 3.0,
                "beta": 0.15,
                "spend_range": (50000, 150000)
            },
            "ctv": {
                "adstock_alpha": 0.35,
                "saturation_k": 50000,
                "saturation_s": 2.5,
                "beta": 0.10,
                "spend_range": (20000, 80000)
            },
            "paid_search": {
                "adstock_alpha": 0.10,
                "saturation_k": 40000,
                "saturation_s": 4.0,
                "beta": 0.12,
                "spend_range": (15000, 60000)
            },
            "paid_social": {
                "adstock_alpha": 0.25,
                "saturation_k": 35000,
                "saturation_s": 3.0,
                "beta": 0.08,
                "spend_range": (10000, 50000)
            },
            "display": {
                "adstock_alpha": 0.15,
                "saturation_k": 20000,
                "saturation_s": 5.0,
                "beta": 0.05,
                "spend_range": (5000, 25000)
            }
        }
        
    def geometric_adstock(self, x, alpha):
        """Applies geometric decay to a time series."""
        adstocked = np.zeros_like(x)
        for t in range(len(x)):
            if t == 0:
                adstocked[t] = x[t]
            else:
                adstocked[t] = x[t] + alpha * adstocked[t-1]
        return adstocked
    
    def hill_saturation(self, x, k, s):
        """Applies Hill function diminishing returns transformation."""
        return (x**s) / (x**s + k**s)

    def generate_spend(self):
        """Generates realistic weekly spend for all channels."""
        weeks = np.arange(self.n_weeks)
        df = pd.DataFrame({"week": weeks})
        
        # Base spend with seasonality
        q4_boost = np.where((weeks % 52 >= 40) & (weeks % 52 <= 51), 1.4, 1.0)
        
        # Linear TV: heavier in Q4
        base_tv = np.random.uniform(*self.channels["linear_tv"]["spend_range"], size=self.n_weeks)
        df["linear_tv_spend"] = base_tv * q4_boost
        
        # CTV: correlated with Linear TV (r=0.6)
        base_ctv = np.random.uniform(*self.channels["ctv"]["spend_range"], size=self.n_weeks)
        df["ctv_spend"] = 0.6 * df["linear_tv_spend"] * (self.channels["ctv"]["spend_range"][1] / self.channels["linear_tv"]["spend_range"][1]) + \
                         0.4 * base_ctv
        
        # Paid Search: reactive to TV with a lag
        search_base = np.random.uniform(*self.channels["paid_search"]["spend_range"], size=self.n_weeks)
        # Add a spike where TV spend is high
        tv_spike = (df["linear_tv_spend"] > df["linear_tv_spend"].mean()) * 0.2 * search_base
        df["paid_search_spend"] = search_base + tv_spike
        
        # Paid Social: independent
        df["paid_social_spend"] = np.random.uniform(*self.channels["paid_social"]["spend_range"], size=self.n_weeks)
        
        # Display: low spend
        df["display_spend"] = np.random.uniform(*self.channels["display"]["spend_range"], size=self.n_weeks)
        
        return df

    def generate_controls(self, df):
        """Generates seasonal terms, holidays, and trend."""
        weeks = df["week"].values
        
        # Fourier Seasonality (2 harmonics)
        df["sin_period_1"] = np.sin(2 * np.pi * weeks / 52.18)
        df["cos_period_1"] = np.cos(2 * np.pi * weeks / 52.18)
        df["sin_period_2"] = np.sin(4 * np.pi * weeks / 52.18)
        df["cos_period_2"] = np.cos(4 * np.pi * weeks / 52.18)
        
        # Holiday Indicators
        # Major US holidays (approximate weekly indices)
        holidays = {
            "new_year": 0, "presidents_day": 7, "memorial_day": 21, 
            "july_4th": 26, "labor_day": 35, "thanksgiving": 47, 
            "black_friday": 47, "christmas": 51
        }
        for name, week_offset in holidays.items():
            df[f"holiday_{name}"] = 0
            for year in range(3):
                df.loc[df["week"] == (week_offset + year*52), f"holiday_{name}"] = 1
                
        # Competitor Activity: noisy sine wave
        df["competitor_proxy"] = 0.5 * np.sin(2 * np.pi * weeks / 26) + np.random.normal(0, 0.1, self.n_weeks)
        
        # Linear Trend: organic growth
        df["trend"] = 1.0 + 0.002 * weeks
        
        return df

    def generate_conversions(self, df):
        """Calculates conversions using a multiplicative log-link model."""
        # Baseline contribution
        log_conversions = np.log(1000) + np.log(df["trend"])
        
        # Seasonality effects
        log_conversions += 0.15 * df["sin_period_1"] + 0.05 * df["cos_period_2"]
        
        # Holiday effects
        holiday_cols = [c for c in df.columns if "holiday_" in c]
        for col in holiday_cols:
            weight = 0.2 if "black_friday" in col else 0.1
            log_conversions += weight * df[col]
            
        # Competitor suppression
        log_conversions -= 0.1 * df["competitor_proxy"]
        
        # Media contributions
        channel_contributions = {}
        for chan, params in self.channels.items():
            spend = df[f"{chan}_spend"].values
            # Adstock
            adstocked = self.geometric_adstock(spend, params["adstock_alpha"])
            # Saturation
            saturated = self.hill_saturation(adstocked, params["saturation_k"], params["saturation_s"])
            # Impact
            contribution = params["beta"] * saturated
            log_conversions += contribution
            channel_contributions[chan] = contribution
            
        # Convert back from log space
        conversions = np.exp(log_conversions)
        
        # Heteroscedastic Noise: variance proportional to mean
        noise_std = 0.05 * conversions
        conversions += np.random.normal(0, noise_std)
        
        df["conversions"] = conversions.astype(int)
        
        # Store true contribution shares for verification
        self.true_contributions = channel_contributions
        
        return df

    def run(self):
        """Run the full simulation pipeline."""
        df = self.generate_spend()
        df = self.generate_controls(df)
        df = self.generate_conversions(df)
        
        # Create directories if needed
        os.makedirs("media_mix_model/data", exist_ok=True)
        
        # Save data
        df.to_csv("media_mix_model/data/simulated_data.csv", index=False)
        
        # Save ground truth
        ground_truth = {
            "channels": self.channels,
            "seed": self.seed
        }
        with open("media_mix_model/data/ground_truth_params.json", "w") as f:
            json.dump(ground_truth, f, indent=4)
            
        print(f"Simulated {self.n_weeks} weeks of data saved to media_mix_model/data/")
        return df

if __name__ == "__main__":
    sim = MMMSimulator()
    df = sim.run()
    print(df.head())
