import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta

class TVAttributionSimulator:
    """
    Simulates high-resolution minute-level web traffic and TV ad airings.
    Includes realistic response curves, organic volatility, and control market correlation.
    """
    
    def __init__(self, n_days=90, seed=42):
        self.n_days = n_days
        self.n_minutes = n_days * 1440
        self.seed = seed
        np.random.seed(seed)
        
        self.start_date = datetime(2024, 1, 1)
        self.minute_index = pd.date_range(self.start_date, periods=self.n_minutes, freq='min')
        
        # Performance parameters
        self.networks = {
            "NBC": {"reach": 1.5, "base_cost": 40},
            "ESPN": {"reach": 1.2, "base_cost": 35},
            "HGTV": {"reach": 0.8, "base_cost": 10},
            "CNN": {"reach": 0.9, "base_cost": 12},
            "Fox News": {"reach": 1.0, "base_cost": 15},
            "Discovery": {"reach": 0.7, "base_cost": 8},
            # ... and so on for 20 networks
        }
        
        self.creatives = {
            "A": 1.0, "B": 1.3, "C": 0.7, "D": 1.1
        }
        
    def get_organic_baseline(self):
        """Generates minute-level sessions with seasonality and trend."""
        minutes = np.arange(self.n_minutes)
        hours = (minutes % 1440) / 60
        days = (minutes // 1440) % 7
        
        # Double-peaked daily pattern (10am-12pm, 7pm-9pm)
        # Peak 1
        daily_1 = 20 * np.exp(-((hours - 11)**2) / 4)
        # Peak 2
        daily_2 = 25 * np.exp(-((hours - 20)**2) / 6)
        # Baseline
        base = 10 + daily_1 + daily_2
        
        # Day of week effect: weekdays (0-4) are higher than weekends (5-6)
        dow_mult = np.where(days < 5, 1.25, 1.0)
        base *= dow_mult
        
        # Slow upward trend
        trend = 1.0 + (0.05 / self.n_minutes) * minutes
        base *= trend
        
        # AR(1) Noise autocorrelation 0.7
        noise = np.zeros(self.n_minutes)
        phi = 0.7
        for t in range(1, self.n_minutes):
            noise[t] = phi * noise[t-1] + np.random.normal(0, 1.5)
            
        return base + noise

    def generate_airings(self):
        """Generates 300 ad airings across the 90 days."""
        airings = []
        network_names = list(self.networks.keys())
        creative_ids = list(self.creatives.keys())
        
        # Track frequency by network and day
        freq_tracker = {}
        
        for i in range(300):
            # random timestamps
            random_min = np.random.randint(0, self.n_minutes)
            ts = self.start_date + timedelta(minutes=random_min)
            network = np.random.choice(network_names)
            day = ts.date()
            
            # frequency tracker
            freq_key = (network, day)
            freq_tracker[freq_key] = freq_tracker.get(freq_key, 0) + 1
            freq = freq_tracker[freq_key]
            
            creative = np.random.choice(creative_ids)
            spot_length = np.random.choice([15, 30])
            
            # Cost calculation based on primetime
            hour = ts.hour
            is_primetime = 19 <= hour <= 23
            cost = self.networks[network]["base_cost"] * (2.0 if is_primetime else 1.0)
            
            # response calculation parameters
            reach = self.networks[network]["reach"]
            creative_mult = self.creatives[creative]
            length_mult = 1.4 if spot_length == 30 else 1.0
            dow_mult = 0.85 if ts.weekday() >= 5 else 1.0
            
            # frequency fatigue
            fatigue = 0.8**(max(0, freq - 3))
            
            total_response = 50 * reach * creative_mult * length_mult * dow_mult * fatigue
            
            airing = {
                "airing_id": i,
                "timestamp": ts,
                "network": network,
                "creative_id": creative,
                "spot_length": spot_length,
                "cost": cost,
                "is_missed": i < 10, # first 10 airings are missed
                "true_total_lift": total_response if i >= 10 else 0,
                "tau": 2.5 # time to peak
            }
            airings.append(airing)
            
        return pd.DataFrame(airings)

    def inject_response_curves(self, sessions, airings):
        """Injects non-linear lift into the session counts."""
        session_vals = sessions
        
        for idx, row in airings.iterrows():
            if row["is_missed"]:
                continue
                
            start_idx = int((row["timestamp"] - self.start_date).total_seconds() // 60)
            
            # Response window 0-15 minutes
            window_len = 15
            if start_idx + window_len >= len(session_vals):
                continue
                
            # lift(t) = A * (t/tau) * exp(1 - t/tau)
            # A is total lift / integral or something related to peak amplitude
            # Area of this curve is A * tau * e
            tau = row["tau"]
            A = row["true_total_lift"] / (tau * np.exp(1))
            
            # Add delay 1-2 minutes
            delay = 1
            for t in range(window_len):
                if t < delay: continue
                t_rel = t - delay
                lift = A * (t_rel/tau) * np.exp(1 - t_rel/tau)
                session_vals[start_idx + t] += lift
                
        return session_vals

    def run(self):
        """Run full simulation."""
        sessions_base = self.get_organic_baseline()
        airings_df = self.generate_airings()
        
        # Inject response
        sessions_treated = self.inject_response_curves(sessions_base.copy(), airings_df)
        
        # Create control market (correlated organic pattern r=0.85)
        # sessions_base is the organic baseline. Control should match it + independent noise.
        # R=0.85 => 0.85*Base + sqrt(1-0.85^2)*IndependentNoise
        control_market = (0.85 * sessions_base) + (np.sqrt(1 - 0.85**2) * np.random.normal(0, np.std(sessions_base), self.n_minutes))
        
        # Create DataFrames
        df_sessions = pd.DataFrame({
            "timestamp": self.minute_index,
            "sessions": sessions_treated.astype(int)
        })
        
        df_control = pd.DataFrame({
            "timestamp": self.minute_index,
            "sessions": control_market.astype(int)
        })
        
        # Create directories
        os.makedirs("tv_attribution/data", exist_ok=True)
        
        # Save files
        df_sessions.to_csv("tv_attribution/data/sessions.csv", index=False)
        airings_df.to_csv("tv_attribution/data/airings.csv", index=False)
        df_control.to_csv("tv_attribution/data/control_market.csv", index=False)
        
        # Save ground truth for validation
        ground_truth = airings_df.to_dict(orient="records")
        # Need to handle Datetime objects for JSON
        for record in ground_truth:
            record["timestamp"] = record["timestamp"].isoformat()
            
        with open("tv_attribution/data/ground_truth.json", "w") as f:
            json.dump(ground_truth, f, indent=4)
            
        print(f"TV Attribution datasets saved to tv_attribution/data/")
        return df_sessions, airings_df

if __name__ == "__main__":
    sim = TVAttributionSimulator()
    sim.run()
