import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

class TVAttributor:
    """
    Spot-level attribution engine for TV ad airings.
    Performs baseline estimation, lift extraction, and parametric curve fitting.
    """
    
    def __init__(self, sessions_df, airings_df):
        self.sessions = sessions_df
        # Ensure timestamp is datetime
        self.sessions["timestamp"] = pd.to_datetime(self.sessions["timestamp"])
        self.sessions = self.sessions.set_index("timestamp")
        
        self.airings = airings_df
        self.airings["timestamp"] = pd.to_datetime(self.airings["timestamp"])
        
    def get_noise_floor(self):
        """Calculates standard deviation of sessions during quiet periods."""
        # Simple implementation: global std
        return self.sessions["sessions"].std()

    def response_model(self, t, A, tau):
        """Parametric response curve: A * (t/tau) * exp(1 - t/tau)."""
        # Avoid division by zero
        return A * (t / (tau + 1e-9)) * np.exp(1 - t / (tau + 1e-9))

    def attribute_spot(self, ts, window_len=15, pre_window=20):
        """
        Attributes incremental sessions to a single spot at timestamp ts.
        """
        # 1. Extract windows
        end_ts = ts + pd.Timedelta(minutes=window_len)
        start_pre = ts - pd.Timedelta(minutes=pre_window)
        
        try:
            pre_data = self.sessions.loc[start_pre : ts - pd.Timedelta(minutes=1)]
            post_data = self.sessions.loc[ts : end_ts]
            
            if len(pre_data) < 10 or len(post_data) < 5:
                return None
        except KeyError:
            return None
            
        # 2. Baseline estimation (Local Linear Trend)
        X_pre = np.arange(len(pre_data)).reshape(-1, 1)
        y_pre = pre_data["sessions"].values
        model = LinearRegression().fit(X_pre, y_pre)
        
        # Extrapolate baseline
        X_post = np.arange(len(pre_data), len(pre_data) + len(post_data)).reshape(-1, 1)
        baseline = model.predict(X_post)
        
        # 3. Observed lift
        observed = post_data["sessions"].values
        lift_series = observed - baseline
        total_lift = np.sum(lift_series)
        
        # 4. Statistical significance (Z-score)
        noise_std = self.get_noise_floor()
        cum_noise_std = noise_std * np.sqrt(len(post_data))
        z_score = total_lift / cum_noise_std
        
        # 5. Curve fitting
        t_vals = np.arange(len(lift_series))
        fitted_params = [0, 0]
        try:
            # Bounds: A > 0, tau > 0
            popt, _ = curve_fit(self.response_model, t_vals, lift_series, p0=[np.max(lift_series), 2.5], bounds=(0, [np.inf, 10]))
            fitted_params = popt
        except:
            pass
            
        return {
            "total_lift": total_lift,
            "z_score": z_score,
            "is_significant": z_score > 2.0,
            "fitted_A": fitted_params[0],
            "fitted_tau": fitted_params[1]
        }

    def run_attribution_pipeline(self):
        """Runs attribution for all airings in the log."""
        results = []
        for idx, row in self.airings.iterrows():
            attr = self.attribute_spot(row["timestamp"])
            if attr:
                res = {**row.to_dict(), **attr}
                results.append(res)
            else:
                results.append(row.to_dict())
                
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Smoke test logic
    print("TVAttributor module loaded.")
