import logging
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# Standard portfolio logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TVAttributor:
    """
    Spot-level attribution engine for linear TV airings.
    
    This system implements a parametric response model to extract incremental 
    sessions from minute-level telemetry. It utilizes local linear baselines 
    and bootstrap resampling to certify the statistical significance of 
    attribution results.
    """
    
    def __init__(self, sessions_df, airings_df):
        self.sessions = sessions_df
        # Standardize timing data
        self.sessions["timestamp"] = pd.to_datetime(self.sessions["timestamp"])
        self.sessions = self.sessions.set_index("timestamp")
        
        self.airings = airings_df
        self.airings["timestamp"] = pd.to_datetime(self.airings["timestamp"])
        
    def get_noise_floor(self, window_mins=30):
        """Calculates the rolling noise floor to define the significance threshold."""
        return self.sessions["sessions"].rolling(window=window_mins).std().median()

    def response_model(self, t, A, tau):
        """
        Parametric Response Curve: A * (t/tau) * exp(1 - t/tau).
        
        This models the 'immediate impact followed by exponential decay' pattern 
        typical of Direct-to-Consumer TV response.
        """
        return A * (t / (tau + 1e-9)) * np.exp(1 - t / (tau + 1e-9))

    def attribute_spot(self, ts, window_len=15, pre_window=20, n_bootstrap=1000):
        """
        Infers incremental lift for a specific airing timestamp (ts).
        
        Methodology:
        1. Local Linear Trend extraction for baseline estimation.
        2. Signal subtraction to isolate the attribution 'spike'.
        3. Bootstrap resampling to generate 95% Confidence Intervals for total lift.
        4. Parametric curve fitting to recover response shape (A, tau).
        """
        # Ensure we have data for the window
        end_ts = ts + pd.Timedelta(minutes=window_len)
        start_pre = ts - pd.Timedelta(minutes=pre_window)
        
        try:
            pre_data = self.sessions.loc[start_pre : ts - pd.Timedelta(minutes=1)]
            post_data = self.sessions.loc[ts : end_ts]
            
            if len(pre_data) < 10 or len(post_data) < 5:
                return None
        except (KeyError, ValueError):
            return None
            
        # 1. Baseline Estimation (Local Linear Extrapolation)
        # We fit a trend to the pre-window and predict it into the post-window
        X_pre = np.arange(len(pre_data)).reshape(-1, 1)
        y_pre = pre_data["sessions"].values
        model = LinearRegression().fit(X_pre, y_pre)
        
        X_post = np.arange(len(pre_data), len(pre_data) + len(post_data)).reshape(-1, 1)
        baseline = model.predict(X_post)
        
        # 2. Extract Obsereved Signal
        observed = post_data["sessions"].values
        lift_series = observed - baseline
        total_lift_obs = np.sum(lift_series)
        
        # 3. Bootstrap Confidence Intervals
        # Following Efron (1979), we resample residuals to quantify uncertainty
        residuals = y_pre - model.predict(X_pre)
        boot_lifts = []
        for _ in range(n_bootstrap):
            boot_res = np.random.choice(residuals, size=len(post_data), replace=True)
            boot_lifts.append(np.sum(lift_series + boot_res))
            
        ci_low, ci_high = np.percentile(boot_lifts, [2.5, 97.5])
        
        # 4. Parametric Curve Fitting (Shape Recovery)
        t_vals = np.arange(len(lift_series))
        fitted_params = [0, 0]
        try:
            # Bounds: Intensity (A) > 0, Decay (tau) > 0
            popt, _ = curve_fit(
                self.response_model, t_vals, lift_series, 
                p0=[np.max(lift_series), 2.5], 
                bounds=(0, [np.inf, window_len])
            )
            fitted_params = popt
        except Exception as e:
            logger.debug(f"Curve fit failed for spot {ts}: {str(e)}")
            
        return {
            "total_lift": total_lift_obs,
            "ci_95": (ci_low, ci_high),
            "is_significant": ci_low > 0, # Lift is significant if CI does not cross zero
            "intensity_A": fitted_params[0],
            "decay_tau": fitted_params[1]
        }

    def run_attribution_pipeline(self):
        """Executes the full attribution sequence for the provided airings log."""
        logger.info(f"Starting attribution for {len(self.airings)} airings...")
        results = []
        
        for idx, row in self.airings.iterrows():
            attr = self.attribute_spot(row["timestamp"])
            if attr:
                res = {**row.to_dict(), **attr}
                results.append(res)
            else:
                results.append({**row.to_dict(), "total_lift": 0.0, "is_significant": False})
                
        df_results = pd.DataFrame(results)
        logger.info(f"Attribution complete. Significant spots found: {df_results['is_significant'].sum()}")
        return df_results

if __name__ == "__main__":
    logger.info("TVAttributor engine ready. Awaiting telemetry ingestion.")
