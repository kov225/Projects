import pandas as pd
import numpy as np
import pymc as pm
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior

class MMMWrapper:
    """
    Wrapper for PyMC-Marketing MMM to simplify fitting and diagnostics.
    Tailored for DTC brand analytics with specific adstock and saturation priors.
    """
    
    def __init__(self, data, target_col, date_col, media_cols, control_cols):
        self.data = data
        self.target_col = target_col
        self.date_col = date_col
        self.media_cols = media_cols
        self.control_cols = control_cols
        self.mmm = None
        
    def build_model(self):
        """Initializes the MMM with custom transformations and priors."""
        
        # User requested specific priors:
        # - channel coefficients: HalfNormal
        # - adstock alpha: Beta(2, 2)
        # - saturation lambda: Gamma
        
        model_config = {
            # Media transformation priors
            "adstock_alpha": {
                "dist": "Beta",
                "kwargs": {"alpha": 2, "beta": 2},
                "dims": "channel"
            },
            "saturation_lam": {
                "dist": "Gamma",
                "kwargs": {"alpha": 3, "beta": 1},
                "dims": "channel"
            },
            # Channel coefficients (beta)
            "saturation_beta": {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
                "dims": "channel"
            },
            # Intercept and control priors
            "intercept": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 2}},
            "gamma_control": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 1}, "dims": "control"}
        }
        
        adstock = GeometricAdstock(l_max=8)
        saturation = LogisticSaturation()
        
        self.mmm = MMM(
            adstock=adstock,
            saturation=saturation,
            date_to_index=True,
            model_config=model_config
        )
        
    def fit(self, tune=2000, draws=2000, chains=4):
        """Fits the model using MCMC."""
        if self.mmm is None:
            self.build_model()
            
        X = self.data[self.media_cols + self.control_cols]
        y = self.data[self.target_col]
        
        # In a real scenario, we'd pass the dataframe. 
        # pymc-marketing MMM.fit expects X and y.
        # Note: we need to handle the date column correctly if required by MMM.
        # MMM typically handles feature data as X.
        
        self.mmm.fit(
            X=X,
            y=y,
            tune=tune,
            draws=draws,
            chains=chains,
            target_accept=0.9
        )
        return self.mmm.idata
    
    def plot_diagnostics(self):
        """Generates standard MCMC diagnostics."""
        import arviz as az
        import matplotlib.pyplot as plt
        
        idata = self.mmm.idata
        
        # Trace plots
        az.plot_trace(idata, var_names=["adstock_alpha", "saturation_lam", "saturation_beta"])
        plt.tight_layout()
        plt.savefig("media_mix_model/figures/mcmc_traces.png")
        plt.close()
        
        # Summary statistics (R-hat, n_eff)
        summary = az.summary(idata, var_names=["adstock_alpha", "saturation_lam", "saturation_beta"])
        summary.to_csv("media_mix_model/data/model_diagnostics.csv")
        
        return summary

if __name__ == "__main__":
    # Smoke test initialization
    # In practice, this is called from the notebooks
    print("MMMWrapper module loaded.")
