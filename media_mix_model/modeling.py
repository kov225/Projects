import logging
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

# Set up researcher-grade logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MMMWrapper:
    """
    A diagnostic-heavy wrapper for PyMC-Marketing Media Mix Models.
    
    This implementation focuses on Bayesian reproducibility, ensuring that 
    MCMC chains have converged before reporting ROI or budget optimizations.
    """
    
    def __init__(self, data, target_col, date_col, media_cols, control_cols):
        self.data = data
        self.target_col = target_col
        self.date_col = date_col
        self.media_cols = media_cols
        self.control_cols = control_cols
        self.mmm = None
        self.idata = None
        
    def build_model(self):
        """Initializes the MMM with hierarchically-inspired priors and transformations."""
        logger.info("Initializing Bayesian MMM architecture with Geometric Adstock and Logistic Saturation.")
        
        # Configuration reflecting common DTC marketing assumptions:
        # - Saturation: HalfNormal priors ensure non-negative contributions.
        # - Adstock: Beta(2, 2) centers the decay at 0.5 for balanced carryover.
        model_config = {
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
            "saturation_beta": {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
                "dims": "channel"
            },
            "intercept": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 2}},
            "gamma_control": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 1}, "dims": "control"}
        }
        
        # We assume a max lag of 8 weeks for adstock (typical for digital-heavy DTC)
        adstock = GeometricAdstock(l_max=8)
        saturation = LogisticSaturation()
        
        self.mmm = MMM(
            adstock=adstock,
            saturation=saturation,
            date_column=self.date_col,
            channel_columns=self.media_cols,
            control_columns=self.control_cols,
            adstock_max_lag=8,
            model_config=model_config
        )
        
    def fit(self, tune=1000, draws=1000, chains=4):
        """Fits the model using NUTS (No-U-Turn Sampler)."""
        if self.mmm is None:
            self.build_model()
            
        logger.info(f"Sampling {draws} draws across {chains} chains (tune={tune})...")
        
        X = self.data[self.media_cols + self.control_cols + [self.date_col]]
        y = self.data[self.target_col]
        
        self.mmm.fit(
            X=X,
            y=y,
            tune=tune,
            draws=draws,
            chains=chains,
            target_accept=0.9,
            random_seed=42
        )
        self.idata = self.mmm.idata
        
        # Post-fit convergence check
        self.check_convergence()
        return self.idata
    
    def check_convergence(self):
        """Verifies MCMC chain stability using R-hat diagnostics."""
        summary = az.summary(self.idata, var_names=["adstock_alpha", "saturation_beta"])
        max_rhat = summary["r_hat"].max()
        
        if max_rhat > 1.05:
            logger.warning(f"CONVERGENCE WARNING: Max R-hat is {max_rhat:.3f}. Chains may not have mixed well.")
        else:
            logger.info(f"Convergence verified. Max R-hat: {max_rhat:.3f}")
            
    def plot_diagnostics(self):
        """Generates technical diagnostic visualizations for the portfolio."""
        if self.idata is None:
            raise ValueError("Model must be fitted before plotting diagnostics.")
            
        logger.info("Generating Bayesian diagnostics and PPC plots.")
        
        # 1. Trace plots for parameter stability
        az.plot_trace(self.idata, var_names=["adstock_alpha", "saturation_beta"])
        plt.tight_layout()
        plt.savefig("media_mix_model/figures/mcmc_traces.png")
        plt.close()
        
        # 2. Posterior Predictive Check (PPC)
        # Verifies that the model generates data similar to observations
        az.plot_ppc(self.idata, kind="kde")
        plt.savefig("media_mix_model/figures/posterior_predictive_check.png")
        plt.close()
        
        # 3. Forest plot for channel coefficients
        az.plot_forest(self.idata, var_names=["saturation_beta"], combined=True)
        plt.savefig("media_mix_model/figures/channel_coefficients_forest.png")
        plt.close()
        
        return az.summary(self.idata)

if __name__ == "__main__":
    logger.info("MMMWrapper initialized. Ready for Bayesian inference.")
