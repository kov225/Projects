# TV Ad Attribution

**Stack:** Python, NumPy, SciPy (non-linear least squares + bootstrap),
Pandas, tfcausalimpact (notebook).

A short window attribution engine for linear TV airings that estimates
incremental site sessions per spot by comparing observed minute level traffic
to a local linear counterfactual. The CausalImpact notebook validates the
campaign level result with a Bayesian Structural Time Series model fit on
correlated control markets.

## Methods

### Local linear counterfactual
For each airing, the script fits an OLS line to the 20 minute pre roll window
and projects it forward across a 15 minute post roll window. The projection
is the counterfactual; the residual against observed traffic is the
**incremental signal**. This handles intra day seasonality and short run
trend without assuming a global time series model.

### Residual bootstrap
Because minute level web traffic is noisy and far from Gaussian, the script
resamples pre airing residuals (Efron, 1979) to build a 95 percent confidence
interval around the per spot lift. A spot is reported as significant only
when the interval excludes zero.

### Parametric response curve
The aggregated lift trajectory is fit to a one parameter Pearson IV style
response curve:

```
L(t) = A * (t / tau) * exp(1 - t / tau)
```

`A` is the peak response, `tau` the time to peak / decay constant. The fit
is done with `scipy.optimize.curve_fit` and gives a clean way to compare
networks: a small `tau` is a fast, short burst response; a large `tau` is
slower but longer lived.

### Cross validation with CausalImpact
`notebooks/03_causal_impact.ipynb` runs Brodersen et al. (2015) BSTS at the
campaign level, using a correlated unaired DMA as the synthetic control.
This is a coarser, campaign level check on whether the per spot lift
aggregates into something the BSTS model can detect from market level data.

## Repository layout

```
attribution.py        Per spot bootstrap lift + parametric curve fit
simulator.py          Minute level session generator with airing impulses
notebooks/
  01_eda.ipynb              Exploratory checks on the simulated traffic
  02_attribution.ipynb      Walkthrough of the per spot pipeline
  03_causal_impact.ipynb    Campaign level BSTS cross check
  04_scorecard.ipynb        Network level summaries
data/                 Airing logs and minute level telemetry
tests/                Unit tests on the bootstrap and curve fitting
```

## Reproduction

```bash
pip install -r requirements.txt
python simulator.py
python attribution.py
```

The script writes a network level scorecard with CPIS (cost per incremental
session), response half life, and a daypart heatmap.

## Caveats

- **"Minute level" refers to telemetry resolution, not causal granularity.**
  The attribution windows are 15 minutes wide; finer windows are too noisy
  for the bootstrap interval to be informative.
- **Synthetic data only.** The lift magnitudes here are useful for sanity
  checking the math, not for benchmarking against any real network.
- **CausalImpact is in a notebook, not in `attribution.py`.** Promoting it
  into the core library is a planned next step.

## References

- Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife.*
  Annals of Statistics.
- Brodersen et al. (2015). *Inferring causal impact using Bayesian
  structural time-series models.* Annals of Applied Statistics.
