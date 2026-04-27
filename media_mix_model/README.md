# Bayesian Media Mix Model

**Stack:** PyMC-Marketing, PyMC, ArviZ, NumPy, Pandas, SciPy.

A Bayesian Media Mix Model (MMM) on simulated direct to consumer marketing
data. The goal is to decompose conversions across channels (Search, Social,
TV, Display, Affiliate) while accounting for two effects that almost always
break frequentist MMMs: the **carryover** of spend across weeks (adstock)
and the **diminishing returns** of additional spend (saturation).

Working in the Bayesian setting matters because adstock and saturation
parameters are weakly identified from spend alone. Informative priors
stabilize the estimates and produce a full posterior over channel ROI rather
than a single point estimate.

## Methods

### Geometric adstock
Carryover is modeled with a single decay parameter `alpha` per channel:

```
Y_t = X_t + alpha * Y_{t-1},   alpha in [0, 1]
```

I use a `Beta(2, 2)` prior, encoding moderate but uncertain carryover
appropriate for digital channels. Heavier tailed channels like TV would be
better served by a `Beta(5, 2)`.

### Logistic saturation
Diminishing returns are modeled as a logistic curve in spend, with channel
specific half saturation and steepness parameters. This prevents the
posterior from extrapolating linearly when a channel is pushed past its
observed range.

### Posterior diagnostics
The fit script reports:

- `R_hat` for every parameter (target < 1.05).
- Effective sample size (bulk and tail) for the channel coefficients.
- Posterior predictive checks against held out weeks.

If any chain fails `R_hat < 1.05`, the script raises rather than producing a
silently bad model.

### Budget optimizer (prototype)
`optimizer.py` formulates allocation as a constrained optimization. The goal
is to maximize expected weekly conversions subject to a total spend budget
and per channel floors and caps, evaluating the objective by sampling from
the posterior of the fitted MMM. The current SciPy `SLSQP` driver is a
working prototype. See *Limitations* below.

## Repository layout

```
modeling.py    PyMC-Marketing wrapper with diagnostic checks
simulator.py   DTC data generator (trend + seasonality + channel specific noise)
optimizer.py   Posterior aware budget allocation (prototype)
notebooks/
  analysis.ipynb  EDA, posterior interpretation, ROI plots
tests/         Unit tests for the simulator and adstock/saturation transforms
```

## Reproduction

```bash
pip install -r requirements.txt

python simulator.py     # generate data/marketing_data.csv
python modeling.py      # fit MMM, save posterior, run diagnostics
python optimizer.py     # allocate a fixed weekly budget across channels
```

## Limitations

- **The optimizer is not finished.** The SLSQP call is wired up, but the
  saturation gradient used in the objective is still being verified, and
  gradient evaluation through the posterior samples is currently slow. I
  expect to either move to a vectorized JAX objective or fall back to
  `scipy.optimize.minimize` with `method="trust-constr"` and finite
  differences.
- **Synthetic data.** The simulator is meant to give the diagnostics
  something to bite on, not to mimic any real brand. Channel ROI plots from
  this run should not be read as substantive marketing claims.
- **Prior sensitivity.** I have not yet run a formal prior predictive check
  sweep across plausible `Beta(a, b)` choices for adstock; that is the next
  thing I want to add.

## References

- Jin et al. (2017). *Bayesian Methods for Media Mix Modeling with Carryover
  and Shape Effects.* Google Research.
- PyMC-Marketing documentation (Aesara backend, MMM module).
