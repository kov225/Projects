# Bing Experimentation Suite

**Status:** Independent study (Spring 2025).
**Stack:** Python, NumPy, SciPy, Statsmodels, Pandas, Plotly Dash, Pytest.

A small experimentation library implementing variance reduction estimators
for online A/B tests on simulated telemetry. The motivation was to understand
the statistical machinery that platforms like Bing, Booking, and Netflix
actually deploy, and to validate end to end that the estimators behave as the
papers claim.

## What is implemented

### CUPED (Deng, Xu, Kohavi, Walker, 2013)
Adjusts a post treatment metric `Y` by a pre experiment covariate `X`:

```
Y_cuped = Y - theta * (X - E[X])
theta*  = Cov(Y, X) / Var(X)
```

The variance of the adjusted estimator is `(1 - rho^2)` times the original,
where `rho` is the sample correlation between `X` and `Y`. The test in
`tests/test_cuped.py` verifies this reduction empirically.

### Post stratification
Treatment effects are estimated within strata (geo, device, tier) and
combined as a weighted average of the within stratum effects. This corrects
accidental imbalances in randomization and follows Miratrix, Sekhon, and Yu
(2013).

### Novelty effect detection
Fits an exponential decay model of the form `lift(t) = a * exp(-b * t) + c`
to the per day lift series. A series is flagged as novelty driven when the
decay constant `b` is statistically distinguishable from zero. The detector
helps distinguish true treatment effects from the transient excitement that
accompanies any new UI.

### A/B test core
Welch's t-test plus a non parametric bootstrap for cases where normality
fails (skewed engagement metrics, zero inflated revenue, etc.). The bootstrap
path is slower but does not require the sampling distribution assumption.

### Variance benchmark
`experiments/variance_benchmark.py` runs all estimators against the same
generated telemetry and reports the empirical variance reduction so the
methods can be compared on identical data.

## Repository layout

```
experiments/
  ab_test.py             Welch's t-test + bootstrap
  cuped.py               CUPED variance reduction
  stratification.py      Post stratification + regression adjustment
  novelty.py             Exponential decay novelty detector
  variance_benchmark.py  Comparative harness
data/
  generate.py            Synthetic telemetry with structured noise
dashboard/               Plotly Dash views (3 pages)
tests/                   11 unit tests covering A/A validity, CUPED, novelty
```

## Running it

```bash
pip install -r requirements.txt
python -m experiments.variance_benchmark
pytest -q
```

The benchmark prints variance reduction for each estimator on a freshly
simulated dataset.

## Validation

- **A/A test.** Under no treatment effect, p-values are uniformly distributed
  on `[0, 1]` (Kolmogorov-Smirnov check in the tests).
- **Power curves.** Minimum detectable effect (MDE) is reported for each
  estimator at fixed sample size.
- **Bootstrap parity.** Frequentist and bootstrap p-values agree within
  Monte Carlo error on well behaved data.

## Limitations and next steps

- An earlier draft of the README mentioned heterogeneous treatment effect
  (HTE) estimation. That is not yet implemented. The suite reports global and
  stratified effects, but no causal tree or meta learner code is in place.
  This is the next addition I plan to make.
- The Plotly Dash views are intentionally minimal; the focus has been on the
  estimators rather than the front end.

## References

- Deng, Xu, Kohavi, Walker (2013). *Improving the sensitivity of online
  controlled experiments by utilizing pre-experiment data.* WSDM.
- Miratrix, Sekhon, Yu (2013). *Adjusting treatment effect estimates by
  post-stratification in randomized experiments.* JRSS-B.
- Kohavi, Tang, Xu (2020). *Trustworthy Online Controlled Experiments.* CUP.
