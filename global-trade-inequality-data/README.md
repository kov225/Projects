# Global Trade Inequality

**Stack:** Python, Pandas, NumPy, SciPy, Matplotlib, Jupyter.
**Co-author:** Group project with Cheema (statistics milestone, Lehigh).

A notebook driven study of structural inequality in international trade,
combined with a smaller follow on study on cross tier software pricing. The
two notebooks share statistical machinery from the same `src/` module so the
inequality measures are computed identically across both.

## Methods

### Gini coefficient
Used to measure the dispersion of trade value across actors:

```
G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mean(x))
```

The normalization makes the index comparable across product categories and
time periods regardless of the absolute scale of trade.

### Herfindahl Hirschman Index (HHI)
Concentration of market share, computed as the sum of squared shares. Used
to flag commodities where a handful of countries dominate exports, which is
a signal of trade bottleneck risk.

### Welch's t-tests on pricing tiers
The companion software pricing notebook uses Welch's t-tests to test whether
licensing prices differ significantly across economic tiers, with effect
sizes reported alongside p-values.

## Repository layout

```
src/
  inequality_metrics.py     Gini and HHI implementations
Cheema_Vennalakanti_Global Trade Inequality.ipynb     Main research notebook
Vennalakanti_Koushik_softwarePricing.ipynb           Pricing follow-up
*.pdf                       Formal write-ups
```

## Reproduction

```bash
python src/inequality_metrics.py        # smoke test on bundled data
jupyter lab                              # open either notebook
```

## Notes

This project is mostly notebook driven: the `src/` module is intentionally
minimal and exists so the inequality calculations can be reused outside the
notebooks. The PDFs in this directory are the formal milestone write-ups
submitted with the project.
