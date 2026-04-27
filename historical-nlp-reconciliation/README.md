# Historical NLP Reconciliation

**Stack:** Python, RapidFuzz, SciPy (`linear_sum_assignment`), NetworkX,
Docker.

A reconciliation pipeline that aligns AI extracted entity records against a
human curated ground truth from 17th century King's Bench plea rolls. The
underlying question is straightforward but messy: when an OCR/extraction
pipeline reads early modern legal text, can the resulting fragments be
matched back to the right historical individuals despite spelling variation,
occupational titles, and fragmented mentions?

## Why the Hungarian algorithm

A greedy "best match first" pass produces local optima but misallocates
records when several extractions plausibly map to the same person. Framing
reconciliation as a bipartite assignment problem and solving it with the
**Hungarian algorithm (Kuhn and Munkres, 1955)** finds the assignment that
minimizes total cost across the whole record set. SciPy's
`linear_sum_assignment` is used for the solver.

## Scoring function

Each candidate pair is scored by a weighted composite:

| Component | Weight | Source |
|-----------|--------|--------|
| Token sorted name similarity | 0.75 | RapidFuzz `token_sort_ratio` |
| County agreement | +15 bonus | exact match on county field |
| Plea type agreement | +10 bonus | semantic match on pleading category |
| Litigant count divergence | -20 penalty | absolute difference in mentioned parties |

The weights are hand tuned against an annotated dev split. The scoring
function is intentionally interpretable: every reconciliation can be traced
back to which terms drove the match.

### Domain stop words

Occupational titles (`Yeoman`, `Husbandman`, `Spinster`, `Gentleman`,
`Esquire`, and so on) and honorifics dominate the token statistics and would
otherwise inflate the similarity score on truly different people. The
preprocessor strips a curated list of these before fuzzy scoring.

## Repository layout

```
src/
  engine.py        Hungarian reconciliation + scoring composite
  accuracy.py      F1 / precision / recall against held out ground truth
data/              Raw and processed JSON record sets
tests/             Unit tests on the heuristics
Dockerfile
docker-compose.yml
```

## Reproduction

```bash
pip install -r requirements.txt
python -m src.engine
# or, fully containerized:
docker compose up
```

The script writes a reconciliation report and computes precision and recall
against the held out ground truth in `data/`.

## Results

Roughly 1,200 unique entities are reconciled across the curated and
extracted sets. Fuzzy matching plus the Hungarian assignment substantially
outperforms a strict equality baseline on F1 (the capstone version of this
project reports F1 of about 0.49 vs. 0.26 on the same split). The
reconstructed network of plaintiffs and defendants is exported as a
NetworkX graph for downstream social network analysis.

## Caveats

- The scoring weights are not learned. A logistic regression layer that fits
  the weights on the dev set would be more honest than the current hand
  tuning, and is the next planned change.
- Plea type semantic matching uses string matching on a small controlled
  vocabulary. It could be replaced with embeddings, but the vocabulary is
  small enough that returns are likely modest.

## References

- Kuhn, H. W. (1955). *The Hungarian method for the assignment problem.*
  Naval Research Logistics Quarterly.
- Munkres, J. (1957). *Algorithms for the assignment and transportation
  problems.* SIAM Journal.
