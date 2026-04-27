# King's Bench Plea Rolls (KB27/799), HTR Reconciliation

**Capstone project, Lehigh University.**
**Stack:** Python, RapidFuzz, SciPy (`linear_sum_assignment`), NetworkX,
BeautifulSoup, Matplotlib, pytest.

The task: reconcile AI transcribed (Handwritten Text Recognition, HTR)
records of 15th century King's Bench plea rolls against a human curated
ground truth, and use the reconciled records to reconstruct a social
network of plaintiffs and defendants. The capstone version of
[historical-nlp-reconciliation](../../historical-nlp-reconciliation),
extended with two slot bipartite matching, an HTML scraper for the ground
truth, and a network analysis layer.

![Litigation Social Network](assets/benchmark.png)

## What makes the data hard

Most academic datasets are clean by construction. These records are not.

- **Medieval naming conventions.** "John" appears in hundreds of unrelated
  cases under variant spellings (*Johannes*, *Johan*, *Joh'n'*).
  Honorifics and occupational titles dominate token statistics.
- **HTR fragmentation.** A single court case spanning two manuscript pages
  becomes two separate HTR records that must be merged back together.
- **No common schema.** The ground truth is HTML scraped from a published
  edition; the HTR output is nested JSON. Both required custom parsers.
- **Threshold sensitivity.** The similarity threshold has to be tuned
  empirically against an annotated dev split.

## Results

| Metric | Result |
|--------|--------|
| Total ground truth cases processed | 909 |
| Unique individuals identified | 1,200+ |
| Matching strategy | two slot bipartite (Hungarian) |
| Strict F1 | 0.262 |
| Fuzzy F1 (threshold = 80) | **0.491** |
| Precision / recall (fuzzy) | 0.494 / 0.489 |
| Improvement over strict baseline | +87.5 percent F1 |
| Tests passing | 5 / 5 (`pytest`) |

The fuzzy F1 of about 0.49 is roughly twice the strict baseline. Most of
the gain came from grouping defendant tokens into "Person" entities before
scoring, rather than from any change to the matching algorithm itself.

## Pipeline

### Similarity engine (`similarity.py`)
Three fuzzy strategies are blended:

- `token_sort_ratio`. Handles word order differences.
- `token_set_ratio`. Handles subset/superset name lists.
- `partial_ratio`. Handles abbreviated names.

A soft size penalty discounts matches where the ground truth lists more
defendants than HTR managed to extract, without hard rejecting them.

### Two slot bipartite matching (`reconciliation.py`)
Each ground truth case is represented by **two rows** in the cost matrix.
That allows the Hungarian solver to assign up to two HTR fragments to a
single ground truth case, which is what the data needs because cases
frequently span page boundaries.

### Network analysis (`analysis.py`, `versatile_digraph.py`)
Reconciled litigants are loaded into a small in house digraph class and
the resulting network is summarized by degree centrality and ego subgraphs.
The visualization in `generate_network_plot.py` was useful for sanity
checks: well connected hubs almost always corresponded to recognizable
gentry families in the source archive.

## Repository layout

```
scraper.py                  HTML ground truth + JSON HTR parsing
similarity.py               Fuzzy similarity engine (3 strategies + size penalty)
reconciliation.py           Two slot Hungarian bipartite matching
analysis.py                 Strict + fuzzy F1, social network builder
versatile_digraph.py        Light digraph class with centrality + ego subgraph
generate_network_plot.py    NetworkX -> litigation_network.png
tests/test_similarity.py    Unit tests on the similarity engine
data/                       Cached ground truth + raw HTR output
```

## Reproduction

```bash
pip install -r requirements.txt
python analysis.py                     # full pipeline + evaluation
python generate_network_plot.py        # social network visualization
pytest tests/ -v
```

## Next steps

- Replace the fuzzy heuristics with name embeddings fine tuned on medieval
  Latin. The current weights are hand tuned; a small learned scorer should
  generalize better across volumes.
- Extend coverage from KB27/799 to the full King's Bench archive (about
  10,000+ records).
- Publish the reconciled dataset and the resulting social network as an
  open academic resource.

## References

- Kuhn, H. W. (1955). *The Hungarian method for the assignment problem.*
- Munkres, J. (1957). *Algorithms for the assignment and transportation
  problems.*
- RapidFuzz documentation (token set / token sort ratio definitions).
