# Latent Recommend

**Stack:** Python, scikit-learn (PCA, K-Means), SciPy, SQLite, Pandas.

A small content based music recommender that uses Spotify acoustic features
rather than play count popularity. The motivating question: if a listener
liked one mainstream pop track, can we surface tracks from less played
artists with similar *sonic texture*, without relying on collaborative
filtering signals that overweight already popular catalog?

## Methods

### 1. Quantifying popularity bias
A Welch's t-test (unequal variances) compares the per track popularity score
between "mainstream" and "niche" buckets, segmented by play count quartile.
The very large effect size motivates moving away from popularity ranked
baselines. This step is mostly a sanity check, but it is honest framing for
why a content based approach is worth running at all.

### 2. Latent factor extraction
Five acoustic features (`acousticness`, `danceability`, `energy`,
`valence`, `tempo`) are standardized and projected with PCA. The first two
or three components capture most of the variance and form the "acoustic
neighborhood" in which similarity is computed.

### 3. Recommendation
Given a seed track, recommendations are the cosine similarity nearest
neighbors in the normalized PCA space, with a popularity aware penalty so
that the top results are not dominated by tracks that were already
mainstream. K-Means clustering is also fit so neighborhoods can be inspected
qualitatively.

## Repository layout

```
src/
  baseline_sml.py         PCA, similarity, K-Means
  data_ingestion.py       SQLite metadata + acoustic feature loader
  data_ingestion_kaggle.py Alternate ingestion path for the Kaggle dump
data/                     SQLite database, fitted models
documentations/           Design notes
```

## Reproduction

```bash
pip install -r requirements.txt
python src/data_ingestion.py     # build data/metadata.db
python src/baseline_sml.py       # fit PCA, K-Means, run example queries
```

## Caveats

- This is a content based recommender, not a collaborative one. It cannot
  learn from listener track interactions, only from track features.
- The five acoustic features are coarse. Adding raw audio embeddings (for
  example CLAP or a small CNN over mel spectrograms) is an obvious next step
  but requires the audio itself, not just metadata.
- The popularity penalty is a heuristic; turning it into a calibrated re
  ranking layer would be a more principled fix.
