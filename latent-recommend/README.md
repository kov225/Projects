# 🎵 Latent Recommend: Acoustic Topological Discovery

This project implements a latent-factor recommendation engine designed to bypass popularity bias in music discovery. It uses dimensionality reduction and unsupervised clustering to map the "acoustic manifold" of tracks, enabling the discovery of niche content based on structural similarity rather than mainstream popularity.

## 🧠 Methodology: Mapping the Acoustic Manifold

Modern recommendation systems often suffer from the "Rich Get Richer" effect (popularity bias), where highly-rated content is over-exposed. Our engine focuses on the *latent acoustic properties* of music.

### 1. Structural Bias Quantification
We utilize **Welch's T-Test** to formally quantify the discrepancy between 'Mainstream' and 'Niche' distributions. This establishes the statistical necessity for a non-popularity-based discovery engine.

### 2. Latent Factor Extraction
We project high-dimensional acoustic features (Acousticness, Danceability, Energy, Valence, Tempo) into a lower-dimensional latent space using **Principal Component Analysis (PCA)**. This isolates the primary variance components that define a 'musical neighborhood'.

### 3. Topological Recommendation
Once the latent manifold is established, recommendations are generated using **Cosine Similarity** in the normalized feature space.
- **Goal**: Given a mainstream pop track, find the nearest neighbors in the 'Ambient' or 'Indie' subspaces.
- **Result**: A cross-genre discovery engine that prioritizes *sonic texture* over *market reach*.

## 🛠️ Project Structure

```text
├── src/
│   ├── baseline_sml.py    # Core Discovery Engine & Latent Mapping
│   ├── data_ingestion.py   # SQL Pipeline for track metadata
├── data/                  # SQLite store and serialized models
└── notebooks/             # Exploratory Analysis & Cluster Visualization
```

## 🚀 Usage

1. **Ingest Data**:
   ```bash
   python src/data_ingestion.py
   ```

2. **Run Discovery Pipeline**:
   ```bash
   python src/baseline_sml.py
   ```

---
*Developed as part of my Applied Data Science & ML Engineering Portfolio.*
