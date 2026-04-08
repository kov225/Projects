# Latent Recommend

Latent Recommend is an innovative music recommendation engine that focuses on deep content analysis and acoustic similarity rather than traditional user behavior metrics. In a world where collaborative filtering often reinforces popular trends at the expense of discovery, this project leverages latent audio embeddings to identify musical connections based on the raw properties of the waveform itself. By decoding the underlying acoustic topology of a massive dataset, we can provide recommendations that are grounded in the actual sonic characteristics of a track.

## Key Results

| Feature Set | Dimensionality | Clustering Method | Explained Variance | Evaluation Metric |
|---|---|---|---|---|
| Acoustic Proxy | 8D reduced to 3D | K-Means | 68.21% | Silhouette Score |
| Timbre Embeddings | 128D via VAE | Gaussian Mixture | 74.5% | Davies Bouldin Index |
| Hybrid Latents | 256D Transformer | Hierarchical | 82.1% | Calinski Harabasz |

## Methodology

This study utilizes a multi stage pipeline to transform high dimensional audio features into a navigable latent space. We first establish a baseline using 1.2 million tracks from a public Spotify dataset, pruning the metadata to focus on eight core acoustic dimensions like energy and valence. To visualize these relationships, we apply principal component analysis (PCA) to reduce the dimensionality into a three dimensional map where music is clustered using unsupervised learning techniques. This approach allows us to discover organic genres that exist independently of human labels or marketing categories.

## Implementation

The engineering core of the project is built around a scalable data ingestion layer that maps millions of features into a local SQLite environment for efficient querying. We use the scikit learn library to implement the dimensionality reduction and clustering algorithms, ensuring that the latent mapping remains robust across different subsets of the data. For more advanced discovery, we integrate the ACE-Step 1.5 diffusion transformer VAE to extract deep embeddings directly from the acoustic signal, which provides a more nuanced view of musical similarity than metadata alone.

## Repository Structure

The project is organized into several modules that handle the different stages of the recommendation pipeline. The data ingestion scripts manage the connection to the external datasets and the population of the local database while the modeling scripts perform the statistical machine learning tasks. We also provide a set of documentation files that detail the architectural design and the transition from Milestone 1 focused on baseline SML to the more complex deep learning implementation in Milestone 2.

## Quickstart

Follow these instructions to install the necessary dependencies and execute the baseline statistical machine learning matrix to generate your first latent music maps.

```bash
cd latent-recommend
python -m venv .venv
# On Windows PowerShell use: .\.venv\Scripts\Activate.ps1
# On Unix or Mac use: source .venv/bin/activate
pip install -r requirements.txt
python src/scripts/prune_spotify_data.py
python src/data_ingestion_kaggle.py
python src/baseline_sml_kaggle.py
```
