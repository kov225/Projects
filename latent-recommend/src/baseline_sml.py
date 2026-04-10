import sqlite3
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

# Professional Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'metadata.db')

def load_data():
    """Retrieves track features from SQL store."""
    if not os.path.exists(DB_PATH):
        logger.error("Data store not found. Ensure ingestion pipeline has executed.")
        raise FileNotFoundError("Database not found.")
    
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT t.track_id, t.track_name, t.popularity, t.genre,
               f.acousticness, f.danceability, f.energy, f.valence, f.tempo
        FROM tracks t
        JOIN acoustic_features f ON t.track_id = f.track_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def analyze_structural_bias(df):
    """
    Quantifies the discrepancy between mainstream and niche distributions.
    
    Hypothesis: The popularity distribution of 'Mainstream' vs 'Indie/Ambient' 
    is not IID, suggesting systemic platform bias.
    """
    logger.info("Executing Hypothesis Testing: Popularity Bias Distribution...")
    
    # Welch's T-Test (does not assume equal variance)
    pop_group = df[df['genre'] == 'Pop-Mainstream']['popularity']
    niche_group = df[df['genre'].isin(['Ambient', 'Dub-Techno'])]['popularity']
    
    t_stat, p_val = stats.ttest_ind(pop_group, niche_group, equal_var=False)
    logger.info(f"Welch's T-Test | t-stat: {t_stat:.3f} | p-value: {p_val:.2e}")
    
    if p_val < 0.05:
        logger.info("Statistical Significance Confirmed: Significant divergence in popularity distributions.")

def run_latent_discovery_engine(df):
    """
    Generates a latent topological map of acoustic features.
    
    Uses PCA for dimensionality reduction and K-Means for unsupervised 
    neighborhood discovery. This bypasses popularity bias by focusing on 
    the raw acoustic manifold.
    """
    logger.info("Projecting acoustic features into latent space...")
    
    feature_cols = ['acousticness', 'danceability', 'energy', 'valence', 'tempo']
    X = df[feature_cols].values
    
    # Z-score normalization for scale independence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Latent Factor Extraction (PCA)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'], df['pca2'] = X_pca[:, 0], X_pca[:, 1]
    
    logger.info(f"Latent Explained Variance: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    # 2. Neighborhood Discovery (Clustering)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    return X_scaled

def recommend_niche_alternatives(df, X_latent, track_index, n_recs=5):
    """
    Implements a Latent Discovery Recommender.
    
    Given a mainstream track, finds the nearest niche 'acoustic neighbors' 
    using cosine similarity in the latent manifest space.
    """
    logger.info(f"Computing latent neighbors for: {df.iloc[track_index]['track_name']}")
    
    # Compute Cosine Similarity across the feature space
    similarities = cosine_similarity(X_latent[track_index].reshape(1, -1), X_latent).flatten()
    
    # Filter for niche tracks only (Ambient/Dub) to demonstrate discovery
    niche_mask = df['genre'].isin(['Ambient', 'Dub-Techno'])
    
    # Get top N niche tracks by similarity
    recommendation_indices = np.argsort(similarities[niche_mask])[::-1][:n_recs]
    recommendations = df[niche_mask].iloc[recommendation_indices]
    
    print("\n" + "="*40)
    print(f"LATENT DISCOVERY: Neighbors for '{df.iloc[track_index]['track_name']}'")
    print("="*40)
    for _, row in recommendations.iterrows():
        print(f"• {row['track_name']} ({row['genre']}) | Similarity: {similarities[row.name]:.3f}")
    print("="*40)

if __name__ == "__main__":
    data = load_data()
    analyze_structural_bias(data)
    X_latent = run_latent_discovery_engine(data)
    
    # Recommend based on a sample pop track if it exists
    pop_samples = data[data['genre'] == 'Pop-Mainstream'].index
    if not pop_samples.empty:
        recommend_niche_alternatives(data, X_latent, track_index=pop_samples[0])
