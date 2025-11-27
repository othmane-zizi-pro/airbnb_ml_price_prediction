"""
Create cluster visualization using PCA for dimensionality reduction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

# Load data
listings = pd.read_csv('listings_with_clusters.csv')

# Load cluster model
with open('cluster_model.pkl', 'rb') as f:
    cluster_data = pickle.load(f)

# Prepare features (same as clustering)
knn_features = [
    'accommodates', 'bedrooms', 'bathrooms',
    'price', 'amenities_count', 'number_of_reviews'
]

# Clean data
listings_clean = listings[knn_features + ['cluster']].fillna(listings[knn_features].median())
X = listings_clean[knn_features]
y = listings_clean['cluster']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Scatter plot with clusters
scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=y, cmap='tab10',
                     alpha=0.6, s=30, edgecolors='none')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax1.set_title('Montreal Airbnb Market Segments (PCA Visualization)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Add cluster centroids
cluster_centers_pca = []
for cluster_id in range(6):
    cluster_points = X_scaled[y == cluster_id]
    if len(cluster_points) > 0:
        center = cluster_points.mean(axis=0)
        center_pca = pca.transform([center])[0]
        cluster_centers_pca.append(center_pca)
        ax1.scatter(center_pca[0], center_pca[1],
                   marker='*', s=500, c='red',
                   edgecolors='black', linewidths=2,
                   zorder=10)
        ax1.annotate(f'C{cluster_id}',
                    (center_pca[0], center_pca[1]),
                    fontsize=10, fontweight='bold',
                    ha='center', va='center')

# Legend
legend1 = ax1.legend(*scatter.legend_elements(num=6),
                    loc="upper right", title="Clusters",
                    framealpha=0.9)
ax1.add_artist(legend1)

# Plot 2: Cluster size distribution
import json
cluster_profiles = json.load(open('cluster_profiles.json'))

cluster_names = [p['name'][:20] + '...' if len(p['name']) > 20 else p['name']
                for p in cluster_profiles]
cluster_sizes = [p['size'] for p in cluster_profiles]
cluster_pcts = [p['pct_of_market'] for p in cluster_profiles]

# Create bar chart
colors = plt.cm.tab10(range(6))
bars = ax2.barh(range(6), cluster_sizes, color=colors, edgecolor='black', linewidth=1.5)

# Add percentage labels
for i, (size, pct) in enumerate(zip(cluster_sizes, cluster_pcts)):
    ax2.text(size + 100, i, f'{pct:.1f}%',
            va='center', fontsize=10, fontweight='bold')

ax2.set_yticks(range(6))
ax2.set_yticklabels([f'C{i}: {name}' for i, name in enumerate(cluster_names)], fontsize=9)
ax2.set_xlabel('Number of Listings', fontsize=12)
ax2.set_title('Market Segment Sizes', fontsize=14, fontweight='bold')
ax2.grid(True, axis='x', alpha=0.3)

# Add total explained variance
total_variance = pca.explained_variance_ratio_.sum()
fig.suptitle(f'Cluster Analysis Visualization (PCA explains {total_variance:.1%} of variance)',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure
plt.savefig('cluster_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Saved cluster_visualization.png")

# Save PCA model for interactive use
with open('cluster_pca.pkl', 'wb') as f:
    pickle.dump({'pca': pca, 'scaler': scaler}, f)
print("✓ Saved cluster_pca.pkl")
