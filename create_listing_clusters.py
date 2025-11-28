"""
LISTING CLUSTERING FOR AIRBNB MONTREAL
=====================================================

This creates ACTIONABLE market segments that hosts can actually use to:
1. Understand their competitive position
2. Optimize mutable characteristics
3. Set realistic price expectations
4. Improve their listing strategically

Business Logic:
- Cluster by IMMUTABLE characteristics (location, size, type)
- Profile by PERFORMANCE (price, ratings, bookings)
- Recommend based on MUTABLE opportunities (amenities, response time, etc.)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pickle
import json

print("="*80)
print("LOADING DATA")
print("="*80)

# Load listings
listings = pd.read_csv('listings.csv')

# Clean price
if listings['price'].dtype == 'object':
    listings['price'] = listings['price'].str.replace('$', '', regex=False)
    listings['price'] = listings['price'].str.replace(',', '', regex=False)
    listings['price'] = pd.to_numeric(listings['price'], errors='coerce')

# Parse bathrooms
if 'bathrooms_text' in listings.columns:
    listings['bathrooms'] = listings['bathrooms_text'].str.extract(r'(\d+\.?\d*)')[0].astype(float)

# Clean host_acceptance_rate
if 'host_acceptance_rate' in listings.columns and listings['host_acceptance_rate'].dtype == 'object':
    listings['host_acceptance_rate'] = listings['host_acceptance_rate'].str.replace('%', '', regex=False)
    listings['host_acceptance_rate'] = pd.to_numeric(listings['host_acceptance_rate'], errors='coerce') / 100

# Count amenities
if 'amenities' in listings.columns:
    listings['amenities_count'] = listings['amenities'].str.count(',') + 1
    listings.loc[listings['amenities'].isna(), 'amenities_count'] = 0

print(f"✓ Loaded {len(listings)} listings")

# ============================================================================
# STEP 1: DEFINE CLUSTERING FEATURES (Immutable + Key Performance)
# ============================================================================

print("\n" + "="*80)
print("STEP 1: FEATURE ENGINEERING FOR CLUSTERING")
print("="*80)

# Immutable characteristics
listings['location_tier'] = listings['neighbourhood_cleansed'].map({
    'Ville-Marie': 3, 'Le Plateau-Mont-Royal': 3,  # Premium
    'Rosemont-La Petite-Patrie': 2, 'Le Sud-Ouest': 2, 'Côte-des-Neiges-Notre-Dame-de-Grâce': 2,  # Mid
}).fillna(1)  # Budget

# Size category
listings['size_category'] = pd.cut(
    listings['accommodates'].fillna(2),
    bins=[0, 2, 4, 6, 100],
    labels=['Studio', 'Small', 'Medium', 'Large']
)

# Property type simplified
def categorize_property(prop_type):
    if pd.isna(prop_type):
        return 'Other'
    prop_lower = str(prop_type).lower()
    if 'entire' in prop_lower or 'apartment' in prop_lower or 'condo' in prop_lower:
        return 'Entire Place'
    elif 'private' in prop_lower:
        return 'Private Room'
    elif 'shared' in prop_lower:
        return 'Shared Room'
    else:
        return 'Other'

listings['property_category'] = listings['room_type'].apply(categorize_property)

# Amenity tier
listings['amenity_tier'] = pd.cut(
    listings['amenities_count'].fillna(0),
    bins=[0, 15, 30, 50, 200],
    labels=['Basic', 'Standard', 'Enhanced', 'Premium']
)

# Quality tier (based on reviews)
listings['quality_tier'] = pd.cut(
    listings['review_scores_rating'].fillna(4.0),
    bins=[0, 4.0, 4.5, 4.8, 5.1],
    labels=['Developing', 'Good', 'Excellent', 'Outstanding']
)

# Clustering features (mix of immutable and performance)
clustering_features = [
    'accommodates',
    'bedrooms',
    'bathrooms',
    'location_tier',
    'amenities_count',
    'number_of_reviews',
    'review_scores_rating',
    'price'
]

# Prepare clustering data
cluster_df = listings[clustering_features].copy()
cluster_df = cluster_df.fillna(cluster_df.median())

print(f"✓ Prepared {len(clustering_features)} features for clustering")

# ============================================================================
# STEP 2: OPTIMAL NUMBER OF CLUSTERS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: FINDING OPTIMAL NUMBER OF CLUSTERS")
print("="*80)

# Standardize
scaler = StandardScaler()
cluster_df_scaled = scaler.fit_transform(cluster_df)

# Test different k values
silhouette_scores = []
for k in range(3, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(cluster_df_scaled)
    score = silhouette_score(cluster_df_scaled, labels)
    silhouette_scores.append((k, score))
    print(f"  k={k}: Silhouette Score = {score:.3f}")

# Choose k with best silhouette score
optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
optimal_score = max(silhouette_scores, key=lambda x: x[1])[1]
print(f"\n✓ Optimal clusters: {optimal_k} (silhouette score: {optimal_score:.3f})")

# ============================================================================
# STEP 3: FINAL CLUSTERING
# ============================================================================

print("\n" + "="*80)
print("STEP 3: CREATING LISTING SEGMENTS")
print("="*80)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
listings['cluster'] = kmeans_final.fit_predict(cluster_df_scaled)

print(f"✓ Assigned {len(listings)} listings to {optimal_k} clusters")

# ============================================================================
# STEP 4: CLUSTER PROFILING & BUSINESS NAMING
# ============================================================================

print("\n" + "="*80)
print("STEP 4: CLUSTER PROFILING")
print("="*80)

cluster_profiles = []

for cluster_id in range(optimal_k):
    cluster_data = listings[listings['cluster'] == cluster_id]

    profile = {
        'cluster_id': int(cluster_id),
        'size': len(cluster_data),
        'pct_of_market': len(cluster_data) / len(listings) * 100,

        # Immutable characteristics
        'avg_accommodates': float(cluster_data['accommodates'].mean()),
        'avg_bedrooms': float(cluster_data['bedrooms'].mean()),
        'avg_bathrooms': float(cluster_data['bathrooms'].mean()),
        'modal_location_tier': int(cluster_data['location_tier'].mode()[0] if len(cluster_data) > 0 else 1),
        'modal_property_type': cluster_data['property_category'].mode()[0] if len(cluster_data) > 0 else 'Other',

        # Performance metrics
        'avg_price': float(cluster_data['price'].median()),
        'price_std': float(cluster_data['price'].std()),
        'avg_reviews': float(cluster_data['number_of_reviews'].mean()),
        'avg_rating': float(cluster_data['review_scores_rating'].mean()),
        'avg_amenities': float(cluster_data['amenities_count'].mean()),

        # Superhost rate
        'superhost_rate': float((cluster_data['host_is_superhost'] == 't').sum() / len(cluster_data) if len(cluster_data) > 0 else 0),

        # Characteristics
        'characteristics': {
            'size': cluster_data['size_category'].mode()[0] if len(cluster_data) > 0 else 'Unknown',
            'location': 'Premium' if cluster_data['location_tier'].mean() > 2.5 else ('Mid-tier' if cluster_data['location_tier'].mean() > 1.5 else 'Budget'),
            'amenities': cluster_data['amenity_tier'].mode()[0] if len(cluster_data) > 0 else 'Standard',
            'quality': cluster_data['quality_tier'].mode()[0] if len(cluster_data) > 0 else 'Good'
        }
    }

    # Business naming logic
    size_name = profile['characteristics']['size']
    location_name = profile['characteristics']['location']
    price_range = 'Budget' if profile['avg_price'] < 80 else ('Mid-Range' if profile['avg_price'] < 150 else 'Premium')

    # Create descriptive name
    if profile['avg_rating'] > 4.7 and profile['superhost_rate'] > 0.3:
        quality_prefix = "Luxury"
    elif profile['avg_rating'] > 4.5:
        quality_prefix = "Quality"
    else:
        quality_prefix = "Value"

    profile['name'] = f"{quality_prefix} {size_name} - {location_name}"
    profile['description'] = f"{price_range} {profile['modal_property_type']}s averaging ${profile['avg_price']:.0f}/night"

    cluster_profiles.append(profile)

    print(f"\n  Cluster {cluster_id}: {profile['name']}")
    print(f"    - {profile['description']}")
    print(f"    - {profile['size']} listings ({profile['pct_of_market']:.1f}% of market)")
    print(f"    - Avg: {profile['avg_accommodates']:.1f} guests, ${profile['avg_price']:.0f}/night, {profile['avg_rating']:.2f}★")

# ============================================================================
# STEP 5: SAVE CLUSTER MODEL & PROFILES
# ============================================================================

print("\n" + "="*80)
print("STEP 5: SAVING CLUSTER ASSETS")
print("="*80)

# Save the KMeans model and scaler
with open('cluster_model.pkl', 'wb') as f:
    pickle.dump({
        'kmeans': kmeans_final,
        'scaler': scaler,
        'features': clustering_features
    }, f)
print("✓ Saved cluster_model.pkl")

# Save cluster profiles
with open('cluster_profiles.json', 'w') as f:
    json.dump(cluster_profiles, f, indent=2)
print("✓ Saved cluster_profiles.json")

# Save listing clusters (for KNN comparables)
listings_export = listings[[
    'id', 'name', 'cluster', 'accommodates', 'bedrooms', 'bathrooms',
    'neighbourhood_cleansed', 'room_type', 'price', 'number_of_reviews',
    'review_scores_rating', 'amenities_count', 'host_is_superhost'
]].copy()

listings_export.to_csv('listings_with_clusters.csv', index=False)
print("✓ Saved listings_with_clusters.csv")

print("\n" + "="*80)
print("CLUSTERING COMPLETE!")
print("="*80)
print(f"\n✓ Created {optimal_k} market segments")
print(f"✓ Silhouette score: {optimal_score:.3f}")
print(f"✓ Ready for Streamlit integration")
