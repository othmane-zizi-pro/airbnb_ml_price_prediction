"""
Listing Intelligence: Comparables & Recommendations
===================================================

Provides actionable insights for Airbnb hosts based on:
1. K-Nearest Neighbors analysis (find similar listings)
2. Cluster-based recommendations (what successful listings in your segment do)
3. Mutable vs Immutable characteristic identification
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import json


class ListingIntelligence:
    def __init__(self):
        """Load cluster data and prepare for analysis"""
        self.listings = pd.read_csv('listings_with_clusters.csv')
        self.cluster_profiles = json.load(open('cluster_profiles.json'))

        # Prepare features for KNN
        self.knn_features = [
            'accommodates', 'bedrooms', 'bathrooms',
            'price', 'amenities_count', 'number_of_reviews'
        ]

        self.listings_clean = self.listings[self.knn_features].fillna(self.listings[self.knn_features].median())

        # Fit KNN model
        self.scaler = StandardScaler()
        self.listings_scaled = self.scaler.fit_transform(self.listings_clean)

        self.knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
        self.knn.fit(self.listings_scaled)

    def find_comparables(self, listing_features):
        """
        Find 5 most similar listings based on KNN

        Args:
            listing_features: dict with keys matching self.knn_features

        Returns:
            DataFrame of 5 comparable listings
        """
        # Prepare input
        input_df = pd.DataFrame([listing_features])[self.knn_features]
        input_scaled = self.scaler.transform(input_df)

        # Find nearest neighbors (6 including the listing itself)
        distances, indices = self.knn.kneighbors(input_scaled)

        # Return top 5 (excluding the first which is the listing itself if it exists)
        comparable_indices = indices[0][1:6]
        comparables = self.listings.iloc[comparable_indices].copy()

        # Add distance score
        comparables['similarity_score'] = 100 * (1 - distances[0][1:6] / distances[0][1:6].max())

        return comparables[[
            'name', 'accommodates', 'bedrooms', 'bathrooms', 'price',
            'number_of_reviews', 'review_scores_rating', 'amenities_count',
            'host_is_superhost', 'neighbourhood_cleansed', 'similarity_score'
        ]]

    def assign_cluster(self, listing_features):
        """
        Determine which cluster a listing belongs to

        Args:
            listing_features: dict with features

        Returns:
            cluster_id (int) and cluster_profile (dict)
        """
        # Load cluster model
        import pickle
        with open('cluster_model.pkl', 'rb') as f:
            cluster_data = pickle.load(f)

        kmeans = cluster_data['kmeans']
        scaler = cluster_data['scaler']
        features = cluster_data['features']

        # Prepare features in correct order
        input_array = np.array([[listing_features.get(f, 0) for f in features]])
        input_scaled = scaler.transform(input_array)

        # Predict cluster
        cluster_id = int(kmeans.predict(input_scaled)[0])
        cluster_profile = self.cluster_profiles[cluster_id]

        return cluster_id, cluster_profile

    def generate_recommendations(self, listing_features, cluster_id):
        """
        Generate actionable recommendations based on cluster benchmarks

        Args:
            listing_features: dict with current listing features
            cluster_id: assigned cluster

        Returns:
            dict with recommendations categorized as immutable vs mutable
        """
        cluster_profile = self.cluster_profiles[cluster_id]
        cluster_listings = self.listings[self.listings['cluster'] == cluster_id]

        recommendations = {
            'immutable_context': [],
            'mutable_opportunities': [],
            'pricing_insights': [],
            'quick_wins': []
        }

        # === IMMUTABLE CHARACTERISTICS ===
        recommendations['immutable_context'].append({
            'category': 'Property Size',
            'your_value': listing_features.get('accommodates', 0),
            'cluster_avg': cluster_profile['avg_accommodates'],
            'insight': f"Your {listing_features.get('bedrooms', 0)}-bedroom property is typical for this segment"
        })

        # === MUTABLE: AMENITIES ===
        your_amenities = listing_features.get('amenities_count', 0)
        cluster_amenities_avg = cluster_profile['avg_amenities']
        cluster_amenities_top25 = cluster_listings['amenities_count'].quantile(0.75)

        if your_amenities < cluster_amenities_avg:
            gap = cluster_amenities_avg - your_amenities
            recommendations['mutable_opportunities'].append({
                'category': 'Amenities',
                'priority': 'HIGH',
                'current': your_amenities,
                'target': int(cluster_amenities_top25),
                'gap': int(gap),
                'action': f"Add {int(gap)} more amenities to match cluster average",
                'impact': 'Amenities strongly correlate with price and bookings',
                'examples': ['Fast WiFi', 'Self check-in', 'Kitchen', 'Workspace', 'Free parking']
            })
        else:
            recommendations['quick_wins'].append({
                'category': 'Amenities',
                'message': f"✓ Your {your_amenities} amenities exceed the cluster average of {cluster_amenities_avg:.0f}",
                'advice': 'Highlight premium amenities in your listing description'
            })

        # === MUTABLE: PRICING ===
        your_price = listing_features.get('price', 0)
        cluster_price_avg = cluster_profile['avg_price']
        cluster_price_std = cluster_profile['price_std']

        price_diff = your_price - cluster_price_avg
        price_diff_pct = (price_diff / cluster_price_avg) * 100 if cluster_price_avg > 0 else 0

        if abs(price_diff_pct) > 15:
            if price_diff > 0:
                recommendations['pricing_insights'].append({
                    'category': 'Premium Pricing',
                    'current_price': your_price,
                    'cluster_avg': cluster_price_avg,
                    'difference': f"+{price_diff_pct:.1f}%",
                    'message': f"You're priced {price_diff_pct:.0f}% above cluster average",
                    'advice': 'Justify premium pricing with superior amenities, photos, and reviews',
                    'risk': 'May reduce booking frequency if not differentiated'
                })
            else:
                recommendations['pricing_insights'].append({
                    'category': 'Under-Priced',
                    'current_price': your_price,
                    'cluster_avg': cluster_price_avg,
                    'opportunity': f"${abs(price_diff):.0f}/night",
                    'message': f"You're priced {abs(price_diff_pct):.0f}% below cluster average",
                    'advice': f"Consider raising price to ${cluster_price_avg:.0f}/night",
                    'impact': f"Potential ${abs(price_diff) * 15:.0f}/month additional revenue (assuming 50% occupancy)"
                })
        else:
            recommendations['quick_wins'].append({
                'category': 'Pricing',
                'message': f"✓ Your ${your_price:.0f}/night pricing is competitive (within 15% of cluster average)",
                'advice': 'Monitor competitor pricing and adjust seasonally'
            })

        # === MUTABLE: REVIEWS & QUALITY ===
        your_reviews = listing_features.get('number_of_reviews', 0)
        cluster_reviews_avg = cluster_profile['avg_reviews']

        if your_reviews < cluster_reviews_avg * 0.5:
            recommendations['mutable_opportunities'].append({
                'category': 'Guest Reviews',
                'priority': 'MEDIUM',
                'current': your_reviews,
                'target': int(cluster_reviews_avg),
                'action': 'Focus on getting more bookings and reviews',
                'impact': 'Reviews build trust and improve search ranking',
                'tips': [
                    'Encourage guests to leave reviews',
                    'Provide exceptional service',
                    'Follow up after checkout'
                ]
            })

        # === SUPERHOST OPPORTUNITY ===
        if listing_features.get('is_superhost', False):
            recommendations['quick_wins'].append({
                'category': 'Superhost Status',
                'message': '⭐ Superhost status achieved!',
                'advice': 'Maintain high standards to keep this competitive advantage'
            })
        else:
            superhost_rate = cluster_profile['superhost_rate'] * 100
            if superhost_rate > 20:
                recommendations['mutable_opportunities'].append({
                    'category': 'Superhost Status',
                    'priority': 'MEDIUM',
                    'current': 'Not Superhost',
                    'target': f'{superhost_rate:.0f}% of cluster are Superhosts',
                    'action': 'Work towards Superhost status',
                    'requirements': [
                        '4.8+ overall rating',
                        '10+ stays or 100+ nights',
                        '<1% cancellation rate',
                        '90% response rate'
                    ],
                    'impact': 'Superhosts earn 20-30% more and get priority placement'
                })

        return recommendations

    def get_cluster_summary(self, cluster_id):
        """Get human-readable cluster summary"""
        profile = self.cluster_profiles[cluster_id]

        return {
            'name': profile['name'],
            'description': profile['description'],
            'market_share': f"{profile['pct_of_market']:.1f}%",
            'avg_price': f"${profile['avg_price']:.0f}",
            'avg_rating': f"{profile['avg_rating']:.2f}★",
            'total_listings': profile['size'],
            'characteristics': profile['characteristics']
        }
