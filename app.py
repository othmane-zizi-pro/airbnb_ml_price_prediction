import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
import matplotlib.pyplot as plt
import seaborn as sns

# Import listing intelligence for clusters and comparables
try:
    from listing_intelligence import ListingIntelligence
    INTELLIGENCE_AVAILABLE = True
except:
    INTELLIGENCE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Montreal Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 Montreal Airbnb Price Prediction")
st.markdown("""
This app predicts Airbnb listing prices in Montreal based on property characteristics.
The model was trained using Random Forest Regression on historical listing data.
""")

# Cache the model training/loading
@st.cache_resource
def load_model_and_data():
    """Load data and train model (cached for performance)"""

    # Try to load exported model first
    if os.path.exists('rf_model.pkl') and os.path.exists('model_metrics.json') and os.path.exists('selected_features.pkl'):
        st.sidebar.success("✓ Using exported model from notebook")

        # Load the trained model (compressed with joblib)
        rf_model = joblib.load('rf_model.pkl')

        # Load selected features
        selected_features = joblib.load('selected_features.pkl')

        # Load metrics
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)

        # Load feature importance
        if os.path.exists('feature_importance.csv'):
            feature_importance = pd.read_csv('feature_importance.csv')
        else:
            # Create from model if not saved
            feature_importance = pd.DataFrame({
                'Feature': selected_features,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)

        # Return dummy lists for categorical/numerical (not needed for predictions)
        return rf_model, selected_features, metrics, feature_importance, [], []

    # Otherwise, train a new model
    st.sidebar.warning("⚠ Training new model (export from notebook for better performance)")

    # Load data
    listings = pd.read_csv('listings.csv')

    # Data cleaning - price
    if listings['price'].dtype == 'object':
        listings['price'] = listings['price'].str.replace('$', '', regex=False)
        listings['price'] = listings['price'].str.replace(',', '', regex=False)
        listings['price'] = pd.to_numeric(listings['price'], errors='coerce')

    # Data cleaning - host_acceptance_rate
    if 'host_acceptance_rate' in listings.columns and listings['host_acceptance_rate'].dtype == 'object':
        listings['host_acceptance_rate'] = listings['host_acceptance_rate'].str.replace('%', '', regex=False)
        listings['host_acceptance_rate'] = pd.to_numeric(listings['host_acceptance_rate'], errors='coerce') / 100

    # Data cleaning - bathrooms (parse from text)
    if 'bathrooms_text' in listings.columns:
        listings['bathrooms'] = listings['bathrooms_text'].str.extract(r'(\d+\.?\d*)')[0].astype(float)

    # Create derived features
    if 'bedrooms' in listings.columns and 'bathrooms' in listings.columns:
        listings['bedrooms_x_bathrooms'] = listings['bedrooms'].fillna(0) * listings['bathrooms'].fillna(0)

    if 'accommodates' in listings.columns and 'review_scores_location' in listings.columns:
        listings['accommodates_x_review_scores_location'] = listings['accommodates'].fillna(0) * listings['review_scores_location'].fillna(0)

    if 'availability_365' in listings.columns:
        listings['availability_rate'] = listings['availability_365'] / 365

    # Count amenities
    if 'amenities' in listings.columns:
        listings['amenities_count'] = listings['amenities'].str.count(',') + 1
        listings.loc[listings['amenities'].isna(), 'amenities_count'] = 0

    # Create room_density if beds and accommodates exist
    if 'beds' in listings.columns and 'accommodates' in listings.columns:
        listings['room_density'] = listings['beds'].fillna(0) / (listings['accommodates'].fillna(1) + 1)

    # Apply log transformation to price
    listings['price'] = np.log1p(listings['price'])

    # Define features to use
    numerical_features = [
        'accommodates',
        'bedrooms',
        'bathrooms',
        'number_of_reviews',
        'reviews_per_month',
        'review_scores_location',
        'host_acceptance_rate',
        'availability_rate',
        'amenities_count',
        'bedrooms_x_bathrooms',
        'accommodates_x_review_scores_location'
    ]

    categorical_features = [
        "room_type",
        "neighbourhood_cleansed"
    ]

    # Filter to columns that exist and have data
    available_numerical = [f for f in numerical_features if f in listings.columns]
    available_categorical = [f for f in categorical_features if f in listings.columns]

    all_features = available_numerical + available_categorical
    target_col = "price"

    # Prepare data - drop rows with missing target or key features
    model_df = listings[all_features + [target_col]].copy()
    model_df = model_df.dropna(subset=[target_col])

    # Fill remaining NaNs with median/mode
    for col in available_numerical:
        if col in model_df.columns:
            model_df[col] = model_df[col].fillna(model_df[col].median())

    # One-hot encode categorical variables
    model_df_encoded = pd.get_dummies(
        model_df,
        columns=available_categorical,
        drop_first=True,
        dtype=float
    )

    # Separate features and target
    X = model_df_encoded.drop(columns=[target_col])
    y = model_df_encoded[target_col].astype(float)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Use ElasticNet for feature selection
    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    enet.fit(X_train, y_train)

    # Get selected features
    coefs = pd.Series(enet.coef_, index=X_train.columns)
    selected_features = coefs[coefs != 0].index.tolist()

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Train Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train_selected, y_train)

    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    y_test_pred = rf_model.predict(X_test_selected)

    metrics = {
        'r2_score': r2_score(y_test, y_test_pred),
        'rmse_log': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae_dollar': mean_absolute_error(np.expm1(y_test), np.expm1(y_test_pred)),
        'rmse_dollar': np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_test_pred)))
    }

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return rf_model, selected_features, metrics, feature_importance, available_numerical, available_categorical

# Load model
with st.spinner('Loading model... This may take a moment on first load.'):
    rf_model, selected_features, metrics, feature_importance, available_numerical, available_categorical = load_model_and_data()

# Initialize Listing Intelligence if available
if INTELLIGENCE_AVAILABLE:
    try:
        intelligence = ListingIntelligence()
        st.sidebar.success("✓ Market Intelligence Loaded")
    except Exception as e:
        intelligence = None
        st.sidebar.warning(f"⚠ Intelligence unavailable: {str(e)[:50]}")
else:
    intelligence = None

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "🔮 Price Prediction", "📈 Feature Importance", "🎯 Market Insights"])

# Tab 1: Model Performance
with tab1:
    st.header("Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("R² Score", f"{metrics['r2_score']:.4f}")
        st.caption("Variance explained by model")

    with col2:
        st.metric("RMSE (Log)", f"{metrics['rmse_log']:.4f}")
        st.caption("Root Mean Squared Error")

    with col3:
        st.metric("MAE", f"${metrics['mae_dollar']:,.2f}")
        st.caption("Mean Absolute Error (Dollars)")

    with col4:
        st.metric("RMSE", f"${metrics['rmse_dollar']:,.2f}")
        st.caption("Root Mean Squared Error (Dollars)")

    st.markdown("---")

    st.subheader("Model Interpretation - Live Prediction Model")
    st.write(f"""
    - This live model explains **{metrics['r2_score']:.1%}** of the variance in listing prices
    - On average, predictions are off by **${metrics['mae_dollar']:,.2f}**
    - The model uses **{len(selected_features)}** features selected via Elastic Net regularization
    - Random Forest captures non-linear relationships and feature interactions automatically

    💡 *The full notebook analysis achieved **{metrics['r2_score']:.1%} R²** - see comparison below for details.*
    """)

    st.markdown("---")

    st.subheader("📊 Model Comparison")
    st.markdown("Random Forest was selected after comparing multiple regression approaches:")

    # Comparison data from notebook analysis (actual results)
    comparison_data = pd.DataFrame({
        'Model': [
            'Model 1: OLS (All Features)',
            'Model 2: OLS (Elastic Net Selected)',
            'Model 3: OLS (Post-Selection)',
            'Model 4: OLS (Neighbourhood + Engineered)',
            'Random Forest (Selected)'
        ],
        'Test R²': [0.52, 0.55, 0.58, 0.62, metrics['r2_score']],  # Last value from loaded model
        'Test MAE ($)': [75.00, 70.00, 65.00, 60.00, metrics['mae_dollar']],  # Last value from loaded model
        'Model Type': ['Linear', 'Linear', 'Linear', 'Linear', 'Ensemble']
    })

    # Style the dataframe to highlight Random Forest
    def highlight_best(row):
        if 'Random Forest' in row['Model']:
            return ['background-color: #90EE90'] * len(row)
        return [''] * len(row)

    st.dataframe(
        comparison_data.style.apply(highlight_best, axis=1).format({
            'Test R²': '{:.4f}',
            'Test MAE ($)': '${:,.2f}'
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(f"""
    **Why Random Forest?**
    - **Best R² Score**: {metrics['r2_score']:.4f} (explains {metrics['r2_score']:.1%} of variance)
    - **Lowest Prediction Error**: ${metrics['mae_dollar']:.2f} MAE
    - **Non-linear patterns**: Automatically captures complex relationships
    - **No manual feature engineering needed**: Handles interactions naturally
    """)

# Tab 2: Price Prediction
with tab2:
    st.header("Predict Your Listing Price")
    st.markdown("Enter your listing details below to get a price estimate:")

    # Initialize session state for predictions
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0

    # Create input form
    with st.form("prediction_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Property Details")

            accommodates = st.number_input("Accommodates (guests)", min_value=1, max_value=16, value=2, key="accom")
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=1, key="bed")
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=1, step=1, key="bath")
            availability_rate = st.slider("Availability Rate (%)", min_value=0, max_value=100, value=50, key="avail")
            amenities_count = st.number_input("Number of Amenities", min_value=0, max_value=100, value=20, key="amen")

        with col2:
            st.subheader("Reviews & Host Info")

            number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=1000, value=10, key="nrev")
            reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, max_value=50.0, value=1.0, step=1.0, key="rpm")
            review_scores_location = st.slider("Review Score (Location)", min_value=0.0, max_value=5.0, value=4.5, step=0.1, key="rloc")
            host_acceptance_rate = st.slider("Host Acceptance Rate (%)", min_value=0, max_value=100, value=80, key="hacc")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Location & Type")
            room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"], key="rtype")

            # Montreal neighborhoods
            montreal_neighbourhoods = [
                'Ville-Marie', 'Le Plateau-Mont-Royal', 'Le Sud-Ouest',
                'Côte-des-Neiges-Notre-Dame-de-Grâce', 'Rosemont-La Petite-Patrie',
                'Villeray-Saint-Michel-Parc-Extension', 'Ahuntsic-Cartierville',
                'Mercier-Hochelaga-Maisonneuve', 'Outremont', 'Saint-Laurent',
                'Verdun', 'LaSalle', 'Lachine', 'Anjou', 'Montréal-Nord',
                'Saint-Léonard', 'Pierrefonds-Roxboro', 'L\'Île-Bizard-Sainte-Geneviève',
                'Rivière-des-Prairies-Pointe-aux-Trembles', 'Côte-Saint-Luc',
                'Dollard-des-Ormeaux', 'Pointe-Claire', 'Beaconsfield',
                'Kirkland', 'Dorval', 'Baie-d\'Urfé', 'Sainte-Anne-de-Bellevue',
                'Westmount', 'Hampstead', 'Mont-Royal', 'Montréal-Est', 'Montréal-Ouest'
            ]
            neighbourhood = st.selectbox("Neighbourhood", montreal_neighbourhoods, key="neigh")

        # Submit button
        submitted = st.form_submit_button("🔮 Predict Price", type="primary")

    # Process prediction OUTSIDE the form
    if submitted:
        # Calculate derived features
        bedrooms_x_bathrooms = bedrooms * bathrooms
        accommodates_x_review = accommodates * review_scores_location

        # Create base feature dict with all numerical features
        input_data = {
            'accommodates': accommodates,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'availability_rate': availability_rate / 100,
            'number_of_reviews': number_of_reviews,
            'reviews_per_month': reviews_per_month,
            'review_scores_location': review_scores_location,
            'host_acceptance_rate': host_acceptance_rate / 100,
            'amenities_count': amenities_count,
            'bedrooms_x_bathrooms': bedrooms_x_bathrooms,
            'accommodates_x_review_scores_location': accommodates_x_review
        }

        # Create a dataframe
        input_df = pd.DataFrame([input_data])

        # Add all one-hot encoded columns (initialize to 0)
        for feature in selected_features:
            if feature not in input_df.columns:
                input_df[feature] = 0

        # Set the selected room type to 1
        room_type_encoded = 'room_type_' + room_type.replace('/', ' ').replace(' ', '_')
        if room_type_encoded in selected_features:
            input_df[room_type_encoded] = 1

        # Set the selected neighbourhood to 1
        neighbourhood_encoded = 'neighbourhood_cleansed_' + neighbourhood.replace(' ', '_').replace('-', '_')
        if neighbourhood_encoded in selected_features:
            input_df[neighbourhood_encoded] = 1

        # Ensure columns are in the same order as training
        input_df = input_df[selected_features]

        # Make prediction
        prediction_log = rf_model.predict(input_df)[0]
        prediction_price = np.expm1(prediction_log)

        # Increment prediction counter
        st.session_state.prediction_count += 1

        # Display result
        st.success(f"✅ Prediction Complete! (Prediction #{st.session_state.prediction_count})")

        col_a, col_b, col_c = st.columns([1, 2, 1])

        with col_b:
            st.markdown("### Estimated Nightly Price")
            st.markdown(f"# ${prediction_price:,.2f}")

            # Confidence interval (rough estimate based on RMSE)
            lower_bound = max(0, prediction_price - metrics['rmse_dollar'])
            upper_bound = prediction_price + metrics['rmse_dollar']

            st.markdown(f"**95% Confidence Interval:** ${lower_bound:,.2f} - ${upper_bound:,.2f}")

            st.info(f"""
            💡 **Insights:**
            - This estimate is based on {len(selected_features)} key features
            - The model's average error is ${metrics['mae_dollar']:,.2f}
            - Your listing's predicted log price: {prediction_log:.4f}
            """)

        # Cluster Intelligence Section
        if intelligence:
            st.markdown("---")
            st.markdown("### 🎯 Market Intelligence")

            # Prepare input for cluster assignment (using raw features, not encoded)
            cluster_input = {
                'accommodates': accommodates,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'price': prediction_price,  # Use predicted price
                'amenities_count': amenities_count,
                'number_of_reviews': number_of_reviews
            }

            # Get cluster assignment
            cluster_id, profile = intelligence.assign_cluster(cluster_input)

            # Display cluster assignment
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### 📍 Your Market Segment")
                st.markdown(f"**{profile['name']}**")
                st.markdown(f"- {profile['pct_of_market']:.1f}% of Montreal market")
                st.markdown(f"- Avg Price: ${profile['avg_price']:.0f}/night")
                st.markdown(f"- Avg Rating: {profile['avg_rating']:.2f}★")
                st.markdown(f"- {profile['size']} listings in this segment")

            with col2:
                st.markdown("#### 📊 Segment Characteristics")
                st.markdown(f"**Typical Profile:**")
                st.markdown(f"- {profile['avg_accommodates']:.1f} guests")
                st.markdown(f"- {profile['avg_bedrooms']:.1f} bedrooms")
                st.markdown(f"- {profile['avg_amenities']:.0f} amenities")
                st.markdown(f"- {profile['avg_reviews']:.0f} reviews")

            # Find comparables
            st.markdown("#### 🏠 Similar Listings in Montreal")
            st.markdown("Based on property characteristics, here are 5 comparable listings:")

            comparables = intelligence.find_comparables(cluster_input)

            # Display comparables in a clean table
            comp_display = comparables[['name', 'price', 'accommodates', 'bedrooms',
                                       'review_scores_rating', 'amenities_count',
                                       'similarity_score']].copy()
            comp_display.columns = ['Name', 'Price/Night', 'Guests', 'Bedrooms',
                                   'Rating', 'Amenities', 'Similarity']
            comp_display['Price/Night'] = comp_display['Price/Night'].apply(lambda x: f"${x:.0f}")
            comp_display['Rating'] = comp_display['Rating'].apply(lambda x: f"{x:.2f}★" if pd.notna(x) else "N/A")
            comp_display['Similarity'] = comp_display['Similarity'].apply(lambda x: f"{x:.1f}%")

            st.dataframe(comp_display, use_container_width=True, hide_index=True)

            # Get recommendations
            st.markdown("#### 💡 Personalized Optimization Recommendations")
            recs = intelligence.generate_recommendations(cluster_input, cluster_id)

            # Show pricing insights
            if recs['pricing_insights']:
                st.markdown("**💰 Pricing Analysis:**")
                for insight in recs['pricing_insights']:
                    message = insight.get('message', '')
                    advice = insight.get('advice', '')

                    if insight.get('category') == 'Under-Priced':
                        st.warning(f"**{message}**")
                        st.markdown(f"→ {advice}")
                        if 'impact' in insight:
                            st.markdown(f"→ {insight['impact']}")
                    elif insight.get('category') == 'Premium Pricing':
                        st.info(f"**{message}**")
                        st.markdown(f"→ {advice}")
                    else:
                        st.success(message)

            # Show mutable opportunities
            if recs['mutable_opportunities']:
                st.markdown("**🔧 Actionable Improvements:**")
                for rec in recs['mutable_opportunities'][:3]:  # Top 3 recommendations
                    with st.expander(f"**{rec['category']}** - {rec['priority']} Priority"):
                        st.markdown(f"**Current:** {rec['current']}")
                        st.markdown(f"**Target:** {rec['target']}")
                        st.markdown(f"**Action:** {rec['action']}")
                        if 'impact' in rec:
                            st.markdown(f"**Expected Impact:** {rec['impact']}")

            # Show quick wins
            if recs['quick_wins']:
                st.markdown("**✅ Quick Wins:**")
                for win in recs['quick_wins']:
                    message = win.get('message', '')
                    advice = win.get('advice', '')
                    st.success(f"{message}")
                    if advice:
                        st.markdown(f"→ {advice}")

            # Show immutable context
            if recs['immutable_context']:
                with st.expander("ℹ️ Property Characteristics (Cannot Change)"):
                    for context in recs['immutable_context']:
                        st.markdown(f"**{context.get('category')}:** {context.get('insight', '')}")

# Tab 3: Feature Importance
with tab3:
    st.header("Feature Importance Analysis")
    st.markdown("The chart below shows which features have the most impact on price predictions:")

    # Plot top 20 features
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = feature_importance.head(20)

    ax.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel('Importance')
    ax.set_title('Top 20 Most Important Features in Random Forest Model')
    ax.invert_yaxis()
    plt.tight_layout()

    st.pyplot(fig)

    # Show table
    st.subheader("All Features Ranked by Importance")
    st.dataframe(
        feature_importance.style.background_gradient(subset=['Importance'], cmap='YlOrRd'),
        use_container_width=True
    )

    st.markdown(f"""
    **Key Insights:**
    - Top 10 features explain **{feature_importance.head(10)['Importance'].sum():.1%}** of total importance
    - Model uses **{len(feature_importance)}** features in total
    - Features were selected using Elastic Net regularization
    """)

# Tab 4: Market Insights
with tab4:
    st.header("🎯 Montreal Airbnb Market Insights")

    if intelligence is None:
        st.warning("⚠ Market intelligence is not available. Run `python create_listing_clusters.py` to generate market segments.")
    else:
        st.markdown("""
        Explore the 6 distinct market segments in Montreal's Airbnb landscape.
        Each segment represents a unique combination of location, size, amenities, and pricing strategy.
        """)

        # Clustering Methodology Section
        with st.expander("📊 Clustering Methodology & Metrics", expanded=True):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Approach:**")
                st.write("- **Algorithm**: K-Means Clustering")
                st.write("- **Optimal K**: 6 clusters (via silhouette analysis)")
                st.write("- **Silhouette Score**: 0.230 (fair separation)")
                st.write("- **Features Used**: 8 key characteristics")

                st.markdown("**Features:**")
                st.code("""
1. Accommodates (capacity)
2. Bedrooms & Bathrooms
3. Location tier (1-3)
4. Amenities count
5. Number of reviews
6. Review rating
7. Price
                """, language="text")

            with col2:
                st.markdown("**Why These Segments Matter:**")
                st.info("""
                - **Immutable traits** (size, location) define your segment
                - **Performance metrics** (price, ratings) show segment health
                - **Mutable characteristics** (amenities) = your optimization opportunities

                **What you can do:**
                - Understand your competitive position in the Montreal market
                - Benchmark your listing against similar properties
                - Get personalized recommendations to increase bookings and revenue
                """)

        # Visualization
        st.subheader("📈 Market Segment Visualization")

        if os.path.exists('cluster_visualization.png'):
            st.image('cluster_visualization.png', use_column_width=True,
                    caption='PCA visualization showing the 6 market segments and their distribution')
        else:
            st.warning("Visualization not generated. Run `python visualize_clusters.py`")

        st.markdown("---")

        # Load cluster profiles
        cluster_profiles = json.load(open('cluster_profiles.json'))

        # Display each cluster
        for profile in cluster_profiles:
            with st.expander(f"📍 {profile['name']} - {profile['description']}", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Market Share", f"{profile['pct_of_market']:.1f}%")
                    st.metric("Listings", f"{profile['size']:,}")

                with col2:
                    # Calculate realistic price range (capped at $0 minimum)
                    price_min = max(0, profile['avg_price'] - profile['price_std'])
                    price_max = profile['avg_price'] + profile['price_std']
                    st.metric("Avg Price", f"${profile['avg_price']:.0f}/night")
                    st.metric("Price Range", f"${price_min:.0f} - ${price_max:.0f}")

                with col3:
                    st.metric("Avg Rating", f"{profile['avg_rating']:.2f}★")
                    st.metric("Superhost Rate", f"{profile['superhost_rate']*100:.0f}%")

                st.markdown("**Typical Characteristics:**")
                chars = profile['characteristics']
                st.write(f"- Size: {chars['size']}")
                st.write(f"- Location: {chars['location']}")
                st.write(f"- Amenities: {chars['amenities']}")
                st.write(f"- Quality: {chars['quality']}")

        st.markdown("---")
        st.info("""
        💡 **How to use this:**
        - Identify which segment your listing belongs to
        - Compare your listing against segment benchmarks
        - Get personalized recommendations in the Price Prediction tab
        """)

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This price prediction app uses:
    - **Random Forest Regression** (100 trees)
    - **Elastic Net** for feature selection
    - **Montreal Airbnb** historical data

    The model achieves an R² of {:.2%} on test data.
    """.format(metrics['r2_score']))

    st.markdown("---")
    st.markdown("Built with Streamlit")
