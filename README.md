# Montreal Airbnb Price Prediction

This project analyzes Montreal Airbnb listings and provides a machine learning model to predict listing prices.

## Project Structure

- `airbnb_ml_model.ipynb` - Jupyter notebook with full analysis and model development
- `listings.csv` - Montreal Airbnb listings dataset
- `app.py` - Streamlit web application for price predictions
- `requirements.txt` - Python package dependencies

## Streamlit App Features

The web app provides three main features:

1. **Model Performance** - View comprehensive metrics and model evaluation
2. **Price Prediction** - Input listing characteristics to get instant price estimates
3. **Feature Importance** - Understand which features most influence pricing

## Running the Streamlit App

### Prerequisites

```bash
pip install -r requirements.txt
```

### Step 1: Export the Trained Model (Recommended)

For best performance and accuracy, export the trained model from the notebook:

1. Open `airbnb_ml_model.ipynb` in Jupyter
2. Run all cells (or at least up to the Random Forest model)
3. Run the **"Export Model for Streamlit Deployment"** cells at the end
4. This creates:
   - `rf_model.pkl` - Trained model
   - `selected_features.pkl` - Feature list
   - `model_metrics.json` - Performance metrics
   - `feature_importance.csv` - Feature rankings

### Step 2: Launch the App

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

**Note**: If model files don't exist, the app will train a new model automatically (but with slightly lower performance).

## Model Details

- **Algorithm**: Random Forest Regression (100 trees)
- **Feature Selection**: Elastic Net regularization
- **Features**: 8 numerical + categorical features (room type, neighbourhood, etc.)
- **Preprocessing**: Log transformation of target variable, one-hot encoding of categorical features

## Usage

1. Navigate to the "Price Prediction" tab
2. Enter your listing details:
   - Property characteristics (bedrooms, bathrooms, etc.)
   - Location information
   - Host and review metrics
3. Click "Predict Price" to get your estimate
4. View the predicted price with confidence interval

## Model Performance

The model is evaluated using:
- R² Score (variance explained)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

All metrics are displayed in the "Model Performance" tab.

## Data Source

Montreal Airbnb listings data from Inside Airbnb (http://insideairbnb.com/)
