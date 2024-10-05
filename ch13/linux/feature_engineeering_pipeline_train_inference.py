# ----------------------------------------
# filename feature_engineering_pipeline_train_inference
# author Partha Deka
# updated by Joyce Weiner
# version 1.1
# revision history 1.0 - initial script
# rev 1.1 - put Timeseries model into py file, save model and pipeline
# as pickle files for deployment
# -----------------------------------------
# Import necessary libraries
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pickle
# Custom transformer for creating lagged features
class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lags=3):
        self.lags = lags

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X.copy())
        for lag in range(1, self.lags + 1):
            df[f'lag_{lag}'] = df['Value'].shift(lag)
        df.dropna(inplace=True)
        return df

# Create a synthetic time series dataset
date_range = pd.date_range(start='1/1/2020', periods=100, freq='D')
data = pd.DataFrame({'Date': date_range, 'Value': np.random.randn(100).cumsum()})
data.set_index('Date', inplace=True)

# Create lag features and corresponding target
lagged_data = LagFeatureTransformer(lags=3).transform(data)
X = lagged_data.drop(columns=['Value'])  # Features: lagged values
y = lagged_data['Value']  # Target: original values shifted by lag

# Train-test split  - set a seed for reproducibility random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state = 0)

# Define the pipeline with scaling and XGBoost model
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),                    # Feature scaling
    ('model', XGBRegressor(objective='reg:squarederror'))  # XGBoost for regression
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# save the pipeline
with open('XBGpipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

# Make predictions
y_pred = pipeline.predict(X_test)

