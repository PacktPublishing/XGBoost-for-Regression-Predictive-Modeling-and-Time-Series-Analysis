# ----------------------------------------
"""
streamlit_app.py
Author: Partha Deka
Revision: 1.0
Revision History:
    - 1.0: Initial script

This Streamlit application predicts house prices using an XGBoost regression model. The application performs the following tasks as already discussed 
in chapters 7, 8 and 12:

1. Loads and displays a dataset of house prices.
2. Preprocesses the data by handling missing values and scaling numeric features.
3. Trains an XGBoost regression model using the preprocessed data.
4. Evaluates the model's performance on a test set.
5. Provides an interface for users to input new data and make predictions.

Classes:
    - MissingValueImputer: Custom transformer for handling missing values in categorical variables.
Functions:
    - load_data: Loads the house pricing dataset from a CSV file.
Streamlit Interface:
    - Displays the dataset preview.
    - Shows the model's performance metrics (R² Score and RMSE).
    - Provides input fields for users to enter new data for prediction.
    - Displays the predicted house price based on user input.
"""
# filename streamlit_app.py
# author Partha Deka
# revision 1.0
# revision history 1.0 - initial script

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for handling missing values in categorical variables
class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, for_missing_string, for_frequent_category):
        self.for_missing_string = for_missing_string
        self.for_frequent_category = for_frequent_category
        self.frequent_categories = {}

    def fit(self, X, y=None):
        # Store the most frequent category for variables with few missing observations
        for var in self.for_frequent_category:
            self.frequent_categories[var] = X[var].mode()[0]
        return self

    def transform(self, X, y=None):
        # Replace missing values with "Missing" for specific variables
        X[self.for_missing_string] = X[self.for_missing_string].fillna('Missing')
        # Replace missing values with the most frequent category for specific variables
        for var in self.for_frequent_category:
            X[var] = X[var].fillna(self.frequent_categories[var])
        return X

# Streamlit Interface
st.title("House Price Prediction with XGBoost")

# Load dataset
@st.cache
def load_data():
    data = pd.read_csv("house_pricing.csv")  # Replace with your local file path
    return data

# Load the data
data = load_data()

# Display the data
st.write("Dataset Preview:")
st.write(data.head())

# Define feature columns
numeric_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'GrLivArea']
categorical_features = ['Neighborhood', 'HouseStyle', 'GarageType', 'SaleCondition']

# Handle missing categorical variables
cat_vars_with_na = [var for var in categorical_features if data[var].isnull().sum() > 0]
for_missing_string = [var for var in cat_vars_with_na if data[var].isnull().mean() > 0.1]
for_frequent_category = [var for var in cat_vars_with_na if data[var].isnull().mean() < 0.1]

# Target column
target = 'SalePrice'

# Split the data into features and target
X = data.drop(columns=[target, 'Id'])
y = data[target]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())                 # Scale features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categorical values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # One-hot encode categorical features
])

# Combine both numeric and categorical preprocessing, along with custom missing value handling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the complete pipeline
pipeline = Pipeline(steps=[
    ('missing_imputer', MissingValueImputer(for_missing_string, for_frequent_category)),  # Custom missing value handling
    ('preprocessor', preprocessor),               # Preprocessing step
    ('model', XGBRegressor(objective='reg:squarederror'))  # XGBoost regression model
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predictions on test set
y_pred = pipeline.predict(X_test)

# Show the model's performance
st.subheader("Model Performance")
st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")

# User Input for Predictions
st.subheader("Make a New Prediction")
input_data = {}

# Loop through numeric features and use number_input for numeric columns
for col in numeric_features:
    input_data[col] = st.number_input(f"Enter {col}:", value=float(X[col].mean()))

# Loop through categorical features and use selectbox for categorical columns
for col in categorical_features:
    unique_values = X[col].unique()
    input_data[col] = st.selectbox(f"Select {col}:", options=unique_values)

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = pipeline.predict(input_df)
    st.write(f"Predicted House Price: ${prediction[0]:,.2f}")

