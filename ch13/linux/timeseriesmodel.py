# ----------------------------------------
# filename timeseriesmodel.py
# purpose deploy a timeseries model using XGBoost
# author Joyce Weiner
# revision 1.0
# revision history 1.0 - initial file
# ----------------------------------------
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def retrain_model(model,X_train, y_train):
    #load the model from pickle file, retrain the model and save to pickle
    with open(model, 'rb') as file:
        loaded_model_pickle = pickle.load(file)
    loaded_model_pickle.fit(X_train, y_train)

    with open(model +'_retrain', 'wb') as file:
        pickle.dump(loaded_model_pickle, file)
    

def predict(model, X_test):
    #use the loaded model to predict a result
    with open(model, 'rb') as file:
        loaded_model_pickle = pickle.load(file)
    predictions_pickle = loaded_model_pickle.predict(X_test)
    return predictions_pickle

# Create a class to hold the results of model evaluation mse, r_squared
class EvalMetrics:
    def __init__(self):
        self.mse = 0
        self.r_squared = 0

# Evaluate the model
def evalmodel(model, X_test, y_test):
    y_pred = predict(model, X_test)
    EvalMetrics.mse = mean_squared_error(y_test, y_pred)
    EvalMetrics.r_squared = r2_score(y_test, y_pred)
    return EvalMetrics


p = EvalMetrics
evalmodel(model,p,X_test, y_test, y_pred)
print(f'Mean Squared Error: {p.mse}')
print(f'R squared: {p.r_squared}')