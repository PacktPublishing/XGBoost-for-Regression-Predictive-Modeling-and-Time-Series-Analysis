# ----------------------------------------
# filename timeseriesmodel.py
# purpose deploy a timeseries model using XGBoost
# author Joyce Weiner
# revision 1.0
# revision history 1.0 - initial file
# ----------------------------------------
# Import necessary libraries
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def retrain_model(model,X_train, y_train):
    #load the model from pickle file, retrain the model and save to pickle
    with open(model, 'rb') as file:
        loaded_model_pickle = pickle.load(file)
    loaded_model_pickle.fit(X_train, y_train)

    base_name, extension = os.path.splitext(model)
    new_file_path = f"{base_name}_retrained{extension}"

    with open(new_file_path, 'wb') as file:
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
