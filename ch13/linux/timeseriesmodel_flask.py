# ----------------------------------------
# filename timeseriesmodel_flask.py
# purpose deploy a timeseries model using XGBoost
# author Joyce Weiner
# revision 1.1
# revision history 1.0 - initial file
# 1.1 take timeseriesmodel and put it into flask api
# ----------------------------------------
# Import necessary libraries
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
from flask import Flask, jsonify, request

app=Flask(__name__)

@app.route("/retrain",methods = ["POST"])
def retrain_model(model="XBGpipeline.pkl"):
    #load the model from pickle file, retrain the model and save to pickle
    with open(model, 'rb') as file:
        loaded_model_pickle = pickle.load(file)

    data = request.get_json()
    df = pd.DataFrame(data)

    # Split the data into features and target variable
    X_train = df.drop('target', axis=1)
    y_train = df['target']
    loaded_model_pickle.fit(X_train, y_train)

    base_name, extension = os.path.splitext(model)
    new_file_path = f"{base_name}_retrained{extension}"

    with open(new_file_path, 'wb') as file:
        pickle.dump(loaded_model_pickle, file)

    return jsonify({'message': 'Model retrained successfully'})
    
@app.route("/predict",methods = ["POST"])
def predict(model="XBGpipeline.pkl"):
    if request.method == "POST":
        X_test = request.get_json()

        #use the loaded model to predict a result
        with open(model, 'rb') as file:
            loaded_model_pickle = pickle.load(file)

        try: 
            predictions_pickle = loaded_model_pickle.predict(X_test)
            
        except Exception as e:
            predictions_pickle = None
            print("An error occurred:", e)
        
        return jsonify({'predictions': predictions_pickle.tolist()})

# Evaluate the model
@app.route("/eval",methods = ["POST"])
def evalmodel(model="XBGpipeline.pkl"):
    #requires json file with y_pred, y_actual
    data = request.get_json()
    df = pd.DataFrame(data)
    y_pred = df.drop('target', axis=1)
    y_actual = df['target']

    # Create a class to hold the results of model evaluation mse, r_squared
    class EvalMetrics:
        def __init__(self):
            self.mse = 0
            self.r_squared = 0
    
    EvalMetrics.mse = mean_squared_error(y_pred, y_actual)
    EvalMetrics.r_squared = r2_score(y_pred, y_actual)
    return jsonify({'Metrics': EvalMetrics.tolist()})