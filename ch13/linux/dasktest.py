# ----------------------------------------
# filename dasktest.py
# purpose demonstrate XGBoost distributed 
# compute functionality using dask
# author Joyce Weiner
# revision 1.0
# revision history 1.0 - initial script
# ----------------------------------------
import pandas as pd
import numpy as np
import time 
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from dask import dataframe as dd
from dask.distributed import Client, LocalCluster
from xgboost import dask as dxgb

def train(client, dtrain):
    train_start = time.time()

    housevalue_dask = xgb.dask.train(
        client,
        {"verbosity": 2, "tree_method": "approx", "objective": "reg:squarederror"},
        dtrain,
        num_boost_round=4,
        evals=[(dtrain, "train")],
    )

    train_end = time.time()

    print("Training time with dask is :{0:.3f}".format((train_end - train_start) * 10**3), "ms")
    return housevalue_dask

def predict(client, model, dtest):
    pred_start = time.time()
    ypred = xgb.dask.predict(client, model, dtest)
    pred_end = time.time()

    print("Prediction time with dask is :{0:.3f}".format((pred_end - pred_start) * 10**3), "ms")

    return ypred

if __name__ =="__main__":
    with LocalCluster(n_workers = 4, threads_per_worker = 1) as cluster:
        with Client(cluster) as client:
            # load the California Housing data set from scikit-learn
            housingX, housingy = datasets.fetch_california_housing (return_X_y=True, as_frame=True)

            X_train, X_test, y_train, y_test = train_test_split(housingX,housingy, test_size=0.2, random_state=17)
            
            dm_X_train = dd.from_pandas(X_train, npartitions=1)
            dm_X_test = dd.from_pandas(X_test, npartitions=1)
            dm_y_train = dd.from_pandas(y_train, npartitions=1)
            dm_y_test = dd.from_pandas(y_test, npartitions=1)

            dtrain= xgb.dask.DaskDMatrix(client, dm_X_train, dm_y_train)
            dtest= xgb.dask.DaskDMatrix(client, dm_X_test, dm_y_test)

            housevalue_dask = train(client, dtrain)
            y_pred = predict(client, housevalue_dask, dtest)
            
            xgb_dask_r2 = r2_score(y_true=dm_y_test, y_pred= y_pred)
            print("XGBoost with dask Rsquared is {0:.2f}".format(xgb_dask_r2))
