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

def main(client):
    # load the California Housing data set from scikit-learn
    housingX, housingy = datasets.fetch_california_housing (return_X_y=True, as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(housingX,housingy, test_size=0.2, random_state=17)

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    train_start = time.time()

    housevalue_dask = dxgb.DaskXGBRegressor()
    housevalue_dask.fit(X_train,y_train, eval_set=[(X_test,y_test),(X_train,y_train)])

    train_end = time.time()

    print("Training time with dask is :{0:.3f}".format((train_end - train_start) * 10**3), "ms")

    pred_start = time.time()
    ypred = housevalue_dask.predict(dtest)
    pred_end = time.time()

    print("Prediction time with dask is :{0:.3f}".format((pred_end - pred_start) * 10**3), "ms")

    xgb_dask_r2 = r2_score(y_true=y_test, y_pred= ypred)
    print("XGBoost Rsquared is {0:.2f}".format(xgb_dask_r2))

if __name__ =="__main__":
    with localCluster(n_workers = 4, threads_per_worker = 1) as cluster:
        with Client(cluster) as client:
            main(client)