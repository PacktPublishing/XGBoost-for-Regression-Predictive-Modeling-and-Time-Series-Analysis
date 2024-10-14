# ----------------------------------------
# filename multithreaded.py
# purpose demonstrate XGBoost multithreaded functionality
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

# load the California Housing data set from scikit-learn
housingX, housingy = datasets.fetch_california_housing (return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(housingX,housingy, test_size=0.2, random_state=17)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

train_start = time.time()

param = {"eta": 0.3, "booster": "gbtree", "nthread": 2}
housevalue_xgb = xgb.train(param, dtrain)
train_end = time.time()

print("Training time with 2 threads is :{0:.3f}".format((train_end - train_start) * 10**3), "ms")

train_start = time.time()

param = {"eta": 0.3, "booster": "gbtree", "nthread": 8}
housevalue_xgb = xgb.train(param, dtrain)
train_end = time.time()

print("Training time with 8 threads is :{0:.3f}".format((train_end - train_start) * 10**3), "ms")

pred_start = time.time()
ypred = housevalue_xgb.predict(dtest)
pred_end = time.time()

print("Prediction time is :{0:.3f}".format((pred_end - pred_start) * 10**3), "ms")

xgb_r2 = r2_score(y_true=y_test, y_pred= ypred)
print("XGBoost Rsquared is {0:.2f}".format(xgb_r2))