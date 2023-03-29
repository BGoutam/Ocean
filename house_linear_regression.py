# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:27:32 2023

@author: Goutam
"""

# loading dependencies
import json
import os
import pickle
import sys
import pandas as pd
import numpy as np
from sklearn import linear_model

def get_input(local=False):
    if local:
        print("Reading local file houseprice_train.csv")

        return "houseprice_train.csv"

    dids = os.getenv("DIDS", None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    for did in dids:
        filename = f"data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {filename}.")

        return filename
def run_linear_regression(local=False):
    filename = get_input(local)
    if not filename:
        print("Could not retrieve filename.")
        return

    train_df = pd.read_csv(filename, header=0)
# loading our data
#train_df = pd.read_csv('houseprice_train.csv')
# viewing few files
    print(train_df.head())

# creating the model object
    model = linear_model.LinearRegression() # y = mx+b
# fitting model with X_train - area, y_train - price
    model.fit(train_df[['area']],train_df.price)
    print("Coeeficient - ",model.coef_)
    print("Intercept - ",model.intercept_)
# predict model values - area = 5000
    print("Model Predict - ",model.predict([[5000]]))
    filename = "HP_Linear_Regression_regression.pickle" if local else "/data/outputs/result"
    with open(filename, "wb") as pickle_file:
        print(f"Pickling results in {filename}")
        pickle.dump(model, pickle_file)

if __name__ == "__main__":
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_linear_regression(local)        
"""
# create an iterator object with write permission - model.pkl
with open('model_pkl', 'wb') as files:
    pickle.dump(model, files)
# load saved model
with open('model_pkl' , 'rb') as f:
    lr = pickle.load(f)
# check prediction

print("Predict after reloading model from pickle ",lr.predict([[5000]])) 
"""

