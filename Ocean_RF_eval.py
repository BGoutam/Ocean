# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:27:32 2023

@author: Goutam Bakshi, Zero2AI 
"""

# loading dependencies
import json
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score
import pickle
#import numpy as np
#from sklearn import linear_model

def get_input(local=False):
    if local:
        print("Reading local file Titanic_train.csv")
        pkl_file = "RF_model_Titanic.pkl"
        return "Titanic_train.csv",pkl_file

    dids = os.getenv("DIDS", None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    for did in dids:
        filename = f"data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {filename}.")
        pkl_file = f"data/inputs/{did}/1"
        print(f"Reading asset file {pkl_file}.")
        return filename,pkl_file
def run_rf_evaluate(local=False):
    filename,pickle_file = get_input(local)
    if not filename:
        print("Could not retrieve filename.")
        return
    if not pickle_file:
        print("Could not retrieve model pickle.")
        return

    loaded_model = pickle.load(open(pickle_file, 'rb'))
    data=pd.read_csv(filename)

    le=LabelEncoder()
    data['Sex']=le.fit_transform(data['Sex'].astype(str))
    data['Embarked']=le.fit_transform(data['Embarked'].astype(str))
    data.drop(['PassengerId'],axis=1,inplace=True)
    data.drop(['Name'],axis=1,inplace=True)
    data.drop(['Ticket'],axis=1,inplace=True)
    data.drop(['Cabin'],axis=1,inplace=True)
    data.drop(['Age'],axis=1,inplace=True)
#print("Sample Data",data.head())


#Divide the dataset into features and target variables
    X=data.iloc[:,1:]
    Y=data['Survived'].ravel()
#Divide the dataset into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.15, train_size=0.85)
    predict = loaded_model.predict(X_test)
    acc_score = accuracy_score(predict, Y_test)
    print("Accuracy Score is :",acc_score)
    f_score = f1_score(Y_test,predict)
    print("F1 score is :",f_score)
    conf_matrix = confusion_matrix(Y_test, predict)
    TP = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TN = conf_matrix[1][1]
    print("True Positive - ", TP)
    print("False positive - ", FP) 
    print("False negative - ", FN)
    print("True Negative - ",TN)
    filename = "RF_evaluate_result.txt" if local else "/data/outputs/result"
    with open(filename,"w") as stats_file:
        stats_file.write("Accuracy Score - ")
        stats_file.write(str(acc_score))
        stats_file.write('\n')
        stats_file.write("F1 Score - ")
        stats_file.write(str(f_score))
        stats_file.write('\n')
        stats_file.write('True Positive -  ')
        stats_file.write(str(TP))
        stats_file.write('False Positive -  ')
        stats_file.write(str(FP))
        stats_file.write('False Negative -  ')
        stats_file.write(str(FN))
        stats_file.write('True Negative -  ')
        stats_file.write(str(TN))
    

if __name__ == "__main__":
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_rf_evaluate(local)        


