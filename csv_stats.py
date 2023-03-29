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
#import numpy as np
#from sklearn import linear_model

def get_input(local=False):
    if local:
        print("Reading local file covid19_cases.csv")

        return "covid19_cases.csv"

    dids = os.getenv("DIDS", None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    for did in dids:
        filename = f"data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {filename}.")

        return filename
def run_csv_stats(local=False):
    filename = get_input(local)
    if not filename:
        print("Could not retrieve filename.")
        return

    df = pd.read_csv(filename)
    rows = len(df.axes[0])
 
# computing number of columns
    cols = len(df.axes[1])
 

    print("Number of Rows: ", rows)
    print("Number of Columns: ", cols)
 
# creating a list of column names by
# calling the .columns
    list_of_column_names = list(df.columns)
 
# displaying the list of column names
    print('List of column names : ',list_of_column_names)
    filename = "csv_stats.txt" if local else "/data/outputs/result"
    with open(filename,"w") as stats_file:
        stats_file.write("Number of Rows:")
        stats_file.write(str(rows))
        stats_file.write('\n')
        stats_file.write("Number of Columns: ")
        stats_file.write(str(cols))
        stats_file.write('\n')
        stats_file.write('List of column names : ')
        stats_file.write(str(list_of_column_names))
    

if __name__ == "__main__":
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_csv_stats(local)        


