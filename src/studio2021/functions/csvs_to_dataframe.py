import pandas as pd
import glob
import os
import studio2021


# for all csv files in folder_x:
#   for each csv:
#       columns a, b, c, ..., x, y, z = 1, 2, 3, ..., 7, 8, 9
path = "C:\Users\papep\Documents\GitHub\msdc-thesis\tool\temp"
def csvToDataframe(path):
    
    all_files = glob.glob(path, "*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files))
    df.shape
    print(df)