import os
import pandas as pd

# print(os.path.dirname(__file__))
dir = os.path.join(os.path.dirname(__file__), "results")
filename = ('1k_results.csv')
filepath = os.path.join(dir, filename)

# dadus = pd.read_csv(filepath)