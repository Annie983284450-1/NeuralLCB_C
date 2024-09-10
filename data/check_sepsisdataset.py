import pandas as pd
import numpy as np
import sys 
import os
sys.path.append(os.getcwd())
df = pd.read_csv('./data/fully_imputed.csv')
print(df.head())
print(df.dtypes)
