import os
import pandas as pd
import numpy as np

def readcsv(path):
    return pd.read_csv(path, low_memory=False)

def writecsv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def joinpath(*args):
    return os.path.join(*args)

def tofloat(x):
    return pd.to_numeric(x, errors="coerce")

def isnan(x):
    return pd.isna(x)
