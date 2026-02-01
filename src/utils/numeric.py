import numpy as np
import pandas as pd

def tofloat(x):
    return pd.to_numeric(x, errors="coerce")

def isnan(x):
    return pd.isna(x)

def clip012(x):
    return np.clip(np.rint(x), 0, 2).astype(np.int8)

def safemean(x):
    m = ~np.isnan(x)
    if not np.any(m):
        return 0.0
    return float(np.mean(x[m]))
