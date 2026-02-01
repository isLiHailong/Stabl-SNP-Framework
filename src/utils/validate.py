import os
import numpy as np

def requirefile(path):
    if (path is None) or (str(path).strip() == ""):
        raise ValueError("path is empty")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

def requiredir(path):
    if (path is None) or (str(path).strip() == ""):
        raise ValueError("path is empty")
    if not os.path.isdir(path):
        raise NotADirectoryError(path)

def requirecols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError("missing columns: " + ", ".join(missing))

def requireunique(values, name="values"):
    arr = np.asarray(values)
    if arr.size == 0:
        raise ValueError(name + " is empty")
    u = np.unique(arr.astype(str))
    if u.size != arr.size:
        raise ValueError(name + " contains duplicates")

def requireshape(x, nd=None, nrows=None, ncols=None, name="array"):
    a = np.asarray(x)
    if nd is not None and a.ndim != int(nd):
        raise ValueError(f"{name} ndim expected {int(nd)} got {a.ndim}")
    if nrows is not None and a.shape[0] != int(nrows):
        raise ValueError(f"{name} nrows expected {int(nrows)} got {a.shape[0]}")
    if ncols is not None and a.shape[1] != int(ncols):
        raise ValueError(f"{name} ncols expected {int(ncols)} got {a.shape[1]}")
    return a

def requirebinary(y, name="y"):
    a = np.asarray(y)
    u = np.unique(a)
    if u.size == 0:
        raise ValueError(name + " is empty")
    ok = set(u.tolist()).issubset({0, 1})
    if not ok:
        raise ValueError(name + " must be binary {0,1}")
