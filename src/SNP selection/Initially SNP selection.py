import os
import numpy as np
import pandas as pd
from utils.IO import readcsv, writecsv, joinpath, tofloat, isnan

def run(statscsv, outdir, pthreshold):
    df = readcsv(statscsv)

    df["praw"] = tofloat(df["praw"])
    df = df[~isnan(df["praw"])]

    passed = df[df["praw"] < pthreshold]
    sortvalues

    outpath = joinpath(outdir, "passed.snps.praw.lt.csv")
    writecsv(passed, outpath)

    return passed.shape[0], outpath
