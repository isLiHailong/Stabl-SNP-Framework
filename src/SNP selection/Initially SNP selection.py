import os
from utils.IO import readcsv, writecsv, joinpath, tofloat, isnan

def run(statscsv, outdir, pthreshold):
    df = readcsv(statscsv)

    if "praw" not in df.columns:
        raise ValueError("Input stats file must contain a 'praw' column.")


    df["praw"] = tofloat(df["praw"])
    df = df[~isnan(df["praw"])]

    passed = df[df["praw"] < pthreshold]
    passed = passed.sort_values("praw")


    os.makedirs(outdir, exist_ok=True)
    outpath = joinpath(outdir, "passed.snps.praw.lt.csv")
    writecsv(passed, outpath)

    return passed.shape[0], outpath
