import os
from utils.IO import readcsv, writecsv, joinpath, tofloat, isnan


def run(statscsv, outdir, pthreshold, snp_column="SNPName"):
    df = readcsv(statscsv)

    if "praw" not in df.columns:
        raise ValueError("Input stats file must contain a 'praw' column.")
    if snp_column not in df.columns:
        raise ValueError(f"Input stats file must contain '{snp_column}' column.")


    df["praw"] = tofloat(df["praw"])
    df = df[~isnan(df["praw"])]

    passed = df[df["praw"] < pthreshold]
    passed = passed.sort_values("praw")


    os.makedirs(outdir, exist_ok=True)
    outpath = joinpath(outdir, "passed.snps.praw.lt.csv")
    writecsv(passed, outpath)
    snplist = passed[[snp_column]].dropna()
    snplist_path = joinpath(outdir, "passed.snps.list.csv")
    writecsv(snplist, snplist_path)

    return passed.shape[0], outpath, snplist_path