import os
import numpy as np
import pandas as pd

from utils.IO import writecsv, joinpath
from utils.numeric2 import chi2test
from utils.numeric3 import build_labels


def _chi2_pval_genotype_vs_group(genotypes, y):
    tab3 = np.zeros((3, 2), dtype=int)
    for g in (0, 1, 2):
        m = genotypes == g
        tab3[g, 0] = int(np.sum(m & (y == 0)))
        tab3[g, 1] = int(np.sum(m & (y == 1)))

    if tab3.sum() == 0:
        return 1.0


    keep = tab3.sum(axis=1) > 0
    tab = tab3[keep, :]
    if tab.shape[0] < 2:
        return 1.0


    try:
        _stat, pval = chi2test(tab)
        return float(pval)
    except Exception:
        return 1.0


def run(
    genotype_file,
    group_file,
    outdir,
    pthreshold=0.001,
    sample_prefix="Y1_",
    snp_column="SNPName",
    chunksize=8000,
):
    reader = pd.read_csv(genotype_file, low_memory=False, chunksize=chunksize)

    sample_cols = None
    labels = None
    use = None
    y = None

    results = []

    for chunk in reader:
        if snp_column not in chunk.columns:
            raise ValueError(f"Genotype file missing {snp_column} column.")

        if sample_cols is None:
            sample_cols = [c for c in chunk.columns if str(c).startswith(sample_prefix)]
            if not sample_cols:
                raise ValueError("No sample columns found with the given prefix.")
            labels = build_labels(group_file, sample_cols)
            use = labels >= 0
            if not np.any(use):
                raise ValueError("No samples matched to groups A/B in group_file.")
            y = labels[use].astype(int)

        data = chunk[sample_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        data = np.where(np.isfinite(data), data, np.nan)

        for idx in range(data.shape[0]):
            snp_name = str(chunk.iloc[idx][snp_column])
            vals = data[idx, use]

            if not np.any(np.isfinite(vals)):
                pval = 1.0
            else:
                vals = vals.astype(float, copy=True)
                mask = np.isfinite(vals)
                mu = float(np.mean(vals[mask]))
                vals[~mask] = mu
                vals = np.clip(np.rint(vals), 0, 2).astype(int)
                pval = _chi2_pval_genotype_vs_group(vals, y)

            results.append({snp_column: snp_name, "praw": float(pval)})

    df = pd.DataFrame(results)
    passed = df[df["praw"] < float(pthreshold)].sort_values("praw")

    os.makedirs(outdir, exist_ok=True)
    outpath = joinpath(outdir, "passed.snps.praw.lt.csv")
    writecsv(passed, outpath)

    snplist = passed[[snp_column]].dropna()
    snplist_path = joinpath(outdir, "passed.snps.list.csv")
    writecsv(snplist, snplist_path)

    return passed.shape[0], outpath, snplist_path