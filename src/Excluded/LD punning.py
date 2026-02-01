import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from utils.IO import writecsv, joinpath, readcsv
from utils.numeric3 import buildsnplist, buildXgenoint
from utils.validate import requirefile


def standardizematrix(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0, ddof=1)
    std = np.where(np.isfinite(std) & (std > 0), std, 0.0)

    Z = X - mean
    valid = std > 0
    if np.any(valid):
        Z[:, valid] /= std[valid]
    Z[:, ~valid] = 0.0

    return Z, std


def r2matrixfromZ(Z: np.ndarray) -> np.ndarray:
    n = int(Z.shape[0])
    denom = max(float(n - 1), 1.0)
    C = (Z.T @ Z) / denom
    R2 = np.square(C)
    np.fill_diagonal(R2, 0.0)
    return R2


def windowpruneonce(
    Zwin: np.ndarray,
    idxs: List[int],
    r2threshold: float,
) -> List[int]:
    if len(idxs) < 2:
        return idxs

    active = np.ones(len(idxs), dtype=bool)

    while True:
        Zact = Zwin[:, active]
        if Zact.shape[1] < 2:
            break

        R2 = r2matrixfromZ(Zact)
        hit = R2 > float(r2threshold)
        if not np.any(hit):
            break

        deg = np.sum(hit, axis=1).astype(int)
        maxdeg = int(np.max(deg))
        cand = np.where(deg == maxdeg)[0]

        if cand.size == 1:
            droplocal = int(cand[0])
        else:
            var = np.nanvar(Zact, axis=0, ddof=1)
            var = np.where(np.isfinite(var), var, -1.0)
            droplocal = int(cand[np.argmin(var[cand])])

        activeidx = np.where(active)[0]
        active[activeidx[droplocal]] = False

    kept = [idxs[i] for i, a in enumerate(active) if a]
    return kept


def normalizechr(x):
    if pd.isna(x):
        return None

    s = str(x).strip()
    s = s.replace("CHR", "").replace("chr", "").replace("Chr", "").strip()

    if s in {"X", "x"}:
        return 23
    if s in {"Y", "y"}:
        return 24

    try:
        return int(float(s))
    except Exception:
        return None


def loadsnppositions(annotationcsv: str, snplist: List[str]) -> pd.DataFrame:
    ann = readcsv(annotationcsv)
    need = {"SNPName", "CHR", "BP"}
    if not need.issubset(set(ann.columns)):
        raise ValueError(f"annotationcsv must contain columns {need}")

    ann = ann[["SNPName", "CHR", "BP"]].copy()
    ann["SNPName"] = ann["SNPName"].astype(str)
    ann["CHR"] = ann["CHR"].map(normalizechr)
    ann["BP"] = pd.to_numeric(ann["BP"], errors="coerce")

    ann = ann.dropna(subset=["CHR", "BP"])
    ann["CHR"] = ann["CHR"].astype(int)
    ann["BP"] = ann["BP"].astype(int)

    ann = ann[ann["SNPName"].isin(set(map(str, snplist)))].copy()
    if ann.empty:
        raise ValueError("No SNPs from snplist found in annotationcsv.")

    ann = ann.drop_duplicates(subset=["SNPName"], keep="first")

    miss = [s for s in snplist if s not in set(ann["SNPName"].tolist())]
    if miss:
        raise ValueError(
            f"Missing SNP positions in annotationcsv (first 10): {miss[:10]}"
        )

    ann["order"] = ann["SNPName"].map(
        {s: i for i, s in enumerate(snplist)}
    ).astype(int)

    ann = ann.sort_values(["CHR", "BP", "order"]).reset_index(drop=True)
    return ann


def ldpruneindeppairwisekb(
    X: np.ndarray,
    snplist: List[str],
    snpposdf: pd.DataFrame,
    r2threshold: float = 0.2,
    windowkb: float = 250.0,
    stepkb: float = 50.0,
) -> Dict[str, object]:
    if len(snplist) != X.shape[1]:
        raise ValueError("snplist length must match X columns.")

    if X.shape[0] < 2 or X.shape[1] < 2:
        return {"kept": snplist, "removed": []}

    if windowkb <= 0 or stepkb <= 0:
        raise ValueError("windowkb and stepkb must be > 0")

    snptocol = {s: i for i, s in enumerate(map(str, snplist))}
    pos = snpposdf.copy()
    pos["colidx"] = pos["SNPName"].map(snptocol).astype(int)

    Z, std = standardizematrix(X)
    keep = np.ones(X.shape[1], dtype=bool)
    keep = keep | (std == 0)

    windowbp = int(round(float(windowkb) * 1000.0))
    stepbp = int(round(float(stepkb) * 1000.0))

    for chrid, sub in pos.groupby("CHR", sort=True):
        bps = sub["BP"].to_numpy(dtype=int)
        cols = sub["colidx"].to_numpy(dtype=int)

        if bps.size < 2:
            continue

        startbp = int(bps.min())
        endbp = int(bps.max())

        left = startbp
        while left <= endbp:
            right = left + windowbp

            inwin = (bps >= left) & (bps <= right)
            wincolsall = cols[inwin].tolist()
            wincols = [j for j in wincolsall if keep[j] and (std[j] > 0)]

            if len(wincols) >= 2:
                Zwin = Z[:, wincols]
                keptcols = windowpruneonce(
                    Zwin,
                    wincols,
                    r2threshold=float(r2threshold),
                )
                dropcols = set(wincols) - set(keptcols)
                for j in dropcols:
                    keep[j] = False

            left += stepbp

    kept = [snplist[i] for i in range(len(snplist)) if keep[i]]
    removed = [snplist[i] for i in range(len(snplist)) if not keep[i]]
    return {"kept": kept, "removed": removed}


def runldfilterkb(
    genotypecsv: str,
    snplistcsv: str,
    annotationcsv: str,
    outdir: str,
    r2threshold: float = 0.2,
    windowkb: float = 250.0,
    stepkb: float = 50.0,
    sampleprefix: str = "Y1_",
    chunksize: int = 8000,
):
    requirefile(genotypecsv)
    requirefile(snplistcsv)
    requirefile(annotationcsv)

    snplist = buildsnplist(snplistcsv)
    Xint, Ximp, samplecols = buildXgenoint(
        genotypecsv,
        snplist,
        sampleprefix=sampleprefix,
        chunksize=chunksize,
    )

    posdf = loadsnppositions(annotationcsv, snplist)

    result = ldpruneindeppairwisekb(
        Ximp,
        snplist,
        posdf,
        r2threshold=float(r2threshold),
        windowkb=float(windowkb),
        stepkb=float(stepkb),
    )

    kept = result["kept"]
    removed = result["removed"]

    os.makedirs(outdir, exist_ok=True)
    keptpath = joinpath(outdir, "ld.kb.kept.snps.list.csv")
    removedpath = joinpath(outdir, "ld.kb.removed.snps.list.csv")

    writecsv(pd.DataFrame({"SNPName": kept}), keptpath)
    writecsv(pd.DataFrame({"SNPName": removed}), removedpath)

    return len(kept), keptpath, removedpath


__all__ = [
    "ldpruneindeppairwisekb",
    "runldfilterkb",
    "loadsnppositions",
]
