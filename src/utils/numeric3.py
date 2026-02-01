import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def buildsnplist(snplistcsv):
    df = pd.read_csv(snplistcsv, low_memory=False)
    if "SNPName" not in df.columns:
        raise ValueError("SNP list file must contain column 'SNPName'")
    return df["SNPName"].astype(str).tolist()


def buildlabels(groupcsv, sampleids):
    part = pd.read_csv(groupcsv, low_memory=False)
    need = {"SampleID", "Group"}
    if not need.issubset(part.columns):
        raise ValueError("Group file must contain SampleID and Group")

    partmap = dict(zip(part["SampleID"].astype(str), part["Group"].astype(str)))

    labels = np.full(len(sampleids), -1, dtype=np.int8)
    for i, sid in enumerate(sampleids):
        g = partmap.get(str(sid), None)
        if g == "A":
            labels[i] = 1
        elif g == "B":
            labels[i] = 0
    return labels


def buildXgenoint(genotypecsv, snplist, sampleprefix="Y1_", chunksize=8000):
    reader = pd.read_csv(genotypecsv, low_memory=False, chunksize=chunksize)

    keep = []
    samplecols = None
    snpset = set(snplist)

    for chunk in reader:
        if samplecols is None:
            if "SNPName" not in chunk.columns:
                raise ValueError("Genotype file missing SNPName")
            samplecols = [c for c in chunk.columns if str(c).startswith(sampleprefix)]
            if not samplecols:
                raise ValueError("No sample columns found")

        sub = chunk[chunk["SNPName"].astype(str).isin(snpset)]
        if len(sub) > 0:
            keep.append(sub[["SNPName"] + samplecols])

    if len(keep) == 0:
        raise ValueError("No SNP rows were found for the given snplist.")

    df = pd.concat(keep, axis=0, ignore_index=True)
    df["SNPName"] = df["SNPName"].astype(str)

    missing = [s for s in snplist if s not in set(df["SNPName"].tolist())]
    if missing:
        raise ValueError(f"Missing SNP rows in genotype file (first 10): {missing[:10]}")

    df = df.set_index("SNPName").loc[snplist]

    G = df[samplecols].to_numpy()
    G = np.where(pd.isna(G), -1, G).astype(np.int16)

    X = G.T

    Ximp = X.astype(float)
    for j in range(Ximp.shape[1]):
        col = Ximp[:, j]
        m = col >= 0
        if not np.any(m):
            col[:] = 0.0
        else:
            mu = float(np.mean(col[m]))
            col[~m] = mu
        Ximp[:, j] = col

    Xint = np.clip(np.rint(Ximp), 0, 2).astype(np.int8)
    return Xint, Ximp, samplecols


def buildXgenoz(Ximp):
    scaler = StandardScaler()
    Xz = scaler.fit_transform(Ximp.astype(float))
    return Xz, scaler


__all__ = ["buildsnplist", "buildlabels", "buildXgenoint", "buildXgenoz"]
