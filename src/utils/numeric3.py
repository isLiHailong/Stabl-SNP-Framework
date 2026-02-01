import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_snplist(snp_list_csv):
    df = pd.read_csv(snp_list_csv, low_memory=False)
    if "SNPName" not in df.columns:
        raise ValueError("SNP list file must contain column 'SNPName'")
    return df["SNPName"].astype(str).tolist()


def build_labels(group_csv, sample_ids):
    part = pd.read_csv(group_csv, low_memory=False)
    need = {"SampleID", "Group"}
    if not need.issubset(part.columns):
        raise ValueError("Group file must contain SampleID and Group")

    part_map = dict(zip(part["SampleID"].astype(str), part["Group"].astype(str)))

    labels = np.full(len(sample_ids), -1, dtype=np.int8)
    for i, sid in enumerate(sample_ids):
        g = part_map.get(str(sid), None)
        if g == "A":
            labels[i] = 1
        elif g == "B":
            labels[i] = 0
    return labels


def build_Xgenoint(genotype_csv, snplist, sample_prefix="Y1_", chunksize=8000):
    reader = pd.read_csv(genotype_csv, low_memory=False, chunksize=chunksize)

    keep = []
    sample_cols = None
    snp_set = set(snplist)

    for chunk in reader:
        if sample_cols is None:
            if "SNPName" not in chunk.columns:
                raise ValueError("Genotype file missing SNPName")
            sample_cols = [c for c in chunk.columns if str(c).startswith(sample_prefix)]
            if not sample_cols:
                raise ValueError("No sample columns found")

        sub = chunk[chunk["SNPName"].astype(str).isin(snp_set)]
        if len(sub) > 0:
            keep.append(sub[["SNPName"] + sample_cols])

    if len(keep) == 0:
        raise ValueError("No SNP rows were found for the given snplist.")

    df = pd.concat(keep, axis=0, ignore_index=True)
    df["SNPName"] = df["SNPName"].astype(str)

    missing = [s for s in snplist if s not in set(df["SNPName"].tolist())]
    if missing:
        raise ValueError(f"Missing SNP rows in genotype file (first 10): {missing[:10]}")

    df = df.set_index("SNPName").loc[snplist]

    G = df[sample_cols].to_numpy()
    G = np.where(pd.isna(G), -1, G).astype(np.int16)

    X = G.T

    X_imp = X.astype(float)
    for j in range(X_imp.shape[1]):
        col = X_imp[:, j]
        m = col >= 0
        if not np.any(m):
            col[:] = 0.0
        else:
            mu = float(np.mean(col[m]))
            col[~m] = mu
        X_imp[:, j] = col

    X_int = np.clip(np.rint(X_imp), 0, 2).astype(np.int8)
    return X_int, X_imp, sample_cols


def build_Xgeno_z(X_imp):
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_imp.astype(float))
    return Xz, scaler


__all__ = ["build_snplist", "build_labels", "build_Xgenoint", "build_Xgeno_z"]
