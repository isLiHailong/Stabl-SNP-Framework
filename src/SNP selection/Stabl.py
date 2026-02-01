import os
import numpy as np
import pandas as pd
from utils.config import DEFAULT_SEED, NUMERIC_STABILITY, STABILITY_SELECTION
from utils.numeric2 import discretizeback, chi2test, snplist, labels, Xgenoint, Xgeno_z

B = 200
SUBSAMPLE_FRAC = 0.7


def sqrtpsd(A):
    A = (A + A.T) / 2.0
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def gaussianequicorrelatedknockoffs(X, ridge=None, eps=None, jitter=None, seed=None):
    ridge = float(NUMERIC_STABILITY["ridge"] if ridge is None else ridge)
    eps = float(NUMERIC_STABILITY["eps"] if eps is None else eps)
    jitter = float(NUMERIC_STABILITY["jitter"] if jitter is None else jitter)
    seed = int(DEFAULT_SEED if seed is None else seed)

    rng = np.random.default_rng(seed)
    n, p = X.shape

    Sigma = np.cov(X, rowvar=False)
    Sigma = (Sigma + Sigma.T) / 2.0
    Sigma = (1.0 - ridge) * Sigma + ridge * np.eye(p)

    w = np.linalg.eigvalsh(Sigma)
    lamin = float(np.min(w))
    if lamin <= eps:
        Sigma = Sigma + (jitter - lamin) * np.eye(p)
        Sigma = (Sigma + Sigma.T) / 2.0
        lamin = float(np.min(np.linalg.eigvalsh(Sigma)))

    sval = min(2.0 * lamin, 1.0)
    s = np.full(p, sval)
    S = np.diag(s)

    invSigma = np.linalg.inv(Sigma)
    C2 = 2.0 * S - S @ invSigma @ S
    C = sqrtpsd(C2)

    A = np.eye(p) - invSigma @ S
    U = rng.standard_normal((n, p))
    return X @ A + U @ C


def choosethetaminfdp(freqreal, freqko, theta_grid=None):
    if theta_grid is None:
        theta_grid = np.linspace(
            float(STABILITY_SELECTION["theta_left"]),
            float(STABILITY_SELECTION["theta_right"]),
            int(STABILITY_SELECTION["theta_size"]),
        )

    best = None
    for t in theta_grid:
        sreal = int(np.sum(freqreal >= t))
        sko = int(np.sum(freqko >= t))
        if sreal == 0:
            continue
        fdpplus = (sko + 1.0) / float(sreal)
        cand = (fdpplus, -float(t), sreal)
        if (best is None) or (cand < best[0]):
            best = (cand, float(t), float(fdpplus), sreal, sko)
    if best is None:
        return None
    return best[1]


def chi2pvalsgenotypevsgroup(Xint, y):
    n, p = Xint.shape
    pvals = np.ones(p)

    if len(np.unique(y)) < 2:
        return pvals

    for j in range(p):
        x = Xint[:, j]
        tab3 = np.zeros((3, 2), dtype=int)
        for g in (0, 1, 2):
            m = (x == g)
            tab3[g, 0] = int(np.sum(m & (y == 0)))
            tab3[g, 1] = int(np.sum(m & (y == 1)))

        if tab3.sum() == 0:
            pvals[j] = 1.0
            continue

        keep = tab3.sum(axis=1) > 0
        tab = tab3[keep, :]

        if tab.shape[0] < 2:
            pvals[j] = 1.0
            continue

        try:
            _, pval = chi2test(tab)
            pvals[j] = float(pval)
        except Exception:
            pvals[j] = 1.0

    return pvals


snp_list = snplist
y = labels
X_int = Xgenoint
Xz = Xgeno_z

Xk = gaussianequicorrelatedknockoffs(Xz)
Xk_int = discretizeback(Xk)

rng = np.random.default_rng(int(DEFAULT_SEED))
n, p = X_int.shape

selcountsreal = np.zeros(p)
selcountsko = np.zeros(p)

pgrid = np.logspace(
    float(STABILITY_SELECTION["pgrid_left"]),
    float(STABILITY_SELECTION["pgrid_right"]),
    int(STABILITY_SELECTION["pgrid_size"]),
)

for b in range(B):
    idx = rng.choice(n, size=int(n * SUBSAMPLE_FRAC), replace=False)
    yb = y[idx]
    Xb = X_int[idx, :]
    Xkb = Xk_int[idx, :]

    preal = chi2pvalsgenotypevsgroup(Xb, yb)
    pko = chi2pvalsgenotypevsgroup(Xkb, yb)

    selectedreal = np.zeros(p, dtype=bool)
    selectedko = np.zeros(p, dtype=bool)

    for pcut in pgrid:
        selectedreal |= (preal <= pcut)
        selectedko |= (pko <= pcut)

    selcountsreal += selectedreal
    selcountsko += selectedko

freqreal = selcountsreal / float(B)
freqko = selcountsko / float(B)

theta = choosethetaminfdp(freqreal, freqko)

selmask = (freqreal >= theta) if theta is not None else np.zeros_like(freqreal, dtype=bool)
selected = [snp_list[i] for i in np.where(selmask)[0]]
