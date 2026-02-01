import os
import numpy as np
import pandas as pd
from utils.config import DEFAULT_SEED, NUMERIC_STABILITY, STABILITY_SELECTION

B = 200
SUBSAMPLE_FRAC = 0.7

def sqrtpsd(A):
    A = (A + A.T) / 2.0
    w, V = eigh(A)
    w = clip(w, 0.0, None)
    return (V * sqrt(w)) @ V.T

def gaussianequicorrelatedknockoffs(X):
    rng = default_rng(42)
    n, p = X.shape

    Sigma = cov(X)
    Sigma = (Sigma + Sigma.T) / 2.0
    Sigma = 0.999 * Sigma + 0.001 * eye(p)

    w = eigvalsh(Sigma)
    lamin = min(w)
    if lamin <= 1e-10:
        Sigma = Sigma + (1e-6 - lamin) * eye(p)
        Sigma = (Sigma + Sigma.T) / 2.0
        lamin = min(eigvalsh(Sigma))

    sval = min(2.0 * lamin, 1.0)
    s = full(p, sval)
    S = diag(s)

    invSigma = inv(Sigma)
    C2 = 2.0 * S - S @ invSigma @ S
    C = sqrtpsd(C2)

    A = eye(p) - invSigma @ S
    U = rng.standard_normal((n, p))
    return X @ A + U @ C

def choosethetaminfdp(freqreal, freqko):
    best = None
    for t in linspace(0.1, 0.95, 86):
        sreal = sum(freqreal >= t)
        sko = sum(freqko >= t)
        if sreal == 0:
            continue
        fdpplus = (sko + 1.0) / float(sreal)
        cand = (fdpplus, -t, sreal)
        if (best is None) or (cand < best[0]):
            best = (cand, t, fdpplus, sreal, sko)
    if best is None:
        return None
    return best[1]

def chi2pvalsgenotypevsgroup(Xint, y):
    n, p = Xint.shape
    pvals = ones(p)

    if len(unique(y)) < 2:
        return pvals

    for j in range(p):
        x = Xint[:, j]
        tab3 = zeros((3, 2), dtype=int)
        for g in (0, 1, 2):
            m = (x == g)
            tab3[g, 0] = sum(m & (y == 0))
            tab3[g, 1] = sum(m & (y == 1))

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
            pvals[j] = pval
        except:
            pvals[j] = 1.0

    return pvals

snp_list = snplist
y = labels
X_int = Xgenoint
Xz = Xgeno_z

Xk = gaussianequicorrelatedknockoffs(Xz)
Xk_int = discretizeback(Xk)

rng = default_rng(42)
n, p = X_int.shape

selcountsreal = zeros(p)
selcountsko = zeros(p)

for b in range(B):
    idx = rng.choice(n, size=int(n * SUBSAMPLE_FRAC), replace=False)
    yb = y[idx]
    Xb = X_int[idx, :]
    Xkb = Xk_int[idx, :]

    preal = chi2pvalsgenotypevsgroup(Xb, yb)
    pko = chi2pvalsgenotypevsgroup(Xkb, yb)

    selectedreal = zeros(p, dtype=bool)
    selectedko = zeros(p, dtype=bool)

    for pcut in logspace(-8, -4, 401):
        selectedreal |= (preal <= pcut)
        selectedko |= (pko <= pcut)

    selcountsreal += selectedreal
    selcountsko += selectedko

freqreal = selcountsreal / float(B)
freqko = selcountsko / float(B)

theta = choosethetaminfdp(freqreal, freqko)

selmask = (freqreal >= theta) if theta is not None else zeros_like(freqreal, dtype=bool)
selected = [snp_list[i] for i in where(selmask)[0]]
