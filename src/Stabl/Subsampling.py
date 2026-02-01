import numpy as np

from utils.config import DEFAULTSEED, STABILITYSELECTION
from utils.numeric2 import chi2test


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
            stat, pval = chi2test(tab)
            pvals[j] = float(pval)
        except Exception:
            pvals[j] = 1.0

    return pvals


def runsubsampling(
    Xint,
    y,
    Bruns,
    subsamplefrac,
    pgrid=None,
    seed=DEFAULTSEED,
):
    rng = np.random.default_rng(int(seed))
    n, p = Xint.shape

    if pgrid is None:
        pgrid = np.logspace(
            float(STABILITYSELECTION["pgridleft"]),
            float(STABILITYSELECTION["pgridright"]),
            int(STABILITYSELECTION["pgridsize"]),
        )
    pgrid = np.asarray(pgrid, dtype=float)

    freqbycutoff = np.zeros((pgrid.size, p), dtype=float)

    for b in range(int(Bruns)):
        idx = rng.choice(n, size=int(n * subsamplefrac), replace=False)
        yb = y[idx]
        Xb = Xint[idx, :]

        pvals = chi2pvalsgenotypevsgroup(Xb, yb)

        for i, cutoff in enumerate(pgrid):
            freqbycutoff[i] += (pvals <= cutoff)

    freqbycutoff /= float(Bruns)
    freqmax = np.max(freqbycutoff, axis=0)

    return {
        "pgrid": pgrid,
        "freqbycutoff": freqbycutoff,
        "freqmax": freqmax,
    }
