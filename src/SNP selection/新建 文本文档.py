import numpy as np

from utils.config import DEFAULT_SEED, STABILITY_SELECTION
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
            _, pval = chi2test(tab)
            pvals[j] = float(pval)
        except Exception:
            pvals[j] = 1.0

    return pvals


def run_subsampling(
    X_int,
    y,
    B_runs,
    subsample_frac,
    pgrid=None,
    seed=DEFAULT_SEED,
):
    rng = np.random.default_rng(int(seed))
    n, p = X_int.shape

    if pgrid is None:
        pgrid = np.logspace(
            float(STABILITY_SELECTION["pgrid_left"]),
            float(STABILITY_SELECTION["pgrid_right"]),
            int(STABILITY_SELECTION["pgrid_size"]),
        )
    pgrid = np.asarray(pgrid, dtype=float)

    freq_by_cutoff = np.zeros((pgrid.size, p), dtype=float)

    for _ in range(int(B_runs)):
        idx = rng.choice(n, size=int(n * subsample_frac), replace=False)
        yb = y[idx]
        Xb = X_int[idx, :]

        pvals = chi2pvalsgenotypevsgroup(Xb, yb)

        for i, cutoff in enumerate(pgrid):
            freq_by_cutoff[i] += (pvals <= cutoff)

    freq_by_cutoff /= float(B_runs)
    freq_max = np.max(freq_by_cutoff, axis=0)

    return {
        "pgrid": pgrid,
        "freq_by_cutoff": freq_by_cutoff,
        "freq_max": freq_max,
    }