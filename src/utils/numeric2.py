import numpy as np

try:
    from scipy.stats import chi2 as _chi2_dist
except Exception:
    _chi2_dist = None


def _kmeans1d_3(x, seed=42, max_iter=50, tol=1e-8):
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 0:
        return np.array([], dtype=int), np.array([np.nan, np.nan, np.nan], dtype=float)

    if np.allclose(x, x[0]):
        return np.zeros(n, dtype=int), np.array([x[0], x[0], x[0]], dtype=float)

    rng = np.random.default_rng(seed)
    qs = np.quantile(x, [0.2, 0.5, 0.8])
    cents = qs.copy()
    if np.any(~np.isfinite(cents)) or np.unique(cents).size < 3:
        lo, hi = np.nanmin(x), np.nanmax(x)
        cents = np.linspace(lo, hi, 3)
    cents = cents.astype(float)

    prev_obj = np.inf
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        d = np.abs(x[:, None] - cents[None, :])
        labels = np.argmin(d, axis=1)

        new_cents = cents.copy()
        for k in range(3):
            m = labels == k
            if np.any(m):
                new_cents[k] = np.mean(x[m])
            else:
                new_cents[k] = x[rng.integers(0, n)]

        d2 = (x - new_cents[labels]) ** 2
        obj = float(np.sum(d2))

        if abs(prev_obj - obj) <= tol * (1.0 + prev_obj):
            cents = new_cents
            break

        prev_obj = obj
        cents = new_cents

    return labels, cents


def discretizeback(X, method="kmeans", seed=42):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    out = np.zeros((n, p), dtype=np.int8)

    if method != "kmeans":
        raise ValueError("Only method='kmeans' is supported in this implementation.")

    for j in range(p):
        col = X[:, j].astype(float, copy=False)
        labels, cents = _kmeans1d_3(col, seed=seed)
        order = np.argsort(cents)
        remap = np.empty(3, dtype=int)
        remap[order[0]] = 0
        remap[order[1]] = 1
        remap[order[2]] = 2
        out[:, j] = remap[labels].astype(np.int8)

    return out


def chi2test(tab, yates_correction=False):
    tab = np.asarray(tab)
    if tab.ndim != 2:
        raise ValueError("tab must be a 2D contingency table.")
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        raise ValueError("tab must have shape at least (2, 2).")
    if np.any(tab < 0):
        raise ValueError("tab must be nonnegative.")
    if not np.issubdtype(tab.dtype, np.integer):
        tab = tab.astype(float)

    n = float(np.sum(tab))
    if n <= 0:
        return 0.0, 1.0

    row = np.sum(tab, axis=1, keepdims=True)
    col = np.sum(tab, axis=0, keepdims=True)
    expected = (row @ col) / n

    mask = expected > 0
    if not np.any(mask):
        return 0.0, 1.0

    obs = tab.astype(float)
    if yates_correction and tab.shape == (2, 2):
        stat = np.sum(((np.abs(obs - expected) - 0.5) ** 2)[mask] / expected[mask])
    else:
        stat = np.sum(((obs - expected) ** 2)[mask] / expected[mask])

    dof = int((tab.shape[0] - 1) * (tab.shape[1] - 1))
    if dof <= 0:
        return float(stat), 1.0

    if _chi2_dist is None:
        raise RuntimeError("scipy is required for chi-square p-value (scipy.stats.chi2).")

    pval = float(_chi2_dist.sf(stat, dof))
    return float(stat), pval



