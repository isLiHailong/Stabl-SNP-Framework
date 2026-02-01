import numpy as np

from utils.config import DEFAULT_SEED, NUMERIC_STABILITY
from utils.numeric3 import build_Xgeno_z, build_Xgenoint, build_labels, build_snplist


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


def load_candidate_genotypes(
    genotype_file,
    group_file,
    snp_list_file,
    sample_prefix="Y1_",
):
    snp_list = build_snplist(snp_list_file)
    X_int, X_imp, sample_cols = build_Xgenoint(
        genotype_file,
        snp_list,
        sample_prefix=sample_prefix,
    )
    labels = build_labels(group_file, sample_cols)
    use = labels >= 0
    y = labels[use].astype(int)
    X_int = X_int[use, :]
    Xz, scaler = build_Xgeno_z(X_imp[use, :])
    return snp_list, y, X_int, Xz, scaler


def run_noise_injection(Xz, seed=DEFAULT_SEED):
    return gaussianequicorrelatedknockoffs(Xz, seed=seed)