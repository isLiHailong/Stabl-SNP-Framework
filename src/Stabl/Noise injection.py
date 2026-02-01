import numpy as np

from utils.config import DEFAULTSEED, NUMERICSTABILITY
from utils.numeric3 import buildXgenoz, buildXgenoint, buildlabels, buildsnplist


def sqrtpsd(A):
    A = (A + A.T) / 2.0
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def gaussianequicorrelatedknockoffs(X, ridge=None, eps=None, jitter=None, seed=None):
    ridge = float(NUMERICSTABILITY["ridge"] if ridge is None else ridge)
    eps = float(NUMERICSTABILITY["eps"] if eps is None else eps)
    jitter = float(NUMERICSTABILITY["jitter"] if jitter is None else jitter)
    seed = int(DEFAULTSEED if seed is None else seed)

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


def loadcandidategenotypes(
    genotypefile,
    groupfile,
    snplistfile,
    sampleprefix="Y1_",
):
    snplist = buildsnplist(snplistfile)
    Xint, Ximp, samplecols = buildXgenoint(
        genotypefile,
        snplist,
        sampleprefix=sampleprefix,
    )
    labels = buildlabels(groupfile, samplecols)
    use = labels >= 0
    y = labels[use].astype(int)

    Xint = Xint[use, :]
    Xz, scaler = buildXgenoz(Ximp[use, :])

    return snplist, y, Xint, Xz, scaler


def runnoiseinjection(Xz, seed=DEFAULTSEED):
    return gaussianequicorrelatedknockoffs(Xz, seed=seed)
