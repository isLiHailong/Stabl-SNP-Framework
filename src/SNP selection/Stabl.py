import numpy as np
from utils.config import DEFAULT_SEED, NUMERIC_STABILITY, STABILITY_SELECTION
from utils.numeric2 import discretizeback, chi2test
from utils.numeric3 import build_snplist, build_labels, build_Xgenoint, build_Xgeno_z
from utils.IO import readcsv, tofloat, isnan

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
        cand = (float(fdpplus), -float(t), int(sreal))
        if (best is None) or (cand < best[0]):
            best = (cand, float(t), float(fdpplus), int(sreal), int(sko))
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


def load_inputs(
    genotype_file,
    group_file,
    sample_prefix="Y1_",
    snp_list_file=None,
    snp_list=None,
):
    if snp_list is None:
        if snp_list_file is None:
            raise ValueError("Either snp_list or snp_list_file must be provided.")
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
    Xz, _scaler = build_Xgeno_z(X_imp[use, :])

    return snp_list, y, X_int, Xz

def build_snplist_from_stats(statscsv, pthreshold, snp_column="SNPName"):
    df = readcsv(statscsv)
    if "praw" not in df.columns:
        raise ValueError("Input stats file must contain a 'praw' column.")
    if snp_column not in df.columns:
        raise ValueError(f"Input stats file must contain '{snp_column}' column.")

    df["praw"] = tofloat(df["praw"])
    df = df[~isnan(df["praw"])]

    passed = df[df["praw"] < pthreshold]
    snplist = passed[snp_column].dropna().astype(str).tolist()
    if not snplist:
        raise ValueError("No SNPs passed the p-value threshold.")
    return snplist




def run_stability_selection(
    snp_list_file=None,
    stats_file=None,
    pthreshold=None,
    snp_column="SNPName",
    genotype_file=None,
    group_file=None,
    sample_prefix="Y1_",
    x_int=None,
    x_z=None,
    y=None,
    snp_list=None,
    B_runs=B,
    subsample_frac=SUBSAMPLE_FRAC,
    seed=DEFAULT_SEED,
):
    if (x_int is None) or (x_z is None) or (y is None) or (snp_list is None):
        if (snp_list_file is None) or (genotype_file is None) or (group_file is None):
            if stats_file is None or pthreshold is None:
                raise ValueError(
                    "Provide either (x_int, x_z, y, snp_list) or provide "
                    "(snp_list_file, genotype_file, group_file) to load inputs. "
                    "Alternatively, pass stats_file with pthreshold to derive snp_list."
                )
        if snp_list is None and snp_list_file is None:
            snp_list = build_snplist_from_stats(
                statscsv=stats_file,
                pthreshold=pthreshold,
                snp_column=snp_column,
            )
        elif snp_list is None:
            snp_list = build_snplist(snp_list_file)
        snp_list, y, x_int, x_z = load_inputs(
            genotype_file=genotype_file,
            group_file=group_file,
            sample_prefix=sample_prefix,
            snp_list_file=snp_list_file,
            snp_list=snp_list,
        )
        
    if x_int.shape[0] != x_z.shape[0]:
        raise ValueError("x_int and x_z must have the same number of rows.")
    if x_int.shape[0] != len(y):
        raise ValueError("x_int and y must have the same number of samples.")
    if x_int.shape[1] != len(snp_list):
        raise ValueError("x_int columns must match snp_list length.")

    Xk = gaussianequicorrelatedknockoffs(x_z, seed=seed)
    Xk_int = discretizeback(Xk, seed=int(seed))

    rng = np.random.default_rng(int(seed))
    n, p = x_int.shape

    selcountsreal = np.zeros(p)
    selcountsko = np.zeros(p)

    pgrid = np.logspace(
        float(STABILITY_SELECTION["pgrid_left"]),
        float(STABILITY_SELECTION["pgrid_right"]),
        int(STABILITY_SELECTION["pgrid_size"]),
    )

    for _ in range(int(B_runs)):
        idx = rng.choice(n, size=int(n * subsample_frac), replace=False)
        yb = y[idx]
        Xb = x_int[idx, :]
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

    freqreal = selcountsreal / float(B_runs)
    freqko = selcountsko / float(B_runs)

    theta = choosethetaminfdp(freqreal, freqko)
    selmask = (freqreal >= theta) if theta is not None else np.zeros_like(freqreal, dtype=bool)
    selected = [snp_list[i] for i in np.where(selmask)[0]]

    return {
        "selected": selected,
        "theta": theta,
        "freqreal": freqreal,
        "freqko": freqko,
    }


if __name__ == "__main__":
    SNP_LIST_FILE = ""
    GENOTYPE_FILE = ""
    GROUP_FILE = ""

    results = run_stability_selection(
        snp_list_file=SNP_LIST_FILE,
        genotype_file=GENOTYPE_FILE,
        group_file=GROUP_FILE,
    )

    if results["theta"] is None:
        print("No threshold found; selection is empty.")
    else:
        print(f"Selected {len(results['selected'])} SNPs with theta={results['theta']:.4f}.")
    print(results["selected"])
