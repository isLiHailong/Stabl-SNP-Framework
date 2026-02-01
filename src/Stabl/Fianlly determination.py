import numpy as np

from utils.config import STABILITY_SELECTION


def fdp_plus(sreal, skn):
    return (1.0 + float(skn)) / max(1.0, float(sreal))


def choose_theta_from_fdp(freqreal, freqko, theta_grid=None):
    if theta_grid is None:
        theta_grid = np.linspace(
            float(STABILITY_SELECTION["theta_left"]),
            float(STABILITY_SELECTION["theta_right"]),
            int(STABILITY_SELECTION["theta_size"]),
        )

    best_fdp = None
    best_theta = None
    for t in theta_grid:
        sreal = int(np.sum(freqreal >= t))
        if sreal == 0:
            continue
        skn = int(np.sum(freqko >= t))
        fdp = fdp_plus(sreal, skn)
        if (best_fdp is None) or (fdp < best_fdp):
            best_fdp = float(fdp)
            best_theta = float(t)

    return best_theta


def summarize_stability(freq_by_cutoff):
    if freq_by_cutoff.ndim != 2:
        raise ValueError("freq_by_cutoff must be a 2D array [cutoff x SNP].")
    return np.max(freq_by_cutoff, axis=0)


def run_determination(
    freq_by_cutoff_real,
    freq_by_cutoff_ko,
    snp_list=None,
    theta_grid=None,
):
    if freq_by_cutoff_real.shape != freq_by_cutoff_ko.shape:
        raise ValueError("Real/knockoff frequency grids must have the same shape.")

    freqreal = summarize_stability(freq_by_cutoff_real)
    freqko = summarize_stability(freq_by_cutoff_ko)

    theta = choose_theta_from_fdp(freqreal, freqko, theta_grid=theta_grid)
    selmask = (freqreal >= theta) if theta is not None else np.zeros_like(freqreal, dtype=bool)

    selected = None
    if snp_list is not None:
        selected = [snp_list[i] for i in np.where(selmask)[0]]

    return {
        "theta": theta,
        "freqreal": freqreal,
        "freqko": freqko,
        "selmask": selmask,
        "selected": selected,
    }