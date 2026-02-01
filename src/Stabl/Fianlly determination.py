import numpy as np

from utils.config import STABILITYSELECTION


def fdpplus(sreal, skn):
    return (1.0 + float(skn)) / max(1.0, float(sreal))


def choosethetafromfdp(freqreal, freqko, thetagrid=None):
    if thetagrid is None:
        thetagrid = np.linspace(
            float(STABILITYSELECTION["thetaleft"]),
            float(STABILITYSELECTION["thetaright"]),
            int(STABILITYSELECTION["thetasize"]),
        )

    bestfdp = None
    besttheta = None

    for t in thetagrid:
        sreal = int(np.sum(freqreal >= t))
        if sreal == 0:
            continue

        skn = int(np.sum(freqko >= t))
        fdp = fdpplus(sreal, skn)

        if (bestfdp is None) or (fdp < bestfdp):
            bestfdp = float(fdp)
            besttheta = float(t)

    return besttheta


def summarizestability(freqbycutoff):
    if freqbycutoff.ndim != 2:
        raise ValueError("freqbycutoff must be a 2D array [cutoff x SNP].")
    return np.max(freqbycutoff, axis=0)


def rundetermination(
    freqbycutoffreal,
    freqbycutoffko,
    snplist=None,
    thetagrid=None,
):
    if freqbycutoffreal.shape != freqbycutoffko.shape:
        raise ValueError("Real/knockoff frequency grids must have the same shape.")

    freqreal = summarizestability(freqbycutoffreal)
    freqko = summarizestability(freqbycutoffko)

    theta = choosethetafromfdp(freqreal, freqko, thetagrid=thetagrid)
    selmask = (
        (freqreal >= theta)
        if theta is not None
        else np.zeros_like(freqreal, dtype=bool)
    )

    selected = None
    if snplist is not None:
        selected = [snplist[i] for i in np.where(selmask)[0]]

    return {
        "theta": theta,
        "freqreal": freqreal,
        "freqko": freqko,
        "selmask": selmask,
        "selected": selected,
    }
