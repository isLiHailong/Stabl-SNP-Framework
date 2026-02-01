import math
from typing import Dict, Tuple, Optional, Union, List, Iterable

from utils.config import HWE_SETTINGS

def _chi2_pvalue_1df(chi2: float) -> float:
    if chi2 < 0:
        return 1.0
    return math.erfc(math.sqrt(chi2 / 2.0))

def hwe_chi2_test(obs_hom_ref: int, obs_het: int, obs_hom_alt: int) -> Tuple[float, float, Tuple[float, float, float]]:
    for x in (obs_hom_ref, obs_het, obs_hom_alt):
        if x < 0:
            raise ValueError("Genotype counts must be non-negative.")
    n = obs_hom_ref + obs_het + obs_hom_alt
    if n == 0:
        return 0.0, 1.0, (0.0, 0.0, 0.0)

    n_alt = 2 * obs_hom_alt + obs_het
    p_alt = n_alt / (2.0 * n)
    p_ref = 1.0 - p_alt

    exp_hom_ref = n * (p_ref ** 2)
    exp_het = 2 * n * p_ref * p_alt
    exp_hom_alt = n * (p_alt ** 2)

    exp = (exp_hom_ref, exp_het, exp_hom_alt)
    chi2 = 0.0
    obs = (obs_hom_ref, obs_het, obs_hom_alt)
    for o, e in zip(obs, exp):
        if e > 0:
            chi2 += (o - e) ** 2 / e

    p_value = _chi2_pvalue_1df(chi2)
    return chi2, p_value, exp

def hwe_exact_test(obs_hom_ref: int, obs_het: int, obs_hom_alt: int) -> float:
    for x in (obs_hom_ref, obs_het, obs_hom_alt):
        if x < 0:
            raise ValueError("Genotype counts must be non-negative.")
    n = obs_hom_ref + obs_het + obs_hom_alt
    if n == 0:
        return 1.0

    obs_hets = obs_het
    rare_copies = 2 * min(obs_hom_ref, obs_hom_alt) + obs_het
    genotypes = n

    if rare_copies < 0 or rare_copies > 2 * genotypes:
        return 1.0

    mid = int(rare_copies * (2 * genotypes - rare_copies) / (2 * genotypes))
    if (mid % 2) != (rare_copies % 2):
        mid += 1

    probs: Dict[int, float] = {}
    probs[mid] = 1.0
    sum_probs = 1.0

    curr = mid
    prob = 1.0
    while curr > 1:
        hom_r = (rare_copies - curr) // 2
        hom_c = genotypes - hom_r - curr
        if curr <= 0 or hom_c <= 0:
            break
        prob *= (curr * (curr - 1.0)) / (4.0 * (hom_r + 1.0) * (hom_c + 1.0))
        curr -= 2
        probs[curr] = prob
        sum_probs += prob

    curr = mid
    prob = 1.0
    while curr <= rare_copies - 2:
        hom_r = (rare_copies - curr) // 2
        hom_c = genotypes - hom_r - curr
        if hom_r <= 0:
            break
        prob *= (4.0 * hom_r * hom_c) / ((curr + 2.0) * (curr + 1.0))
        curr += 2
        probs[curr] = prob
        sum_probs += prob

    for k in list(probs.keys()):
        probs[k] /= sum_probs

    def _obs_to_rare_hets():
        if obs_hom_ref <= obs_hom_alt:
            return obs_hets
        return obs_hets

    obs_hets_rare = _obs_to_rare_hets()
    if obs_hets_rare not in probs:
        keys = sorted(probs.keys())
        closest = min(keys, key=lambda x: abs(x - obs_hets_rare))
        obs_hets_rare = closest

    p_obs = probs[obs_hets_rare]
    p_value = sum(p for h, p in probs.items() if p <= p_obs + 1e-15)
    p_value = min(max(p_value, 0.0), 1.0)
    return p_value

def _normalize_missing_values(
    missing_values: Optional[Iterable[Union[int, float]]],
) -> Tuple[Union[int, float], ...]:
    if missing_values is None:
        return tuple(HWE_SETTINGS["missing_values"])
    return tuple(missing_values)

def hwe_from_genotype_vector(
    g: Union[List[Optional[int]], List[Optional[float]]],
    missing_values: Optional[Iterable[Union[int, float]]] = None,
) -> Dict[str, object]:
    missing_values = _normalize_missing_values(missing_values)
    obs_hom_ref = obs_het = obs_hom_alt = 0
    n_missing = 0

    for x in g:
        if x is None:
            n_missing += 1
            continue
        if isinstance(x, float) and math.isnan(x):
            n_missing += 1
            continue
        if x in missing_values:
            n_missing += 1
            continue
        if x == 0:
            obs_hom_ref += 1
        elif x == 1:
            obs_het += 1
        elif x == 2:
            obs_hom_alt += 1
        else:
            raise ValueError(f"Unexpected genotype code: {x} (expected 0/1/2 or missing).")

    chi2, p_chi2, exp = hwe_chi2_test(obs_hom_ref, obs_het, obs_hom_alt)
    p_exact = hwe_exact_test(obs_hom_ref, obs_het, obs_hom_alt)

    n = obs_hom_ref + obs_het + obs_hom_alt
    n_alt = 2 * obs_hom_alt + obs_het
    maf = min(n_alt / (2.0 * n), 1.0 - (n_alt / (2.0 * n))) if n > 0 else 0.0

    return {
        "n": n,
        "missing": n_missing,
        "counts": {"hom_ref": obs_hom_ref, "het": obs_het, "hom_alt": obs_hom_alt},
        "maf": maf,
        "expected": {"hom_ref": exp[0], "het": exp[1], "hom_alt": exp[2]},
        "chi2": chi2,
        "p_chi2": p_chi2,
        "p_exact": p_exact,
    }

def hwe_from_counts(obs_hom_ref: int, obs_het: int, obs_hom_alt: int) -> Dict[str, object]:
    chi2, p_chi2, exp = hwe_chi2_test(obs_hom_ref, obs_het, obs_hom_alt)
    p_exact = hwe_exact_test(obs_hom_ref, obs_het, obs_hom_alt)

    n = obs_hom_ref + obs_het + obs_hom_alt
    n_alt = 2 * obs_hom_alt + obs_het
    maf = min(n_alt / (2.0 * n), 1.0 - (n_alt / (2.0 * n))) if n > 0 else 0.0

    return {
        "n": n,
        "counts": {"hom_ref": obs_hom_ref, "het": obs_het, "hom_alt": obs_hom_alt},
        "maf": maf,
        "expected": {"hom_ref": exp[0], "het": exp[1], "hom_alt": exp[2]},
        "chi2": chi2,
        "p_chi2": p_chi2,
        "p_exact": p_exact,
    }


