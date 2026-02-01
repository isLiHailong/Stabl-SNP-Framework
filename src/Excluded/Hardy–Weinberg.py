import math
from typing import Dict, Tuple, Optional, Union, List, Iterable

from utils.config import HWESETTINGS


def chi2pvalue1df(chi2: float) -> float:
    if chi2 < 0:
        return 1.0
    return math.erfc(math.sqrt(chi2 / 2.0))


def hwechi2test(
    obshomref: int,
    obshet: int,
    obshomalt: int
) -> Tuple[float, float, Tuple[float, float, float]]:
    for x in (obshomref, obshet, obshomalt):
        if x < 0:
            raise ValueError("Genotype counts must be non-negative.")

    n = obshomref + obshet + obshomalt
    if n == 0:
        return 0.0, 1.0, (0.0, 0.0, 0.0)

    nalt = 2 * obshomalt + obshet
    palt = nalt / (2.0 * n)
    pref = 1.0 - palt

    exphomref = n * (pref ** 2)
    exphet = 2 * n * pref * palt
    exphomalt = n * (palt ** 2)

    exp = (exphomref, exphet, exphomalt)
    obs = (obshomref, obshet, obshomalt)

    chi2 = 0.0
    for o, e in zip(obs, exp):
        if e > 0:
            chi2 += (o - e) ** 2 / e

    pvalue = chi2pvalue1df(chi2)
    return chi2, pvalue, exp


def hweexacttest(
    obshomref: int,
    obshet: int,
    obshomalt: int
) -> float:
    for x in (obshomref, obshet, obshomalt):
        if x < 0:
            raise ValueError("Genotype counts must be non-negative.")

    n = obshomref + obshet + obshomalt
    if n == 0:
        return 1.0

    obshets = obshet
    rarecopies = 2 * min(obshomref, obshomalt) + obshet
    genotypes = n

    if rarecopies < 0 or rarecopies > 2 * genotypes:
        return 1.0

    mid = int(rarecopies * (2 * genotypes - rarecopies) / (2 * genotypes))
    if (mid % 2) != (rarecopies % 2):
        mid += 1

    probs: Dict[int, float] = {}
    probs[mid] = 1.0
    sumprobs = 1.0

    curr = mid
    prob = 1.0
    while curr > 1:
        homr = (rarecopies - curr) // 2
        homc = genotypes - homr - curr
        if curr <= 0 or homc <= 0:
            break
        prob *= (curr * (curr - 1.0)) / (4.0 * (homr + 1.0) * (homc + 1.0))
        curr -= 2
        probs[curr] = prob
        sumprobs += prob

    curr = mid
    prob = 1.0
    while curr <= rarecopies - 2:
        homr = (rarecopies - curr) // 2
        homc = genotypes - homr - curr
        if homr <= 0:
            break
        prob *= (4.0 * homr * homc) / ((curr + 2.0) * (curr + 1.0))
        curr += 2
        probs[curr] = prob
        sumprobs += prob

    for k in list(probs.keys()):
        probs[k] /= sumprobs

    obshetsrare = obshets
    if obshetsrare not in probs:
        keys = sorted(probs.keys())
        obshetsrare = min(keys, key=lambda x: abs(x - obshetsrare))

    pobs = probs[obshetsrare]
    pvalue = sum(p for p in probs.values() if p <= pobs + 1e-15)
    return min(max(pvalue, 0.0), 1.0)


def normalizemissingvalues(
    missingvalues: Optional[Iterable[Union[int, float]]],
) -> Tuple[Union[int, float], ...]:
    if missingvalues is None:
        return tuple(HWESETTINGS["missingvalues"])
    return tuple(missingvalues)


def hwefromgenotypevector(
    g: Union[List[Optional[int]], List[Optional[float]]],
    missingvalues: Optional[Iterable[Union[int, float]]] = None,
) -> Dict[str, object]:
    missingvalues = normalizemissingvalues(missingvalues)

    obshomref = 0
    obshet = 0
    obshomalt = 0
    nmissing = 0

    for x in g:
        if x is None:
            nmissing += 1
            continue
        if isinstance(x, float) and math.isnan(x):
            nmissing += 1
            continue
        if x in missingvalues:
            nmissing += 1
            continue
        if x == 0:
            obshomref += 1
        elif x == 1:
            obshet += 1
        elif x == 2:
            obshomalt += 1
        else:
            raise ValueError(f"Unexpected genotype code: {x}")

    chi2, pchi2, exp = hwechi2test(obshomref, obshet, obshomalt)
    pexact = hweexacttest(obshomref, obshet, obshomalt)

    n = obshomref + obshet + obshomalt
    nalt = 2 * obshomalt + obshet
    maf = min(nalt / (2.0 * n), 1.0 - nalt / (2.0 * n)) if n > 0 else 0.0

    return {
        "n": n,
        "missing": nmissing,
        "counts": {
            "homref": obshomref,
            "het": obshet,
            "homalt": obshomalt,
        },
        "maf": maf,
        "expected": {
            "homref": exp[0],
            "het": exp[1],
            "homalt": exp[2],
        },
        "chi2": chi2,
        "pchi2": pchi2,
        "pexact": pexact,
    }


def hwefromcounts(
    obshomref: int,
    obshet: int,
    obshomalt: int
) -> Dict[str, object]:
    chi2, pchi2, exp = hwechi2test(obshomref, obshet, obshomalt)
    pexact = hweexacttest(obshomref, obshet, obshomalt)

    n = obshomref + obshet + obshomalt
    nalt = 2 * obshomalt + obshet
    maf = min(nalt / (2.0 * n), 1.0 - nalt / (2.0 * n)) if n > 0 else 0.0

    return {
        "n": n,
        "counts": {
            "homref": obshomref,
            "het": obshet,
            "homalt": obshomalt,
        },
        "maf": maf,
        "expected": {
            "homref": exp[0],
            "het": exp[1],
            "homalt": exp[2],
        },
        "chi2": chi2,
        "pchi2": pchi2,
        "pexact": pexact,
    }
