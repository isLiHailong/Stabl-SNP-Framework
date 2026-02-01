from pathlib import Path
import importlib.util
from typing import Iterable, Optional, Union, Dict, List

from utils.config import HWESETTINGS
from utils.numeric3 import buildsnplist, buildXgenoint
from utils.validate import requirefile


HWEMODULENAME = "hardyweinberginternal"
HWEPATH = Path(__file__).resolve().parents[1] / "SNP excluded" / "Hardy–Weinberg.py"

spec = importlib.util.spec_from_file_location(HWEMODULENAME, HWEPATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load Hardy–Weinberg module from {HWEPATH}.")

hwe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hwe)


def hwescreenfromgenotype(
    genotypecsv: str,
    snplistcsv: str,
    sampleprefix: str = "Y1_",
    missingvalues: Optional[Iterable[Union[int, float]]] = None,
    pthreshold: Optional[float] = None,
    method: Optional[str] = None,
) -> Dict[str, object]:
    requirefile(genotypecsv)
    requirefile(snplistcsv)

    missingvalues = (
        tuple(HWESETTINGS["missingvalues"])
        if missingvalues is None
        else tuple(missingvalues)
    )

    pthreshold = float(
        HWESETTINGS["pthreshold"] if pthreshold is None else pthreshold
    )

    method = str(HWESETTINGS["method"] if method is None else method).lower()
    if method not in {"exact", "chi2"}:
        raise ValueError("method must be 'exact' or 'chi2'.")

    snplist = buildsnplist(snplistcsv)
    Xint, Ximp, samplecols = buildXgenoint(
        genotypecsv,
        snplist,
        sampleprefix=sampleprefix,
    )

    kept: List[str] = []
    removed: List[str] = []
    stats: List[Dict[str, object]] = []

    for snp, col in zip(snplist, Xint.T):
        res = hwe.hwefromgenotypevector(
            col.tolist(),
            missingvalues=missingvalues,
        )

        pvalue = res["pexact"] if method == "exact" else res["pchi2"]

        res = dict(res)
        res["snp"] = snp
        res["pvalue"] = pvalue
        stats.append(res)

        if pvalue >= pthreshold:
            kept.append(snp)
        else:
            removed.append(snp)

    return {
        "keptsnps": kept,
        "removedsnps": removed,
        "stats": stats,
        "pthreshold": pthreshold,
        "method": method,
    }


__all__ = ["hwescreenfromgenotype"]
