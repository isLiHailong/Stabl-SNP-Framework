from pathlib import Path
import importlib.util
from typing import Iterable, Optional, Union, Dict, List

from utils.config import HWE_SETTINGS
from utils.numeric3 import build_snplist, build_Xgenoint
from utils.validate import requirefile


_HWE_MODULE_NAME = "hardy_weinberg_internal"
_HWE_PATH = Path(__file__).resolve().parent / "Hardy–Weinberg.py"
_spec = importlib.util.spec_from_file_location(_HWE_MODULE_NAME, _HWE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load Hardy–Weinberg module from {_HWE_PATH}.")
_hwe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hwe)


def hwe_screen_from_genotype(
    genotype_csv: str,
    snp_list_csv: str,
    sample_prefix: str = "Y1_",
    missing_values: Optional[Iterable[Union[int, float]]] = None,
    p_threshold: Optional[float] = None,
    method: Optional[str] = None,
) -> Dict[str, object]:
    requirefile(genotype_csv)
    requirefile(snp_list_csv)

    missing_values = (
        tuple(HWE_SETTINGS["missing_values"])
        if missing_values is None
        else tuple(missing_values)
    )
    p_threshold = float(
        HWE_SETTINGS["p_threshold"] if p_threshold is None else p_threshold
    )
    method = str(HWE_SETTINGS["method"] if method is None else method).lower()
    if method not in {"exact", "chi2"}:
        raise ValueError("method must be 'exact' or 'chi2'.")

    snp_list = build_snplist(snp_list_csv)
    X_int, _X_imp, _sample_cols = build_Xgenoint(
        genotype_csv,
        snp_list,
        sample_prefix=sample_prefix,
    )

    kept: List[str] = []
    removed: List[str] = []
    stats: List[Dict[str, object]] = []

    for snp, col in zip(snp_list, X_int.T):
        res = _hwe.hwe_from_genotype_vector(col.tolist(), missing_values=missing_values)
        p_value = res["p_exact"] if method == "exact" else res["p_chi2"]
        res = dict(res)
        res["snp"] = snp
        res["p_value"] = p_value
        stats.append(res)

        if p_value >= p_threshold:
            kept.append(snp)
        else:
            removed.append(snp)

    return {
        "kept_snps": kept,
        "removed_snps": removed,
        "stats": stats,
        "p_threshold": p_threshold,
        "method": method,
    }


__all__ = ["hwe_screen_from_genotype"]