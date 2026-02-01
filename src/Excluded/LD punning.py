import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from utils.IO import writecsv, joinpath, readcsv
from utils.numeric3 import build_snplist, build_Xgenoint
from utils.validate import requirefile


def _standardize_matrix(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0, ddof=1)
    std = np.where(np.isfinite(std) & (std > 0), std, 0.0)
    Z = X - mean
    valid = std > 0
    if np.any(valid):
        Z[:, valid] /= std[valid]
    Z[:, ~valid] = 0.0
    return Z, std


def _r2_matrix_from_Z(Z: np.ndarray) -> np.ndarray:
    n = int(Z.shape[0])
    denom = max(float(n - 1), 1.0)
    C = (Z.T @ Z) / denom
    R2 = np.square(C)
    np.fill_diagonal(R2, 0.0)
    return R2


def _window_prune_once(
    Zwin: np.ndarray,
    idxs: List[int],
    r2_threshold: float,
) -> List[int]:
    if len(idxs) < 2:
        return idxs

    active = np.ones(len(idxs), dtype=bool)

    while True:
        Zact = Zwin[:, active]
        if Zact.shape[1] < 2:
            break

        R2 = _r2_matrix_from_Z(Zact)
        hit = R2 > float(r2_threshold)
        if not np.any(hit):
            break

        deg = np.sum(hit, axis=1).astype(int)
        max_deg = int(np.max(deg))
        cand = np.where(deg == max_deg)[0]

        if cand.size == 1:
            drop_local = int(cand[0])
        else:
            var = np.nanvar(Zact, axis=0, ddof=1)
            var = np.where(np.isfinite(var), var, -1.0)
            drop_local = int(cand[np.argmin(var[cand])])

        active_indices = np.where(active)[0]
        active[active_indices[drop_local]] = False

    kept = [idxs[i] for i, a in enumerate(active) if a]
    return kept


def _normalize_chr(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = s.replace("CHR", "").replace("chr", "").replace("Chr", "").strip()
    if s in {"X", "x"}:
        return 23
    if s in {"Y", "y"}:
        return 24
    try:
        return int(float(s))
    except Exception:
        return None


def load_snp_positions(annotation_csv: str, snp_list: List[str]) -> pd.DataFrame:
    ann = readcsv(annotation_csv)
    need = {"SNPName", "CHR", "BP"}
    if not need.issubset(set(ann.columns)):
        raise ValueError(f"annotation_csv must contain columns {need}")

    ann = ann[["SNPName", "CHR", "BP"]].copy()
    ann["SNPName"] = ann["SNPName"].astype(str)
    ann["CHR"] = ann["CHR"].map(_normalize_chr)
    ann["BP"] = pd.to_numeric(ann["BP"], errors="coerce")

    ann = ann.dropna(subset=["CHR", "BP"])
    ann["CHR"] = ann["CHR"].astype(int)
    ann["BP"] = ann["BP"].astype(int)

    ann = ann[ann["SNPName"].isin(set(map(str, snp_list)))].copy()
    if ann.empty:
        raise ValueError("No SNPs from snp_list found in annotation_csv.")

    ann = ann.drop_duplicates(subset=["SNPName"], keep="first")
    miss = [s for s in snp_list if s not in set(ann["SNPName"].tolist())]
    if miss:
        raise ValueError(
            f"Missing SNP positions in annotation_csv (first 10): {miss[:10]}"
        )

    ann["order"] = ann["SNPName"].map({s: i for i, s in enumerate(snp_list)}).astype(
        int
    )
    ann = ann.sort_values(["CHR", "BP", "order"]).reset_index(drop=True)
    return ann


def ld_prune_indep_pairwise_kb(
    X: np.ndarray,
    snp_list: List[str],
    snp_pos_df: pd.DataFrame,
    r2_threshold: float = 0.2,
    window_kb: float = 250.0,
    step_kb: float = 50.0,
) -> Dict[str, object]:
    if len(snp_list) != X.shape[1]:
        raise ValueError("snp_list length must match X columns.")
    if X.shape[0] < 2 or X.shape[1] < 2:
        return {"kept": snp_list, "removed": []}

    if window_kb <= 0 or step_kb <= 0:
        raise ValueError("window_kb and step_kb must be > 0")

    snp_to_col = {s: i for i, s in enumerate(map(str, snp_list))}
    pos = snp_pos_df.copy()
    pos["col_idx"] = pos["SNPName"].map(snp_to_col).astype(int)

    Z, std = _standardize_matrix(X)
    keep = np.ones(X.shape[1], dtype=bool)
    keep = keep | (std == 0)

    window_bp = int(round(float(window_kb) * 1000.0))
    step_bp = int(round(float(step_kb) * 1000.0))

    for _chr_id, sub in pos.groupby("CHR", sort=True):
        bps = sub["BP"].to_numpy(dtype=int)
        cols = sub["col_idx"].to_numpy(dtype=int)

        if bps.size < 2:
            continue

        start_bp = int(bps.min())
        end_bp = int(bps.max())

        left = start_bp
        while left <= end_bp:
            right = left + window_bp

            in_win = (bps >= left) & (bps <= right)
            win_cols_all = cols[in_win].tolist()
            win_cols = [j for j in win_cols_all if keep[j] and (std[j] > 0)]

            if len(win_cols) >= 2:
                Zwin = Z[:, win_cols]
                kept_cols = _window_prune_once(
                    Zwin, win_cols, r2_threshold=float(r2_threshold)
                )
                drop_cols = set(win_cols) - set(kept_cols)
                for j in drop_cols:
                    keep[j] = False

            left += step_bp

    kept = [snp_list[i] for i in range(len(snp_list)) if keep[i]]
    removed = [snp_list[i] for i in range(len(snp_list)) if not keep[i]]
    return {"kept": kept, "removed": removed}


def run_ld_filter_kb(
    genotype_csv: str,
    snp_list_csv: str,
    annotation_csv: str,
    outdir: str,
    r2_threshold: float = 0.2,
    window_kb: float = 250.0,
    step_kb: float = 50.0,
    sample_prefix: str = "Y1_",
    chunksize: int = 8000,
):
    requirefile(genotype_csv)
    requirefile(snp_list_csv)
    requirefile(annotation_csv)

    snp_list = build_snplist(snp_list_csv)
    _X_int, X_imp, _sample_cols = build_Xgenoint(
        genotype_csv,
        snp_list,
        sample_prefix=sample_prefix,
        chunksize=chunksize,
    )

    pos_df = load_snp_positions(annotation_csv, snp_list)

    result = ld_prune_indep_pairwise_kb(
        X_imp,
        snp_list,
        pos_df,
        r2_threshold=float(r2_threshold),
        window_kb=float(window_kb),
        step_kb=float(step_kb),
    )

    kept = result["kept"]
    removed = result["removed"]

    os.makedirs(outdir, exist_ok=True)
    kept_path = joinpath(outdir, "ld.kb.kept.snps.list.csv")
    removed_path = joinpath(outdir, "ld.kb.removed.snps.list.csv")

    writecsv(pd.DataFrame({"SNPName": kept}), kept_path)
    writecsv(pd.DataFrame({"SNPName": removed}), removed_path)

    return len(kept), kept_path, removed_path


__all__ = ["ld_prune_indep_pairwise_kb", "run_ld_filter_kb", "load_snp_positions"]