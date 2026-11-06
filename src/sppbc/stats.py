"""Agreement between the decomposition and BOA over many series (one row per patient).

Plain words. For each patient the ratio decomposition / BOA is formed; its logarithm is averaged (geometric mean
ratio) and its spread gives the 95 % limits of agreement (Bland-Altman analysis of log-transformed values, Euser
2008). Confidence intervals come from resampling patients with replacement (bootstrap, percentile intervals).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20261017

#: comparison name -> (decomposition column, BOA column) in the table written by ``sppbc collect``
COMPARISONS = {
    "Subcutaneous region": ("cat_sat_region_de_fat_ml", "cat_sat_region_boa_adipose_ml"),
    "Body cavities": ("cat_cavity_de_fat_ml", "cat_cavity_boa_adipose_ml"),
    "Fat balance": ("fat_balance_de_fat_l", "fat_balance_boa_adipose_l"),
    "Muscle region": ("cat_muscle_region_de_fat_ml", "cat_muscle_region_boa_adipose_ml"),
    "Subcutaneous region, single-energy fraction (70 keV)": ("cat_sat_region_de_fat_single_energy_ml", "cat_sat_region_boa_adipose_ml"),
    "Body cavities, single-energy fraction (70 keV)": ("cat_cavity_de_fat_single_energy_ml", "cat_cavity_boa_adipose_ml"),
    "Fat balance, single-energy fraction (70 keV)": ("fat_balance_de_fat_single_energy_l", "fat_balance_boa_adipose_l"),
    "Fat balance, without interface": ("fat_balance_de_fat_l_without_interface", "fat_balance_boa_adipose_l_without_interface"),
}


def ccc(x, y) -> float:
    mx, my = x.mean(), y.mean()
    return float(2 * ((x - mx) * (y - my)).mean() / (x.var() + y.var() + (mx - my) ** 2))


def icc_a1(x, y) -> float:
    Y = np.c_[x, y]
    n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def agree(de: np.ndarray, ref: np.ndarray) -> dict:
    """Log-ratio Bland-Altman, nonparametric limits, proportional bias, CCC, ICC(A,1)."""
    ok = (de > 0) & (ref > 0)
    r = np.log(de[ok] / ref[ok])
    lm = np.log(np.sqrt(de[ok] * ref[ok]))
    m, s = r.mean(), r.std(ddof=1)
    rel = de / ref - 1.0
    return {"n_pairs": int(ok.sum()), "n_nonpositive": int((~ok).sum()),
            "gmr_pct": 100 * (np.exp(m) - 1), "loa_low_pct": 100 * (np.exp(m - 1.96 * s) - 1),
            "loa_up_pct": 100 * (np.exp(m + 1.96 * s) - 1),
            "np_loa_low_pct": 100 * np.percentile(rel, 2.5), "np_loa_up_pct": 100 * np.percentile(rel, 97.5),
            "median_rel_pct": 100 * np.median(rel), "q25_rel_pct": 100 * np.percentile(rel, 25),
            "q75_rel_pct": 100 * np.percentile(rel, 75), "prop_slope_log": float(np.polyfit(lm, r, 1)[0]),
            "ccc": ccc(de, ref), "icc_a1": icc_a1(de, ref), "mean_de": float(de.mean()), "mean_ref": float(ref.mean())}


def with_ci(de: np.ndarray, ref: np.ndarray, boot: int = 2000, seed: int = SEED) -> dict:
    point = agree(de, ref)
    rng = np.random.default_rng(seed)
    n = len(de)
    bs = []
    for _ in range(boot):
        i = rng.integers(0, n, n)
        try:
            bs.append(agree(de[i], ref[i]))
        except Exception:
            continue
    out = {}
    for k, v in point.items():
        out[k] = v
        if isinstance(v, float) and bs:
            vals = np.array([b[k] for b in bs if np.isfinite(b.get(k, np.nan))])
            if vals.size:
                out[k + "_ci"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return out


def agreement_table(table: pd.DataFrame, boot: int = 2000, comparisons: dict | None = None) -> pd.DataFrame:
    rows = []
    for name, (a, b) in (comparisons or COMPARISONS).items():
        if a not in table or b not in table:
            continue
        d = table[[a, b]].dropna()
        if len(d) < 3:
            continue
        res = with_ci(d[a].to_numpy(float), d[b].to_numpy(float), boot=boot)
        rows.append({"comparison": name, "n": len(d), **{k: (v if not isinstance(v, list) else f"{v[0]:.3f} to {v[1]:.3f}")
                                                          for k, v in res.items()}})
    return pd.DataFrame(rows)
