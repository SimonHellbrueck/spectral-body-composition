"""Two-basis (water/iodine) decomposition of the SPP low/high images, version 0.

Model: in water-calibrated HU, every voxel is a water-equivalent part plus iodine,
    HU(E) = W + I * k(E),   k(E) = (mu/rho)_iodine(E) / (mu/rho)_water(E)   [HU per mg/ml],
so a measured pair (L, H) with effective iodine sensitivities kL, kH gives
    I = (L - H) / (kL - kH)                     iodine map, mg/ml
    W = (kL * H - kH * L) / (kL - kH)           water-equivalent image (= VNC of this model), HU
    VMI(E) = c(E) * L + (1 - c(E)) * H,  c(E) = (k(E) - kH) / (kL - kH).
kL and kH are calibrated from the data itself (no vendor software):
    1. the scanner's VMI at E_ref is c_ref * L + (1 - c_ref) * H  (exact linear fit),
    2. the ratio r = kL / kH is the slope of the iodine branch in the (H, L) plane.
Then kH = k(E_ref) / (1 + c_ref (r - 1)) and kL = r * kH.

Known limitation (v0): tissues whose effective Z differs from water (fat, bone) are not
water + iodine. Fat appears with negative iodine and a VNC that is too high; bone appears as iodine.
"""
from __future__ import annotations

import numpy as np

try:
    import xraylib
except ImportError:  # pragma: no cover
    xraylib = None


def iodine_water_ratio(energy_kev) -> np.ndarray:
    """k(E) = (mu/rho)_I / (mu/rho)_H2O, total attenuation incl. coherent scattering (xraylib)."""
    if xraylib is None:
        raise ImportError("xraylib required")
    e = np.atleast_1d(np.asarray(energy_kev, dtype=float))
    out = np.array([xraylib.CS_Total(53, float(x)) / xraylib.CS_Total_CP("H2O", float(x)) for x in e])
    return out if np.ndim(energy_kev) else float(out[0])


def calibrate(c_ref: float, r: float, e_ref: float = 70.0) -> dict:
    """Effective iodine sensitivities (HU per mg/ml) of L and H from c_ref and r = kL/kH."""
    k_ref = iodine_water_ratio(e_ref)
    k_h = k_ref / (1.0 + c_ref * (r - 1.0))
    k_l = r * k_h
    return {"kL": float(k_l), "kH": float(k_h), "c_ref": float(c_ref), "e_ref": float(e_ref), "r": float(r)}


def vmi_weight(energy_kev, k_l: float, k_h: float):
    return (iodine_water_ratio(energy_kev) - k_h) / (k_l - k_h)


def effective_energy(k_value: float, lo: float = 33.5, hi: float = 200.0) -> float:
    """Energy above the iodine K edge at which k(E) equals k_value (k is monotone there)."""
    e = np.linspace(lo, hi, 3000)
    k = iodine_water_ratio(e)
    return float(np.interp(-k_value, -k, e))


def vmi(L, H, energy_kev, k_l, k_h):
    c = vmi_weight(energy_kev, k_l, k_h)
    return c * L + (1.0 - c) * H


def decompose_water_iodine(L, H, k_l, k_h):
    """Return (water-equivalent HU, iodine mg/ml)."""
    d = k_l - k_h
    return (k_l * H - k_h * L) / d, (L - H) / d


def baseline_difference(h, fat_lh):
    """Expected L-H of iodine-free tissue as a function of its H value (piecewise linear).

    Air (-1000, L=H) -> fat anchor (L_f, H_f, measured) -> water (0, L=H) -> soft tissue (L=H).
    """
    l_f, h_f = fat_lh
    d_f = l_f - h_f
    h = np.asarray(h, dtype=np.float64)
    d = np.zeros_like(h)
    m1 = (h > -1000.0) & (h <= h_f)
    m2 = (h > h_f) & (h < 0.0)
    d[m1] = d_f * (h[m1] + 1000.0) / (h_f + 1000.0)
    d[m2] = d_f * h[m2] / h_f
    return d


def iodine_fat_baseline(L, H, k_l, k_h, fat_lh, n_iter: int = 4):
    """Iodine (mg/ml) as distance from the iodine-free tissue baseline along the iodine direction.

    Solves (L - H) - I (kL - kH) = D(H - I kH) by fixed-point iteration (contraction ~0.2 per step).
    Equals the 2-material estimate for water-like tissue (D = 0) and gives 0 for pure fat.
    """
    L = np.asarray(L, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    dk = k_l - k_h
    d = L - H
    i = d / dk
    for _ in range(n_iter):
        i = (d - baseline_difference(H - i * k_h, fat_lh)) / dk
    return i


def two_direction_solve(L, H, base, d1, d2):
    """Solve (L, H) = base + a*d1 + b*d2 for (a, b); base, d1, d2 are (L, H) pairs (2x2 linear system)."""
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < 1e-9:
        raise ValueError("directions are parallel")
    dl = np.asarray(L, dtype=np.float64) - base[0]
    dh = np.asarray(H, dtype=np.float64) - base[1]
    a = (dl * d2[1] - dh * d2[0]) / det
    b = (d1[0] * dh - d1[1] * dl) / det
    return a, b


def fat_soft_iodine(L, H, fat_lh, soft_lh, k_l, k_h):
    """3-material decomposition fat / soft tissue / iodine (volume conservation of fat + soft tissue).

    (L, H) = S + f (F - S) + I (kL, kH): f = fat volume fraction (0..1), I = iodine (mg/ml).
    Valid for parenchyma (e.g. liver); fluid, air, lung and bone are outside the triangle.
    """
    d_fat = (fat_lh[0] - soft_lh[0], fat_lh[1] - soft_lh[1])
    return two_direction_solve(L, H, soft_lh, d_fat, (k_l, k_h))


def calcium_soft_iodine(L, H, soft_lh, kca_l, kca_h, k_l, k_h):
    """3-material decomposition calcium (hydroxyapatite) / soft tissue / iodine.

    (L, H) = S + Ca (kCaL, kCaH) + I (kL, kH): Ca in mg/ml HA, I in mg/ml iodine.
    """
    return two_direction_solve(L, H, soft_lh, (kca_l, kca_h), (k_l, k_h))




def calibrate_direction(k_ref: float, r: float, c_ref: float) -> tuple[float, float]:
    """(kL, kH) of a basis material from its sensitivity at the reference energy and its slope r = kL/kH."""
    k_h = k_ref / (1.0 + c_ref * (r - 1.0))
    return r * k_h, k_h


def vnc_from_iodine(L, H, iodine, c_ref, k_ref):
    """Virtual non-contrast image at the reference energy: VMI(E_ref) minus the iodine contribution."""
    return c_ref * np.asarray(L) + (1.0 - c_ref) * np.asarray(H) - np.asarray(iodine) * k_ref


def iodine_branch_slope(L, H, h_range=(120.0, 380.0), min_ratio=1.8, bin_width=10.0, min_count=200):
    """Slope dL/dH of the steep (iodine) branch: median L per H bin, then least squares.

    Uses voxels with H in h_range and L > min_ratio * H (separates the iodine branch from bone).
    Returns (slope, intercept, n_voxels, n_bins).
    """
    L = np.asarray(L, dtype=np.float64).ravel()
    H = np.asarray(H, dtype=np.float64).ravel()
    m = (H >= h_range[0]) & (H < h_range[1]) & (L > min_ratio * H)
    Lm, Hm = L[m], H[m]
    edges = np.arange(h_range[0], h_range[1] + bin_width, bin_width)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (Hm >= a) & (Hm < b)
        if sel.sum() >= min_count:
            xs.append(np.median(Hm[sel]))
            ys.append(np.median(Lm[sel]))
    if len(xs) < 3:
        raise ValueError("iodine branch not found (too few populated bins)")
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept), int(m.sum()), len(xs)


# ICRU-44 compositions (mass fractions) and densities, for physics consistency checks
TISSUES = {
    "adipose": ({"H": .114, "C": .598, "N": .007, "O": .278, "Na": .001, "S": .001, "Cl": .001}, 0.95),
    "muscle": ({"H": .102, "C": .143, "N": .034, "O": .710, "Na": .001, "P": .002, "S": .003, "Cl": .001, "K": .004}, 1.05),
    "blood": ({"H": .102, "C": .110, "N": .033, "O": .745, "Na": .001, "P": .001, "S": .002, "Cl": .003, "K": .002, "Fe": .001}, 1.06),
    "liver": ({"H": .102, "C": .139, "N": .030, "O": .716, "Na": .002, "P": .003, "S": .003, "Cl": .002, "K": .003}, 1.06),
}
HA_FORMULA, HA_DENSITY = "Ca10(PO4)6(OH)2", 3.16      # hydroxyapatite, g/cm³


def hydroxyapatite_hu_per_mgml(energy_kev: float) -> float:
    """HU change per mg/ml hydroxyapatite dissolved in water (displaced water volume included)."""
    e = float(energy_kev)
    c = 0.1                                            # 100 mg/ml, linear regime
    mu = xraylib.CS_Total_CP(HA_FORMULA, e) * c + xraylib.CS_Total_CP("H2O", e) * (1.0 - c / HA_DENSITY)
    return 1000.0 * (mu / xraylib.CS_Total_CP("H2O", e) - 1.0) / 100.0


def tissue_hu(tissue: str, energy_kev: float) -> float:
    """Monoenergetic HU of an ICRU tissue (total attenuation, water = 0 HU)."""
    comp, rho = TISSUES[tissue]
    mu = sum(w * xraylib.CS_Total(xraylib.SymbolToAtomicNumber(el), float(energy_kev)) for el, w in comp.items()) * rho
    mu_w = xraylib.CS_Total_CP("H2O", float(energy_kev))
    return 1000.0 * (mu / mu_w - 1.0)
