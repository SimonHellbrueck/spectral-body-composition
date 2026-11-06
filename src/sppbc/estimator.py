"""Compartment estimator for anatomy-guided body composition from photon-counting CT basis images (primary estimator).

Plain words. Every voxel is written as a mixture of fat (the fixed reference adipose tissue of the calibration),
one base tissue chosen by the segmentation (liver, muscle, blood or water) and dissolved iodine. With two energy
images this is two equations for two unknowns (fat fraction f, iodine c), so the solution is exact and linear
(linear three-material decomposition). The linear solution is not limited to 0..100 %: image noise then averages
out over a compartment, and compartment means equal the solution of the mean image values (unbiased estimator).
Limiting each voxel to 0..100 % (box-constrained solution) shifts lean compartments up and pure fat down.

Air is the only material that is modelled differently, and only where air is evident: the smoothed 70 keV value is
below v_lung (-150 HU) or the 3x3x3 neighbourhood contains air (unsmoothed 70 keV below -400 HU). There, voxels
on the soft-tissue side of the soft-tissue/air line are soft tissue + air (fat 0, lung and gas borders), voxels on
the fat side are fat + air (fat = 1 - air, unbounded). Elsewhere a voxel beyond the fat anchor stays in the
linear system (f > 1): with two energies fat + iodine cannot be told from soft tissue + air + iodine, so moving
such voxels to an air triple without evidence of air would read iodinated fat as soft tissue (as the bounded
triple selection in mmd.reconcile does).

The system (linear, fat/air, soft/air) is chosen on smoothed L0/H0, because the choice needs low noise; the
solution itself uses the unsmoothed L0/H0, so compartment means are not mixed with neighbouring tissue through the
smoothing kernel (the segmentation masks are defined on the unsmoothed image).

References: linear three-material decomposition with volume conservation, Liu X et al. Med Phys 2009;36:1602
(doi:10.1118/1.3097632); multi-material decomposition with triple selection and box constraints, Mendonca PRS et al.
IEEE Trans Med Imaging 2014;33:99-116 (doi:10.1109/TMI.2013.2281719).
"""
from __future__ import annotations

import numpy as np

SYS_LIN, SYS_FATAIR, SYS_SOFTAIR = 1, 2, 3
SYS_NAMES = {SYS_LIN: "fat/base/iodine (linear)", SYS_FATAIR: "fat/air/iodine", SYS_SOFTAIR: "base/air/iodine"}


def _solve(dl, dh, d1, d2):
    det = d1[0] * d2[1] - d1[1] * d2[0]
    return (dl * d2[1] - dh * d2[0]) / det, (d1[0] * dh - d1[1] * dl) / det


def linear(L, H, acode, anch: dict, F, dI):
    """Unbounded fat fraction f and iodine c (mg/ml) of the fat / base-tissue / iodine system per voxel.

    L, H: HU arrays (same shape); acode: anchor code per voxel (keys of anch); anch: code -> (L, H) of the base tissue.
    """
    L = np.asarray(L, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    ac = np.asarray(acode)
    f = np.empty(L.shape)
    c = np.empty(L.shape)
    for code in np.unique(ac):
        s = ac == code
        S = anch[int(code)]
        f[s], c[s] = _solve(L[s] - S[0], H[s] - S[1], (F[0] - S[0], F[1] - S[1]), dI)
    return f, c


def lung_iodine(L, H, acode, anch: dict, A, dI):
    """Iodine of the base-tissue / air / iodine system (decides the side of the soft-tissue/air line)."""
    L = np.asarray(L, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    ac = np.asarray(acode)
    out = np.empty(L.shape)
    for code in np.unique(ac):
        s = ac == code
        S = anch[int(code)]
        _, out[s] = _solve(L[s] - S[0], H[s] - S[1], (A[0] - S[0], A[1] - S[1]), dI)
    return out


def classify(Ls, Hs, near_air, acode, anch: dict, p: dict) -> np.ndarray:
    """System code per voxel from smoothed L0/H0 (Ls, Hs) and the near-air mask (see module docstring)."""
    Ls = np.asarray(Ls, dtype=np.float64)
    Hs = np.asarray(Hs, dtype=np.float64)
    na = np.asarray(near_air, bool)
    vmi = p["c_ref"] * Ls + (1.0 - p["c_ref"]) * Hs
    f_s, _ = linear(Ls, Hs, acode, anch, p["F"], p["dI"])
    soft_air_ok = lung_iodine(Ls, Hs, acode, anch, p["A"], p["dI"]) >= -p.get("fat_air_iod", 0.4)
    low = vmi < p.get("v_lung", -150.0)
    air_side = low | (na & ((f_s > 1.0) | ((f_s >= p.get("pv_fat_min", 0.5)) & soft_air_ok)))
    sys = np.full(Ls.shape, SYS_LIN, dtype=np.uint8)
    sys[air_side & soft_air_ok] = SYS_SOFTAIR
    sys[air_side & ~soft_air_ok] = SYS_FATAIR
    return sys


def solve(L, H, sys, acode, anch: dict, p: dict) -> dict:
    """Fat fraction, iodine and air fraction per voxel in the given systems, all unbounded."""
    L = np.asarray(L, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    sys = np.asarray(sys)
    ac = np.asarray(acode)
    F, A, dI = p["F"], p["A"], p["dI"]
    f = np.zeros(L.shape)
    c = np.zeros(L.shape)
    air = np.zeros(L.shape)
    m = sys == SYS_LIN
    if m.any():
        f[m], c[m] = linear(L[m], H[m], ac[m], anch, F, dI)
    m = sys == SYS_FATAIR
    if m.any():
        a, c[m] = _solve(L[m] - F[0], H[m] - F[1], (A[0] - F[0], A[1] - F[1]), dI)
        f[m], air[m] = 1.0 - a, a
    m = sys == SYS_SOFTAIR
    if m.any():
        for code in np.unique(ac[m]):
            s = m & (ac == code)
            S = anch[int(code)]
            air[s], c[s] = _solve(L[s] - S[0], H[s] - S[1], (A[0] - S[0], A[1] - S[1]), dI)
    return {"fat": f, "iod": c, "air": air}


def box(L, H, sys, acode, anch: dict, p: dict, metric=None) -> dict:
    """Box-constrained comparator: in the linear system f in [0, 1] and iodine >= 0 (least squares in the given
    metric, Euclidean by default); voxels in the air systems keep the unbounded air solution of ``solve``.

    This is the plain bounded version of the same model, used to isolate the effect of per-voxel bounds.
    """
    from .mmd import _box_solve
    L = np.asarray(L, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    out = solve(L, H, sys, acode, anch, p)
    Q = np.eye(2) if metric is None else np.asarray(metric, dtype=np.float64)
    m = np.asarray(sys) == SYS_LIN
    ac = np.asarray(acode)
    for code in np.unique(ac[m]):
        s = m & (ac == code)
        S = anch[int(code)]
        F = p["F"]
        a, c, _ = _box_solve(L[s] - S[0], H[s] - S[1], (F[0] - S[0], F[1] - S[1]), p["dI"], (0.0, 1.0), (0.0, np.inf), Q)
        out["fat"][s], out["iod"][s] = a, c
    return out


def anchor_sensitivity(n: float, sum_l: float, sum_h: float, S, F, dI, shift=(0.0, 0.0), fat_shift=(0.0, 0.0)):
    """Mean fat fraction and iodine of linear-system voxels for shifted anchors, from their sums alone.

    The linear solution is affine in (L, H), so the mean over n voxels equals the solution at the mean values.
    shift / fat_shift: (dL, dH) added to the base-tissue anchor S / the fat anchor F.
    """
    if n <= 0:
        return float("nan"), float("nan")
    S2 = (S[0] + shift[0], S[1] + shift[1])
    F2 = (F[0] + fat_shift[0], F[1] + fat_shift[1])
    f, c = _solve(sum_l / n - S2[0], sum_h / n - S2[1], (F2[0] - S2[0], F2[1] - S2[1]), dI)
    return float(f), float(c)
