"""Reconciled multi-material decomposition of the SPP low/high pair (A4 v2).

Plain words: every voxel is explained by at most three materials from a small library - air, fat and
soft tissue as volume parts, iodine and calcium (hydroxyapatite) as dissolved additions. Which triple
applies follows from where the voxel lies in the (L, H) plane. Voxels outside every triple (noise, tissue
that differs from the anchors) are moved to the nearest allowed solution, distances measured in units of
the image noise (multi-material decomposition with triplet selection, after Mendonca PRS, Lamb P,
Sahani DV, IEEE Trans Med Imaging 2014;33(1):99-116, doi:10.1109/TMI.2013.2281719 - at most three materials
per voxel out of a larger library; here with additive iodine/calcium directions; constrained least squares
with the noise covariance as metric).

Triples (base point + two directions, bounds), chosen per voxel:
    air side = VMI70 < v_lung or f > 1, or (next to air and f >= pv_fat_min and lung-triple iodine >= -fat_air_iod):
      lung    lung-triple iodine >= -fat_air_iod   S + a (A - S) + I dI   a in [0, 1], I >= 0
      fat/air otherwise                            F + a (A - F) + I dI   a in [0, 1], I >= 0
    fat/soft  0 <= f <= 1                          S + f (F - S) + I dI   f in [0, 1], I >= 0
    calcium   f < 0                                S + Ca dCa + I dI      Ca >= 0,     I >= 0
f is the free fat fraction of the fat/soft/iodine solution. Fat/soft and calcium share the ray dI from S,
fat/soft and fat/air the ray dI from F (up to iodine 0.9 mg/ml), so the maps are continuous there.
Soft tissue + air (lung, partial volume next to gas) and fat have the same density near -100 HU but differ
by about 25 HU in L - H; the iodine test along the soft-air line separates them.
In the calcium triple, amounts below ca_shrink are blended towards Ca = 0 (soft thresholding): tissue only a
little denser than the soft-tissue anchor (liver, clotted blood) would otherwise get some calcium and too
little iodine.

Effective atomic number and electron density follow from the resulting composition (electron-weighted
power law, exponent 2.94). The constants come from the calibration file (see calibration/).
"""
from __future__ import annotations

import numpy as np

from .decomposition import HA_DENSITY, HA_FORMULA, TISSUES

try:
    import xraylib
except ImportError:  # pragma: no cover
    xraylib = None

N_A = 6.02214076e23
BETA = 2.94
AIR = ({"C": 0.000124, "N": 0.755268, "O": 0.231781, "Ar": 0.012827}, 0.001205)   # dry air (NIST)
TRI_NONE, TRI_LUNG, TRI_FATAIR, TRI_FSI, TRI_SCI = 0, 1, 2, 3, 4
TRI_NAMES = {TRI_LUNG: "Luft/Weichteil/Iod", TRI_FATAIR: "Fett/Luft/Iod", TRI_FSI: "Fett/Weichteil/Iod",
             TRI_SCI: "Weichteil/Kalk/Iod"}

# tissue classes for display and volumetry
CLASS_THR = {"air": 0.5, "air_none": 0.97, "fat": 0.5, "iod": 1.0, "ca": 80.0}
FAT_EDGES = [0.5, 0.6, 0.7, 0.8, 0.9]
IOD_EDGES = [1.0, 2.0, 3.0, 4.5, 6.0, 8.0]
CA_EDGES = [80.0, 150.0, 300.0, 500.0, 800.0]
CLS_AIR, CLS_SOFT, CLS_FAT0, CLS_IOD0, CLS_CA0 = 1, 2, 10, 20, 30


def class_lut() -> dict:
    """code -> [German name, [r, g, b], alpha 0..255]."""
    lut = {"0": ["Luft", [0, 0, 0], 0], "1": ["Lunge/Luftgemisch", [45, 70, 140], 150],
           "2": ["Weichteil (Rest)", [95, 150, 125], 110]}
    fat_rgb = [[205, 175, 60], [220, 190, 40], [235, 205, 25], [245, 215, 10], [255, 230, 0]]
    iod_rgb = [[250, 185, 150], [250, 145, 110], [245, 105, 80], [235, 60, 60], [215, 20, 60], [255, 0, 120]]
    ca_rgb = [[110, 165, 255], [145, 195, 255], [185, 220, 255], [220, 238, 255], [255, 255, 255]]

    def rng(edges, i, fmt, unit):
        lo = edges[i]
        return f"≥ {fmt(lo)} {unit}" if i == len(edges) - 1 else f"{fmt(lo)}–{fmt(edges[i + 1])} {unit}"
    pct = lambda x: f"{100 * x:.0f}"
    num = lambda x: f"{x:g}".replace(".", ",")
    for i, c in enumerate(fat_rgb):
        lut[str(CLS_FAT0 + i)] = [f"Fett {rng(FAT_EDGES, i, pct, '%')}", c, 200]
    for i, c in enumerate(iod_rgb):
        lut[str(CLS_IOD0 + i)] = [f"Iod {rng(IOD_EDGES, i, num, 'mg/ml')}", c, 210]
    for i, c in enumerate(ca_rgb):
        lut[str(CLS_CA0 + i)] = [f"Kalk {rng(CA_EDGES, i, num, 'mg/ml')}", c, 220]
    return lut


def electron_props(comp: dict) -> tuple[float, float]:
    """(electrons per gram, electron-weighted mean of Z**BETA) of a composition {symbol: mass fraction}."""
    tot = float(sum(comp.values()))
    e_mol, zp = 0.0, 0.0
    for el, w in comp.items():
        z = xraylib.SymbolToAtomicNumber(el)
        e = (w / tot) * z / xraylib.AtomicWeight(z)
        e_mol += e
        zp += e * z ** BETA
    return e_mol * N_A, zp / e_mol


def material_constants() -> dict:
    """Electron densities and Z**BETA means of the library materials (for Zeff and rho_e)."""
    cp = xraylib.CompoundParser(HA_FORMULA)
    ha = {xraylib.AtomicNumberToSymbol(z): w for z, w in zip(cp["Elements"], cp["massFractions"])}
    out = {}
    for name, (comp, rho) in {"air": AIR, "fat": TISSUES["adipose"], "soft": TISSUES["liver"]}.items():
        e_g, zp = electron_props(comp)
        out[name] = {"e_ml": e_g * rho, "zp": zp, "rho": rho}
    for name, comp in {"iodine": {"I": 1.0}, "calcium": ha}.items():
        e_g, zp = electron_props(comp)
        out[name] = {"e_mg": e_g / 1000.0, "zp": zp}
    out["calcium"]["density_mg_ml"] = HA_DENSITY * 1000.0
    out["water_e_ml"] = electron_props({"H": 2 * 1.008, "O": 15.999})[0] * 1.0
    out["beta"] = BETA
    return out


def noise_metric(sl: float, sh: float, rho: float) -> np.ndarray:
    """Inverse noise covariance of (L, H), normalised to unit determinant (only ratios matter)."""
    cov = np.array([[sl * sl, rho * sl * sh], [rho * sl * sh, sh * sh]], dtype=np.float64)
    q = np.linalg.inv(cov)
    return q / np.sqrt(np.linalg.det(q))


def params(cal: dict, source: str = "low") -> dict:
    """Decomposition constants from a calibration dictionary (calibration/*.json, with its 'mmd' sub-block)."""
    src = "orig" if source == "orig" else "low"
    m = cal.get("mmd", {})
    nz = (m.get("noise") or {}).get(src) or ([14.44, 15.20, -0.028] if src == "orig" else [2.839, 2.523, 0.932])
    return {"S": tuple(cal["soft"][src]), "F": tuple(cal["fat"][src]), "A": tuple(m.get("air", (-1000.0, -1000.0))),
            "dI": (cal["kL"], cal["kH"]), "dCa": (cal["calcium"]["kL"], cal["calcium"]["kH"]),
            "c_ref": cal["c_ref"], "v_lung": m.get("v_lung", -150.0), "fat_air_iod": m.get("fat_air_iod", 0.4),
            "near_air_vmi": m.get("near_air_vmi", -400.0), "pv_fat_min": m.get("pv_fat_min", 0.5),
            "ca_shrink": m.get("ca_shrink", 80.0),
            "Q": noise_metric(*nz), "ha_mg_ml": HA_DENSITY * 1000.0}


def _solve(dl, dh, d1, d2):
    det = d1[0] * d2[1] - d1[1] * d2[0]
    return (dl * d2[1] - dh * d2[0]) / det, (d1[0] * dh - d1[1] * dl) / det


def _qform(rl, rh, Q):
    return Q[0, 0] * rl * rl + 2.0 * Q[0, 1] * rl * rh + Q[1, 1] * rh * rh


def _box_solve(dl, dh, d1, d2, b1, b2, Q):
    """Least squares (L, H) = a d1 + c d2 with a in b1, c in b2 in the metric Q (exact for 2 unknowns).

    Free solution where it is inside the box; otherwise the best of the four edge solutions
    (one variable at its bound, the other optimal and clamped). Returns a, c, objective (0 inside).
    """
    a, c = _solve(dl, dh, d1, d2)
    a = np.asarray(a, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    obj = np.zeros_like(a)
    out = ~((a >= b1[0]) & (a <= b1[1]) & (c >= b2[0]) & (c <= b2[1]))
    if not out.any():
        return a, c, obj
    xl, xh = dl[out], dh[out]
    best = np.full(xl.shape, np.inf)
    ba, bc = np.zeros_like(xl), np.zeros_like(xl)
    q11 = _qform(d1[0], d1[1], Q)
    q22 = _qform(d2[0], d2[1], Q)
    for fixed, bound, other in ((0, b1[0], b2), (0, b1[1], b2), (1, b2[0], b1), (1, b2[1], b1)):
        if not np.isfinite(bound):
            continue
        dfix, dfree, qq = (d1, d2, q22) if fixed == 0 else (d2, d1, q11)
        rl, rh = xl - bound * dfix[0], xh - bound * dfix[1]
        t = (dfree[0] * (Q[0, 0] * rl + Q[0, 1] * rh) + dfree[1] * (Q[1, 0] * rl + Q[1, 1] * rh)) / qq
        t = np.clip(t, other[0], other[1])
        o = _qform(rl - t * dfree[0], rh - t * dfree[1], Q)
        better = o < best
        best[better] = o[better]
        if fixed == 0:
            ba[better], bc[better] = bound, t[better]
        else:
            ba[better], bc[better] = t[better], bound
    a[out], c[out], obj[out] = ba, bc, best
    return a, c, obj


def near_air_mask(L, H, p: dict) -> np.ndarray:
    """Voxels whose 3x3x3 neighbourhood contains air (unsmoothed VMI70 < near_air_vmi)."""
    from scipy import ndimage as ndi
    vmi = p["c_ref"] * np.asarray(L, dtype=np.float64) + (1.0 - p["c_ref"]) * np.asarray(H, dtype=np.float64)
    return ndi.binary_dilation(vmi < p["near_air_vmi"], structure=np.ones((3,) * vmi.ndim, bool))


def reconcile(L, H, p: dict, near_air=None) -> dict:
    """Per-voxel composition: volume fractions fat/soft/air, iodine and calcium (mg/ml), triple code, misfit.

    ``near_air`` (bool array, see near_air_mask) marks voxels next to air. There, a voxel that would be
    mostly fat with iodine is read as soft tissue + air (partial volume at lung or gas borders), because
    both mixtures lie close together in the (L, H) plane and enhancing fat is implausible.
    ``misfit`` is the noise-weighted distance (normalised metric) between measurement and model; 0 where
    the voxel lies inside its triple.
    """
    L = np.asarray(L, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    S, F, A, dI, dCa, Q = p["S"], p["F"], p["A"], p["dI"], p["dCa"], p["Q"]
    shp = L.shape
    nr = np.zeros(L.size, bool) if near_air is None else np.asarray(near_air, bool).ravel()
    L, H = L.ravel(), H.ravel()
    n = L.size
    fat, soft, air = np.zeros(n), np.zeros(n), np.zeros(n)
    iod, ca, mis = np.zeros(n), np.zeros(n), np.zeros(n)
    tri = np.zeros(n, dtype=np.uint8)
    vmi = p["c_ref"] * L + (1.0 - p["c_ref"]) * H
    low = vmi < p["v_lung"]
    dS = (L - S[0], H - S[1])
    dFS = (F[0] - S[0], F[1] - S[1])
    dAS = (A[0] - S[0], A[1] - S[1])
    f_free, _ = _solve(dS[0], dS[1], dFS, dI)
    # air side (below v_lung, or beyond the fat anchor): soft tissue + air (lung, partial volume next to gas)
    # unless the voxel is clearly on the fat side of the soft-air line (free iodine of the lung triple below
    # -fat_air_iod; pure fat lies at about -1.3 mg/ml), then fat + air
    _, i_lung = _solve(dS[0], dS[1], dAS, dI)
    soft_air_ok = i_lung >= -p.get("fat_air_iod", 0.4)
    air_side = low | (f_free > 1.0) | (nr & (f_free >= p.get("pv_fat_min", 0.5)) & soft_air_ok)
    fat_air = air_side & ~soft_air_ok
    inf = np.inf
    groups = {
        TRI_LUNG: (air_side & soft_air_ok, S, dAS, dI, (0.0, 1.0), (0.0, inf)),
        TRI_FATAIR: (fat_air, F, (A[0] - F[0], A[1] - F[1]), dI, (0.0, 1.0), (0.0, inf)),
        TRI_FSI: (~air_side & (f_free >= 0.0), S, dFS, dI, (0.0, 1.0), (0.0, inf)),
        TRI_SCI: (~air_side & (f_free < 0.0), S, dCa, dI, (0.0, inf), (0.0, inf)),
    }
    for code, (m, base, d1, d2, b1, b2) in groups.items():
        if not m.any():
            continue
        a, c, o = _box_solve(L[m] - base[0], H[m] - base[1], d1, d2, b1, b2, Q)
        tri[m], iod[m], mis[m] = code, c, o
        if code == TRI_LUNG:
            air[m], soft[m] = a, 1.0 - a
        elif code == TRI_FATAIR:
            air[m], fat[m] = a, 1.0 - a
        elif code == TRI_FSI:
            fat[m], soft[m] = a, 1.0 - a
        else:
            # small calcium amounts are more likely denser soft tissue (dense liver, blood clot): blend towards
            # Ca = 0 with iodine from the projection onto the iodine ray (keeps L - H, i.e. iodine unbiased);
            # weight Ca / ca_shrink, continuous at Ca = 0 and at Ca = ca_shrink (soft thresholding)
            cs = p.get("ca_shrink", 80.0)
            if cs > 0:
                dl, dh = L[m] - base[0], H[m] - base[1]
                qd = _qform(d2[0], d2[1], Q)
                t = np.maximum((d2[0] * (Q[0, 0] * dl + Q[0, 1] * dh) + d2[1] * (Q[1, 0] * dl + Q[1, 1] * dh)) / qd, 0.0)
                w = np.clip(a / cs, 0.0, 1.0)
                a = w * a
                iod[m] = (1.0 - w) * t + w * c
            ca[m], soft[m] = a, 1.0 - a / p["ha_mg_ml"]
    r = lambda x: x.reshape(shp)
    return {"fat": r(fat), "soft": r(soft), "air": r(air), "iod": r(iod), "ca": r(ca), "tri": r(tri),
            "misfit": r(np.sqrt(mis))}


def zeff_rhoe(m: dict, mc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Effective atomic number (power law, BETA) and electron density relative to water."""
    parts = [(m["fat"], mc["fat"]["e_ml"], mc["fat"]["zp"]), (m["soft"], mc["soft"]["e_ml"], mc["soft"]["zp"]),
             (m["air"], mc["air"]["e_ml"], mc["air"]["zp"]), (m["iod"], mc["iodine"]["e_mg"], mc["iodine"]["zp"]),
             (m["ca"], mc["calcium"]["e_mg"], mc["calcium"]["zp"])]
    e = sum(x * k for x, k, _ in parts)
    ez = sum(x * k * z for x, k, z in parts)
    with np.errstate(invalid="ignore", divide="ignore"):
        zeff = np.where(e > 0, ez / np.where(e > 0, e, 1.0), np.nan) ** (1.0 / mc["beta"])
    return zeff, e / mc["water_e_ml"]


def classes(m: dict, thr: dict | None = None) -> np.ndarray:
    """Tissue class codes (see class_lut): air > calcium > fat > iodine > soft tissue, graded by amount."""
    t = dict(CLASS_THR, **(thr or {}))
    out = np.full(m["fat"].shape, CLS_SOFT, dtype=np.int16)
    io = m["iod"] >= t["iod"]
    out[io] = CLS_IOD0 + np.searchsorted(IOD_EDGES, m["iod"][io], side="right") - 1
    fa = m["fat"] >= t["fat"]
    out[fa] = CLS_FAT0 + np.searchsorted(FAT_EDGES, m["fat"][fa], side="right") - 1
    cc = m["ca"] >= t["ca"]
    out[cc] = CLS_CA0 + np.searchsorted(CA_EDGES, m["ca"][cc], side="right") - 1
    out[m["air"] >= t["air"]] = CLS_AIR
    out[m["air"] >= t["air_none"]] = 0                  # (almost) pure air: transparent
    return out


IOD_MASKED = -32768






def smooth3(x: np.ndarray) -> np.ndarray:
    """3-point mean along each axis with clipped edges (= 3x3x3 box mean inside)."""
    y = np.asarray(x, dtype=np.float64)
    for ax in range(y.ndim):
        s = np.swapaxes(y, 0, ax)
        o = s.copy()
        o[1:-1] = (s[:-2] + s[1:-1] + s[2:]) / 3.0
        o[0] = (s[0] + s[1]) / 2.0
        o[-1] = (s[-1] + s[-2]) / 2.0
        y = np.swapaxes(o, 0, ax)
    return y


def class_group(code):
    """Coarse group of a class code: 1 air, 2 soft, 3 fat, 4 iodine, 5 calcium."""
    c = np.asarray(code)
    return np.select([c == CLS_AIR, c == CLS_SOFT, (c >= CLS_FAT0) & (c < CLS_FAT0 + 10),
                      (c >= CLS_IOD0) & (c < CLS_IOD0 + 10), c >= CLS_CA0], [1, 2, 3, 4, 5], 0)
