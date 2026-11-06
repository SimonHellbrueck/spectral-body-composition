"""Anatomy-guided multi-material decomposition (A4 v3): the soft-tissue anchor follows the segmentation.

Plain words: A4 v2 (mmd.reconcile) explains the non-fat, non-air part of every voxel with one soft tissue
(ICRU-46 liver). Muscle is less dense than liver, and urine, bile and CSF are water, so these read as partly fat
(for example, muscle about 14 % instead of 10 %, urine 38 % instead of 5 %). Here the segmentation (BOA body regions and
tissues, TotalSegmentator v2 'total') decides which soft tissue fills the non-fat part of each voxel. Everything
else (triples, noise metric, near-air rule, calcium soft threshold) is exactly mmd.reconcile.

Anchors: HU of the reference tissues at the effective energies of the calibration (ICRU-44/46 compositions,
xraylib), water at 0/0 HU by definition of the HU scale. The fat anchor is taken per scan from the scan's own
subcutaneous fat (mode of L0/H0 in the BOA SAT mask), i.e. "fat" means adipose tissue of the same scan.

    anchor  reference                  used for (category)
    liver   ICRU-46 liver, 1.06 g/ml    parenchymal organs, cavity soft tissue, everything not listed below
    muscle  ICRU-44 skeletal muscle     BOA muscle region and TS muscles; BOA subcutaneous region (skin, septa)
    blood   ICRU-44 blood               heart and large vessels (TS)
    water   0 / 0 HU                    urinary bladder, gallbladder, kidney cysts, spinal canal

Categories (one per voxel, first match wins) serve the fat balances: bone, lung, canal, gi, fluid, blood and
outside are excluded from body-composition fat; sat_region, muscle_region, cavity, parenchyma and other count.
"""
from __future__ import annotations

import numpy as np

from . import mmd
from .decomposition import tissue_hu

ANCHOR_NAMES = {0: "liver", 1: "muscle", 2: "blood", 3: "water"}
ANCHOR_SOURCE = {"liver": "liver", "muscle": "muscle", "blood": "blood"}

CATS = {0: "outside", 1: "bone", 2: "lung", 3: "canal", 4: "gi", 5: "fluid", 6: "blood", 7: "parenchyma",
        8: "muscle_region", 9: "sat_region", 10: "cavity", 11: "other"}
CAT_ANCHOR = {0: 0, 1: 0, 2: 0, 3: 3, 4: 0, 5: 3, 6: 2, 7: 0, 8: 1, 9: 1, 10: 0, 11: 0}
CAT_IN_FAT_BALANCE = {7, 8, 9, 10, 11}

# TotalSegmentator v2 'total' label IDs (verified on BOA 1.0.2 output by volume match)
TS = {
    "bone": [25] + list(range(26, 51)) + list(range(69, 79)) + list(range(91, 118)),
    "lung": [10, 11, 12, 13, 14, 16],
    "canal": [79],
    "gi": [6, 15, 18, 19, 20],
    "fluid": [4, 21, 23, 24],
    "blood": [51] + list(range(52, 69)),
    "parenchyma": [1, 2, 3, 5, 7, 8, 9, 17, 22],
    "muscle": list(range(80, 90)),
}
# BOA BCA codes
BOA_TISSUE = {1: "muscle", 2: "bone", 3: "SAT", 4: "VAT", 5: "IMAT", 6: "PAT", 7: "EAT"}
BOA_ADIPOSE = (3, 4, 5, 6, 7)
BOA_REGION = {1: "subcutaneous tissue", 2: "muscle", 3: "abdominal cavity", 4: "thoracic cavity", 5: "bone",
              6: "glands", 7: "pericardium", 8: "breast implant", 9: "mediastinum", 10: "brain", 11: "spinal cord"}


def anchors(cal: dict, fat_lh: tuple[float, float] | None = None) -> dict:
    """Anchor (L, H) in HU per anchor code at the effective energies of the calibration block."""
    eL, eH = cal["e_eff"]["L"], cal["e_eff"]["H"]
    out = {c: (float(tissue_hu(ANCHOR_SOURCE[n], eL)), float(tissue_hu(ANCHOR_SOURCE[n], eH)))
           for c, n in ANCHOR_NAMES.items() if n in ANCHOR_SOURCE}
    out[3] = (0.0, 0.0)
    return out


def categories(body: np.ndarray, tissues: np.ndarray, regions: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Category code per voxel (see CATS), first match wins."""
    cat = np.full(body.shape, 11, dtype=np.uint8)
    order = [
        (10, np.isin(regions, (3, 6, 7, 9))),
        (9, regions == 1),
        (8, (regions == 2) | np.isin(total, TS["muscle"])),
        (7, np.isin(total, TS["parenchyma"])),
        (6, np.isin(total, TS["blood"])),
        (5, np.isin(total, TS["fluid"])),
        (4, np.isin(total, TS["gi"])),
        (3, (regions == 11) | np.isin(total, TS["canal"])),
        (2, (regions == 4) | np.isin(total, TS["lung"])),
        (1, (regions == 5) | (tissues == 2) | np.isin(total, TS["bone"])),
    ]
    for code, m in order:                     # later entries overwrite earlier ones (= higher precedence)
        cat[m] = code
    cat[~body] = 0
    return cat


def anchor_codes(cat: np.ndarray) -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for c, a in CAT_ANCHOR.items():
        lut[c] = a
    return lut[cat]


def fat_anchor_from_sat(L: np.ndarray, H: np.ndarray, sat: np.ndarray, vmi: np.ndarray, c_ref: float,
                        box: float = 5.0) -> dict:
    """Per-scan fat anchor: mode of (L, H) in the SAT mask (1-HU bins), refined by the mean within +-box HU."""
    s = sat & (vmi > -150) & (vmi < -60)
    if s.sum() < 1000:
        return {"ok": False, "n": int(s.sum())}
    l, h = L[s].astype(np.float64), H[s].astype(np.float64)
    hist, le, he = np.histogram2d(l, h, bins=(np.arange(-170, -40.5, 1.0), np.arange(-150, -20.5, 1.0)))
    i, j = np.unravel_index(int(np.argmax(hist)), hist.shape)
    l0, h0 = 0.5 * (le[i] + le[i + 1]), 0.5 * (he[j] + he[j + 1])
    near = (np.abs(l - l0) <= box) & (np.abs(h - h0) <= box)
    return {"ok": True, "n": int(s.sum()), "mode": [float(l0), float(h0)],
            "L": float(l[near].mean()), "H": float(h[near].mean()), "vmi70": float(c_ref * l[near].mean() + (1 - c_ref) * h[near].mean())}


def reconcile_anat(L, H, p: dict, near_air, acode, anch: dict) -> dict:
    """mmd.reconcile with the soft-tissue anchor chosen per voxel (acode -> anch[acode]); same output keys."""
    L, H = np.asarray(L, dtype=np.float64).ravel(), np.asarray(H, dtype=np.float64).ravel()
    na = np.asarray(near_air, bool).ravel()
    ac = np.asarray(acode).ravel()
    n = L.size
    out = {k: np.zeros(n) for k in ("fat", "soft", "air", "iod", "ca", "misfit")}
    out["tri"] = np.zeros(n, dtype=np.uint8)
    for code in np.unique(ac):
        sel = ac == code
        pp = dict(p)
        pp["S"] = anch[int(code)]
        m = mmd.reconcile(L[sel], H[sel], pp, na[sel])
        for k in out:
            out[k][sel] = m[k]
    return out
