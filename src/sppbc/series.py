"""Per-series analysis: anatomy-guided material decomposition of the SPP basis images and comparison with the
threshold segmentation (BOA) in identical voxels.

Plain words. The SPP series of a photon-counting CT examination carries, in every DICOM file, the 70-keV image
(pixel data) and a container with the low- and high-energy basis images (L/H and the spectrally denoised L0/H0).
Each voxel is described as reference adipose tissue + one iodine-free base tissue + iodine; the segmentation chooses
the base tissue (muscle, blood, water or liver). Fat volumes of the decomposition are summed in exactly the voxels
in which BOA counts adipose tissue (-190 to -30 HU), so both methods are compared on the same domain.

Inputs: the SPP DICOM folder of one series, the BOA output for the 70-keV image of that series (tissues.nii.gz,
body_regions.nii.gz, total.nii.gz; computed on the NIfTI written by ``sppbc export-nifti``) and a calibration file.
Output: a dictionary of aggregates (no images, no DICOM header fields).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import estimator as ce
from . import mmd
from . import mmd_anat as ma
from .dicom import load_ct_series
from .spp import load_spp_series

VERSION = "1.0"            # numerics identical to the primary estimator used for the C1 manuscript
TS_L3 = 29                 # TotalSegmentator v2 label of the third lumbar vertebra
FLOOR_HU = -1024           # the 70-keV image is floored at -1024 HU (as the image given to BOA)
SEG_FILES = ("tissues", "body_regions", "total")
N_GROUP = {"cat": 12, "tis": 8, "reg": 256, "ts": 118}
ORGAN_GROUPS = {"autochthon": [86, 87], "iliopsoas": [88, 89], "gluteal": list(range(80, 86)), "liver": [5],
                "spleen": [1], "pancreas": [7], "kidneys": [2, 3], "aorta": [52], "portal vein": [64], "IVC": [63]}
KEYS = ("n", "fat", "iod", "v70", "vnc", "thr", "boa", "fat_se", "n_lin", "if", "if_fat", "if_boa", "if_thr", "if_se")
# Quality control of the 70-keV weight c refitted from L/H (VMI70 = c L + (1 - c) H):
QC_MAX_DC, QC_MAX_SD = 0.005, 1.0      # rule of the manuscript: series outside are flagged and excluded from the agreement
STOP_MAX_DC, STOP_MAX_SD = 0.01, 3.0   # gross mismatch (wrong grid, tube voltage or calibration): the analysis stops


def qc_pass(c_fit: float, c_cal: float, resid_sd: float) -> bool:
    return abs(c_fit - c_cal) <= QC_MAX_DC and resid_sd <= QC_MAX_SD


def body_mask_slice(v2d: np.ndarray) -> np.ndarray:
    """Largest connected component above -500 HU with holes filled (body outline of one slice)."""
    lab, n = ndi.label(v2d > -500.0, structure=np.ones((3, 3), bool))
    if n == 0:
        return np.zeros(v2d.shape, bool)
    areas = np.bincount(lab.ravel())[1:]
    return ndi.binary_fill_holes(lab == int(np.argmax(areas)) + 1)


def read_series(dicom_dir) -> dict:
    """70-keV image and L0/H0 of one SPP series, indexed (i, j, k) = (column, row, slice ascending)."""
    ser = load_ct_series(dicom_dir)
    vols, _, hdrs = load_spp_series(ser.files, names=("L0", "H0"))
    pos = (ser.affine_lps @ np.c_[np.zeros((len(hdrs), 2)), np.arange(len(hdrs)), np.ones(len(hdrs))].T)[:3].T
    if not np.allclose([h["position"] for h in hdrs], pos, atol=0.01):
        raise ValueError("SPP container positions do not match the DICOM slice positions")
    return {"v70": np.maximum(ser.hu, FLOOR_HU).astype(np.int16), "L0": vols["L0"], "H0": vols["H0"],
            "files": list(ser.files), "affine_ras": ser.affine_ras, "spacing": ser.spacing, "kvp": ser.meta.get("KVP")}


def read_masks(seg_dir, shape: tuple, affine_ras: np.ndarray) -> dict:
    import nibabel as nib
    out = {}
    for nm in SEG_FILES:
        img = nib.load(str(Path(seg_dir) / f"{nm}.nii.gz"))
        if tuple(img.shape) != tuple(shape) or not np.allclose(img.affine, affine_ras, atol=0.01):
            raise ValueError(f"{nm}.nii.gz is not on the grid of the 70-keV image (export it with 'sppbc export-nifti')")
        out[nm] = np.asanyarray(img.dataobj).astype(np.uint8)
    return out


def _vmi70_check(files, ks, v70all, c_ref):
    """Refit the 70-keV weight c from L/H on three slices: VMI70 = c L + (1 - c) H (grid and calibration check)."""
    lh, _, _ = load_spp_series([files[k] for k in ks], names=("L", "H"))
    rows = []
    for i, kz in enumerate(ks):
        v = v70all[:, :, kz].astype(np.float64)
        b = (v > -500) & (v < 2000)
        rows.append(np.c_[lh["L"][:, :, i][b], lh["H"][:, :, i][b], v[b]])
    lh3 = np.vstack(rows)
    X = np.c_[lh3[:, 0] - lh3[:, 1], np.ones(len(lh3))]
    coef, *_ = np.linalg.lstsq(X, lh3[:, 2] - lh3[:, 1], rcond=None)
    resid = lh3[:, 2] - lh3[:, 1] - X @ coef
    return float(coef[0]), float(resid.std())


def analyse(dicom_dir, seg_dir, calibration: dict, slab: int = 32, force: bool = False) -> dict:
    """force=True continues past the stop limits (for phantoms, whose 70-keV weight may differ from the patient calibration)."""
    t0 = time.time()
    s = read_series(dicom_dir)
    v70all = s["v70"]
    nx, ny, nz = v70all.shape
    zooms = s["spacing"]
    vox_ml = zooms[0] * zooms[1] * zooms[2] / 1000.0
    px_cm2 = zooms[0] * zooms[1] / 100.0
    seg = read_masks(seg_dir, v70all.shape, s["affine_ras"])
    p = mmd.params(calibration, "low")
    c = p["c_ref"]
    k70 = c * p["dI"][0] + (1.0 - c) * p["dI"][1]
    L0, H0 = s["L0"], s["H0"]

    ks = (nz // 4, nz // 2, (3 * nz) // 4)
    c_fit, c_sd = _vmi70_check(s["files"], ks, v70all, c)
    if not force and (c_sd > STOP_MAX_SD or abs(c_fit - c) > STOP_MAX_DC):
        raise ValueError(f"VMI70 check failed (c {c_fit:.4f}, residual SD {c_sd:.2f} HU): grid, tube voltage or calibration "
                         "mismatch (use --force only for phantoms)")

    fa = ma.fat_anchor_from_sat(L0, H0, seg["tissues"] == 3, v70all, c)     # reported only (SAT density)
    anch = ma.anchors(calibration)
    anch[0] = tuple(p["S"])                                                   # liver anchor of the calibration
    F70 = c * p["F"][0] + (1.0 - c) * p["F"][1]
    s70_lut = np.array([c * anch[k][0] + (1.0 - c) * anch[k][1] for k in range(4)])

    kl3 = None
    zz = np.nonzero((seg["total"] == TS_L3).any(axis=(0, 1)))[0]
    if zz.size:
        w = np.array([(seg["total"][:, :, k] == TS_L3).sum() for k in zz], float)
        kl3 = int(round(float((zz * w).sum() / w.sum())))

    body_all = np.stack([body_mask_slice(v70all[:, :, k].astype(np.float64)) for k in range(nz)], axis=2)
    acc = {g: {k: np.zeros(n) for k in KEYS} for g, n in N_GROUP.items()}
    st3 = np.ones((3, 3, 3), bool)
    l3, nbody = None, 0

    for z0 in range(0, nz, slab):
        z1 = min(nz, z0 + slab)
        e0, e1 = max(0, z0 - 1), min(nz, z1 + 1)
        L = L0[:, :, e0:e1].astype(np.float64)
        H = H0[:, :, e0:e1].astype(np.float64)
        na = mmd.near_air_mask(L, H, p)
        Ls, Hs = mmd.smooth3(L), mmd.smooth3(H)
        s_ = slice(z0 - e0, z0 - e0 + (z1 - z0))
        Lr, Hr = L[:, :, s_], H[:, :, s_]                          # unsmoothed L0/H0: solution
        Ls, Hs, na = Ls[:, :, s_], Hs[:, :, s_], na[:, :, s_]       # smoothed: choice of the system
        v70 = v70all[:, :, z0:z1].astype(np.float64)
        body = body_all[:, :, z0:z1]
        m0, m1 = max(0, z0 - 3), min(nz, z1 + 3)
        sm = slice(z0 - m0, z0 - m0 + (z1 - z0))
        gas_out = ~body_all[:, :, m0:m1] | (v70all[:, :, m0:m1] < -400)
        iface = (ndi.binary_dilation(gas_out, structure=st3, iterations=2) & ~gas_out)[:, :, sm]
        T, R, O = (seg[nm][:, :, z0:z1] for nm in SEG_FILES)
        cat = ma.categories(body, T, R, O)
        ac = ma.anchor_codes(cat)
        bi = np.nonzero(body.ravel())[0]
        if bi.size == 0:
            continue
        Lb, Hb, nab, acb = Ls.ravel()[bi], Hs.ravel()[bi], na.ravel()[bi], ac.ravel()[bi]
        Lrb, Hrb = Lr.ravel()[bi], Hr.ravel()[bi]
        sysv = ce.classify(Lb, Hb, nab, acb, anch, p)
        prim = ce.solve(Lrb, Hrb, sysv, acb, anch, p)
        islin = sysv == ce.SYS_LIN
        v = v70.ravel()[bi]
        thr = ((v >= -190) & (v <= -30)).astype(float)
        boa = np.isin(T.ravel()[bi], ma.BOA_ADIPOSE).astype(float)
        s70 = s70_lut[acb.astype(np.int64)]
        f_se = np.where(islin, (s70 - (c * Lrb + (1.0 - c) * Hrb)) / (s70 - F70), prim["fat"])   # 70-keV value alone
        ifb = iface.ravel()[bi]
        vals = {"fat": prim["fat"], "iod": prim["iod"], "v70": v, "vnc": v - prim["iod"] * k70, "thr": thr, "boa": boa,
                "fat_se": f_se, "n_lin": islin.astype(float), "if": ifb.astype(float),
                "if_fat": np.where(ifb, prim["fat"], 0.0), "if_boa": np.where(ifb, boa, 0.0),
                "if_thr": np.where(ifb, thr, 0.0), "if_se": np.where(ifb, f_se, 0.0)}
        labs = {"cat": cat.ravel()[bi], "tis": T.ravel()[bi], "reg": R.ravel()[bi], "ts": O.ravel()[bi]}
        for g, lab in labs.items():
            nl = N_GROUP[g]
            lab = lab.astype(np.int64)
            acc[g]["n"] += np.bincount(lab, minlength=nl)[:nl]
            for k, x in vals.items():
                acc[g][k] += np.bincount(lab, weights=x, minlength=nl)[:nl]
        nbody += int(bi.size)

        if kl3 is not None and z0 <= kl3 < z1:
            k = kl3 - z0
            sel = np.nonzero(body[:, :, k].ravel())[0]
            full = np.ravel_multi_index(np.unravel_index(sel, body[:, :, k].shape) + (np.full(sel.size, k),), body.shape)
            pos = np.searchsorted(bi, full)
            fk, ik, vk = prim["fat"][pos], prim["iod"][pos], v[pos]
            Tk, Rk = T[:, :, k].ravel()[sel], R[:, :, k].ravel()[sel]
            mus, rmus = Tk == 1, Rk == 2
            mean = lambda x, m: float(x[m].mean()) if m.any() else None
            l3 = {"slice": kl3, "SMA_cm2": float(mus.sum() * px_cm2), "SMD_HU": mean(vk, mus),
                  "SMD_iodine_free_HU": mean(vk - ik * k70, mus), "muscle_ff_mean": mean(fk, mus),
                  "muscle_iod_mean": mean(ik, mus), "muscle_region_ff_mean": mean(fk, rmus),
                  "IMAT_cm2": float((Tk == 5).sum() * px_cm2), "SAT_cm2": float((Tk == 3).sum() * px_cm2),
                  "VAT_cm2": float((Tk == 4).sum() * px_cm2),
                  "de_fat_muscle_region_cm2": float(fk[rmus].sum() * px_cm2),
                  "de_fat_sat_region_cm2": float(fk[Rk == 1].sum() * px_cm2)}

    def entry(g, idx) -> dict | None:
        a = {k: float(np.sum(acc[g][k][idx])) for k in KEYS}
        n = a["n"]
        if n == 0:
            return None
        return {"n_voxels": int(n), "volume_ml": n * vox_ml, "de_fat_ml": a["fat"] * vox_ml, "de_ff_mean": a["fat"] / n,
                "iodine_mean_mg_ml": a["iod"] / n, "vmi70_mean_HU": a["v70"] / n, "vmi70_iodine_free_mean_HU": a["vnc"] / n,
                "boa_adipose_ml": a["boa"] * vox_ml, "hu_window_ml": a["thr"] * vox_ml,
                "de_fat_single_energy_ml": a["fat_se"] * vox_ml, "linear_system_share": a["n_lin"] / n,
                "interface_ml": a["if"] * vox_ml, "interface_de_fat_ml": a["if_fat"] * vox_ml,
                "interface_boa_adipose_ml": a["if_boa"] * vox_ml, "interface_hu_window_ml": a["if_thr"] * vox_ml,
                "interface_de_fat_single_energy_ml": a["if_se"] * vox_ml}

    def table(g, names: dict) -> dict:
        return {nm: e for i, nm in names.items() if (e := entry(g, i)) is not None}

    cb = sorted(ma.CAT_IN_FAT_BALANCE)
    bal = {k: float(sum(acc["cat"][k][i] for i in cb)) * vox_ml / 1000.0 for k in ("boa", "thr", "fat", "fat_se", "if_fat", "if_boa", "if_se")}
    return {
        "software": f"sppbc {VERSION}", "kvp": calibration.get("kvp"), "calibration": calibration.get("name"),
        "seconds": round(time.time() - t0, 1),
        "grid": {"shape": [nx, ny, nz], "voxel_ml": vox_ml, "spacing_mm": [float(x) for x in zooms]},
        "qa": {"qc_pass": qc_pass(c_fit, c, c_sd),
               "qc_rule": f"|c_fit - c_cal| <= {QC_MAX_DC} and residual SD <= {QC_MAX_SD} HU",
               "vmi70_weight_c_fit": c_fit, "vmi70_resid_sd_hu": c_sd, "vmi70_weight_c_cal": c,
               "k70_hu_per_mg_ml": k70, "fat_anchor_reference": [float(x) for x in p["F"]],
               "fat_anchor_sat_mode": fa, "anchors_LH": {ma.ANCHOR_NAMES[k]: [float(x) for x in v] for k, v in anch.items()}},
        "totals": {"body_l": nbody * vox_ml / 1000.0,
                   "fat_balance": {"boa_adipose_l": bal["boa"], "hu_window_l": bal["thr"], "de_fat_l": bal["fat"],
                                   "de_fat_single_energy_l": bal["fat_se"],
                                   "de_fat_l_without_interface": bal["fat"] - bal["if_fat"],
                                   "boa_adipose_l_without_interface": bal["boa"] - bal["if_boa"],
                                   "de_fat_single_energy_l_without_interface": bal["fat_se"] - bal["if_se"]}},
        "categories": table("cat", ma.CATS),
        "boa_tissues": table("tis", {0: "none", **ma.BOA_TISSUE}),
        "boa_regions": table("reg", {0: "none", **ma.BOA_REGION}),
        "organs": {nm: e for nm, ids in ORGAN_GROUPS.items() if (e := entry("ts", ids)) is not None},
        "l3": l3,
    }
