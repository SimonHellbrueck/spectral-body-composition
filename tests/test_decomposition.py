"""Synthetic checks of the decomposition (no image data needed)."""
import json
from pathlib import Path

import numpy as np

from sppbc import estimator as ce
from sppbc import mmd
from sppbc import mmd_anat as ma

CAL = json.loads((Path(__file__).resolve().parents[1] / "src" / "sppbc" / "calibration" / "naeotom_alpha_140kV.json").read_text())


def _voxels(f, iod, code, p, anch):
    S, F, (kL, kH) = anch[code], p["F"], p["dI"]
    L = f * F[0] + (1 - f) * S[0] + iod * kL
    H = f * F[1] + (1 - f) * S[1] + iod * kH
    return L, H


def test_linear_solution_recovers_fat_and_iodine():
    p = mmd.params(CAL, "low")
    anch = ma.anchors(CAL)
    anch[0] = tuple(p["S"])
    f = np.array([-0.1, 0.0, 0.05, 0.3, 0.9, 1.1])
    iod = np.array([0.0, 0.3, 1.0, 0.0, 0.5, 0.0])
    for code in (0, 1, 2, 3):
        L, H = _voxels(f, iod, code, p, anch)
        acode = np.full(f.size, code)
        sys_ = ce.classify(L, H, np.zeros(f.size, bool), acode, anch, p)
        assert (sys_ == ce.SYS_LIN).all()
        res = ce.solve(L, H, sys_, acode, anch, p)
        assert np.allclose(res["fat"], f, atol=1e-9) and np.allclose(res["iod"], iod, atol=1e-9)


def test_single_energy_fraction_without_iodine_equals_decomposition():
    p = mmd.params(CAL, "low")
    anch = ma.anchors(CAL)
    anch[0] = tuple(p["S"])
    c = p["c_ref"]
    f = np.linspace(-0.1, 1.1, 13)
    L, H = _voxels(f, np.zeros_like(f), 1, p, anch)
    s70 = c * anch[1][0] + (1 - c) * anch[1][1]
    f70 = c * p["F"][0] + (1 - c) * p["F"][1]
    assert np.allclose((s70 - (c * L + (1 - c) * H)) / (s70 - f70), f, atol=1e-9)
