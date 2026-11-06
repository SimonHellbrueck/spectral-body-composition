"""collect and agreement on synthetic result files (no image data)."""
import json

import numpy as np
import pandas as pd

from sppbc import cli


def _result(case, scale, size, qc):
    comp = lambda de, boa: {"de_fat_ml": de * size, "boa_adipose_ml": boa * size, "de_fat_single_energy_ml": 0.95 * de * size}
    return {"case": case, "kvp": 140, "calibration": "test",
            "qa": {"qc_pass": qc, "vmi70_weight_c_fit": 0.54, "vmi70_resid_sd_hu": 0.3},
            "totals": {"body_l": 40.0 * size,
                       "fat_balance": {"de_fat_l": 20.0 * scale * size, "boa_adipose_l": 21.0 * size,
                                       "de_fat_single_energy_l": 19.0 * scale * size,
                                       "de_fat_l_without_interface": 19.5 * scale * size,
                                       "boa_adipose_l_without_interface": 20.0 * size}},
            "categories": {"sat_region": comp(14000.0 * scale, 15000.0), "cavity": comp(4500.0 * scale, 4200.0),
                           "muscle_region": comp(2100.0 * scale, 1700.0)},
            "boa_tissues": {}, "organs": {}, "l3": {"SMA_cm2": 150.0}}


def test_collect_and_agreement(tmp_path, capsys):
    res = tmp_path / "res"
    res.mkdir()
    for i, s in enumerate(np.linspace(0.95, 1.05, 6)):
        size = 1.0 + 0.15 * ((i * 5) % 6)  # body size differs between cases, independent of the ratio
        (res / f"case{i}.json").write_text(json.dumps(_result(f"case{i}", float(s), size, qc=(i != 3))))
    cli.main(["collect", "--input", str(res), "--out", str(tmp_path / "table.csv")])
    cli.main(["agreement", "--table", str(tmp_path / "table.csv"), "--boot", "50", "--out", str(tmp_path / "agr.csv")])
    agr = pd.read_csv(tmp_path / "agr.csv")
    assert "1 of 6 cases excluded" in capsys.readouterr().out
    assert set(agr["n"]) == {5}
    assert {"Subcutaneous region", "Body cavities", "Fat balance"} <= set(agr["comparison"])
