"""Quality-control rule of the 70-keV weight (same limits as in the manuscript)."""
from sppbc.series import QC_MAX_DC, QC_MAX_SD, STOP_MAX_DC, STOP_MAX_SD, qc_pass


def test_qc_limits():
    assert (QC_MAX_DC, QC_MAX_SD, STOP_MAX_DC, STOP_MAX_SD) == (0.005, 1.0, 0.01, 3.0)
    assert qc_pass(0.5426 + 0.0048, 0.5426, 0.78)
    assert not qc_pass(0.5426 - 0.0051, 0.5426, 0.30)
    assert not qc_pass(0.5426, 0.5426, 1.2)
