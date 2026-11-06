"""Agreement statistics on synthetic numbers."""
import numpy as np

from sppbc.stats import agree, with_ci


def test_constant_ratio():
    ref = np.linspace(1.0, 20.0, 50)
    r = agree(1.1 * ref, ref)
    assert abs(r["gmr_pct"] - 10.0) < 1e-9 and abs(r["loa_low_pct"] - 10.0) < 1e-9 and abs(r["loa_up_pct"] - 10.0) < 1e-9


def test_bootstrap_interval_contains_point():
    rng = np.random.default_rng(1)
    ref = rng.uniform(1, 10, 200)
    de = ref * np.exp(rng.normal(0.05, 0.1, 200))
    r = with_ci(de, ref, boot=200)
    lo, hi = r["gmr_pct_ci"]
    assert lo < r["gmr_pct"] < hi
