"""Command line: python -m sppbc <command> ...

  export-nifti  70-keV image of an SPP series as NIfTI (input for the BOA segmentation)
  series        one series: decomposition and comparison with BOA -> JSON
  batch         many series from a table (case, dicom, seg, kvp) -> one JSON per case
  collect       JSON files -> one table (CSV), one row per case
  agreement     table -> agreement between decomposition and BOA (geometric mean ratio, limits of agreement)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PKG = Path(__file__).resolve().parent
CAL_DIR = PKG / "calibration"
DEFAULT_CAL = {140: CAL_DIR / "naeotom_alpha_140kV.json", 120: CAL_DIR / "naeotom_alpha_120kV.json"}


def load_calibration(kvp: int | None, path: str | None) -> dict:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if kvp not in DEFAULT_CAL:
        raise SystemExit("tube voltage unknown: give --kvp 140 or 120, or --calibration <file>")
    return json.loads(DEFAULT_CAL[kvp].read_text(encoding="utf-8"))


def cmd_export_nifti(a):
    import nibabel as nib
    import numpy as np
    from .dicom import load_ct_series
    from .series import FLOOR_HU
    ser = load_ct_series(a.dicom)
    img = nib.Nifti1Image(np.maximum(ser.hu, FLOOR_HU).astype(np.int16), ser.affine_ras)
    img.header.set_xyzt_units("mm")
    nib.save(img, str(a.out))
    print(f"written {a.out} {ser.hu.shape}")


def cmd_series(a):
    from .series import analyse
    res = analyse(a.dicom, a.seg, load_calibration(a.kvp, a.calibration), slab=a.slab, force=a.force)
    res["case"] = a.case
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    fb = res["totals"]["fat_balance"]
    qc = "" if res["qa"]["qc_pass"] else "  QC FAILED (excluded from agreement)"
    print(f"{a.case}: fat balance decomposition {fb['de_fat_l']:.2f} L, BOA {fb['boa_adipose_l']:.2f} L "
          f"({res['seconds']:.0f} s){qc}")


def cmd_batch(a):
    import pandas as pd
    cases = pd.read_csv(a.table, dtype=str).fillna("")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def run(r):
        dst = out / f"{r['case']}.json"
        if dst.exists() and not a.redo:
            return r["case"], "exists"
        args = [sys.executable, "-m", "sppbc", "series", "--dicom", r["dicom"], "--seg", r["seg"], "--case", r["case"],
                "--out", str(dst)]
        if r.get("kvp"):
            args += ["--kvp", r["kvp"]]
        if r.get("calibration"):
            args += ["--calibration", r["calibration"]]
        p = subprocess.run(args, capture_output=True, text=True)
        if p.returncode:
            (out / f"{r['case']}.error.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
            return r["case"], "failed"
        return r["case"], "done"

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for case, status in ex.map(run, [r for _, r in cases.iterrows()]):
            print(f"{case}: {status}", flush=True)


def flatten(res: dict) -> dict:
    row = {"case": res.get("case"), "kvp": res.get("kvp"), "calibration": res.get("calibration")}
    for k, v in res["qa"].items():
        if isinstance(v, (int, float)):
            row[f"qa_{k}"] = v
    row["body_l"] = res["totals"]["body_l"]
    for k, v in res["totals"]["fat_balance"].items():
        row[f"fat_balance_{k}"] = v
    for grp, pre in (("categories", "cat"), ("boa_tissues", "tis"), ("organs", "org")):
        for nm, d in res[grp].items():
            for k, v in d.items():
                row[f"{pre}_{nm.replace(' ', '_')}_{k}"] = v
    for k, v in (res.get("l3") or {}).items():
        row[f"l3_{k}"] = v
    return row


def cmd_collect(a):
    import pandas as pd
    rows = [flatten(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(Path(a.input).glob("*.json"))]
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{len(rows)} cases -> {a.out}")


def cmd_agreement(a):
    import pandas as pd
    from .stats import agreement_table
    tab = pd.read_csv(a.table)
    if "qa_qc_pass" in tab and not a.include_qc_fail:
        ok = tab["qa_qc_pass"].astype(str).str.lower().isin(["true", "1", "1.0"])
        print(f"quality control: {int((~ok).sum())} of {len(tab)} cases excluded "
              f"(70-keV weight off by more than 0.005 or residual SD above 1.0 HU)")
        tab = tab[ok]
    t = agreement_table(tab, boot=a.boot)
    t.to_csv(a.out, index=False)
    cols = ["comparison", "n", "gmr_pct", "gmr_pct_ci", "loa_low_pct", "loa_up_pct"]
    print(t[[c for c in cols if c in t]].to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sppbc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export-nifti")
    e.add_argument("--dicom", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_export_nifti)
    s = sub.add_parser("series")
    s.add_argument("--dicom", required=True, help="folder with the SPP DICOM series")
    s.add_argument("--seg", required=True, help="folder with tissues/body_regions/total.nii.gz of BOA")
    s.add_argument("--kvp", type=int, default=None)
    s.add_argument("--calibration", default=None)
    s.add_argument("--case", default="case")
    s.add_argument("--slab", type=int, default=32)
    s.add_argument("--force", action="store_true", help="continue past the stop limits (phantoms only)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_series)
    b = sub.add_parser("batch")
    b.add_argument("--table", required=True, help="CSV with columns case, dicom, seg, kvp (optional: calibration)")
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=3)
    b.add_argument("--redo", action="store_true")
    b.set_defaults(fn=cmd_batch)
    c = sub.add_parser("collect")
    c.add_argument("--input", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_collect)
    g = sub.add_parser("agreement")
    g.add_argument("--table", required=True)
    g.add_argument("--boot", type=int, default=2000)
    g.add_argument("--include-qc-fail", action="store_true", help="keep cases that fail the quality control")
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_agreement)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
