"""DICOM CT series reading with explicit geometry checks.

Conventions:
- Patient coordinates are DICOM LPS in mm.
- Volumes are returned in DICOM index order ``[i, j, k]`` = (column, row, slice),
  i.e. ``vol[i, j, k]`` is pixel (row j, column i) of the k-th slice after sorting
  slices by their position along the slice normal (ascending).
- ``affine_lps`` maps (i, j, k, 1) to LPS mm; ``affine_ras`` (for NIfTI) is
  ``diag(-1, -1, 1, 1) @ affine_lps``.

Only technical tags are read into the returned metadata. Identifying tags
(names, IDs, dates, UIDs) are never copied into ``CTSeries.meta``.
"""
from __future__ import annotations

import dataclasses
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

#: Technical tags copied into ``CTSeries.meta`` (first slice; constancy checked separately).
TECH_TAGS = (
    "Manufacturer", "ManufacturerModelName", "SoftwareVersions", "SeriesDescription", "ImageType",
    "Rows", "Columns", "PixelSpacing", "SliceThickness", "RescaleSlope", "RescaleIntercept",
    "RescaleType", "BitsAllocated", "BitsStored", "PixelRepresentation", "ConvolutionKernel",
    "KVP", "XRayTubeCurrent", "Exposure", "ExposureTime", "FilterType", "FocalSpots",
    "ReconstructionDiameter", "SpiralPitchFactor", "RevolutionTime", "SingleCollimationWidth",
    "TotalCollimationWidth", "ContrastBolusAgent", "PatientPosition", "BodyPartExamined",
)

#: HU below this value are treated as outside the reconstruction field of view.
#: Siemens SPP series use RescaleIntercept -8192, stored value 0 outside the FOV circle.
OUT_OF_FOV_HU = -2048

#: Elements larger than this are read lazily (pydicom ``defer_size``), e.g. tag (0049,1010).
DEFER = 64 * 1024


@dataclasses.dataclass
class CTSeries:
    hu: np.ndarray            # int16, shape (n_col, n_row, n_slice), HU
    affine_lps: np.ndarray    # 4x4
    meta: dict                # technical tags + geometry summary
    files: list               # file paths in slice order (k ascending)

    @property
    def affine_ras(self) -> np.ndarray:
        return LPS_TO_RAS @ self.affine_lps

    @property
    def spacing(self) -> tuple:
        return tuple(float(np.linalg.norm(self.affine_lps[:3, a])) for a in range(3))

    @property
    def fov_mask(self) -> np.ndarray:
        """True inside the reconstruction field of view."""
        return self.hu > OUT_OF_FOV_HU


def _tech_value(v):
    if isinstance(v, pydicom.multival.MultiValue):
        return [str(x) for x in v]
    return str(v)


def find_dicom_files(folder: str | os.PathLike) -> list[Path]:
    """All files below ``folder`` that pydicom can parse as DICOM (header only)."""
    out = []
    for dp, _, fn in os.walk(folder):
        for f in sorted(fn):
            p = Path(dp) / f
            try:
                pydicom.dcmread(p, stop_before_pixels=True, specific_tags=["SOPClassUID"])
            except Exception:
                continue
            out.append(p)
    return out


def load_ct_series(folder: str | os.PathLike, series_index: int = 0, dtype=np.int16) -> CTSeries:
    """Read one CT series from ``folder`` into an HU volume with LPS affine.

    Raises ``ValueError`` if the slices do not form a regular grid (constant
    orientation, constant in-plane origin along the normal, uniform spacing,
    one image per position).
    """
    files = find_dicom_files(folder)
    if not files:
        raise ValueError(f"no DICOM files in {folder}")
    # defer_size keeps large private blobs (SPP container, ~0.8 MB/slice) out of memory
    headers = [pydicom.dcmread(p, stop_before_pixels=True, defer_size=DEFER) for p in files]
    series = Counter(str(h.get("SeriesInstanceUID", "")) for h in headers)
    if len(series) > 1:
        warnings.warn(f"{len(series)} series in folder; using series_index={series_index}")
    uid = sorted(series)[series_index]
    pairs = [(p, h) for p, h in zip(files, headers) if str(h.get("SeriesInstanceUID", "")) == uid]

    iop = np.array([float(x) for x in pairs[0][1].ImageOrientationPatient])
    for _, h in pairs:
        if not np.allclose([float(x) for x in h.ImageOrientationPatient], iop, atol=1e-5):
            raise ValueError("ImageOrientationPatient not constant")
    row_dir, col_dir = iop[:3], iop[3:]          # direction of increasing column index / row index
    normal = np.cross(row_dir, col_dir)
    pos = np.array([[float(x) for x in h.ImagePositionPatient] for _, h in pairs])
    dist = pos @ normal
    order = np.argsort(dist)
    dist, pos = dist[order], pos[order]
    pairs = [pairs[i] for i in order]
    steps = np.diff(dist)
    if len(steps) and (np.any(steps <= 1e-4)):
        raise ValueError("duplicate slice positions (more than one image per position)")
    if len(steps) and np.ptp(steps) > 1e-3:
        raise ValueError(f"non-uniform slice spacing: {np.unique(np.round(steps, 4))[:6]}")
    inplane = pos - np.outer(dist - dist[0], normal)
    if np.max(np.abs(inplane - inplane[0])) > 1e-3:
        raise ValueError("slice origins not aligned along the normal (gantry tilt or shift)")
    dz = float(steps.mean()) if len(steps) else float(pairs[0][1].get("SliceThickness", 1.0))

    h0 = pairs[0][1]
    dr, dc = (float(x) for x in h0.PixelSpacing)   # row spacing (between rows), column spacing
    affine = np.eye(4)
    affine[:3, 0] = row_dir * dc
    affine[:3, 1] = col_dir * dr
    affine[:3, 2] = normal * dz
    affine[:3, 3] = pos[0]

    n_row, n_col = int(h0.Rows), int(h0.Columns)
    hu = np.empty((n_col, n_row, len(pairs)), dtype=dtype)
    slopes, inters = set(), set()
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    n_clipped = 0
    for k, (p, _) in enumerate(pairs):
        ds = pydicom.dcmread(p, defer_size=DEFER)
        s, b = float(ds.get("RescaleSlope", 1.0)), float(ds.get("RescaleIntercept", 0.0))
        slopes.add(s); inters.add(b)
        img = ds.pixel_array.astype(np.float64) * s + b
        if info is not None:
            n_clipped += int(np.count_nonzero((img < info.min) | (img > info.max)))
            img = np.clip(np.rint(img), info.min, info.max)
        hu[:, :, k] = img.T.astype(dtype)

    meta = {t: _tech_value(h0.get(t)) for t in TECH_TAGS if h0.get(t) is not None}
    meta.update({
        "n_slices": len(pairs), "shape_ijk": list(hu.shape), "spacing_mm": [dc, dr, dz],
        "slice_normal_lps": normal.tolist(), "rescale_slopes": sorted(slopes), "rescale_intercepts": sorted(inters),
        "n_values_clipped_to_dtype": n_clipped, "n_series_in_folder": len(series),
        "instance_numbers_ascending_with_k": bool(np.all(np.diff([int(h.get("InstanceNumber", 0)) for _, h in pairs]) > 0)),
    })
    return CTSeries(hu=hu, affine_lps=affine, meta=meta, files=[str(p) for p, _ in pairs])


def world_lps(affine_lps: np.ndarray, ijk) -> np.ndarray:
    """Voxel index (i, j, k) to LPS mm."""
    ijk = np.atleast_2d(np.asarray(ijk, dtype=float))
    return (affine_lps @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
