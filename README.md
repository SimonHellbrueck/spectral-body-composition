# sppbc – body composition from SPP photon-counting CT data

Reads the spectral post-processing (SPP) data of Siemens NAEOTOM Alpha examinations directly from the DICOM files.
Each voxel is split into reference adipose tissue, an iodine-free base tissue and iodine (anatomy-guided material
decomposition). The resulting fat volumes are compared with the threshold-based volumes of the Body and Organ Analysis
(BOA) in identical voxels.

The repository contains code and calibration constants only: no images, no tables, no examination data.

## What it does

1. **Read** – every DICOM file of an SPP series holds the 70-keV image (pixel data) and a container with the low- and
   high-energy basis images (L/H and the spectrally denoised L0/H0). `sppbc` decodes the container itself; no vendor
   software is needed.
2. **Segment** – BOA (version 1.0.2 with TotalSegmentator v2, not included) runs on the 70-keV image, exported with
   `sppbc export-nifti`.
3. **Decompose** – the segmentation chooses the base tissue of each voxel (skeletal muscle in muscle and subcutaneous
   tissue, blood in heart and vessels, water in fluid-filled organs and the spinal canal, liver elsewhere). The two
   energy images give two equations for the fat fraction and the iodine concentration, solved exactly and without
   limiting single voxels to 0–100 % fat. Air is modelled only where attenuation or neighbourhood shows it.
4. **Compare** – fat volumes of the decomposition and BOA adipose volumes (−190 to −30 HU) are summed over the same
   voxels: subcutaneous region, body cavities (abdominal cavity, mediastinum, pericardium and thyroid region without
   organ masks; the thyroid itself belongs to the parenchyma), muscle region and the fat balance over all soft-tissue
   compartments. Bowel, vessels, fluid-filled organs and bone are excluded. Also reported: a single-energy fat fraction from the
   70-keV value alone, variants without interface voxels, and L3 muscle values (area, density, iodine-free density,
   fat fraction, iodine).
5. **Agreement** – over many patients: geometric mean ratio and 95 % limits of agreement of the ratio decomposition/BOA
   (Bland–Altman analysis of log-transformed values) with bootstrap confidence intervals, CCC and ICC.

## Installation

```
conda env create -f environment.yml
conda activate sppbc
pip install -e .
python -m pytest
```

## Use

```
sppbc export-nifti --dicom <SPP series folder> --out <case>/vmi70.nii.gz
#   run BOA on vmi70.nii.gz; its output folder holds tissues.nii.gz, body_regions.nii.gz, total.nii.gz
sppbc series --dicom <SPP series folder> --seg <BOA output folder> --kvp 140 --case <label> --out results/<label>.json
sppbc batch --table cases.csv --out results --workers 3      # cases.csv: case,dicom,seg,kvp
sppbc collect --input results --out table.csv
sppbc agreement --table table.csv --out agreement.csv
```

- The tube voltage is not stored in the SPP DICOM files; give it with `--kvp` (140 or 120) or name a calibration file
  with `--calibration`.
- Use neutral case labels (for example, consecutive numbers). The JSON files contain aggregates only.
- `agreement` treats every row as one patient; keep one examination per patient in the table.
- One series of about 900 slices takes 3–6 minutes and about 4 GB of memory.

## Calibration

`src/sppbc/calibration/` holds the constants for NAEOTOM Alpha, Qr40, QIR 3, 1.0 mm, at 140 kV and 120 kV: 70-keV
weight, iodine slopes of L and H, effective energies, reference adipose tissue and liver base tissue. They were
derived from contrast-enhanced clinical scans of this protocol (140 kV: a single scan; 120 kV: pooled scans). Other protocols need their own
calibration.

## Quality control

Every series is checked by refitting the 70-keV weight from L/H on three slices.

- Rule of the manuscript: a series whose weight deviates by more than 0.005 from the calibration, or whose fit residual
  exceeds a standard deviation of 1.0 HU, is flagged (`qa.qc_pass = false`) and left out by `agreement`
  (keep it with `--include-qc-fail`).
- Beyond 0.01 or 3 HU the grid, the tube voltage or the calibration is wrong, and `series` stops. `--force` continues;
  use it only for phantoms, whose weight may differ from the patient calibration.

## Validation

- L/H/L0/H0 read from the DICOM files are bit-identical to the reference decoding; container positions equal the
  DICOM slice positions.
- `export-nifti` reproduces the image on which BOA was run in the reference analysis exactly.
- For a test examination, all compartment volumes, the fat balance, the single-energy fraction and the L3 values
  equal the reference analysis to within the rounding of the stored reference values (relative difference ≤ 6×10⁻⁵).
- `tests/` checks the decomposition and the statistics on synthetic numbers. `tests/test_no_identifiers.py` fails
  if any file contains pseudonyms, drive paths, dates or patient-related DICOM tags.

## Limitations

There is no in-vivo reference standard: the agreement with BOA describes how the methods differ, not which one is
correct. The fat fraction is the volume fraction of reference adipose tissue, not a proton-density fat fraction.
With two energies, fat with iodine cannot be told apart from soft tissue with air and iodine.

## Related publication

Simon Hellbrueck, Julian Kausch, Hamed Mirbagheri, Abraham Cortes, Juan Camilo Garcés Echeverri, Gebhard Oestreicher, Christian Scheurig-Muenkler, Thomas Kroencke, Josua A. Decker.
*Body composition on contrast-enhanced photon-counting CT: anatomy-guided material decomposition versus threshold segmentation.* Manuscript submitted, 2026.
This repository holds the analysis code for that article. The quality-control limits, the base-tissue
assignment and the sensitivity analyses implemented here are the ones described in its Methods and
Supplementary Methods.

## Data availability

No imaging data are included. The CT examinations cannot be shared publicly because of data
protection regulations; aggregated per-patient results are available from the corresponding author on
reasonable request. The calibration files under `src/sppbc/calibration/` carry no patient information.

## Citation

Please cite the article above and this software. Machine-readable metadata are in `CITATION.cff`;
the archived release carries its own DOI.

## License

MIT, see `LICENSE`.
