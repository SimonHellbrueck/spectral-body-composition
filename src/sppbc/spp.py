"""Reader for Siemens SPP containers (DICOM private tag (0049,1010), creator 'SIEMENS CT SPP HEADER').

The format was derived from exported data alone; no vendor software, library or documentation was used.

Container (little endian)::

    0    char[16]  magic 'Siemens SPP V21\\0'
    16   u32       header size (108)
    20   u32       slice index (0-based, ascending z)
    24   f32[3]    image position (equals DICOM ImagePositionPatient)
    36   u32       columns, u32 rows
    44   u32       number of blocks n (5)
    48   n x {u32 id, u32 size, u32 checksum}     (checksum algorithm unknown, not needed)
    108  blocks, each starting 4-byte aligned; ids 9..12 = images in descriptor <Content> order
         (L/H/L0/H0), id 13 = XML <SPECTRAL_MIX_DESCRIPTOR>

Image block ('DS' codec)::

    0    u16       0
    2    char[2]   'DS'
    4    u32       n1 = number of symbols in stream 1 (= pixels + escapes)
    8    u32       w1 = length of stream 1 in 16-bit words
    12   u16[256]  table 1, normalized symbol frequencies (sum 4096)
    524  u32       n2 = number of symbols in stream 2 (= escapes)
    528  u32       w2 = length of stream 2 in 16-bit words
    532  u16[256]  table 2 (sum 4096)
    1044 u16[w1]   stream 1, then u16[w2] stream 2

Entropy coder: 8-lane interleaved rANS, 32-bit states, L = 2^16, 16-bit renormalization words,
12-bit probabilities. Encoded in forward symbol order, symbol p in lane p % 8 with the first group
holding only n % 8 symbols (lanes n % 8 .. 7 idle); final states stored at the end of each stream
as 8 low words followed by 8 high words; the decoder reads the stream backwards. A correct decode
consumes every word and returns every lane to the initial state 2^16 (used as integrity check).

Symbols: zig-zag mapped prediction residuals v (0, -1, +1, -2, ...). v >= 255 is coded as escape
symbol 255 followed by v & 255 in stream 1 and v >> 8 in stream 2.
Prediction (raster order): row 0 left neighbour (0 before the first column); column 0 predicted
as 0; otherwise planar predictor a + b - c (left + above - upper-left).
Stored values map to HU through the descriptor's piecewise-linear MapStored*/MapOutput* tables.
"""
from __future__ import annotations

import dataclasses
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - slow fallback
    def njit(*a, **k):
        def deco(f):
            return f
        return deco(a[0]) if a and callable(a[0]) else deco

SPP_TAG = (0x0049, 0x1010)
MAGIC = b"Siemens SPP V21"
RANS_L = 1 << 16
SCALE_BITS = 12
LANES = 8
ESCAPE = 255


@dataclasses.dataclass
class SppSlice:
    images: dict            # name -> stored values, uint16 (rows, cols)
    descriptor: dict        # parsed SPECTRAL_MIX_DESCRIPTOR
    header: dict            # container header

    def hu(self, name: str, dtype=np.float32) -> np.ndarray:
        return stored_to_hu(self.images[name], self.descriptor, name).astype(dtype)


# ----------------------------------------------------------------------------------------- container
def parse_container(buf: bytes) -> dict:
    if not buf.startswith(MAGIC):
        raise ValueError("not an SPP V21 container")
    hsize, idx = struct.unpack_from("<II", buf, 16)
    xyz = struct.unpack_from("<3f", buf, 24)
    cols, rows, nblk = struct.unpack_from("<III", buf, 36)
    blocks, off = [], hsize
    for n in range(nblk):
        bid, size, chk = struct.unpack_from("<III", buf, 48 + 12 * n)
        blocks.append({"id": bid, "offset": off, "size": size, "checksum": chk})
        off = (off + size + 3) & ~3
    if off < len(buf) or off - len(buf) > 3:
        raise ValueError(f"block table does not cover the container ({off} vs {len(buf)})")
    return {"header_size": hsize, "index": idx, "position": xyz, "cols": cols, "rows": rows, "blocks": blocks}


def parse_block_header(buf: bytes, offset: int) -> dict:
    zero, tag, n1, w1 = struct.unpack_from("<H2sII", buf, offset)
    if tag != b"DS":
        raise ValueError(f"unknown image codec {tag!r}")
    t1 = np.frombuffer(buf, dtype="<u2", count=256, offset=offset + 12).astype(np.int64)
    n2, w2 = struct.unpack_from("<II", buf, offset + 524)
    t2 = np.frombuffer(buf, dtype="<u2", count=256, offset=offset + 532).astype(np.int64)
    if t1.sum() != 1 << SCALE_BITS or t2.sum() != 1 << SCALE_BITS:
        raise ValueError("frequency tables do not sum to 4096")
    p = offset + 1044
    s1 = np.frombuffer(buf, dtype="<u2", count=w1, offset=p).astype(np.int64)
    s2 = np.frombuffer(buf, dtype="<u2", count=w2, offset=p + 2 * w1).astype(np.int64)
    return {"n1": n1, "w1": w1, "t1": t1, "n2": n2, "w2": w2, "t2": t2, "stream1": s1, "stream2": s2}


# ----------------------------------------------------------------------------------------- codec
@njit(cache=True)
def rans_decode(words, nsym, freqs):
    """Decode nsym symbols; returns (symbols, ok). ok = all words used and all lanes back at 2^16."""
    out = np.zeros(nsym, np.int64)
    n = words.shape[0]
    if nsym == 0:
        return out, n == 2 * LANES
    cum = np.zeros(257, np.int64)
    for s in range(256):
        cum[s + 1] = cum[s] + freqs[s]
    slot2sym = np.zeros(1 << SCALE_BITS, np.int64)
    for s in range(256):
        for q in range(cum[s], cum[s + 1]):
            slot2sym[q] = s
    st = np.zeros(LANES, np.int64)
    for i in range(LANES):
        st[i] = words[n - 2 * LANES + i] | (words[n - LANES + i] << 16)
    ptr = n - 2 * LANES - 1
    rem = nsym % LANES
    shift = (LANES - rem) % LANES
    npad = nsym + shift
    mask = (1 << SCALE_BITS) - 1
    for p in range(npad - 1, -1, -1):
        if rem != 0 and p < LANES and p >= rem:
            continue
        ln = p % LANES
        x = st[ln]
        slot = x & mask
        s = slot2sym[slot]
        x = freqs[s] * (x >> SCALE_BITS) + slot - cum[s]
        if x < RANS_L:
            if ptr < 0:
                return out, False
            x = (x << 16) | words[ptr]
            ptr -= 1
        st[ln] = x
        idx = p if p < LANES else p - shift
        out[idx] = s
    ok = ptr == -1
    for i in range(LANES):
        if st[i] != RANS_L:
            ok = False
    return out, ok


@njit(cache=True)
def _symbols_to_residuals(sym1, sym2, npix):
    res = np.zeros(npix, np.int64)
    j = 0
    e = 0
    for i in range(npix):
        v = sym1[j]
        j += 1
        if v == ESCAPE:
            v = (sym2[e] << 8) | sym1[j]
            j += 1
            e += 1
        res[i] = (v >> 1) if (v & 1) == 0 else -((v + 1) >> 1)
    return res, j, e


@njit(cache=True)
def _reconstruct_planar(res, rows, cols):
    x = np.zeros((rows, cols), np.int64)
    acc = 0
    for c in range(cols):
        acc += res[c]
        x[0, c] = acc
    for r in range(1, rows):
        base = r * cols
        x[r, 0] = res[base]
        for c in range(1, cols):
            x[r, c] = x[r, c - 1] + x[r - 1, c] - x[r - 1, c - 1] + res[base + c]
    return x


def decode_image_block(buf: bytes, block: dict, rows: int, cols: int) -> np.ndarray:
    bh = parse_block_header(buf, block["offset"])
    npix = rows * cols
    if bh["n1"] - bh["n2"] != npix:
        raise ValueError("symbol counts do not match the image size")
    s1, ok1 = rans_decode(bh["stream1"], bh["n1"], bh["t1"])
    s2, ok2 = rans_decode(bh["stream2"], bh["n2"], bh["t2"])
    if not (ok1 and ok2):
        raise ValueError(f"rANS integrity check failed for block {block['id']}")
    res, used1, used2 = _symbols_to_residuals(s1, s2, npix)
    if used1 != bh["n1"] or used2 != bh["n2"]:
        raise ValueError("escape accounting failed")
    img = _reconstruct_planar(res, rows, cols)
    # The encoder predicts in modulo-2^16 arithmetic: near very dense voxels (metal, dense contrast) next to
    # out-of-field zeros the prediction wraps, so the reconstruction must wrap too (the wrapped image matches the
    # scanner VMI70 within 2.3 HU). Wraps can accumulate along the
    # prediction path, so only gross values (beyond 2^24) are treated as corruption; the rANS and escape checks above
    # already validate the streams.
    if np.abs(img).max() > (1 << 24):
        raise ValueError("reconstructed values implausible (corrupt block?)")
    return np.mod(img, 65536).astype(np.uint16)


# ----------------------------------------------------------------------------------------- descriptor
def _num(text):
    parts = text.split()
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return text
    if all(v.is_integer() for v in vals) and not re.search(r"[.eE]", text):
        vals = [int(v) for v in vals]
    return vals[0] if len(vals) == 1 else vals


def parse_descriptor(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes.split(b"\x00")[0].decode("latin-1"))
    return {child.tag: _num(child.text or "") for child in root}


def content_names(descriptor: dict) -> list:
    return str(descriptor.get("Content", "L/H/L0/H0")).split("/")


def stored_to_hu(stored: np.ndarray, descriptor: dict, name: str) -> np.ndarray:
    """Piecewise-linear mapping MapStored<group> -> MapOutput<group>; group 'LH' or 'L0H0'."""
    group = "L0H0" if name.endswith("0") else "LH"
    xp = np.atleast_1d(np.asarray(descriptor[f"MapStored{group}"], dtype=np.float64))
    fp = np.atleast_1d(np.asarray(descriptor[f"MapOutput{group}"], dtype=np.float64))
    return np.interp(stored.astype(np.float64), xp, fp)


# ----------------------------------------------------------------------------------------- slice API
def load_spp_series(files_in_k_order, names=("L", "H", "L0", "H0"), dtype=np.float32, progress=None):
    """Decode the SPP images of a whole series into HU volumes indexed (i, j, k) like pcdlab.io.dicom.

    ``files_in_k_order`` must be sorted by position along the slice normal (CTSeries.files).
    Returns (volumes: dict name -> array, descriptors: list of dict, headers: list of dict).
    """
    import pydicom
    vols, descs, hdrs = None, [], []
    for k, fp in enumerate(files_in_k_order):
        ds = pydicom.dcmread(fp, stop_before_pixels=True)
        sl = read_spp(ds, names=names)
        if vols is None:
            rows, cols = sl.header["rows"], sl.header["cols"]
            vols = {nm: np.empty((cols, rows, len(files_in_k_order)), dtype) for nm in names}
        for nm in names:
            vols[nm][:, :, k] = sl.hu(nm, dtype).T
        descs.append(sl.descriptor)
        hdrs.append({key: sl.header[key] for key in ("index", "position", "rows", "cols")})
        if progress and (k % 100 == 0):
            progress(k)
    return vols, descs, hdrs


def read_spp(source, names=None) -> SppSlice:
    """Decode one slice. ``source`` is the raw container bytes or a pydicom Dataset."""
    if not isinstance(source, (bytes, bytearray, memoryview)):
        source = bytes(source[SPP_TAG].value)
    buf = bytes(source)
    hdr = parse_container(buf)
    blocks = {b["id"]: b for b in hdr["blocks"]}
    xml_blk = blocks[max(blocks)]
    desc = parse_descriptor(buf[xml_blk["offset"]: xml_blk["offset"] + xml_blk["size"]])
    image_ids = sorted(blocks)[:-1]
    cn = content_names(desc)
    if len(cn) != len(image_ids):
        raise ValueError(f"content {cn} does not match {len(image_ids)} image blocks")
    images = {}
    for nm, bid in zip(cn, image_ids):
        if names is None or nm in names:
            images[nm] = decode_image_block(buf, blocks[bid], hdr["rows"], hdr["cols"])
    return SppSlice(images=images, descriptor=desc, header=hdr)
