import json
import sys
from pathlib import Path

import cupy
import fsspec
import numcodecs
import tiffslide
import zarr
from fsspec.implementations.reference import ReferenceFileSystem
from nvjpeg2k_numcodecs import NvJpeg2k
from tiffslide._kerchunk import to_kerchunk


def register_nvjpeg2k_codec():
    numcodecs.register_codec(NvJpeg2k)


def get_zarr_array(fn: str | Path, *, level: int = 0) -> zarr.Array:
    """return a zarr Array"""

    fn = Path(fn).absolute()
    assert fn.suffix == ".svs"
    ts = tiffslide.open_slide(fn)
    kc = to_kerchunk(ts, urlpath=fn)

    # build zarr array
    fs: ReferenceFileSystem = fsspec.filesystem(
        "reference",
        fo=kc,
    )
    m = fs.get_mapper(f"s0/{level}")

    # cupy backed array
    arr = zarr.Array(
        store=m,
        read_only=True,
    )
    return arr


def get_nvjpeg2k_zarr_array(fn: str | Path, *, level: int = 0) -> zarr.Array:
    """return a zarr Array using the nvjpeg2k decoder for chunk decompression"""

    fn = Path(fn).absolute()
    assert fn.suffix == ".svs"
    ts = tiffslide.open_slide(fn)
    kc = to_kerchunk(ts, urlpath=fn)

    # this is currently targeting svs files...
    zkey_zarray = f"s0/{level}/.zarray"

    refs = kc["refs"]
    array_info = json.loads(refs[zkey_zarray])

    # replace the compressor with nvjpeg2k (currently requires shape reordering)
    array_info["compressor"] = {"id": "nvjpeg2k", "blocking": False, "num_cuda_streams": 8}
    _sy, _sx, _sz = array_info["shape"]
    _cy, _cx, _cz = array_info["chunks"]
    array_info["chunks"] = [_cz, _cy, _cx]
    array_info["shape"] = [_sz, _sy, _sx]
    refs[zkey_zarray] = json.dumps(array_info)

    # need to remap all chunk indicies
    keys = [k for k in refs if k.startswith(f"s0/{level}")]
    for key in keys:
        prefix, idxs = key.rsplit("/", maxsplit=1)
        if idxs in {".zarray", ".zgroup", ".zattrs"}:
            continue
        iy, ix, iz = idxs.split(".")
        new_key = f"{prefix}/{iz}.{iy}.{ix}"
        refs[new_key] = refs.pop(key)

    # build zarr array
    fs: ReferenceFileSystem = fsspec.filesystem(
        "reference",
        fo=kc,
    )
    m = fs.get_mapper(f"s0/{level}")

    # cupy backed array
    arr = zarr.Array(
        store=m,
        read_only=True,
        meta_array=cupy.empty(()),
    )
    return arr
