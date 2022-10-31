import json
import pathlib
import sys

import fsspec
import numcodecs
import tiffslide
import zarr
from fsspec.implementations.reference import ReferenceFileSystem

# from tiffslide._kerchunk import from_kerchunk
from tiffslide._kerchunk import to_kerchunk

IMAGE = pathlib.Path("CMU-1-JP2K-33005.svs").absolute()

ts = tiffslide.open_slide(IMAGE)
kc = to_kerchunk(ts, urlpath=IMAGE)

if sys.argv[1] == "cpu":
    meta_array = None
    pass

elif sys.argv[1] == "gpu":
    import cupy

    from nvjpeg2k_numcodecs import NvJpeg2k

    old_zarray = kc["refs"]["s0/0/.zarray"]
    z = json.loads(old_zarray)

    numcodecs.register_codec(NvJpeg2k)
    z["compressor"] = {"id": "nvjpeg2k", "blocking": True}
    z["chunks"] = [3, 240, 240]
    z["shape"] = [z["shape"][2], *z["shape"][:2]]

    meta_array = cupy.empty(())

    refs = kc["refs"]
    keys = [k for k in refs if k.startswith("s0/0")]
    for key in keys:
        prefix, idxs = key.rsplit("/", maxsplit=1)
        if idxs in {".zarray", ".zgroup", ".zattrs"}:
            continue
        iy, ix, iz = idxs.split(".")
        new_key = f"{prefix}/{iz}.{iy}.{ix}"
        refs[new_key] = refs.pop(key)
        print(key, "-->", new_key)

    kc["refs"]["s0/0/.zarray"] = json.dumps(z)
else:
    raise ValueError("argv[1] not in {'cpu', 'gpu'}")


fs: ReferenceFileSystem = fsspec.filesystem(
    "reference",
    fo=kc,
)
m = fs.get_mapper("s0/0")

# cupy backed array ??
arr = zarr.Array(store=m, read_only=True, meta_array=meta_array)
print(arr.info)

arr[:, :240, :240]
