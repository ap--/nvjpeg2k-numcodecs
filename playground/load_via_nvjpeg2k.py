import json
import pathlib
import sys

import numcodecs
import tiffslide
from tiffslide._kerchunk import from_kerchunk
from tiffslide._kerchunk import to_kerchunk

IMAGE = pathlib.Path("CMU-1-JP2K-33005.svs").absolute()

ts = tiffslide.open_slide(IMAGE)
kc = to_kerchunk(ts, urlpath=IMAGE)

old_zarray = kc["refs"]["s0/0/.zarray"]
z = json.loads(old_zarray)

if sys.argv[1] == "cpu":
    pass
elif sys.argv[1] == "gpu":
    from nvjpeg2k_numcodecs import NvJpeg2k

    numcodecs.register_codec(NvJpeg2k)
    z["compressor"] = {"id": "nvjpeg2k"}
else:
    raise ValueError("argv[1] not in {'cpu', 'gpu'}")

kc["refs"]["s0/0/.zarray"] = json.dumps(z)

gpu_ts = from_kerchunk(kc)

# cupy backed array ??
arr = gpu_ts.zarr_group["0"]
print(arr.info)
