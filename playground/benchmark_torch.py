import time
import zarr
from torch import as_tensor

from _common_bench import register_nvjpeg2k_codec
from _common_bench import get_nvjpeg2k_zarr_array
from _common_bench import get_zarr_array


register_nvjpeg2k_codec()


@contextlib.contextmanager
def timer(name: str):
    t0 = time.monotonic()
    yield
    duration = time.monotonic() - t0
    print(name, "took:", duration, "seconds")


def bench(arr: zarr.Array):
    tile_size = 240

    if arr.shape[0] == 3:  
        _, height, width = arr.shape
        for x, y in itertools.product(
                range(0, width - tile_size + 1, tile_size),
                range(0, height - tile_size + 1, tile_size)
                ):
            tile = arr[:,y:y+tile_size,x:x+tile_size]
            t = as_tensor(tile, device="cuda")
    elif arr.shape[2] == 3:
        height, width, _ = arr.shape
        for x, y in itertools.product(
                range(0, width - tile_size + 1, tile_size),
                range(0, height - tile_size + 1, tile_size)
                ):
            tile = arr[y:y+tile_size,x:x+tile_size, :]
            t = as_tensor(tile, device="cuda")
    else:
        raise AssertionError


if __name__ == "__main__":
    print("entering main")
    import sys

    if sys.argv[1] == "gpu":
        PATH = sys.argv[2]
      
        with timer("init gpu"):
            arr = get_nvjpeg2k_zarr_array(PATH)

        with timer("bench gpu"):
            bench(arr)

    elif sys.argv[1] == "cpu":
        PATH = sys.argv[2]
        
        with timer("init cpu"):
            arr = get_zarr_array(PATH)

        with timer("bench cpu"):
            bench(arr)

    else:
        print("argv[1] must be gpu or cpu")
        raise SystemExit(1)

    import numcodecs.ndarray_like as n
    print(n._CachedProtocolMeta._instancecheck_cache)
