"""nvjpeg2k numcodec for decoding directly on nvidia gpus"""
try:
    from nvjpeg2k_numcodecs._version import __version__
except ImportError:
    __version__ = "not-installed"

from nvjpeg2k_numcodecs.nvjpeg2k import NvJpeg2k

__all__ = [
    "NvJpeg2k",
]
