from typing import Optional
from typing import Union

from numcodecs.abc import Codec

from nvjpeg2k_numcodecs._nvjpeg2k import NvJpeg2kContext
from nvjpeg2k_numcodecs._nvjpeg2k import Stream
from nvjpeg2k_numcodecs._nvjpeg2k import nvjpeg2k_decode

# waiting for: https://peps.python.org/pep-0688/
BufferLike = Union[bytes, bytearray, memoryview]


class NvJpeg2k(Codec):
    """NvJpeg2000 codec for numcodecs"""

    codec_id = "nvjpeg2k"

    def __init__(self, blocking: bool = False) -> None:
        self._ctx = NvJpeg2kContext()
        self._stream = Stream(non_blocking=not blocking)

    def encode(self, buf: BufferLike) -> None:
        """Encode data in `buf`.

        Parameters
        ----------
        buf : buffer-like
            Data to be encoded. Can be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : buffer-like
            Encoded data. Can be any object supporting the new-style buffer
            protocol.
        """
        raise NotImplementedError("todo")

    def decode(self, buf: BufferLike, out: Optional[BufferLike] = None) -> BufferLike:
        """Decode data in `buf`.

        Parameters
        ----------
        buf : buffer-like
            Encoded data. Can be any object supporting the new-style buffer
            protocol.
        out : buffer-like, optional
            Writeable buffer to store decoded data. N.B. if provided, this buffer
            must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : buffer-like
            Decoded data. Can be any object supporting the new-style
            buffer protocol.
        """
        return nvjpeg2k_decode(  # type: ignore
            buf, out=_flat(out), ctx=self._ctx, stream=self._stream
        )


# from imagecodecs.numcodecs import _flat
def _flat(out: Optional[BufferLike]) -> Optional[BufferLike]:
    """Return numpy array as contiguous view of bytes if possible."""
    if out is None:
        return None
    view = memoryview(out)
    if view.readonly or not view.contiguous:
        return None
    return view.cast("B")
