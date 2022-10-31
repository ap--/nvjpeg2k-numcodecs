# distutils: language = c++

import cupy
from cupy.cuda.stream import Stream

from libc.stdint cimport intptr_t
from libcpp.vector cimport vector

from nvjpeg2k_numcodecs cimport _nvjpeg2k as nvjpeg2k

_status_error_msg = {
    nvjpeg2k.NVJPEG2K_STATUS_SUCCESS: "SUCCESS",
    nvjpeg2k.NVJPEG2K_STATUS_NOT_INITIALIZED: "NOT_INITIALIZED",
    nvjpeg2k.NVJPEG2K_STATUS_INVALID_PARAMETER: "INVALID_PARAMETER",
    nvjpeg2k.NVJPEG2K_STATUS_BAD_JPEG: "BAD_JPEG",
    nvjpeg2k.NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED: "JPEG_NOT_SUPPORTED",
    nvjpeg2k.NVJPEG2K_STATUS_ALLOCATOR_FAILURE: "ALLOCATOR_FAILURE",
    nvjpeg2k.NVJPEG2K_STATUS_EXECUTION_FAILED: "EXECUTION_FAILED",
    nvjpeg2k.NVJPEG2K_STATUS_ARCH_MISMATCH: "ARCH_MISMATCH",
    nvjpeg2k.NVJPEG2K_STATUS_INTERNAL_ERROR: "INTERNAL_ERROR",
    nvjpeg2k.NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED: "IMPLEMENTATION_NOT_SUPPORTED",
}
_cuda_error_msg = {}


cdef int dev_malloc(void **p, size_t s):
    return <int>cudaMalloc(p, s)

cdef int dev_free(void *p):
    return <int>cudaFree(p)

cdef int host_malloc(void **p, size_t s, unsigned int f):
    return <int>cudaHostAlloc(p, s, f)

cdef int host_free(void *p):
    return <int>cudaFreeHost(p)


cdef raise_if_nvjpeg2k_error(status: nvjpeg2k.nvjpeg2kStatus_t, name: str = " "):
    if status != nvjpeg2k.NVJPEG2K_STATUS_SUCCESS:
        msg = _status_error_msg[status]
        raise RuntimeError(f"nvJPEG2000 status error:{name}{msg!r}")


cdef raise_if_cuda_error(cuda_error: nvjpeg2k.cudaError_t, name: str = " "):
    if cuda_error != 0:
        # msg = _cuda_error_msg[cuda_error]
        msg = int(cuda_error)
        raise RuntimeError(f"nvJPEG2000 cuda error:{name}{msg!r}")


cdef int cudaStreamDefault = 0x00  # Default stream flag
cdef int cudaStreamNonBlocking = 0x01  # Stream does not synchronize with stream 0 (the NULL stream)

cdef int cudaEventDefault = 0x00  # Default event flag
cdef int cudaEventBlockingSync = 0x01  # Event uses blocking synchronization
cdef int cudaEventDisableTiming = 0x02  # Event will not record timing data
cdef int cudaEventInterprocess = 0x04  # Event is suitable for interprocess use. cudaEventDisableTiming must be set


cdef class NvJpeg2kContext:

    cdef nvjpeg2kHandle_t handle = NULL
    cdef nvjpeg2kDecodeState_t decode_state = NULL
    cdef nvjpeg2kStream_t jpeg2k_stream = NULL

    def __init__(self):
        # device and host allocators
        cdef nvjpeg2kDeviceAllocator_t dev_allocator
        dev_allocator.device_malloc = &dev_malloc
        dev_allocator.device_free = &dev_free
        cdef nvjpeg2kPinnedAllocator_t pinned_allocator
        pinned_allocator.pinned_malloc = &host_malloc
        pinned_allocator.pinned_free = &host_free

        cdef nvjpeg2kStatus_t status

        status = nvjpeg2k.nvjpeg2kCreate(
            nvjpeg2k.NVJPEG2K_BACKEND_DEFAULT,
            &dev_allocator,
            &pinned_allocator,
            &self.handle,
        )
        raise_if_nvjpeg2k_error(status)

        status = nvjpeg2kDecodeStateCreate(
            self.handle,
            &self.decode_state,
        )
        raise_if_nvjpeg2k_error(status)

        status = nvjpeg2kStreamCreate(&self.jpeg2k_stream)
        raise_if_nvjpeg2k_error(status)

    def __dealloc__(self):
        cdef nvjpeg2kStatus_t status

        if self.jpeg2k_stream != NULL:
            status = nvjpeg2kStreamDestroy(self.jpeg2k_stream)
            raise_if_nvjpeg2k_error(status)

        if self.decode_state != NULL:
            status = nvjpeg2kDecodeStateDestroy(self.decode_state)
            raise_if_nvjpeg2k_error(status)

        if self.handle != NULL:
            status = nvjpeg2kDestroy(self.handle)
            raise_if_nvjpeg2k_error(status)


def nvjpeg2k_decode(
    buf,
    out=None,
    rgb_output: int = 0,
    ctx: NvJpeg2kContext = None,
    stream: Stream = None,
):
    cdef nvjpeg2kStatus_t status
    cdef cudaStream_t cuda_stream

    cdef unsigned char* buffer
    cdef size_t length

    cdef nvjpeg2kDecodeParams_t decode_params = NULL

    cdef int bytes_per_element = 1
    cdef nvjpeg2kImage_t output_image
    cdef nvjpeg2kImageInfo_t image_info

    cdef vector[nvjpeg2kImageComponentInfo_t] image_comp_info
    cdef vector[void *] decode_output_pixel_data
    cdef vector[size_t] decode_output_pitch

    if buf is out:
        raise ValueError("cannot decode in-place")

    buffer = buf
    length = len(buf)

    if ctx is None:
        ctx = NvJpeg2kContext()

    if stream is None:
        stream = Stream(non_blocking=True)
    cuda_stream = <cudaStream_t> <intptr_t> stream.ptr

    cudaStreamSynchronize(cuda_stream)

    try:
        status = nvjpeg2kDecodeParamsCreate(&decode_params)
        raise_if_nvjpeg2k_error(status)

        status = nvjpeg2kDecodeParamsSetRGBOutput(decode_params, rgb_output)
        raise_if_nvjpeg2k_error(status)

        status = nvjpeg2kStreamParse(
            ctx.handle,
            <unsigned char*> buffer,
            length,
            0,
            0,
            ctx.jpeg2k_stream
        )
        raise_if_nvjpeg2k_error(status)

        status = nvjpeg2kStreamGetImageInfo(ctx.jpeg2k_stream, &image_info)
        raise_if_nvjpeg2k_error(status)

        image_comp_info.resize(image_info.num_components)
        for c in range(image_info.num_components):
            status = nvjpeg2kStreamGetImageComponentInfo(
                ctx.jpeg2k_stream,
                &image_comp_info[c],
                c
            )
            raise_if_nvjpeg2k_error(status)

        decode_output_pixel_data.resize(image_info.num_components)
        output_image.pixel_data = decode_output_pixel_data.data()
        decode_output_pitch.resize(image_info.num_components)
        output_image.pitch_in_bytes = decode_output_pitch.data()
        output_image.num_components = image_info.num_components

        if 8 < image_comp_info[0].precision <= 16:
            output_image.pixel_type = NVJPEG2K_UINT16
            bytes_per_element = 2

        elif image_comp_info[0].precision == 8:
            output_image.pixel_type = NVJPEG2K_UINT8
            bytes_per_element = 1

        else:
            raise RuntimeError(f"nvJPEG2000 precision not supported: '{image_comp_info[0].precision!r}'")

        # shapes
        shape = (image_info.num_components, image_info.image_height, image_info.image_width)
        dtype = f"u{bytes_per_element}"

        # >>> generate output array
        if out is None:
            out = cupy.empty(shape, dtype=dtype, order="C")
        elif isinstance(out, cupy.ndarray):
            if out.shape != shape:
                raise ValueError("out has incorrect shape")
        else:
            raise NotImplementedError("todo: not implemented yet...")

        decode_output_pitch.resize(image_info.num_components)
        output_image.pitch_in_bytes = decode_output_pitch.data()
        output_image.num_components = image_info.num_components

        for c in range(image_info.num_components):
            output_image.pixel_data[c] = <void *> <intptr_t> out[c, :, :].data.ptr
            output_image.pitch_in_bytes[c] = image_info.image_width * bytes_per_element

        # decode the image
        status = nvjpeg2kDecodeImage(
            ctx.handle,
            ctx.decode_state,
            ctx.jpeg2k_stream,
            decode_params,
            &output_image,
            cuda_stream,
        )
        raise_if_nvjpeg2k_error(status, "decodeImage")

    finally:
        if decode_params != NULL:
            status = nvjpeg2kDecodeParamsDestroy(decode_params)
            raise_if_nvjpeg2k_error(status, "paramsDestroy")

    return out
