# distutils: language = c++

import numpy

cimport numpy
from libcpp.vector cimport vector

from nvjpeg2k_numcodecs cimport _nvjpeg2k as nvjpeg2k

numpy.import_array()


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


cdef inline raise_if_nvjpeg2k_error(status: nvjpeg2k.nvjpeg2kStatus_t):
    if status != nvjpeg2k.NVJPEG2K_STATUS_SUCCESS:
        msg = _status_error_msg[status]
        raise RuntimeError(f"nvJPEG2000 status error: {msg!r}")


cdef inline raise_if_cuda_error(cuda_error: nvjpeg2k.cudaError_t):
    if cuda_error != 0:
        msg = _cuda_error_msg[cuda_error]
        raise RuntimeError(f"nvJPEG2000 cuda error: {msg!r}")


cdef int cudaStreamDefault = 0x00  # Default stream flag
cdef int cudaStreamNonBlocking = 0x01  # Stream does not synchronize with stream 0 (the NULL stream)

cdef int cudaEventDefault = 0x00  # Default event flag
cdef int cudaEventBlockingSync = 0x01  # Event uses blocking synchronization
cdef int cudaEventDisableTiming = 0x02  # Event will not record timing data
cdef int cudaEventInterprocess = 0x04  # Event is suitable for interprocess use. cudaEventDisableTiming must be set

cdef class NvJpeg2kContext:

    cdef nvjpeg2kHandle_t handle
    cdef nvjpeg2kDecodeState_t decode_state
    cdef nvjpeg2kStream_t jpeg2k_stream
    cdef cudaStream_t stream

    def __init__(self):
        # device and host allocators
        cdef nvjpeg2kDeviceAllocator_t dev_allocator
        dev_allocator.device_malloc = &dev_malloc
        dev_allocator.device_free = &dev_free
        cdef nvjpeg2kPinnedAllocator_t pinned_allocator
        pinned_allocator.pinned_malloc = &host_malloc
        pinned_allocator.pinned_free = &host_free

        cdef nvjpeg2kStatus_t status
        cdef cudaError_t cuda_error

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

        # stream for decoding
        cuda_error = cudaStreamCreateWithFlags(&self.stream, cudaStreamNonBlocking)
        raise_if_cuda_error(cuda_error)

    def __dealloc__(self):
        cdef nvjpeg2kStatus_t status
        cdef cudaError_t cuda_error

        if self.stream != NULL:
            cuda_error = cudaStreamDestroy(self.stream)
            raise_if_cuda_error(cuda_error)

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
):
    cdef nvjpeg2kStatus_t status
    cdef cudaError_t cuda_error
    cdef cudaEvent_t startEvent = NULL
    cdef cudaEvent_t stopEvent = NULL

    cdef unsigned char* buffer
    cdef size_t length

    if buf is out:
        raise ValueError('cannot decode in-place')

    buffer = buf
    length = len(buf)

    if ctx is None:
        ctx = NvJpeg2kContext()

    cuda_error = cudaStreamSynchronize(ctx.stream)
    raise_if_cuda_error(cuda_error)

    cuda_error = cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync)
    raise_if_cuda_error(cuda_error)

    cuda_error = cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync)
    raise_if_cuda_error(cuda_error)

    cdef nvjpeg2kDecodeParams_t decode_params
    status = nvjpeg2kDecodeParamsCreate(&decode_params)
    raise_if_nvjpeg2k_error(status)

    # >>> if (NVJPEG2K_VER_MAJOR == 0 && NVJPEG2K_VER_MINOR >= 3)
    # >>>    420 and 422 subsampling are enabled in nvJPEG2k v 0.3.0
    status = nvjpeg2kDecodeParamsSetRGBOutput(decode_params, rgb_output)
    raise_if_nvjpeg2k_error(status)

    cdef int bytes_per_element = 1
    cdef nvjpeg2kImage_t output_image
    cdef nvjpeg2kImageInfo_t image_info

    cdef vector[nvjpeg2kImageComponentInfo_t] image_comp_info
    cdef vector[unsigned short*] decode_output_u16
    cdef vector[unsigned char*] decode_output_u8
    cdef vector[size_t] decode_output_pitch

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

    decode_output_pitch.resize(image_info.num_components)
    output_image.pitch_in_bytes = decode_output_pitch.data()
    output_image.num_components = image_info.num_components

    if 8 < image_comp_info[0].precision <= 16:
        decode_output_u16.resize(image_info.num_components)
        output_image.pixel_data = <void **>decode_output_u16.data()
        output_image.pixel_type = NVJPEG2K_UINT16
        bytes_per_element = 2

    elif image_comp_info[0].precision == 8:
        decode_output_u8.resize(image_info.num_components)
        output_image.pixel_data = <void **>decode_output_u8.data()
        output_image.pixel_type = NVJPEG2K_UINT8
        bytes_per_element = 1

    else:
        raise RuntimeError(f"nvJPEG2000 precision not supported: '{image_comp_info[0].precision!r}'")


    if rgb_output:
        for c in range(image_info.num_components):
            cuda_error = cudaMallocPitch(
                &output_image.pixel_data[c],
                &output_image.pitch_in_bytes[c],
                image_info.image_width * bytes_per_element,
                image_info.image_height,
            )
            raise_if_cuda_error(cuda_error)

    else:
        for c in range(image_info.num_components):
            cuda_error = cudaMallocPitch(
                &output_image.pixel_data[c],
                &output_image.pitch_in_bytes[c],
                image_comp_info[c].component_width * bytes_per_element,
                image_comp_info[c].component_height,
            )
            raise_if_cuda_error(cuda_error)

    cuda_error = cudaEventRecord(startEvent, ctx.stream)
    raise_if_cuda_error(cuda_error)

    # decode the image
    status = nvjpeg2kDecodeImage(
        ctx.handle,
        ctx.decode_state,
        ctx.jpeg2k_stream,
        decode_params,
        &output_image,
        ctx.stream,
    )

    cuda_error = cudaEventRecord(stopEvent, ctx.stream)
    raise_if_cuda_error(cuda_error)

    cuda_error = cudaEventSynchronize(stopEvent)
    raise_if_cuda_error(cuda_error)

    cdef float loop_time = 0
    cuda_error = cudaEventElapsedTime(&loop_time, startEvent, stopEvent)

    # generate output array
    shape = (image_info.image_height, image_info.image_width, 3)
    dtype = f"u{bytes_per_element}"
    if out is None:
        out = numpy.empty(shape, dtype=dtype)
    elif isinstance(out, numpy.ndarray):
        if out.shape != shape:
            raise ValueError()
    else:
        count = 1
        for s in shape:
            count *= s
        out = numpy.frombuffer(out, dtype=dtype, count=count)
        out.shape = shape

    status = nvjpeg2kDecodeParamsDestroy(decode_params)
    raise_if_nvjpeg2k_error(status)
    cuda_error = cudaEventDestroy(startEvent)
    raise_if_cuda_error(cuda_error)
    cuda_error = cudaEventDestroy(stopEvent)
    raise_if_cuda_error(cuda_error)

    return out
