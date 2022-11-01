
from libc.stdint cimport int32_t
from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport uint32_t


cdef extern from "library_types.h":
    cdef enum libraryPropertyType:
        MAJOR_VERSION
        MINOR_VERSION
        PATCH_LEVEL

    ctypedef libraryPropertyType libraryPropertyType_t


cdef extern from "cuda_runtime_api.h":
    cdef enum cudaError:
        cudaSuccess = 0
        cudaErrorMissingConfiguration = 1
        cudaErrorMemoryAllocation = 2
        cudaErrorInitializationError = 3
        cudaErrorLaunchFailure = 4
        cudaErrorPriorLaunchFailure = 5
        cudaErrorLaunchTimeout = 6
        cudaErrorLaunchOutOfResources = 7
        cudaErrorInvalidDeviceFunction = 8
        cudaErrorInvalidConfiguration = 9
        cudaErrorInvalidDevice = 10
        cudaErrorInvalidValue = 11
        cudaErrorInvalidPitchValue = 12
        cudaErrorInvalidSymbol = 13
        cudaErrorMapBufferObjectFailed = 14
        cudaErrorUnmapBufferObjectFailed = 15
        cudaErrorInvalidHostPointer = 16
        cudaErrorInvalidDevicePointer = 17
        cudaErrorInvalidTexture = 18
        cudaErrorInvalidTextureBinding = 19
        cudaErrorInvalidChannelDescriptor = 20
        cudaErrorInvalidMemcpyDirection = 21
        cudaErrorAddressOfConstant = 22
        cudaErrorTextureFetchFailed = 23
        cudaErrorTextureNotBound = 24
        cudaErrorSynchronizationError = 25
        cudaErrorInvalidFilterSetting = 26
        cudaErrorInvalidNormSetting = 27
        cudaErrorMixedDeviceExecution = 28
        cudaErrorCudartUnloading = 29
        cudaErrorUnknown = 30
        cudaErrorNotYetImplemented = 31
        cudaErrorMemoryValueTooLarge = 32
        cudaErrorInvalidResourceHandle = 33
        cudaErrorNotReady = 34
        cudaErrorInsufficientDriver = 35
        cudaErrorSetOnActiveProcess = 36
        cudaErrorInvalidSurface = 37
        cudaErrorNoDevice = 38
        cudaErrorECCUncorrectable = 39
        cudaErrorSharedObjectSymbolNotFound = 40
        cudaErrorSharedObjectInitFailed = 41
        cudaErrorUnsupportedLimit = 42
        cudaErrorDuplicateVariableName = 43
        cudaErrorDuplicateTextureName = 44
        cudaErrorDuplicateSurfaceName = 45
        cudaErrorDevicesUnavailable = 46
        cudaErrorInvalidKernelImage = 47
        cudaErrorNoKernelImageForDevice = 48
        cudaErrorIncompatibleDriverContext = 49
        cudaErrorPeerAccessAlreadyEnabled = 50
        cudaErrorPeerAccessNotEnabled = 51
        cudaErrorDeviceAlreadyInUse = 54
        cudaErrorProfilerDisabled = 55
        cudaErrorProfilerNotInitialized = 56
        cudaErrorProfilerAlreadyStarted = 57
        cudaErrorProfilerAlreadyStopped = 58
        cudaErrorAssert = 59
        cudaErrorTooManyPeers = 60
        cudaErrorHostMemoryAlreadyRegistered = 61
        cudaErrorHostMemoryNotRegistered = 62
        cudaErrorOperatingSystem = 63
        cudaErrorPeerAccessUnsupported = 64
        cudaErrorLaunchMaxDepthExceeded = 65
        cudaErrorLaunchFileScopedTex = 66
        cudaErrorLaunchFileScopedSurf = 67
        cudaErrorSyncDepthExceeded = 68
        cudaErrorLaunchPendingCountExceeded = 69
        cudaErrorNotPermitted = 70
        cudaErrorNotSupported = 71
        cudaErrorHardwareStackError = 72
        cudaErrorIllegalInstruction = 73
        cudaErrorMisalignedAddress = 74
        cudaErrorInvalidAddressSpace = 75
        cudaErrorInvalidPc = 76
        cudaErrorIllegalAddress = 77
        cudaErrorInvalidPtx = 78
        cudaErrorInvalidGraphicsContext = 79
        cudaErrorNvlinkUncorrectable = 80
        cudaErrorJitCompilerNotFound = 81
        cudaErrorCooperativeLaunchTooLarge = 82
        cudaErrorStartupFailure = 0x7f
        cudaErrorApiFailureBase = 10000
    ctypedef cudaError cudaError_t

    ctypedef struct CUevent_st
    ctypedef CUevent_st* cudaEvent_t

    ctypedef struct CUstream_st
    ctypedef CUstream_st* cudaStream_t

    cdef cudaError_t cudaMalloc(void **devPtr, size_t size)
    cdef cudaError_t cudaFree(void *devPtr)
    cdef cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
    cdef cudaError_t cudaFreeHost(void *ptr)
    cdef cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)

    cdef cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
    cdef cudaError_t cudaStreamDestroy(cudaStream_t stream)
    cdef cudaError_t cudaStreamSynchronize(cudaStream_t stream) nogil

    cdef cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
    cdef cudaError_t cudaEventDestroy(cudaEvent_t event)
    cdef cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
    cdef cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
    cdef cudaError_t cudaEventSynchronize(cudaEvent_t event)


cdef extern from "nvjpeg2k.h":

    # Prototype for device memory allocation, modelled after cudaMalloc()
    ctypedef int (*nvjpeg2kDeviceMalloc)(void**, size_t)

    # Prototype for device memory release
    ctypedef int (*nvjpeg2kDeviceFree)(void*)

    # Prototype for pinned memory allocation, modelled after cudaHostAlloc()
    ctypedef int (*nvjpeg2kPinnedMalloc)(void**, size_t, unsigned int flags)
    # Prototype for device memory release
    ctypedef int (*nvjpeg2kPinnedFree)(void*)

    ctypedef struct nvjpeg2kDeviceAllocator_t:
        nvjpeg2kDeviceMalloc device_malloc
        nvjpeg2kDeviceFree device_free

    ctypedef struct nvjpeg2kPinnedAllocator_t:
        nvjpeg2kPinnedMalloc pinned_malloc
        nvjpeg2kPinnedFree   pinned_free

    ctypedef enum nvjpeg2kStatus_t:
        NVJPEG2K_STATUS_SUCCESS = 0
        NVJPEG2K_STATUS_NOT_INITIALIZED = 1
        NVJPEG2K_STATUS_INVALID_PARAMETER = 2
        NVJPEG2K_STATUS_BAD_JPEG = 3
        NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED = 4
        NVJPEG2K_STATUS_ALLOCATOR_FAILURE = 5
        NVJPEG2K_STATUS_EXECUTION_FAILED = 6
        NVJPEG2K_STATUS_ARCH_MISMATCH = 7
        NVJPEG2K_STATUS_INTERNAL_ERROR = 8
        NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED = 9

    ctypedef enum nvjpeg2kBackend_t:
        NVJPEG2K_BACKEND_DEFAULT = 0

    ctypedef enum nvjpeg2kColorSpace_t:
        NVJPEG2K_COLORSPACE_NOT_SUPPORTED = -1
        NVJPEG2K_COLORSPACE_UNKNOWN       = 0
        NVJPEG2K_COLORSPACE_SRGB          = 1
        NVJPEG2K_COLORSPACE_GRAY          = 2
        NVJPEG2K_COLORSPACE_SYCC          = 3

    ctypedef struct nvjpeg2kImageComponentInfo_t:
        uint32_t component_width
        uint32_t component_height
        uint8_t  precision
        uint8_t  sgn

    ctypedef struct nvjpeg2kImageInfo_t:
        uint32_t image_width
        uint32_t image_height
        uint32_t tile_width
        uint32_t tile_height
        uint32_t num_tiles_x  # no of tiles in horizontal direction
        uint32_t num_tiles_y  # no of tiles in vertical direction
        uint32_t num_components

    ctypedef enum nvjpeg2kImageType_t:
        NVJPEG2K_UINT8 = 0
        NVJPEG2K_UINT16 = 1

    ctypedef struct nvjpeg2kImage_t:
        void **pixel_data
        size_t *pitch_in_bytes
        nvjpeg2kImageType_t pixel_type
        uint32_t num_components

    DEF NVJPEG2K_MAXRES = 33

    ctypedef enum nvjpeg2kProgOrder:
        NVJPEG2K_LRCP = 0
        NVJPEG2K_RLCP = 1
        NVJPEG2K_RPCL = 2
        NVJPEG2K_PCRL = 3
        NVJPEG2K_CPRL = 4

    ctypedef enum nvjpeg2kBitstreamType:
        NVJPEG2K_STREAM_J2K = 0
        NVJPEG2K_STREAM_JP2 = 1

    # contains parameters present in the COD and SIZ headers of the JPEG 2000 bitstream
    ctypedef struct nvjpeg2kEncodeConfig_t:
        nvjpeg2kBitstreamType stream_type
        nvjpeg2kColorSpace_t color_space
        uint16_t rsiz
        uint32_t image_width
        uint32_t image_height
        uint32_t enable_tiling
        uint32_t tile_width
        uint32_t tile_height
        uint32_t num_components
        nvjpeg2kImageComponentInfo_t *image_comp_info
        uint32_t enable_SOP_marker
        uint32_t enable_EPH_marker
        nvjpeg2kProgOrder prog_order
        uint32_t num_layers
        uint32_t mct_mode
        uint32_t num_resolutions
        uint32_t code_block_w
        uint32_t code_block_h
        uint32_t encode_modes
        uint32_t irreversible
        uint32_t enable_custom_precincts
        uint32_t precint_width[NVJPEG2K_MAXRES]
        uint32_t precint_height[NVJPEG2K_MAXRES]

    cdef struct nvjpeg2kHandle
    ctypedef nvjpeg2kHandle* nvjpeg2kHandle_t

    cdef struct nvjpeg2kDecodeState
    ctypedef nvjpeg2kDecodeState* nvjpeg2kDecodeState_t

    cdef struct nvjpeg2kStream
    ctypedef nvjpeg2kStream* nvjpeg2kStream_t

    cdef struct nvjpeg2kDecodeParams
    ctypedef nvjpeg2kDecodeParams* nvjpeg2kDecodeParams_t

    cdef struct nvjpeg2kEncoder
    ctypedef nvjpeg2kEncoder* nvjpeg2kEncoder_t

    cdef struct nvjpeg2kEncodeState
    ctypedef nvjpeg2kEncodeState* nvjpeg2kEncodeState_t

    cdef struct nvjpeg2kEncodeParams
    ctypedef nvjpeg2kEncodeParams* nvjpeg2kEncodeParams_t


    # returns library's property values, such as MAJOR_VERSION, MINOR_VERSION or PATCH_LEVEL
    # noinspection PyShadowingBuiltins
    cdef nvjpeg2kStatus_t nvjpeg2kGetCudartProperty(libraryPropertyType type, int *value)

    # returns CUDA Toolkit property values that was used for building library,
    # such as MAJOR_VERSION, MINOR_VERSION or PATCH_LEVEL
    # noinspection PyShadowingBuiltins
    cdef nvjpeg2kStatus_t nvjpeg2kGetProperty(libraryPropertyType type, int *value)

    cdef nvjpeg2kStatus_t nvjpeg2kCreateSimple(nvjpeg2kHandle_t *handle)

    cdef nvjpeg2kStatus_t nvjpeg2kCreate(
        nvjpeg2kBackend_t backend,
        nvjpeg2kDeviceAllocator_t *device_allocator,
        nvjpeg2kPinnedAllocator_t *pinned_allocator,
        nvjpeg2kHandle_t *handle
    )

    cdef nvjpeg2kStatus_t nvjpeg2kDestroy(nvjpeg2kHandle_t handle)

    cdef nvjpeg2kStatus_t nvjpeg2kDecodeStateCreate(
        nvjpeg2kHandle_t handle,
        nvjpeg2kDecodeState_t *decode_state
    )

    cdef nvjpeg2kStatus_t nvjpeg2kDecodeStateDestroy(nvjpeg2kDecodeState_t decode_state)

    cdef nvjpeg2kStatus_t nvjpeg2kStreamCreate(nvjpeg2kStream_t *stream_handle)

    cdef nvjpeg2kStatus_t nvjpeg2kStreamDestroy(nvjpeg2kStream_t stream_handle)

    cdef nvjpeg2kStatus_t nvjpeg2kStreamParse(
        nvjpeg2kHandle_t handle,
        const unsigned char *data,
        size_t length,
        int save_metadata,
        int save_stream,
        nvjpeg2kStream *stream_handle
    ) nogil

    cdef nvjpeg2kStatus_t nvjpeg2kStreamGetImageInfo(
        nvjpeg2kStream_t stream_handle,
        nvjpeg2kImageInfo_t* image_info
    ) nogil

    cdef nvjpeg2kStatus_t nvjpeg2kStreamGetImageComponentInfo(
        nvjpeg2kStream_t stream_handle,
        nvjpeg2kImageComponentInfo_t* component_info,
        uint32_t component_id
    ) nogil

    cdef nvjpeg2kStatus_t nvjpeg2kStreamGetResolutionsInTile(
        nvjpeg2kStream_t stream_handle,
        uint32_t tile_id,
        uint32_t* num_res
    )

    cdef nvjpeg2kStatus_t nvjpeg2kStreamGetTileComponentDim(
        nvjpeg2kStream_t stream_handle,
        uint32_t component_id,
        uint32_t tile_id,
        uint32_t* tile_width,
        uint32_t* tile_height
    )

    cdef nvjpeg2kStatus_t nvjpeg2kStreamGetResolutionComponentDim(
        nvjpeg2kStream_t stream_handle,
        uint32_t component_id,
        uint32_t tile_id,
        uint32_t res_level,
        uint32_t* res_width,
        uint32_t* res_height
    )

    cdef nvjpeg2kStatus_t nvjpeg2kStreamGetColorSpace(
        nvjpeg2kStream_t stream_handle,
        nvjpeg2kColorSpace_t* color_space
    )

    cdef nvjpeg2kStatus_t nvjpeg2kDecodeParamsCreate(
        nvjpeg2kDecodeParams_t *decode_params
    )

    cdef nvjpeg2kStatus_t nvjpeg2kDecodeParamsDestroy(nvjpeg2kDecodeParams_t decode_params)


    cdef nvjpeg2kStatus_t nvjpeg2kDecodeParamsSetDecodeArea(
        nvjpeg2kDecodeParams_t decode_params,
        uint32_t start_x,
        uint32_t end_x,
        uint32_t start_y,
        uint32_t end_y
    )

    cdef nvjpeg2kStatus_t nvjpeg2kDecodeParamsSetRGBOutput(
        nvjpeg2kDecodeParams_t decode_params,
        int32_t force_rgb
    )

    cdef nvjpeg2kStatus_t nvjpeg2kDecode(
        nvjpeg2kHandle_t handle,
        nvjpeg2kDecodeState_t decode_state,
        nvjpeg2kStream_t jpeg2k_stream,
        nvjpeg2kImage_t* decode_output,
        cudaStream_t stream
    )

    cdef nvjpeg2kStatus_t nvjpeg2kDecodeImage(
        nvjpeg2kHandle_t handle,
        nvjpeg2kDecodeState_t decode_state,
        nvjpeg2kStream_t jpeg2k_stream,
        nvjpeg2kDecodeParams_t decode_params,
        nvjpeg2kImage_t* decode_output,
        cudaStream_t stream
    ) nogil

    cdef nvjpeg2kStatus_t nvjpeg2kDecodeTile(
        nvjpeg2kHandle_t handle,
        nvjpeg2kDecodeState_t decode_state,
        nvjpeg2kStream_t jpeg2k_stream,
        nvjpeg2kDecodeParams_t decode_params,
        uint32_t tile_id,
        uint32_t num_res_levels,
        nvjpeg2kImage_t* decode_output,
        cudaStream_t stream
    )


# Encoder APIs
# nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncoderCreateSimple(nvjpeg2kEncoder_t *enc_handle);

# nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncoderDestroy(nvjpeg2kEncoder_t enc_handle);

# nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeStateCreate(
#     nvjpeg2kEncoder_t enc_handle,
#     nvjpeg2kEncodeState_t *encode_state
# );

# nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeStateDestroy(nvjpeg2kEncodeState_t encode_state);

# nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeParamsCreate(nvjpeg2kEncodeParams_t *encode_params);

# nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeParamsDestroy(nvjpeg2kEncodeParams_t encode_params);

# nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeParamsSetEncodeConfig(
#     nvjpeg2kEncodeParams_t encode_params,
#     nvjpeg2kEncodeConfig_t* encoder_config
# );

# nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeParamsSetQuality(
#     nvjpeg2kEncodeParams_t encode_params,
#     double target_psnr
# );

# nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncode(
#     nvjpeg2kEncoder_t enc_handle,
#     nvjpeg2kEncodeState_t encode_state,
#     const nvjpeg2kEncodeParams_t encode_params,
#     const nvjpeg2kImage_t *input_image,
#     cudaStream_t stream
# );


# nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeRetrieveBitstream(
#     nvjpeg2kEncoder_t enc_handle,
#     nvjpeg2kEncodeState_t encode_state,
#     unsigned char *compressed_data,
#     size_t *length,
#     cudaStream_t stream
# );
