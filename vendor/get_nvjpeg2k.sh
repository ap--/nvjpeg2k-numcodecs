#!/bin/sh

NVJPEG2K_ARCHIVE=libnvjpeg_2k-linux-x86_64-0.6.0.28-archive.tar.xz

curl -o ${NVJPEG2K_ARCHIVE} https://developer.download.nvidia.com/compute/libnvjpeg-2k/redist/libnvjpeg_2k/linux-x86_64/${NVJPEG2K_ARCHIVE}

mkdir nvjpeg2k
tar xvf ${NVJPEG2K_ARCHIVE} -C nvjpeg2k --strip-components 1

