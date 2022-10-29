from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

setup(
    name="nvjpeg2k_numcodecs",
    ext_modules=cythonize(
        Extension(
            "nvjpeg2k_numcodecs._nvjpeg2k",
            sources=["nvjpeg2k_numcodecs/*.pyx"],
            libraries=['cudart'],
            extra_objects=[
                "./vendor/nvjpeg2k/lib/libnvjpeg2k_static.a",
            ],
            include_dirs=[
                "./vendor/nvjpeg2k/include",
            ],
        ),
        compiler_directives={
            "language_level": 3,
        },
    ),
)
