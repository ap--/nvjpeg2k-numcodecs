from Cython.Build import cythonize
from setuptools import setup

setup(
    name="nvjpeg2k_numcodecs",
    ext_modules=cythonize(
        "nvjpeg2k_numcodecs/*.pyx",
        include_path=["vendor/libnvjpeg_2k/include"],
        compiler_directives={
            "language_level": 3,
        },
    ),
)
