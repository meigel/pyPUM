from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "marchingcube",
    ext_modules = cythonize("*.pyx"),
)
