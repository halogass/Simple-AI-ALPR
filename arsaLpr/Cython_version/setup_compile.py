from setuptools import setup
from Cython.Build import cythonize

files = ["arsalpr_cuda.pyx", "server.pyx"]

setup(
    ext_modules=cythonize("server.pyx"),
    zip_safe=False,
)
