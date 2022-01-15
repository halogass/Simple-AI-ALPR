from setuptools import setup
from Cython.Build import cythonize

setup(
    name='darknet',
    ext_modules=cythonize("darknet.pyx"),
    zip_safe=False,
)
