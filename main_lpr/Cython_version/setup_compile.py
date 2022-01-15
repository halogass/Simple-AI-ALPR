from setuptools import setup
from Cython.Build import cythonize

files = ["arsai.pyx", "arsalpr.pyx"]

setup(
    ext_modules=cythonize(files),
    zip_safe=False,
)
