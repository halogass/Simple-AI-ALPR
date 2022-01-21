from setuptools import setup
from Cython.Build import cythonize

files = ["arsalpr.pyx", "serverAi.pyx"]

setup(
    ext_modules=cythonize(files),
    zip_safe=False,
)
