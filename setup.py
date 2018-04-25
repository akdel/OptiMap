from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='cross correlation',
    ext_modules=cythonize("cross_corr.pyx"),
    include_dirs=[numpy.get_include()]

)
