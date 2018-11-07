from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='OptiMap',
      version='1.0',
      description='Bionano Tiff to Optical Map Signals',
      author='Mehmet Akdel',
      author_email='mehmet.akdel@wur.nl',
      url='https://gitlab.com/akdel/',
      packages=['OptiMap'],
      install_requires=["sqlalchemy", "numpy", "scipy", "intervaltree", "matplotlib", "cython"],
      # ext_modules=cythonize("cross_corr.pyx"),
      include_dirs=[numpy.get_include()])
