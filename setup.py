from distutils.core import setup

setup(name='OptiMap',
      version='1.0',
      description='Bionano Tiff to Optical Map Signals',
      author='Mehmet Akdel',
      author_email='mehmet.akdel@wur.nl',
      url='https://gitlab.com/akdel/',
      packages=['OptiMap'],
      install_requires=["sqlalchemy", "numpy", "numba", "scipy", "intervaltree", "matplotlib", "cython"],)
