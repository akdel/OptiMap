from distutils.core import setup

setup(name='OptiMap',
      version='1.0',
      description='An optical map molecule alignment tool.',
      author='Mehmet Akdel',
      author_email='mehmet.akdel@wur.nl',
      url='https://gitlab.com/akdel/',
      scripts=["bin/OptiMap-deep",
               "bin/OptiMap-naive",
               "bin/OptiMap-sparse",
               "bin/OptiSpeed"],
      packages=['OptiMap', "OptiMap/OptiSpeed"],
      install_requires=["sqlalchemy", "numpy", "numba", "scipy", "intervaltree",
                        "matplotlib", "cython", "Simple-LSH", "fire", "ray"])
