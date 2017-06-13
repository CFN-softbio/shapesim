#!/usr/bin/env python
import numpy as np

#from distutils.core import setup
import setuptools
from Cython.Build import cythonize

setuptools.setup(name='shapesim',
    version='1.0',
    author='Julien Lhermitte',
    description="Meso cluster analysis",
    include_dirs=[np.get_include()],
    author_email='lhermitte@bnl.gov',
#   install_requires=['six', 'numpy'],  # essential deps only
    ext_modules = cythonize("shapesim/tools/rotate.pyx"),
    keywords='simulation SAXS xray',
    license='BSD',
)
