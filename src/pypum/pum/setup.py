from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup (
	name='PUM module',
	cmdclass={'build_ext': build_ext},
	ext_modules=[
		Extension("monomialbasis_cy",
		["monomialbasis_cy.pyx"],
		include_dirs=[np.get_include()]),
                Extension("pu_cy",
                ["pu_cy.pyx"],
                include_dirs=[np.get_include()]),
                Extension("pufunction_cy",
                ["pufunction_cy.pyx"],
                include_dirs=[np.get_include()]),
                Extension("affinemap_cy",
                ["affinemap_cy.pyx"],
                include_dirs=[np.get_include()]),
])
