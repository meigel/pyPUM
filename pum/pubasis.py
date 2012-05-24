from pypum.pum.basis import BasisSet

import numpy as np


class PUBasis(object):
	def __init__(self, pu, basis, with_pu=True):
		super(PUBasis, self).__init__(basis=basis)
		self._pu = pu
		self._with_pu = with_pu

	def indices(self):
		return self._pu.indices()

	def __call__(self, x, id):
		vf = self.basis.eval(x, id)
		if self._with_pu:
			vpu = self._pu.eval(x, id)
			vf = [vpu * v for v in vf]
		return vf
	
	def dx(self, x, id):
		pass

	def dim(self, id):
		return self._basis.dim(id)

	@property
	def basis(self):
		return self._basis
	
	@property
	def pu(self):
		return self._pu

	@property
	def with_pu(self):
		return self._with_pu
	
	@with_pu.setter
	def with_pu(self, val):
		self._with_pu = val
