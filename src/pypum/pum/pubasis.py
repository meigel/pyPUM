from __future__ import division
import numpy as np

from pypum.pum.basis import BasisSet
from pypum.geom.affinemap import AffineMap


class PUBasis(BasisSet):
	"""PU basis as product of a basis on the reference cube [0,1]^d and a PU subject to some nTree."""
	
	def __init__(self, pu, basis, with_pu=True):
		super(PUBasis, self).__init__(basis=basis)
		self._pu = pu
		self._with_pu = with_pu

	def indices(self):
		return self._pu.indices()

	def __call__(self, x, id):
		vf = 0
		node = self._pu._tree[id]
		if node.bbox.is_inside(x, scaling=self._pu._scaling):
			tx = AffineMap.eval_inverse_map(node.bbox, x, scaling=self._pu._scaling)
			vf = [f(tx) for f in self[id]]
			if self._with_pu:
				vpu = self._pu(x, id)
				vf = [vpu * v for v in vf]
		return vf
	
	def dx(self, x, id):
		node = self._pu._tree[id]
		vfdx = np.zeros(node.dim)
		if node.bbox.is_inside(x, scaling=self._pu._scaling):
			tx = AffineMap.eval_inverse_map(node.bbox, x, scaling=self._pu._scaling)
			vfdx = [f.dx(x) for f in self[id]]
			vfdx *= 1 / (self._pu.scaling * self._pu._tree[id].size)			# scale basis gradients
			if self._with_pu:
				vf = [f(x) for f in self[id]]
				vpu = self._pu(x, id)
				vpudx = self._pu.dx(x, id)										# pu gradients are already scaled
				vfdx = [fdx * vpu + f * vpudx for f, fdx in zip(vf, vfdx)]
		return vfdx

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
