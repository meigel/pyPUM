from __future__ import division
import numpy as np

from pypum.pum.basis import BasisSet


class PUBasis(BasisSet):
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
			tx = AffineMap.eval_inverse_map(node.bbox, x)
			vf = self.basis.eval(tx, id)
			if self._with_pu:
				vpu = self._pu.eval(x, id)
				vf = [vpu * v for v in vf]
		return vf
	
	def dx(self, x, id):
		node = self._pu._tree[id]
		vfdx = np.zeros(node.dim)
		if node.bbox.is_inside(x, scaling=self._pu._scaling):
			tx = AffineMap.eval_inverse_map(node.bbox, x)
			vfdx = self.basis.dx(x, id)
			vfdx *= 1 / self._pu._tree[id].size		# scale basis gradients
			if self._with_pu:
				vf = self.basis(x, id)
				vpu = self._pu(x, id)
				vpudx = self._pu.dx(x, id)			# pu gradients are already scaled
				vfdx = [fdx * pu + f * pudx for f, fdx, pu, pudx in zip(vf, vfdx, vpu, vpudx)]
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
