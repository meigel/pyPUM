from __future__ import division
import numpy as np

from pypum.geom.affinemap import AffineMap

# TODO: decorator vectorisation in PUBasisHelper 


class PUBasisHelper(object):
	"""Defines a specific patch basis of a PUBasis. Used in assembly."""
	
	def __init__(self, pubasis, id):
		self._pubasis = pubasis
		self._id = id

	@property
	def dim(self):
		return self._pubasis.dim(self._id)

	@property
	def basis(self):
		return self._pubasis.basis[id]

	@property
	def pu(self):
		return self._pubasis.pu[self._id]

#	@vectorize
	def __call__(self, x, baseid=None):
		if isinstance(x, (list, tuple)):
			return np.array([self._pubasis(cx, self._id, baseid) for cx in x])
		else:
#			return [1.0] * len(x)
			return self._pubasis(x, self._id, baseid)

#	@vectorize		
	def dx(self, x, baseid=None):
		if isinstance(x, (list, tuple)):
			return [self._pubasis.dx(cx, self._id, baseid) for cx in x]
		else:
#			return [np.ones(2)] * len(x)
			return self._pubasis.dx(x, self._id, baseid)


class PUBasis(object):
	"""PU basis as product of a basis on the reference cube [0,1]^d and a PU subject to some nTree."""
	
	def __init__(self, pu, basisset, with_pu=True):
		self._basisset = basisset
		self._pu = pu
		self._with_pu = with_pu

	def indices(self):
		return self._pu.indices()

	def __call__(self, x, id, baseid=None):
		# NOTE: pu functions are evaluated in physical coordinates, basis functions in reference coordinates
		vf = 0.0
		node = self._pu._tree[id]
		if node.bbox.is_inside(x, scaling=self._pu._scaling):
			if baseid is None:
				# evaluate all base functions at once
				tx = AffineMap.eval_inverse_map(node.bbox, x, scaling=self._pu._scaling)
				vf = [f(tx) for f in self._basisset[id]]
				if self._with_pu:
					vpu = self._pu(x, id)
					vf = [vpu * v for v in vf]
			else:
				# evaluate single basis
				tx = AffineMap.eval_inverse_map(node.bbox, x, scaling=self._pu._scaling)
				vf = self._basisset[id][baseid](tx)
				if self._with_pu:
					vpu = self._pu(x, id)
					vf = vpu * vf
		return vf
	
	def dx(self, x, id, baseid=None):
		# NOTE: pu functions are evaluated in physical coordinates, basis functions in reference coordinates
		node = self._pu._tree[id]
		vfdx = np.zeros(node.dim)
		if node.bbox.is_inside(x, scaling=self._pu._scaling):
			if baseid is None:	
				# evaluate all base functions at once
				tx = AffineMap.eval_inverse_map(node.bbox, x, scaling=self._pu._scaling)
				vfdx = [f.dx(tx) for f in self._basisset[id]]
				vfdx *= 1 / (self._pu.scaling * self._pu._tree[id].size)			# scale basis gradients
				if self._with_pu:
					vf = [f(tx) for f in self._basisset[id]]
					vpu = self._pu(x, id)
					vpudx = self._pu.dx(x, id)										# pu gradients are already scaled
					vfdx = [fdx * vpu + f * vpudx for f, fdx in zip(vf, vfdx)]
			else:
				# evaluate single basis
				vfdx = np.zeros(node.dim)
				tx = AffineMap.eval_inverse_map(node.bbox, x, scaling=self._pu._scaling)
				vfdx = self._basisset[id][baseid](tx)
				vfdx *= 1 / (self._pu.scaling * self._pu._tree[id].size)			# scale basis gradients
				if self._with_pu:
					vf = self._basisset[id][baseid](tx)
					vpu = self._pu(x, id)
					vpudx = self._pu.dx(x, id)										# pu gradients are already scaled
					vfdx = vfdx * vpu + vf * vpudx
		return vfdx
	
	def __getitem__(self, id):
		return PUBasisHelper(self, id)
	
	def dim(self, id):
		return self._basisset.dim(id)

	@property
	def basis(self):
		return self._basisset
	
	@property
	def pu(self):
		return self._pu

	@property
	def with_pu(self):
		return self._with_pu
	
	@with_pu.setter
	def with_pu(self, val):
		self._with_pu = val
