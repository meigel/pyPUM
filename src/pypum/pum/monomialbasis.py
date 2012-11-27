import itertools as iter
import numpy as np

from pypum.pum.monomialbasis_cy import eval_monomial, eval_monomial_dx
from pypum.utils.math import prod

class MonomialBasis(object):
	"""cython optimised monomial basis."""
	def __init__(self, degree, dim):
		self._degree = degree
		self._dim = dim
		self._idx = [idx for idx in iter.product(range(self._degree + 1), repeat=self._dim)]

	@property
	def dim(self):
		return len(self._idx)

	@property
	def geomdim(self):
		return self._dim

	@property
	def idx(self):
		return self._idx

	def __len__(self):
		return self.dim

	def __call__(self, x, bid, gradient, y=None, ty=None):
		assert bid >= 0 and bid < self.dim
		N = x.shape[0]
		returny = False
		if y is None:
			returny = True
			if gradient:
				y = np.zeros_like(x)
			else:
				y = np.zeros((N,))
		if ty is None:
				ty = np.zeros((N,))
		# flatten variables
		fx = x.view()
		fx.shape = prod(fx.shape)
		fy = y.view()
		fy.shape = prod(fy.shape)
		# call optimised evaluation
		if gradient:
			eval_monomial_dx(fx, np.array(self._idx[bid]), fy, ty)
#			y.shape = (N, self.geomdim) 
		else:
			eval_monomial(fx, np.array(self._idx[bid]), fy, ty)
		if returny:
			return y
