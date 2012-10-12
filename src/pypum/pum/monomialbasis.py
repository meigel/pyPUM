import itertools as iter
import numpy as np

from pypum.pum.monomialbasis_cy import eval_monomial, eval_monomial_dx


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
		if y is None:
			if gradient:
				y = np.zeros_like(x)
			else:
				y = np.zeros((N,))
		if ty is None:
				ty = np.zeros((N,))
		# call optimised evaluation
		if gradient:
			y = eval_monomial_dx(x.flatten(), np.array(self._idx[bid]), y.flatten(), ty.flatten())
			y.shape = (N, self.geomdim) 
		else:
			y = eval_monomial(x.flatten(), np.array(self._idx[bid]), y.flatten(), ty.flatten())
		return y
