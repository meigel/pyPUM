import itertools as iter
import numpy as np

from pypum.pum.monomialbasis_cy import eval_monomial, eval_monomial_dx, eval_test


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

	def __len__(self):
		return self.dim

	def __call__(self, x, bid, gradient, y=None, ty=None):
		assert bid >= 0 and bid < self.dim
		returny = False
		if y is None:
			returny = True
			if gradient:
				y = np.zeros_like(x)
			else:
				y = np.zeros_like(x[:, 0])
		if ty is None:
			if gradient:
				ty = np.zeros_like(x)
			else:
				ty = np.zeros_like(x[:, 0])
		# call optimised evaluation
		if gradient:
			y = eval_monomial_dx(self.geomdim, x.flatten(), np.array(self._idx[bid]), y.flatten(), ty.flatten())
		else:
			y = eval_monomial(self.geomdim, x.flatten(), np.array(self._idx[bid]), y, ty)
		if returny:
			return y
