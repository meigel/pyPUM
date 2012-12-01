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

	def __call__(self, _x, bid, gradient, y=None):
		assert bid >= 0 and bid < self.dim
		x = _x.view()
		N = x.shape[0]
		if len(x.shape) == 1:
			x = x[:, None]
		assert len(x.shape) == 2
		if y is None:
			if gradient:
				cy = np.zeros_like(x)
			else:
				cy = np.zeros((N,))
		else:
			cy = y.view()
			if gradient:
				if len(cy.shape) == 1:
					cy = cy[:, None]
		
		# call optimised evaluation
		if gradient:
			return eval_monomial_dx(x, np.array(self._idx[bid]), cy)
		else:
			return eval_monomial(x, np.array(self._idx[bid]), cy)
