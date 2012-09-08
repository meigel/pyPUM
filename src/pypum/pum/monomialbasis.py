import itertools as iter
import numpy as np

import pypum.pum.monomialbasis_cy

class MonomialBasis(object):
	"""cython optimised monomial basis."""
	def __init__(self, degree, dim):
		self._degree = degree
		self._dim = dim
		self._idx = [idx for idx in iter.product(range(self._degree + 1), repeat=self._dim)]

	@property
	def dim(self):
		return len(self._idx)

	def __len__(self):
		return self.dim

	def __call__(x, bid, gradient, y=None, ty=None):
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
			monomialbasis_cy.eval_monomial_dx(x, self._idx[bid], y, ty)
		else:
			monomialbasis_cy.eval_monomial(x, self._idx[bid], y, ty)
		if returny:
			return y
