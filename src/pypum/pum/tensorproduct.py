import operator
import itertools as iter
import numpy as np

from pypum.pum.function import Function

import logging
logger = logging.getLogger(__name__)


class TensorProduct(Function):
    """Tensor product of arbitrary many functions."""
    
    def __init__(self, funcs):
        assert all([f.dim == 1 for f in funcs])
        self._funcs = funcs
        super(TensorProduct, self).__init__(dim=self.dim, codim=1)
    
    @property
    def dim(self):
        return len(self._funcs)
    
    def _f(self, x):
        def prod(lst):
            return reduce(operator.mul, lst)
        return prod([f(cx) for f, cx in zip(self._funcs, x)])
    
    def _dx(self, x):
        return np.array([f.dx(cx) for f, cx in zip(self._funcs, x)])
    
    @classmethod
    def create_basis(cls, funcs, dim):
        return [TensorProduct(f) for f in iter.product(funcs, repeat=dim)]
