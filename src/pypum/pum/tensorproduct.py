import operator
import itertools as iter
import numpy as np

import logging
logger = logging.getLogger(__name__)


class TensorProduct(object):
    """Tensor product of arbitrary many functions."""
    
    def __init__(self, funcs):
#        assert all([f.dim == 1 for f in funcs])
        self._funcs = funcs
    
    @property
    def dim(self):
        return len(self._funcs)
    
    def __call__(self, x):
        def prod(lst):
            return reduce(operator.mul, lst)
        return prod([f(x[:, d]) for d, f in enumerate(self._funcs)])
    
    def dx(self, x):
        val = np.vstack([f.dx(x[:, d]) for d, f in enumerate(self._funcs)])
        return val.T
    
    @classmethod
    def create_basis(cls, funcs, dim):
        return [TensorProduct(f) for f in iter.product(funcs, repeat=dim)]
