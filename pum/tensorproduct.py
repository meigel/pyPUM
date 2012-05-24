from pypum.pum.function import Function

import numpy as np
import logging
logger = logging.getLogger(__name__)


class TensorProduct(Function):
    def __init__(self, funcs):
        assert all([f.dim == 1 for f in funcs])
        self._funcs = funcs
        super(TensorProduct, self).__init__(dim=self.dim, codim=1)
    
    @property
    def dim(self):
        return len(self._funcs)
    
    def __call__(self, *x):
        import operator
        def prod(lst):
            return reduce(operator.mul, lst)
        return prod([f(cx) for f, cx in zip(self._funcs, *x)])
    
    def dx(self, *x):
        return np.diag([f.dx(cx) for f, cx in zip(self._funcs, *x)])
    
    @classmethod
    def create_basis(cls, funcs, dim):
        return [TensorProduct(f) for f in iter.product(funcs, dim)]
