from __future__ import division
import numpy as np
from collections import Iterable

from pypum.utils.box import Box
from pypum.utils.decorators import vectorize 

import logging
logger = logging.getLogger(__name__)


class AffineMap(object):
    """Affine map from reference cube [0,1]^d to physical cube."""
    
    def __init__(self, box):
        self._box = box
        self._p = np.array([box.pos[d][0] for d in box.dim])
        self._A = np.diag([box.size[d] for d in box.dim])
        self._Ainv = np.invert(self._A)

    def __call__(self, x):
        return self.map(x)

#    @vectorize('x')
    def map(self, x):
        return self._A.dot(x) + self._p
    
#    @vectorize('y')
    def inverse_map(self, y):
        return self.Ainv.dot(y - self._p)

    @staticmethod
    def eval_map(box, x, scaling=1.0):
        if not isinstance(scaling, Iterable): 
            scaling = np.array([scaling] * box.dim)
        dx = box.size * scaling 
        y = x * dx + (box.center - box.size / 2)
        if len(y) == 1:
            return y[0]
        else:
            return y

    @staticmethod
    def eval_inverse_map(box, y, scaling=1.0):
        if not isinstance(scaling, Iterable): 
            scaling = np.array([scaling] * box.dim)
        assert all(scaling >= 1.0)
        dx = box.size * scaling
        x = (y - box.center + box.size / 2) / dx
        if len(x) == 1:
            return x[0]
        else:
            return x
