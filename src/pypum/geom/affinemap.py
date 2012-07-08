from __future__ import division
import numpy as np

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
#    @vectorize('x')
    def eval_map(box, x, scaling=1.0):
        if scaling != 1.0:
            box *= scaling
        y = [box.pos[d][0] for d in box.dim] + x * box.size
        return y

    @staticmethod
#    @vectorize('y')
    def eval_inverse_map(box, y, scaling=1.0):
        if scaling != 1.0:
            box *= scaling
        x = (y - [box.pos[d][0] for d in range(box.dim)]) / box.size
        return x
