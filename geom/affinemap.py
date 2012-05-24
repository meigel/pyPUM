from pypum.utils.box import Box
import numpy as np

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

    def map(self, x):
        return self._A.dot(x) + self._p
    
    def inverse_map(self, y):
        return self.Ainv.dot(y - self._p)

    @staticmethod
    def eval_map(box, x):
        y = [box.pos[d][0] for d in box.dim] + x * box.size
        return y

    @staticmethod
    def eval_inverse_map(box, y):
        x = (y - [box.pos[d][0] for d in range(box.dim)]) / box.size
        return x
