import itertools as iter
import numpy as np

import pypum.pum_cy

weighttypes = {'bspline1':1, 'bspline2':2, 'bspline3':3}

class PU2(object):
    """cython optimised partition of unity."""
    def __init__(self, type='bspline3'):
        self._type = weighttypes[type]

    def __call__(x, bbox, gradient, y=None, ty=None):
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
            pum_cy.eval_pu_dx(x, bbox, y, ty, type=self._type)
        else:
            pum_cy.eval_pu(x, bbox, y, ty, type=self._type)
        if returny:
            return y
