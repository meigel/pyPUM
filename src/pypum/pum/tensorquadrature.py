import scipy.special.orthogonal as sso
import itertools as iter
import operator as op

def GaussLegendre(degree):
    return sso.ps_roots(degree)

class TensorQuadrature(object):
    """Tensor quadrature from 1d quadrature rule."""
    
    def __init__(self, rule=GaussLegendre):
        """Initialise with a 1d quadrature rule returning x,w.
            Note that the reference interval is [-1,1]."""
        self._rule = rule
        self._cache = {}
    
    def __getitem__(self, degree):
        try:
            x, w = self._cache[degree]
        except Exception:
            x, w = self._rule(degree)
            self._cache[degree] = (x, w)
        return x, w

    def transformed(self, box, degree):
        """Return points and weights of tensor quadrature on arbitrary box."""
        x, w = self[degree]
        # transform points to box and adjust weights
        xw = [(box.pos[d][0] + (box.size[d] * x), box.size[d] * w) for d in range(box.dim)]
        nq = len(x)
        idx = [i for i in iter.product(range(nq), repeat=box.dim)]
        tx = [[xw[d][0][i[d]] for d in range(box.dim)] for i in idx]
        tw = [reduce(op.mul, [xw[d][1][i[d]] for d in range(box.dim)]) for i in idx]
        return tx, tw


#from pypum.utils.box import Box
##b = Box(((0, 1), (1 / 2, 1)))
#b = Box(((0, 1), (1, 1)))
#q = TensorQuadrature()
#q.transformed(b, 3)
