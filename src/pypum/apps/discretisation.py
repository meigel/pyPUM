from __future__ import division
import numpy as np

from pypum.pum.function import Function
from pypum.utils.math import inner


class default_NeumannBC(Function):
    def __init__(self, dim=None, codim=None):
        super(default_NeumannBC, self).__init__(dim, codim)
    def _f(self, x):
        return np.ones(len(x))


class ReactionDiffusion(object):
    """Discretisation of the second order elliptic reaction-diffusion problem
        -D \Laplace u + ru = f
    """
    def __init__(self, D, r, f=lambda x: 1.0, isNeumannBC=lambda bndbox: True, g=default_NeumannBC()):
        self._D = D
        self._r = r
        self._f = f
        self._isNeumannBC = isNeumannBC
        self._g = g

    def lhs(self, A, idx1, idx2, basis1, basis2, quad, intbox, boundary):
        # NOTE/TODO: the quadrature degree should depend on the weight function, the basis degree, coefficients and the equation 
        D = self._D
        r = self._r
        tx, w = quad.transformed(intbox, basis1.dim)
        tx = [np.array(cx) for cx in tx]    # convert listst to arrays
        for bid1, j in enumerate(range(idx1, idx1 + basis1.dim)):    
            for bid2, k in enumerate(range(idx2, idx2 + basis2.dim)):
                # operator matrix with diffusion and reaction
                # debug---
#                print "AAA-0", tx
#                print "AAA-1", basis1.dx(tx, bid1), type(basis1.dx(tx, bid1))
#                print "AAA-2", basis1(tx, bid1), type(basis1(tx, bid1))
#                print "AAA-3", inner(basis1.dx(tx, bid1), basis2.dx(tx, bid2))
#                v1 = inner(basis1.dx(tx, bid1), basis2.dx(tx, bid2))
#                print "1---", v1
#                v1 = D * inner(basis1.dx(tx, bid1), basis2.dx(tx, bid2))
#                print "2---", v1
#                print basis1(tx, bid1)
#                print basis2(tx, bid2)
#                v2 = basis1(tx, bid1) * basis2(tx, bid2)
#                print "3---", v1
#                v2 = r * basis1(tx, bid1) * basis2(tx, bid2)
#                print "4---"
#                print D * inner(basis1.dx(tx, bid1), basis2.dx(tx, bid2)) + r * basis1(tx, bid1) * basis2(tx, bid2)
                # ---debug
                val = D * inner(basis1.dx(tx, bid1), basis2.dx(tx, bid2)) + r * basis1(tx, bid1) * basis2(tx, bid2)
                print "LHS (", j, k, "):", A[j, k], "+=", sum(w * val)
                A[j, k] += sum(w * val)
                if idx1 != idx2:    # symmetric operator
                    A[k, j] += sum(w * val)
    
    def rhs(self, b, idx2, basis2, quad, intbox, boundary):
        f = self._f
        g = self._g
        tx, w = quad.transformed(intbox, basis2.dim)
        tx = [np.array(cx) for cx in tx]    # convert listst to arrays
        for bid2, k in enumerate(range(idx2, idx2 + basis2.dim)):
            # source term
            val = f(tx) * basis2(tx, bid2)
            print "LHS (", k, "):", b[k], "+=", sum(w * val)
            b[k] += sum(w * val) 
            # evaluate boundary integrals
            if boundary:
#                print "BOUNDARY quad"
                for bndbox, normal in boundary:
#                    print normal, bndbox
                    if self._isNeumannBC(bndbox):
                        txb, wb = quad.transformed(bndbox, basis2.dim)
                        txb = [np.array(cx) for cx in txb]    # convert listst to arrays
                        valb = inner(g(txb), [normal] * len(txb)) * basis2(txb, bid2)
                        print "LHS-BC (", k, "):", b[k], "+=", sum(wb * valb)
                        b[k] += sum(wb * valb)
