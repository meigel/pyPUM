from __future__ import division
import numpy as np

from pypum.pum.function import Function
from pypum.geom.affinemap import AffineMap
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

    def lhs(self, A, idx1, idx2, nid, bbox, pu, basis1, basis2, quad, intbox, boundary):
        # NOTE/TODO: the quadrature degree should depend on the weight function, the basis degree, coefficients and the equation
        d = bbox.dim 
        dim1 = len(basis1)
        dim2 = len(basis2)
        D = self._D
        r = self._r
        # get quadrature points for pu
        tx, w = quad.transformed(intbox, dim1)
        tx = np.array([np.array(cx) for cx in tx])                    # convert lists to arrays
        # get patch reference points for basis
        px = AffineMap.eval_inverse_map(bbox, tx, scaling=pu.scaling)
        N = len(tx)
        # evaluate pu
        puf = pu(tx, id=nid)
        pufd = np.repeat(puf, d, axis=0)
        pufd.shape = (N, d)
        Dpuf = pu.dx(tx, id=nid)
        
        for bid1, j in enumerate(range(idx1, idx1 + dim1)):
            # evaluate basis1
            b1 = basis1[bid1](px)
            b1d = np.repeat(b1, d, axis=0)
            b1d.shape = (N, d)
            Db1 = basis1[bid1].dx(px)
            
            for bid2, k in enumerate(range(idx2, idx2 + dim2)):
                # evaluate basis2
                b2 = basis2[bid2](px)
                b2d = np.repeat(b2, d, axis=0)
                b2d.shape = (N, d)
                Db2 = basis2[bid2].dx(px)
                
                # debug---
#                print puf
#                print Dpuf
#                print b1
#                print b2
#                print Db1
#                print Db2
                # ---debug
                
                # prepare discretisation parts
                pub1 = puf * b1
                pub2 = puf * b2
                Dpub1 = Dpuf * b1d + pufd * Db1 
                Dpub2 = Dpuf * b2d + pufd * Db2
                                       
                # operator matrix with diffusion and reaction
                val = D * inner(Dpub1, Dpub2) + r * pub1 * pub2
#                print "LHS (", j, k, "):", A[j, k], "+=", sum(w * val)
                
                # add to matrix
                A[j, k] += sum(w * val)
                if idx1 != idx2:            # symmetric operator
                    A[k, j] += sum(w * val)

    def rhs(self, b, idx2, nid, bbox, pu, basis2, quad, intbox, boundary):
        dim2 = len(basis2)
        f = self._f
        g = self._g
        # get quadrature points for pu
        tx, w = quad.transformed(intbox, dim2)
        tx = np.array([np.array(cx) for cx in tx])                    # convert lists to arrays
        # get patch reference points for basis
        px = AffineMap.eval_inverse_map(bbox, tx, scaling=pu.scaling)
        # evaluate pu
        puf = pu(tx, id=nid)
        
        for bid2, k in enumerate(range(idx2, idx2 + dim2)):
            # evaluate basis2
            b2 = basis2[bid2](px)
            # compute source rhs
            val = f(tx) * puf * b2
            # add to rhs vector
#            print "LHS (", k, "):", b[k], "+=", sum(w * val)
            b[k] += sum(w * val)
             
            # evaluate boundary integrals
            if boundary:
#                print "BOUNDARY quad"
                for bndbox, normal in boundary:
#                    print normal, bndbox
                    if self._isNeumannBC(bndbox):
                        # get surface quadrature points for pu
                        txb, wb = quad.transformed(bndbox, dim2)
                        txb = np.array([np.array(cx) for cx in txb])          # convert lists to arrays
                        # get patch reference points for basis
                        pxb = AffineMap.eval_inverse_map(bbox, tx, scaling=pu.scaling)
                        
                        # add to rhs vector
                        valb = inner(g(txb), [normal] * len(txb)) * puf * b2
#                        print "LHS-BC (", k, "):", b[k], "+=", sum(wb * valb)
                        b[k] += sum(wb * valb)
