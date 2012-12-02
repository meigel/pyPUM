from __future__ import division

import numpy as np
import numpy.matlib as ml

from pypum.pum.affinemap_cy import affine_map_inverse
from pypum.pum.function import Function
import pypum.utils.math as m


class default_NeumannBC(Function):
    def __init__(self, dim=None, codim=None):
        super(default_NeumannBC, self).__init__(dim, codim)
    def _f(self, x):
        return np.ones(len(x))


class ReactionDiffusion(object):
    """Discretisation of the second order elliptic reaction-diffusion problem
        -D \Laplace u + ru = f
    """
    def __init__(self, basedegree, D, r, f=lambda x: 1.0, isNeumannBC=lambda bndbox: True, g=default_NeumannBC()):
        self._D = D
        self._r = r
        self._f = f
        self._isNeumannBC = isNeumannBC
        self._g = g
        self._quaddegree = max((basedegree - 1) ** 2 + 1, 2)
        self._init_helpers()

    def _init_helpers(self, hs=50):
        # evaluation helper variables
        self._hs = hs
        self._b1 = np.empty((hs, 1))
        self._Db1 = np.empty((hs * 3, 1))
        self._b2 = np.empty((hs, 1))
        self._Db2 = np.empty((hs * 3, 1))

    def lhs(self, A, idx1, idx2, pu, basis1, basis2, quad, intbox, boundary):
        import time
        geomdim = intbox.dim
        T = [time.time()]  # === 1 ===
        # NOTE/TODO: the quadrature degree should depend on the weight function, the basis degree, coefficients and the equation
        D = self._D
        r = self._r
        # get quadrature points for pu
        tx, w = quad.transformed(intbox, self._quaddegree)
        tx = np.array([np.array(cx) for cx in tx])              # convert lists to arrays
        N = len(tx)
#        print "QUADDEGREE", self._quaddegree, N            
        # get patch reference points for basis
        txv = tx.view()
        txv.shape = m.prod(tx.shape)
        px = np.zeros_like(tx)
        pxv = px.view()
        pxv.shape = txv.shape
        affine_map_inverse(geomdim, pu._bbox[0:geomdim], txv, pxv)
        T.append(time.time()) # === 2 ===
        
        if self._hs < tx.shape[0]:
            self._init_helpers(tx.shape[0])
        # reuse result vectors
        _b1 = self._b1.view()
        _Db1 = self._Db1.view()
        _b2 = self._b2.view()
        _Db2 = self._Db2.view()
        
        # evaluate pu
        puf = pu(tx, gradient=False)
        pufd = ml.repmat(puf, geomdim, 1)
        pufd = pufd.T
        T.append(time.time()) # === 3 ===
        Dpuf = pu(tx, gradient=True)
        T.append(time.time()) # === 4 ===
        dT = [T[i + 1] - T[i] for i in range(len(T) - 1)]
        print "TIMINGS A: ", dT, "with", N, "quadrature points for dim", geomdim
        T = [time.time()]
        
        for bid1, j in enumerate(range(idx1, idx1 + basis1.dim)):
            # evaluate basis1
            T = [time.time()] # === 1 ===
            b1 = basis1(px, bid1, gradient=False, y=_b1[:, 0])
            b1d = ml.repmat(b1, geomdim, 1)
            b1d = b1d.T
            Db1 = basis1(px, bid1, gradient=True, y=_Db1)
            T.append(time.time())
#            dT = [T[i + 1] - T[i] for i in range(len(T) - 1)]
#            print "TIMINGS B: ", dT
            T = [time.time()]
            
            for bid2, k in enumerate(range(idx2, idx2 + basis2.dim)):
                # evaluate basis2
                T = [time.time()] # === 1 ===
                b2 = basis2(px, bid2, gradient=False, y=_b2[:, 0])
                b2d = ml.repmat(b2, geomdim, 1)
                b2d = b2d.T
                Db2 = basis2(px, bid2, gradient=True, y=_Db2)
                T.append(time.time()) # === 2 ===
                
#                # debug---
                print "px", px.shape, px
                print "puf", puf.shape, puf
                print "Dpuf", Dpuf.shape, Dpuf
                print "b1", b1.shape, b1
                print "b2", b2.shape, b2
                print "Db1", Db1.shape, Db1
                print "Db2", Db2.shape, Db2
#                # ---debug
                
                # prepare discretisation parts
                T.append(time.time()) # === 3 ===
                pub1 = puf * b1
                pub2 = puf * b2
                T.append(time.time()) # === 4 ===
                Dpub1 = m.mult(Dpuf, b1d) + m.mult(pufd, Db1) 
                Dpub2 = m.mult(Dpuf, b2d) + m.mult(pufd, Db2)
                T.append(time.time()) # === 5 ===
                                       
                # operator matrix with diffusion and reaction
                val = D * m.inner(Dpub1, Dpub2) + r * m.mult(pub1, pub2)
                T.append(time.time()) # === 6 ===
#                print "LHS (", j, k, "):", A[j, k], "+=", sum(w * val)
                
                # add to matrix
                print "\\\\\\\\\\\\\\"
                print pub1.shape
                print m.mult(puf, b1)
                print Dpub1.shape
                print m.mult(Dpuf, b1d)
                print "//////////////"
                print "ooooooooooooooooooooooooooooooooooo"
                print len(w)
                print val.shape
                print "1", w
                print "2", val
                print "3", m.mult(w, val)
                print "4", sum(m.mult(w, val))
                print "ooooooooooooooooooooooooooooooooooo"
                raise Exception()

                A[j, k] += sum(m.mult(w, val))
                if idx1 != idx2:            # symmetric operator
                    A[k, j] += sum(m.mult(w, val))
                T.append(time.time()) # === 7 ===
#                dT = [T[i + 1] - T[i] for i in range(len(T) - 1)]
#                print "TIMINGS C: ", dT

    def rhs(self, b, idx2, pu, basis2, quad, intbox, boundary):
        import time
        geomdim = intbox.dim
        f = self._f
        g = self._g
        # get quadrature points for pu
        tx, w = quad.transformed(intbox, self._quaddegree)
        tx = np.array([np.array(cx) for cx in tx])                    # convert lists to arrays
        # get patch reference points for basis
        txv = tx.view()
        txv.shape = m.prod(tx.shape)
        px = np.zeros_like(tx)
        pxv = px.view()
        pxv.shape = txv.shape
        affine_map_inverse(geomdim, pu._bbox[0:geomdim], txv, pxv)
        
        # reuse result vectors
#        _b1 = self._b1 
#        _Db1 = self._Db1
        _b2 = self._b2
        _Db2 = self._Db2
        
        # evaluate pu
        puf = pu(tx, gradient=False)
        
        for bid2, k in enumerate(range(idx2, idx2 + basis2.dim)):
            # evaluate basis2
            b2 = basis2(px, bid2, gradient=False, y=_b2[:, 0])
            # compute source rhs
            val = m.mult(f(tx), puf, b2)
            # add to rhs vector
#            print "LHS (", k, "):", b[k], "+=", sum(w * val)
            b[k] += sum(m.mult(w, val))
             
            # evaluate boundary integrals
            if boundary:
#                print "BOUNDARY quad"
                for bndbox, normal in boundary:
#                    print normal, bndbox
                    if self._isNeumannBC(bndbox):
                        # get surface quadrature points for pu
                        txb, wb = quad.transformed(bndbox, self._quaddegree)
                        txb = np.array([np.array(cx) for cx in txb])          # convert lists to arrays
                        # get patch reference points for basis
                        txbv = txb.view()
                        txbv.shape = m.prod(txb.shape)
                        pxb = np.zeros_like(txb)
                        pxbv = pxb.view()
                        pxbv.shape = txbv.shape
                        affine_map_inverse(geomdim, pu._bbox[0:geomdim], txbv, pxbv)
                        
                        # evaluate boundary terms
                        pufb = pu(txb, gradient=False, Noffset=tx.shape[0])
                        b2b = basis2(pxb, bid2, gradient=False)
                        gb = g(txb)
                        
                        # add to rhs vector
                        valb = m.inner(gb, [normal] * len(txb)) * m.mult(pufb, b2b)
#                        print "LHS-BC (", k, "):", b[k], "+=", sum(wb * valb)
                        b[k] += sum(m.mult(wb, valb))
