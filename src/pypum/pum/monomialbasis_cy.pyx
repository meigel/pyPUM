# cython: cdivision=True
from __future__ import division

import cython
import numpy as np
cimport numpy as np


cdef enum:
    MAXN = 50000

_ty = np.ndarray((MAXN))


@cython.boundscheck(False)
@cython.wraparound(False)
def eval_monomial(np.float64_t[:,:] x not None, int[:] idx, np.float64_t[:] y not None):
    cdef np.float64_t[:] ty = _ty
    cdef unsigned int d, j, N, geomdim, n
    geomdim = idx.shape[0]
    N = x.shape[0]
    y[:] = 1.0
    for d in range(geomdim):
        ty[:] = 1.0
        for _ in range(idx[d]):
            for n in range(N):
                ty[n] *= x[n,d]
        for n in range(N):
            y[n] = y[n]*ty[n]
    return np.asarray(y[:N])

@cython.boundscheck(False)
@cython.wraparound(False)
def eval_monomial_dx(np.float64_t[:,:] x not None, int[:] idx, np.float64_t[:,:] y not None):
    cdef np.float64_t[:] ty = _ty
    cdef unsigned int d, j, N, geomdim, n
    geomdim = len(idx)
    N = x.shape[0]
    for d in range(geomdim):
        if idx[d] == 0:
            ty[:] = 0.0
        else:
            ty[:] = idx[d]
            for _ in range(idx[d]-1):
                for n in range(N):
                    ty[n] *= x[n,d]
        for n in range(N):
            y[n,d] = ty[n]
    ry = np.asarray(y[:N,:geomdim])
##    ry.shape = ((N,geomdim))
    return ry

@cython.boundscheck(False)
@cython.wraparound(False)
def test_mv(np.float64_t[:] ty):
    N = ty.shape[0]
    cdef int i
    for i in range(N):
        ty[i] = i**2+1
    return ty[:<int>(N/2)]
