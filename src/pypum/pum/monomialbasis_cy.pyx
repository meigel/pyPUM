# cython: cdivision=True

import cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def eval_test(unsigned int geomdim, np.ndarray[np.float64_t, ndim=1] x not None, idx, np.ndarray[np.float64_t, ndim=1] y not None, np.ndarray[np.float64_t, ndim=1] ty):
    cdef unsigned int d, j, N
    N = x.shape[0]/geomdim
    for d in range(geomdim):
        print x[d::geomdim]


@cython.boundscheck(False)
@cython.wraparound(False)
def eval_monomial(unsigned int geomdim, np.ndarray[np.float64_t, ndim=1] x not None, idx, np.ndarray[np.float64_t, ndim=1] y not None, np.ndarray[np.float64_t, ndim=1] ty):
    cdef unsigned int d, j, N
    N = x.shape[0]/geomdim
    if not (y.shape[0] is N and y.shape[0] is 1):
        y = np.zeros((N,))
    if ty is None or ty.shape is not y.shape:
        ty = np.zeros_like(y)
    y[:] = 1.0
    for d in range(geomdim):
        ty[:] = 1.0
        for j in range(idx[d]):
            ty *= x[d::geomdim]
        y *= ty
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def eval_monomial_dx(unsigned int geomdim, np.ndarray[np.float64_t, ndim=1] x not None, idx, np.ndarray[np.float64_t, ndim=1] y not None, np.ndarray[np.float64_t, ndim=1] ty):
    cdef unsigned int d, j, N
    N = x.shape[0]/geomdim
    if not (y.shape[0] is N*geomdim and y.shape[0] is 1):
        y = np.zeros((N*geomdim,))
    if ty is None:
        ty = np.zeros((N,))
    y[:] = 1.0
    for d in range(geomdim):
        if idx[d] == 0:
            ty[:] = 0.0
        else:
            ty[:N] = idx[d]
            for j in range(idx[d]-1):
                ty[:N] = ty[:N]*x[d::geomdim]
            y[d::geomdim] = ty[:N]
    return y
