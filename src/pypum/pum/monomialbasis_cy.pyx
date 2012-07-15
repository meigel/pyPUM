# cython: cdivision=True

import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def eval_monomial(np.ndarray[np.float64_t] x not None, idx, gradient, np.ndarray[np.float64_t, ndim=1] y not None, np.ndarray[np.float64_t, ndim=1] ty):
    ty = np.zeros_like(x[:,0])
    cdef unsigned int d, j
    y[:] = 1.0
    for d in range(x.shape[1]):
        ty[:] = 1.0
        for j in range(idx[d]):
            ty *= x[:,d]
        y *= ty


@cython.boundscheck(False)
@cython.wraparound(False)
def eval_monomial_dx(np.ndarray[np.float64_t] x not None, idx, gradient, np.ndarray[np.float64_t] y not None, np.ndarray[np.float64_t] ty):
    if ty is None:
        ty = np.zeros_like(x)
    cdef unsigned int d, j
    y[:] = 1.0
    for d in range(x.shape[1]):
        if idx[d] == 0:
            ty[:] = 0.0
        else:
            ty[:] = idx[d]
            for j in range(idx[d]-1):
                ty *= x[:,d]
            y[:,d] = ty
