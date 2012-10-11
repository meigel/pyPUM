# cython: cdivision=True

import cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def affine_map(np.ndarray[np.float64_t, ndim=2] box not None, np.ndarray[np.float64_t] x not None, np.ndarray[np.float64_t] y not None):
    """Map from [0,1]^d to box."""
#    cdef np.ndarray[double] y = np.zeros_like(x)
    cdef float w, p
    cdef Py_ssize_t d, i, dim, N
    dim = box.shape[0]
    N = x.shape[0]/dim
    for d in range(dim):
        w = box[d,1] - box[d,0]
        p = box[d,0]
        for i in range(N):
            y[i*dim+d] = p + x[i*dim+d] * w


@cython.boundscheck(False)
@cython.wraparound(False)
def affine_map_inverse(np.ndarray[np.float64_t, ndim=2] box not None, np.ndarray[np.float64_t] y not None, np.ndarray[np.float64_t] x not None):
    """Map from box to [0,1]^d."""
#    cdef np.ndarray[double] x = np.zeros_like(y)
    cdef float w, p
    cdef Py_ssize_t d, i, dim, N
    dim = box.shape[0]
    N = x.shape[0]/dim
    for d in range(dim):
        w = box[d,1] - box[d,0]
        p = box[d,0]
        for i in range(N):
            x[i*dim+d] = (y[i*dim+d] - p) / w
