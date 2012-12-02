# distutils: language = c++
# distutils: sources = marchingcube.cpp

#
#   cython wrapper for the marchingcube API
#

import cython
cimport numpy as np

cdef extern from "marchingcube.hpp":
    ctypedef int (*levelsetfunc)(double *point, int dim, void *user_data)
    int call_levelset(levelsetfunc user_func, void *user_data)
    int get_cells(double *array, int *m, int *n)


@cython.boundscheck(False)
@cython.wraparound(False)
def test_callback(f):
    print "(cython) test_callback"
    return call_levelset(callback, <void*>f)
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int callback(double *point, int dim, void *f):
    print "(cython) callback..."
    p = [point[d] for d in range(dim)]
    return (<object>f)(p)

@cython.boundscheck(False)
@cython.wraparound(False)
def decompose(np.ndarray[double, ndim=1, mode="c"] data):
    print "(cython) decompose"
    cdef int m, n
    m, n = data.shape[0], 1
    r = get_cells(&data[0], &m, &n)
    return (r, m, n)
