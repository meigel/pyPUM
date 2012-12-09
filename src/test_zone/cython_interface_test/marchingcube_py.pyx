# distutils: language = c++
# distutils: sources = marchingcube.cpp

#
#   cython wrapper for the marchingcube API
#

import cython
cimport numpy as np

cdef extern from "marchingcube.hpp":
    int get_cells(int dim, int *vertex_vals, int order, double *cell_data, int *m, double *facet_data, int *n)


@cython.boundscheck(False)
@cython.wraparound(False)
def decompose(int dim, np.ndarray[int, ndim=1, mode="c"] vertex_vals, int order, np.ndarray[double, ndim=1, mode="c"] cell_data, np.ndarray[double, ndim=1, mode="c"] facet_data):
    print "(cython) decompose"
    cdef int m, n
    m, n = cell_data.shape[0], facet_data.shape[0]
    if get_cells(dim, &vertex_vals[0], order, &cell_data[0], &m, &facet_data[0], &n) != 0:
        raise Exception("cut-cell interface call was not successful")
    return m, n
