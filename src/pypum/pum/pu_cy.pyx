# cython: cdivision=True
from __future__ import division

import cython
import numpy as np
cimport numpy as np

import logging
logger = logging.getLogger(__name__)

weighttypes = {'bspline1':0, 'bspline2':1, 'bspline3':2}

class PU(object):
    """cython optimised partition of unity."""
    def __init__(self, tree, weighttype='bspline3', scaling=1.25):
        self._tree = tree
        self._weighttype = weighttypes[weighttype]
        self._scaling = scaling
        # allocate bbox temporary memory
        self._bbox = np.ndarray((100*3, 2))
        self._Nbbox = None

    @property
    def scaling(self):
        return self._scaling
    
    @property
    def tree(self):
        return self._tree
    
    @property
    def indices(self):
        return self._tree.leafs()
    
    def get_node(self, id):
        return self._tree[id]
    
    def get_bbox(self, id):
        return self._tree[id].bbox * self._pu.scaling
    
    def get_neighbours(self, id):
        neighbours = self._tree.find_neighbours(id, scaling=self._scaling)
        return neighbours

    def get_active_neighbours(self, id, x):
        neighbours = [nid for nid in self._tree.find_neighbours(id) if self._tree[nid].bbox.is_inside(x, scaling=self._scaling)]
        return neighbours

    def prepare_neighbours(self, id):
        '''Find and prepare neighbours of patch for pu evaluation.'''
        D = self._tree.bbox.dim
        # get and convert neighbours of patch 
        neighbours = [id] + self.get_neighbours(id)
        self._Nbbox = len(neighbours)
        for d in range(D):
            for i, pid in enumerate(neighbours):
                self._bbox[i*D+d, :] = self._tree[pid].bbox._pos[d]

    def __call__(self, x, gradient, y=None):
        '''Evaluate pu or gradient of pu. Note that prepare_neighbours has to be called first.'''
        dim = x.shape[1]
        N = x.shape[0]
        returny = False
        if y is None:
            returny = True
            if gradient:
                y = np.zeros_like(x)
            else:
                y = np.zeros_like(x[:, 0])
        # call optimised evaluation
        if gradient:
            eval_pu_dx(dim, x.T.flatten(), self._Nbbox, self._bbox, y.T.flatten(), self._weighttype)
        else:
            eval_pu(dim, x.T.flatten(), self._Nbbox, self._bbox, y.T.flatten(), self._weighttype)
        if returny:
            return y


# =============================================================
# ================== cython optimised code ====================
# =============================================================

# define look-up for weight functions
ctypedef double (*wfT)(double x)
cdef wfT[3] wf
cdef wfT[3] Dwf
wf[:3] = [bspline1, bspline2, bspline3]
Dwf[:3] = [bspline1dx, bspline2dx, bspline3dx]

# internal helper variables to avoid memory allocation for temporaries
cdef unsigned int MAXN = 200        # max number points
cdef unsigned int MAXB = 50         # max number neighbours
cdef unsigned int MAXD = 3          # max dimension
cdef double _puy[200][50]           # pu
cdef double _Dpuy[200][50]          # gradient pu
cdef double _Dy[200]                # gradient pu
_tx = np.ndarray((200*3,))          # transformed x


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void mapinv(double a, double b, np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y):
    '''Map from [a,b] to [-1,1].'''
    cdef double w
    cdef Py_ssize_t j
    w = b - a
    for j in range(x.shape[0]):
        y[j] = 2. * (x[j] - a) / w - 1.0

@cython.boundscheck(False)
@cython.wraparound(False)
#@cython.cdivision(True)
cdef eval_pu(unsigned int dim, np.ndarray[np.float64_t, ndim=1] x, unsigned int Nbbox, np.ndarray[np.float64_t, ndim=2] bbox, np.ndarray[np.float64_t, ndim=1] y, unsigned int type):
    global _tx, _puy
    cdef wfT f
    f = wf[type]                                # weight function
    cdef unsigned int N = x.shape[0] / dim      # number points
    cdef unsigned int b, d, j
    cdef double v
    
    for b in range(Nbbox):                      # iterate patches
        for d in range(dim):                    # iterate dimensions
            # transform points to patch
            mapinv(bbox[b * dim + d,0], bbox[b * dim + d,1], x[d * N:(d + 1) * N], _tx)
            if d > 0:
                for j in range(N):          # iterate points
                    v = f(_tx[j])
                    print "V=", v, _tx[j]
                    _puy[j][b] *= v
                    if b == 0:
                        y[j] *= v
            else:
                for j in range(N):          # iterate points
                    v = f(_tx[j])
                    print "V0=", v, _tx[j]
                    _puy[j][b] = y[j]
                    if b == 0:
                        y[j] = v 
    # sum up
    for j in range(N):                      # iterate points
        for b in range(Nbbox-1):            # iterate patches
            _puy[j][0] += _puy[j][b+1]
        if abs(_puy[j][0]) > 1e-8:
            y[j] /= _puy[j][0]
        else:
            y[j] = 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
#@cython.cdivision(True)
cdef eval_pu_dx(unsigned int dim, np.ndarray[np.float64_t, ndim=1] x, unsigned int Nbbox, np.ndarray[np.float64_t, ndim=2] bbox, np.ndarray[np.float64_t, ndim=1] y, unsigned int type):
    global _tx, _puy, _Dpuy, _Dy
    cdef wfT f, Df 
    f = wf[type]
    Df = Dwf[type]
    cdef unsigned int N = x.shape[0] / dim      # number points
    cdef unsigned int b, d, j
    cdef double v, Dv
    for c in range(dim):                        # derivative component
        for b in range(Nbbox):                  # iterate patches
            for d in range(dim):                # iterate dimensions
                # transform points to patch
                mapinv(bbox[b * dim + d,0], bbox[b * dim + d,1], x[d * N:(d + 1) * N], _tx)
                for j in range(N):              # iterate points
                    # evaluate weight and possibly gradient
                    v = f(_tx[j])
                    if d == c:
                        Dv = Df(_tx[j])
                    else:
                        Dv = v
                        
                    # setup or multiply by component
                    if d > 0:
                        _puy[j][b] *= v
                        _Dpuy[j][b] *= Dv
                        if b == 0:
                            y[j] *= v
                            _Dy[j] *= Dv
                    else:
                        _puy[j][b] = v
                        _Dpuy[j][b] = Dv
                        if b == 0:
                            y[j] = v
                            _Dy[j] = Dv
        
        # sum up
        for j in range(N):                      # iterate points
            for b in range(Nbbox-1):            # iterate patches
                _puy[j][0] += _puy[j][b+1]
                _Dpuy[j][0] += _Dpuy[j][b+1]
            if abs(_puy[j][0]) > 1e-8:
                y[j*dim+c] = (_Dy[j] * _puy[j][0] - y[j*dim+c] * _Dpuy[j][0]) / (_puy[j][0] * _puy[j][0])
            else:
                y[j*dim+c] = 0.0


# spline functions
# ================

cdef inline double bspline1(double x):
    cdef double y
    x = abs(x)
    if x < 1.:
        y = 1. - x
    else:
        y = 0.
    return y

cdef inline double bspline1dx(double x):
    cdef double y
    if x < 0 and x > -1:
        y = 1.
    elif x > 0 and x < 1:
        y = -1.
    else:
        y = 0.
    return y        

cdef inline double bspline2(double x):
    cdef double y
    if x <= -1 or x >= 1:
        return 0.
    x = abs((x + 1) / 2.0)
    if x <= 1 / 3.0:
        y = 6 * x * x
    elif x <= 2 / 3.0:
        y = 6 * (1 / 9.0 + 2 / 3.0 * (x - 1 / 3.0) - 2 * (x - 1 / 3.0) * (x - 1 / 3.0))
    elif x <= 1:
        y = 6 * ((1 - x) * (1 - x))
    return y

cdef inline double bspline2dx(double x):
    cdef double y
    if x <= -1 or x >= 1:
        return 0.
    x = abs((x + 1) / 2.0)
    if x <= 1 / 30:
        y = 0.5 * 12 * x
    elif x <= 2 / 3.0:
        y = 0.5 * 6 * (2 / 3.0 - 4 * (x - 1 / 3.0))
    elif x <= 1:
        y = 0.5 * -12 * (1 - x)
    return y

cdef inline double bspline3(double x):
    cdef double y
    if x <= -1 or x >= 1:
        return 0.
    x = abs((x + 1) / 2.0)
    if x <= 1 / 4.0:
        y = 16 * x * x * x;
    elif x <= 2 / 4.0:
        y = 16 * (1 / 64.0 + 3 / 16.0 * (x - 1 / 4.0) + 3 / 4.0 * (x - 1 / 4.0) * (x - 1 / 4.0) - 3 * (x - 1 / 4.0) * (x - 1 / 4.0) * (x - 1 / 4.0))
    elif x <= 3 / 4.0:
        y = 16 * (1 / 64.0 + 3 / 16.0 * (3 / 4.0 - x) + 3 / 4.0 * (3 / 4.0 - x) * (3 / 4.0 - x) - 3.*(3 / 4.0 - x) * (3 / 4.0 - x) * (3 / 4.0 - x))
    elif x <= 1:
        y = 16 * ((1 - x) * (1 - x) * (1 - x))
    return y

cdef inline double bspline3dx(double x):
    cdef double y, s
    if x <= -1 or x >= 1:
        return 0.
    x = abs((x + 1) / 2.0)
    if x <= 1 / 4.0:
        y = 0.5 * 3 * 16 * x * x
    elif x <= 2 / 4.0:
        y = 0.5 * 16 * (3 / 16.0 + 6 / 4.0 * (x - 1 / 4.0) - 9 * (x - 1 / 4.0) * (x - 1 / 4.0))
    elif x <= 3 / 4.0:
        y = 0.5 * 16 * (-3 / 16.0 + -6 / 4.0 * (3 / 4.0 - x) + 9 * (3 / 4.0 - x) * (3 / 4.0 - x))
    elif x <= 1:
        s = -1 + 2 * <int>(x < 0)
        y = s * 0.5 * 3 * 16 * (1 - x) * (1 - x)
    return y
