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
        self._prepared_neighbours = None

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
#        print "========ACTIVE NEIGHBOURS for", x
#        for nid in self._tree.find_neighbours(id):
#            print nid, ":", self._tree[nid].bbox,
#            print "IS_INSIDE...", self._tree[nid].bbox.is_inside(x, scaling=self._scaling)
        neighbours = [nid for nid in self._tree.find_neighbours(id) if self._tree[nid].bbox.is_inside(x, scaling=self._scaling)]
        return neighbours

    def prepare_neighbours(self, id, onlyself=False):
        '''Find and prepare neighbours of patch for pu evaluation.'''
        D = self._tree.bbox.dim
        # get and convert neighbours of patch
        if not onlyself: 
            self._prepared_neighbours = [id] + self.get_neighbours(id)
        else:
            self._prepared_neighbours = [id]
        self._Nbbox = len(self._prepared_neighbours)
        for d in range(D):
            for i, pid in enumerate(self._prepared_neighbours):
                w = self._tree[pid].bbox._size[d]*self._scaling
                self._bbox[i*D+d, 0] = self._tree[pid].bbox._center[d]-w/2
                self._bbox[i*D+d, 1] = self._tree[pid].bbox._center[d]+w/2
                
#        print "PREPARED", self._Nbbox, "NEIGHBOURS", self._prepared_neighbours
#        print self._bbox[:self._Nbbox*D, :]

    def __call__(self, _x, gradient, onlyweight=False):
        '''Evaluate pu or gradient of pu. Note that prepare_neighbours has to be called first.'''
        x = _x
        if len(x.shape) == 1:
            x = _x.view()
            x.shape = (len(_x),1)
        N = x.shape[0]
        print "XXXX", x.shape, x
        geomdim = x.shape[1]
        
        # call optimised evaluation
        if gradient:
            return eval_pu_dx(geomdim, x.T.flatten(), self._Nbbox, self._bbox, self._weighttype, onlyweight)
        else:
            return eval_pu(geomdim, x.T.flatten(), self._Nbbox, self._bbox, self._weighttype, onlyweight)


# =============================================================
# ================== cython optimised code ====================
# =============================================================


# define look-up for weight functions
ctypedef double (*wfT)(double)
cdef wfT[3] wf
cdef wfT[3] Dwf
wf[:3] = [bspline1, bspline2, bspline3]
Dwf[:3] = [bspline1dx, bspline2dx, bspline3dx]


# internal helper variables to avoid memory allocation for temporaries
cdef enum:
    MAXN = 20000      # max number points
    MAXB = 50         # max number neighbours
    MAXD = 3          # max dimension
#cdef np.float64_t _puy[MAXN*MAXB]           # pu
#cdef np.float64_t _Dpuy[MAXN*MAXB*MAXD]     # gradient pu
#cdef np.float64_t _Dy[MAXN*MAXB*MAXD]       # gradient pu
#cdef np.float64_t _y[MAXN*MAXB]             # pu
#cdef np.float64_t _tx[MAXN]                 # transformed x
_puy = np.ndarray((MAXN,MAXB))              # pu
_Dpuy = np.ndarray((MAXN,MAXB,MAXD))        # gradient pu
_Dy = np.ndarray((MAXN,MAXB,MAXD))          # gradient pu
_y = np.ndarray((MAXN,MAXB))                # pu
_tx = np.ndarray((MAXN))                    # transformed x
_w =  np.ndarray((MAXN))                    # summed pu weights
_Dw =  np.ndarray((MAXN,MAXD))              # summed pu weights


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void mapinv(double a, double b, np.float64_t[:] x, np.float64_t[:] y):
    '''Map from [a,b] to [-1,1].'''
    cdef double w
    cdef Py_ssize_t j
    w = b - a
    for j in range(x.shape[0]):
        y[j] = 2. * (x[j] - a) / w - 1.0


@cython.boundscheck(False)
@cython.wraparound(False)
#@cython.cdivision(True)
cdef eval_pu(unsigned int dim, np.float64_t[:] x, unsigned int Nbbox, np.float64_t[:,:] bbox, unsigned int type, unsigned int onlyweight):
    global _tx, _puy, _w
    cdef wfT f
    f = wf[type]                                # weight function
    cdef unsigned int N = x.shape[0] / dim      # number points
    cdef unsigned int b, d, j
    cdef double v
    assert N < MAXN
    
    # setup memoryviews
    cdef np.float64_t[:,:] v_puy = _puy
    cdef np.float64_t[:,:] v_y = _y
    cdef np.float64_t[:]   v_w = _w
 
    # A prepare pu evaluations
    # ------------------------   
    for b in range(Nbbox):                      # iterate patches
        for d in range(dim):                    # iterate dimensions
            # transform points to patch
            mapinv(bbox[b * dim + d,0], bbox[b * dim + d,1], x[d * N:(d + 1) * N], _tx)
            print "MAPINV", bbox[b * dim + d,0], bbox[b * dim + d,1], x[d * N:(d + 1) * N], _tx[0]
            # evaluate weights
            if d > 0:
                for j in range(N):          # iterate points
                    v = f(_tx[j])
                    print "V=", v, _tx[j]
                    v_puy[j][b] *= v
            else:
                for j in range(N):          # iterate points
                    v = f(_tx[j])
                    print "V0=", v, _tx[j]
                    v_puy[j][b] = v
    # B sum up
    # --------
    if onlyweight == 0:
        for j in range(N):                      # iterate points
            # sum up weight
            v_w[j] = _puy[j][0]
            for b in range(Nbbox-1):            # iterate patches
                v_w[j] += _puy[j][b+1]
            # devide pu by sum
            if abs(v_w[j]) > 1e-8:
                for b in range(Nbbox):          # iterate patches
                    v_puy[j][b] /= v_w[j]
            else:
                for b in range(Nbbox):          # iterate patches
                    v_puy[j][b] = 0.0
    return np.asarray(v_puy[:N,:Nbbox])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef eval_pu_dx(unsigned int dim, np.float64_t[:] x, unsigned int Nbbox, np.float64_t[:,:] bbox, unsigned int type, unsigned int onlyweight):
    global _tx, _puy, _Dpuy, _Dy, _y, _w
    cdef wfT f, Df 
    f = wf[type]
    Df = Dwf[type]
    cdef unsigned int N = x.shape[0] / dim      # number points
    cdef unsigned int b, d, j
    cdef double v, Dv
        
    # setup memoryviews
    cdef np.float64_t[:,:]   v_puy = _puy
    cdef np.float64_t[:,:,:] v_Dpuy = _Dpuy
    cdef np.float64_t[:,:]   v_y = _y
    cdef np.float64_t[:,:,:] v_Dy = _Dy
    cdef np.float64_t[:]     v_w = _w
    cdef np.float64_t[:,:]   v_Dw = _Dw
    
    for c in range(dim):                        # derivative component
        # A prepare pu evaluations
        # ------------------------
        for b in range(Nbbox):                  # iterate patches
            for d in range(dim):                # iterate dimensions
                # transform points to patch
                mapinv(bbox[b * dim + d,0], bbox[b * dim + d,1], x[d * N:(d + 1) * N], _tx)
                for j in range(N):              # iterate points
                    # evaluate weight and possibly gradient
                    v = f(_tx[j])
                    Dv = Df(_tx[j])
                    # setup or multiply by component
                    if d > 0:
                        v_puy[j,b] *= v
                        if c == d:
                            v_Dpuy[j,b,c] *= Dv
                        else:
                            v_Dpuy[j,b,c] *= v
                    else:
                        v_puy[j,b] = v
                        if c == d:
                            v_Dpuy[j,b,c] = Dv
                        else:
                            v_Dpuy[j,b,c] = v
        
        # B sum up
        # --------
        if onlyweight == 0:
            for c in range(dim):
                for j in range(N):                      # iterate points
                    if c == 0:
                        v_w[j] = v_puy[j,0]
                    v_Dw[j,:] = v_Dpuy[j,0,:]
                    for b in range(Nbbox-1):            # iterate patches
                        if c == 0:
                            v_w[j,] += _puy[j,b+1]
                        v_Dw[j,0] += _Dpuy[j,b+1,c]
               
            for b in range(Nbbox):      
                for c in range(dim):
                    for j in range(N):                      # iterate points
                        if abs(v_w[j]) > 1e-8:
                            v_Dy[j,b,c] = (v_Dpuy[j,b,c] * v_w[j] - v_puy[j,b] * v_Dw[j,b]) / (v_w[j] * v_w[j])
                        else:
                            v_Dy[j,b,c] = 0.0
            # return memoryview
            return np.asarray(v_Dy[:N, :Nbbox, :dim])
        else:
            return np.asarray(v_Dpuy[:N, :Nbbox, :dim])


# spline functions
# ================

cdef inline double bspline1(double x):
    x = abs(x)
    if x < 1.:
        y = 1. - x
    else:
        y = 0.
    return y

cdef inline double bspline1dx(double x):
    if x < 0 and x > -1:
        y = 1.
    elif x > 0 and x < 1:
        y = -1.
    else:
        y = 0.
    return y

cdef inline double bspline2(double x):
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
    cdef double s
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
