from __future__ import division

from pypum.geom.affinemap import AffineMap
#from pypum.utils.decorators import cache

import numpy as np
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

# TODO: implement proper expiring cache with decorator
# note that usually only one neighbour set would have to be cached
# since assembly is carried out quadrature patch-wise

# TODO: use vectorize decorator


class PU(object):
    """Partition of Unity on nTree."""
    
    def __init__(self, tree, weightfunc, scaling=1.3, cache_neighbours=True):
        self._tree = tree
        self._scaling = scaling
        self._weightfunc = weightfunc
        self._cache_active = cache_neighbours
        self._neighbourcache = defaultdict(lambda: None)
    
    @property
    def scaling(self):
        return self._scaling
    
    @property
    def tree(self):
        return self._tree
    
    @property
    def cache_active(self):
        return self._cache_active
    
    def indices(self):
        return self._tree.leafs()
    
    def get_node(self, id):
        return self._tree[id]
    
    def get_bbox(self, id):
        return self._tree[id].bbox * self._pu.scaling
    
    def _eval_weight(self, x, id, gradient=False):
        if not gradient:
            y = 0
        else:
            y = np.zeros(self._tree.bbox.dim)
        # check if x is inside patch and evaluate weight function or gradient
        node = self._tree[id]
        if node.bbox.is_inside(x, scaling=self._scaling):
            tx = AffineMap.eval_inverse_map(node.bbox, x, scaling=self._scaling)
            if not gradient:
                y = self._weightfunc(tx)
            else:
                y = self._weightfunc.dx(tx)
                y *= 1 / (self._scaling * node.bbox.size)
        return y
    
    def __call__(self, x, id=None):
        if isinstance(x, (list, tuple)):
            val = np.array([self._eval(cx, id) for cx in x])
        else:
            val = self._eval(x, id) 
        return val

    def dx(self, x, id):
        if isinstance(x, (list, tuple)):
            val = np.array([self._eval(cx, id, gradient=True) for cx in x])
        else:
            val = self._eval(x, id, gradient=True) 
        return val

    def _eval(self, x, id, gradient=False):
        if id is None:
            id = self._tree.find_node(x)
        if not gradient:            
        # PU function evaluation
        # ----------------------
            val = 0
            vals = []
            for cid in [id] + self.get_neighbours(id):
                if cid != id:
                    vals.append(self._eval_weight(x, cid))
                else:
                    val = self._eval_weight(x, cid)
                    vals.append(val)
#            print vals
            vals = sum(vals)
            if vals == 0.0:
                return 0.0
            else:
                return val / vals
        else:
        # PU gradient evaluation
        # ----------------------
            val = 0
            valdx = None
            vals = []
            valsdx = []
            for cid in [id] + self.get_neighbours(id):
                if cid != id:
                    vals.append(self._eval_weight(x, cid, gradient=False))
                    valsdx.append(self._eval_weight(x, cid, gradient=True))
                else:
                    val = self._eval_weight(x, cid, gradient=False)
                    vals.append(val)
                    valdx = self._eval_weight(x, cid, gradient=True)
                    valsdx.append(valdx)
#            print "PU-1", val, vals
#            print "PU-2", valdx, valsdx
            vals = sum(vals)
            valsdx = sum(valsdx)
#            print "PU-3", vals, valsdx
            if np.all(vals == 0.0):
                return np.zeros(self._tree.bbox.dim)
            else:
                return (valdx * vals + val * valsdx) / vals ** 2

    def get_neighbours(self, id):
        neighbours = self._neighbourcache[id]
        if not self._cache_active or neighbours is None:
            neighbours = self._tree.find_neighbours(id, scaling=self._scaling)
            if self._cache_active:
                self._neighbourcache[id] = neighbours
        return neighbours

    def get_active_neighbours(self, id, x):
        neighbours = [nid for nid in self._tree.find_neighbours(id) if self._tree[nid].bbox.is_inside(x, scaling=self._scaling)]
        return neighbours

    def clear_cache(self):
        self._neighbourcache.clear()
    
