from __future__ import division

from pypum.geom.affinemap import AffineMap
#from pypum.utils.decorators import cache

import numpy as np
import numpy.matlib as ml
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

# TODO: implement proper expiring cache with decorator
# note that usually only one neighbour set would have to be cached
# since assembly is carried out quadrature patch-wise


PU_EPS = 1e-8


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
        N = x.shape[0]
        if not gradient:
            y = np.zeros((N, 1))
        else:
            y = np.zeros((N, self._tree.bbox.dim))
        # check if x is inside patch and evaluate weight function or gradient
        node = self._tree[id]
#        inside = node.bbox.is_inside(x, scaling=self._scaling)
        tx = AffineMap.eval_inverse_map(node.bbox, x, scaling=self._scaling)
        if not gradient:
            y = self._weightfunc(tx)
        else:
            y = self._weightfunc.dx(tx)
            y *= 1 / (self._scaling * node.bbox.size)
#            inside = ml.repmat(inside, 2, 1)
#            inside = inside.T
#        return y * inside
        return y
    
    def __call__(self, x, id=None):
        if isinstance(x, (list, tuple)):
            x = np.array(x)
        return self._eval(x, id, gradient=False)

    def dx(self, x, id):
        if isinstance(x, (list, tuple)):
            x = np.array(x)
        return self._eval(x, id, gradient=True)

    def _eval(self, x, id, gradient=False):
        if id is None:
            assert False
#            id = self._tree.find_node(x)
        if not gradient:
#            return x[:, 0]            
        # PU function evaluation
        # ----------------------
            val = None
            vals = np.zeros_like(x[:, 0])
            for cid in [id] + self.get_neighbours(id):
                if cid != id:
                    vals += self._eval_weight(x, cid)
                else:
                    val = self._eval_weight(x, cid)
                    vals += val
#            print vals
            idx = np.nonzero(np.abs(val) >= PU_EPS)
            r = np.zeros_like(vals)
            r[idx] = val[idx] / vals[idx]
            return r
        else:
#            return x            
        # PU gradient evaluation
        # ----------------------
            val = None
            valdx = None
            vals = np.zeros_like(x[:, 0])
            valsdx = np.zeros_like(x)
            for cid in [id] + self.get_neighbours(id):
                if cid != id:
                    vals += self._eval_weight(x, cid, gradient=False)
                    valsdx += self._eval_weight(x, cid, gradient=True)
                else:
                    val = self._eval_weight(x, cid, gradient=False)
                    vals += val
                    valdx = self._eval_weight(x, cid, gradient=True)
                    valsdx += valdx
#            print "PU-1", val, vals
#            print "PU-2", valdx, valsdx
#            print "PU-3", vals, valsdx
            idx = (np.nonzero(np.abs(vals) >= PU_EPS))[0]
            d = x.shape[1]
            vald = val[idx].repeat(d)
            vald.shape = (len(idx), d)
            valsd = vals[idx].repeat(d)
            valsd.shape = (len(idx), d)
            r = np.zeros_like(x)
            r[idx, :] = (valdx[idx, :] * valsd + vald * valsdx[idx, :]) / valsd ** 2
            return r

    def get_neighbours(self, id):
        neighbours = self._neighbourcache[id]
        if not self._cache_active or neighbours is None:
            logger.debug("PU neighbours have to be retrieved from tree")
            neighbours = self._tree.find_neighbours(id, scaling=self._scaling)
            if self._cache_active:
                self._neighbourcache[id] = neighbours
        else:
            logger.debug("PU neighbours found in cache")
        return neighbours

    def get_active_neighbours(self, id, x):
        neighbours = [nid for nid in self._tree.find_neighbours(id) if self._tree[nid].bbox.is_inside(x, scaling=self._scaling)]
        return neighbours

    def clear_cache(self):
        self._neighbourcache.clear()
    
