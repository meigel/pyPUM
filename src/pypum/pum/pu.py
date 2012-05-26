from pypum.geom.affinemap import AffineMap

from collections import defaultdict
import logging
logger = logging.getLogger(__name__)


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
    
    def _eval_weight(self, x, id):
        y = 0
        node = self._tree[id]
        if node.bbox.is_inside(x, scaling=self._scaling):
            tx = AffineMap.eval_inverse_map(node.bbox, x)
            y = self._weightfunc(tx)
        return y
    
    def __call__(self, x, id=None):
        if id is None:
            id = self._tree.find_node(x)
        val = 0
        vals = []
        for cid in [id] + self.get_neighbours(id):
            if cid != id:
                vals.append(self._eval_weight(x, cid))
            else:
                val = self._eval_weight(x, cid)
                vals.append(val)
#        print vals
        vals = sum(vals)
        return val / vals

    def dx(self, x, id):
        pass

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
    
