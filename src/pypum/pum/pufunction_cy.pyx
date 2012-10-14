

class PUFunction(object):
    def __init__(self, sol, tree, pu, basisset, dof):
        self._sol = sol
        self._tree = tree
        self._pu = pu
        self._scaling = pu._scaling
        self._basisset = basisset
        self._dof = dof
        self._geomdim = self._tree.dim
        self._currentid = -1
    
    def __call__(self, x, gradient=False):
        assert not gradient     # TODO...
        
        # identify home patch
        nid = self._tree.find_node(x)
        if nid != self._currentid:
            self._pu.prepare_neighbours(nid)
            self._currentid = nid
        
        # get neighbours and evaluate all pu functions at x
        nids = self._pu._prepared_neighbours
        y = self._pu(x, gradient=False, all_pus=True)
        idx = [self._dof[cid] for cid in nids]
    
        # evaluate base functions on neighbour patches
        bbox = self._pu._bbox
        for i, cid in enumerate(nids):
            basis = self._basisset[cid]
            # transform x to patch
#            tx = affine_map(x, bbox[i*self._geomdim:(i+1)*self._geomdim,:])
#            yb = basis(x, gradient=False)
            y = 0
        
        return y
