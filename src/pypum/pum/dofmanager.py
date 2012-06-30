

class DofManager(object):
    """Indexing of degrees of freedom of discrete basis."""
    
    def __init__(self, ids, basisset):
        self._basisset = basisset
        self._init(ids)
    
    def dim(self, id=None):
        if id is None:
            return self._dim
        else:
            return self._basisset.dim(id)
    
    def indices(self):
        return self._dofs.iterkeys()
        
    def _init(self, ids):
        c = 0
        self._dofs = {}
        for id in ids:
            self._dofs[id] = c
            c += self.dim(id)
        self._dim = c               # overall size
    
    def __getitem__(self, id):
        return self._dofs[id]
