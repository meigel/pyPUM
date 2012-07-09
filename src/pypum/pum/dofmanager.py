

class DofManager(object):
    """Indexing of degrees of freedom of discrete basis."""
    
    def __init__(self, ids, basisset, components=1):
        self._basisset = basisset
        self._components = components
        self._dim = None
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
            c += self.dim(id) * self._components
        self._dim = c               # overall size
    
    def __getitem__(self, id):
        return self._dofs[id]

    def __str__(self):
        return "DofManager with dimension " + str(self.dim()) + " and indices " + " ".join([str(id) for id in self.indices()])\
                 + " with dimensions " + " ".join([str(self.dim(id) * self._components) for id in self.indices()]) 
