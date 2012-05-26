

class DofManager(object):
    """Indexing of degrees of freedom of discrete basis."""
    
    def __init__(self, basisset):
        self._basisset = basisset
        self._init()
    
    @property
    def dim(self, id=None):
        if id is None:
            return self._dim
        else:
            return self._basisset.dim(id)
    
    def indices(self):
        return self._basisset.indices()
        
    def _init(self):
        c = 0
        self._dofs = {}
        for id in self.indices():
            self._dofs[id] = c
            c += self.dim(id)
        self._dim = c
    
    def __getitem__(self, id):
        return self._dofs[id]
