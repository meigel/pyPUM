from pypum.pum.function import FunctionSet

class BasisSet(FunctionSet):
    def __init__(self, basis=None):
        self._basis = {}
        if basis is not None:
            self._basis[-1] = basis
    
    def set_basis(self, basis, id=-1):
        self._basis[id] = basis
    
    def __getitem__(self, id):
        try:
            return self._basis[id]
        except:
            return self._basis[-1]
    
    def dim(self, id):
        return len(self[id])
