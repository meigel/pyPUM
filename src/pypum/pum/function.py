import abc
import collections
import numpy as np


class Function(object):
    def __init__(self, dim, codim, vectorised=False, discontinuities=None):
        self._dim = dim
        self._codim = codim
        self._vectorised = vectorised
        self._discontinuities = discontinuities

    def __call__(self, x):
        if self._vectorised:
            return self._f(x)
        else:
            if isinstance(x, collections.Iterable):
                return np.array([self._f(tx) for tx in x])
            else:
                return self._f(x)
    
    def dx(self, x):
        if self._vectorised:
            return self._dx(x)
        else:
            if isinstance(x, collections.Iterable):
                return [self._dx(tx) for tx in x]
            else:
                return self._dx(x)

    def ndx(self, x, eps=1e-8):
        # TODO: implicit vectorisation
        def _xd(d, s):
            xd = x
            xd[d] += s * eps
            return xd
        f = self.eval
        df = [f(_xd(d, 1)) - f(_xd(d, -1)) / (2 * eps) for d in range(self.dim)]
        return df

    @abc.abstractmethod
    def _f(self, x):
        """This is the method which has to be overwritten!"""
        pass
    
    def _dx(self, x):
        return self.ndx(x)      # call numerical derivative evaluation by default

    @property
    def dim(self):
        return self._dim
    
    @property
    def codim(self):
        return self._codim


class FunctionSet(object):
    @abc.abstractmethod
    def __getitem__(self, id):
        pass    
    
    def __call__(self, x, id):
        return [f(x) for f in self[id]]
    
    def dx(self, x, id):
        return [f.dx(x) for f in self[id]]


class FunctionSum(Function):
    def __init__(self, funcs, dim=None, codim=None):
        if dim is None:
            assert len(funcs > 0)
            dim = funcs[0].dim 
        if codim is None:
            assert len(funcs > 0)
            codim = funcs[0].codim 
        super(FunctionSum, self).__init__(dim, codim)
        self._funcs = funcs
    
    def _f(self, x):
        return sum([f(x) for f in self._funcs])
    
    def dx(self, x):
        return sum([f.dx(x) for f in self._funcs])


class FunctionProduct(Function):
    def __init__(self, funcs, dim=None, codim=None):
        if dim is None:
            assert len(funcs > 0)
            dim = funcs[0].dim 
        if codim is None:
            assert len(funcs > 0)
            codim = funcs[0].codim 
        super(FunctionProduct, self).__init__(dim, codim)
        self._funcs = funcs
    
    def _f(self, x):
        import operator
        def prod(lst):
            return reduce(operator.mul, lst)
        return prod([f(x) for f in self._funcs])
    
    def dx(self, x):
        pass


class FunctionDivision(Function):
    def __init__(self, funcs, dim=None, codim=None):
        if dim is None:
            assert len(funcs > 0)
            dim = funcs[0].dim 
        if codim is None:
            assert len(funcs > 0)
            codim = funcs[0].codim 
        super(FunctionDivision, self).__init__(dim, codim)
        self._funcs = funcs
    
    def _f(self, x):
        pass
    
    def dx(self, x):
        pass


class FunctionComposition(Function):
    def __init__(self, funcs, dim=None, codim=None):
        if dim is None:
            assert len(funcs > 0)
            dim = funcs[0].dim 
        if codim is None:
            assert len(funcs > 0)
            codim = funcs[0].codim 
        super(FunctionComposition, self).__init__(dim, codim)
        self._funcs = funcs
    
    def _f(self, x):
        pass
    
    def dx(self, x):
        pass
