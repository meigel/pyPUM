from spuq.utils.type_check import takes, anything
from types import FunctionType, GeneratorType

class ParametricArray(object):
    """Dynamically growing array based on a generator or a callable."""
    Empty = object()

    @takes(anything, (FunctionType, GeneratorType))
    def __init__(self, func):
        self._vals = []
        self._func = func

    def __len__(self):
        return len(self._vals)

    def __call__(self, i):
        return self.__getitem__(i)

    def __getitem__(self, i):
        if i >= len(self._vals):
            self._grow(i)
        val = self._vals[i]
        if val is ParametricArray.Empty:
            assert not isinstance(self._func, GeneratorType)
            val = self._func(i)
            self._vals[i] = val
        return val

    def _grow(self, i):
        l = len(self._vals)
        if i >= l:
            if callable(self._func):
                self._vals += [ParametricArray.Empty] * (i - l + 1)
            else:
                self._vals += [self._func.next() for _ in range(i - l + 1)]

    def __str__(self):
        return str(self._vals)
    
