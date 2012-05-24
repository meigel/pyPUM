# taken from http://machineawakening.blogspot.com/2011/03/making-numpy-ndarrays-hashable.html
from numpy import ndarray, array
from hashlib import sha1

class hashable_ndarray(ndarray):
    def __new__(cls, values):
        this = ndarray.__new__(cls, shape=values.shape, dtype=values.dtype)
        this[...] = values
        return this

    def __init__(self, values):
        self.__hash = int(sha1(self).hexdigest(), 16)

    def __eq__(self, other):
        return all(ndarray.__eq__(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.__hash

#    def __setitem__(self, key, value):
#        raise Exception('hashable arrays are read-only')

    def add(self, pos, val):
        a = array(self)
        try:
            a[pos] += val
        except Exception:
            a.resize(pos+1)
            a[pos] = val
        return hashable_ndarray(a)
