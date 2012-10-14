from __future__ import division
import itertools as iter
import numpy as np
import collections
from exceptions import Exception
from pypum.utils.type_check import takes, anything, list_of, optional

import logging
logger = logging.getLogger(__name__)


class DisjointException(Exception):
    def __init__(self, b1=None, b2=None):
        self._b1 = b1
        self._b2 = b2

    def __str__(self):
        return "Objects do not intersect:", self._b1, self._b2


class Box(object):
    """Box class in arbitrary dimension."""
    
    def __init__(self, pos):
        """Construct box with specified position and size.
        The argument is a set of intervals, e.g., for a 2d box [0,1]x[-1,1]
        box = Box([[0,1],[-1,1]])
        """
        self._pos = pos
        self._size = None
        self._center = None
        self._init_data()

    def _init_data(self):
        self._size = np.array([(p[1] - p[0]) for p in self.pos])
        assert not np.any(self._size < 0.0)
        self._center = np.array([(p[0] + p[1]) / 2 for p in self.pos])

    def copy(self):
        """Return copy."""
        return Box(self.pos)

    def __getitem__(self, d):
        assert d >= 0 and d < self.dim
        return self._pos[d]

    @takes(anything, "Box")
    def do_intersect(self, other, scaling=[1, 1]):
        """Check for intersection with other box."""
        assert self.dim == other.dim and scaling >= 1
        if not isinstance(scaling, collections.Iterable):
            scaling = (scaling, scaling)
        _intersect = lambda p1, p2, dx1, dx2: (p1[0] - dx1 <= p2[1] + dx2 and p2[0] - dx2 <= p1[1] + dx1)
        dx1 = map(lambda x: x * float(scaling[0] - 1), self.size)
        dx2 = map(lambda x: x * float(scaling[1] - 1), other.size)
        for p1, p2, d1, d2 in zip(self.pos, other.pos, dx1, dx2):
            if not _intersect(p1, p2, d1, d2):
                return False
        return True 

    @takes(anything, "Box")
    def intersect(self, other, scaling=[1, 1]):
        """Intersect with other box, return intersection box."""
        assert self.dim == other.dim and scaling >= 1
        if not isinstance(scaling, collections.Iterable):
            scaling = (scaling, scaling)
        if not self.do_intersect(other, scaling=scaling):
            raise DisjointException(self, other)
        _intersection = lambda p1, p2, dx1, dx2: (max(p1[0] - dx1, p2[0] - dx2), min(p1[1] + dx1, p2[1] + dx2))
        dx1 = map(lambda x: x * float(scaling[0] - 1), self.size)
        dx2 = map(lambda x: x * float(scaling[1] - 1), other.size)
        pos = [_intersection(p1, p2, d1, d2) for p1, p2, d1, d2 in zip(self.pos, other.pos, dx1, dx2)]
        return Box(pos)

    def is_inside(self, _p, trueinside=True, scaling=1):
        """Check point inclusion in box."""
#        assert (isinstance(p, np.ndarray) and p.shape[1] == self.dim) or len(x) == self.dim
        dx = np.array(self.size) * scaling
        p = _p 
        if len(p.shape) == 1:
            import copy
            p = copy.copy(_p)
            N = 1
            p.shape = (1, p.shape[0])
        else:
            N = p.shape[0]
        val = np.ones((N, 1)).all(axis=1)
        for d in range(self.dim):
            if trueinside:
                tval = np.column_stack((self.center[d] - dx[d] / 2 < p[:, d], p[:, d] < self.center[d] + dx[d] / 2))
#                print "A IS_INSIDE", d, ":"
#                print self.center[d] - dx[d] / 2, p[:, d], self.center[d] + dx[d] / 2 
#                print tval, tval.all(axis=1), val
                val *= tval.all(axis=1)
            else:
                tval = np.column_stack((self.center[d] - dx[d] / 2 <= p[:, d], p[:, d] <= self.center[d] + dx[d] / 2))
#                print "B IS_INSIDE", d, ":"
#                print self.center[d] - dx[d] / 2, p[:, d], self.center[d] + dx[d] / 2 
#                print tval, tval.all(axis=1), val
                val *= tval.all(axis=1)
        return val

    def split(self):
        """Split box in all dimensions, return 2**d new boxes."""
        p = np.array([self[d][0] for d in range(self.dim)])
        dx = self._size / 2.0
        boxes = []
        for idx in [i for i in iter.product(range(2), repeat=self.dim)]:
            pos = p + np.array(idx) * dx
            pos = [[a, a + dx[d]] for d, a in enumerate(pos)]
            boxes.append(Box(pos))
        return boxes

#    def scaled_pos(self, scaling=1.0):
#        p = np.array(self.pos)
#        dx = np.matlib.repmat(np.array(self.size) * (scaling - 1.0) / 2, 2, 1)
#        print "SP", p, dx
#        dx [:, 0] *= -1
#        p += dx 
#        return p

    @property
    def pos(self):
        return self._pos
    
    @property
    def center(self):
        return self._center
    
    @property
    def size(self):
        return self._size
    
    @property
    def dim(self):
        return len(self._size)

    @takes(anything, "Box")
    def __eq__(self, other):
        return self.pos == other.pos

    def __iadd__(self, a):
        """Translate box position."""
        for d in range(self.dim):
            self._pos[d][0] += a[d]
            self._pos[d][1] += a[d]
        return self

    def __isub__(self, a):
        """Translate box position."""
        return self.__iadd__(-a)

    def __imul__(self, a):
        """Scale box size."""
        if not isinstance(a, collections.Iterable):
            a = [a] * self.dim
        for d in range(self.dim):
            self._size[d] *= a[d]
            self._pos[d][0] = self._center[d] - self._size[d] / 2
            self._pos[d][1] = self._center[d] + self._size[d] / 2
        return self

    def __idiv__(self, a):
        """Scale box size."""
        return self.__imul__(1 / a)

    def __add__(self, a):
        b = self.copy()
        b += a
        return b

    def __sub__(self, a):
        return self.__add__(-a)

    def __mul__(self, a):
        b = self.copy()
        b *= a
        return b

    def __div__(self, a):
        b = self.copy()
        b /= a
        return b

    def __str__(self):
        return "Box[" + str(self.dim) + "]   pos = " + str(self._pos) + "   size = " + str(self._size) + "   center = " + str(self.center)
