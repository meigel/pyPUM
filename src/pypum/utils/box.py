from __future__ import division
import itertools as iter
import numpy as np
from exceptions import Exception
from types import ListType
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
        if not isinstance(scaling, ListType):
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
        if not istype(scaling, ListType):
            scaling = (scaling, scaling)
        if not self.do_intersect(other, scaling=scaling):
            raise DisjointException(self, other)
        _intersection = lambda p1, p2, dx1, dx2: (max(p1[0] - dx1, p2[0] - dx2), min(p1[1] + dx1, p2[1] + dx2))
        dx1 = map(lambda x: x * float(scaling[0] - 1), self.size)
        dx2 = map(lambda x: x * float(scaling[1] - 1), other.size)
        pos = [_intersection(p1, p2, d1, d2) for p1, p2, d1, d2 in zip(self.pos, other.pos, dx1, dx2)]
        return Box(pos)

    def is_inside(self, point, trueinside=False, scaling=1):
        """Check point inclusion in box."""
        assert len(point) == self.dim and scaling >= 1
        dx = map(lambda x: x * float(scaling - 1), self.size)
        if trueinside:
            return all([p1[0] - d < p2 and p2 < p1[1] + d for p1, p2, d in zip(self._pos, point, dx)])
        else:
            return all([p1[0] - d <= p2 and p2 <= p1[1] + d for p1, p2, d in zip(self._pos, point, dx)])

    def split(self):
        """Split box in all dimensions, return 2**d new boxes."""
        p = np.array([self[d][0] for d in range(self.dim)])
        dx = self._size / 2.0
        boxes = []
        for idx in [i for i in iter.product(range(2), repeat=self.dim)]:
            pos = p + np.array(idx) * dx
            pos = [(a, a + dx[d]) for d, a in enumerate(pos)]
            boxes.append(Box(pos))
        return boxes

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
        """Translate box position"""
        for d in range(self.dim):
            self._pos[d][0] += a[d]
            self._pos[d][1] += a[d]
        return self

    def __isub__(self, a):
        """Translate box position"""
        for d in range(self.dim):
            self._pos[d][0] -= a[d]
            self._pos[d][1] -= a[d]
        return self

    def __imul__(self, a):
        """Scale box size."""
        for d in range(self.dim):
            self._pos[d][0] *= a[d]
            self._pos[d][1] *= a[d]
        return self

    def __idiv__(self, a):
        """Scale box size."""
        for d in range(self.dim):
            self._pos[d][0] /= a[d]
            self._pos[d][1] /= a[d]
        return self

    def __add__(self, a):
        b = self.copy()
        b += a
        return b

    def __sub__(self, a):
        b = self.copy()
        b -= a
        return b

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
