from pypum.utils.box import Box
from pypum.utils.ntree import nTree
from pypum.pum.pu_cy import PU
from pypum.pum.monomialbasis import MonomialBasis
from pypum.pum.basis import BasisSet
from pypum.utils.testing import *

import numpy as np
import logging
logger = logging.getLogger(__name__)


def test_pu():
    bbox = Box([[0, 1], [0, 1]])
    tree = nTree(bbox=bbox)
    pu = PU(tree, weighttype='bspline1', scaling=1.25)
    pu.tree.refine(1)
    for id in pu.indices:
        node = pu.get_node(id)
        cn = node.center
        print "node ", node, cn
        neighbours = pu.get_neighbours(id)
        active_neighbours = pu.get_active_neighbours(id, cn)
        print "neighbours", neighbours
        print "active neighbours", active_neighbours
        pu.prepare_neighbours(id)
        y = pu(cn, gradient=False)
        print "center f(", cn, ") =", y
        Dy = pu(cn, gradient=True)
        print "center Df(", cn, ") =", Dy

def test_pu_basis():
    # setup PU
    bbox = Box([[0, 1], [0, 1]])
    tree = nTree(bbox=bbox)
    pu = PU(tree, weighttype='bspline1', scaling=1.25)
    pu.tree.refine(1)
    # setup monom basis
    maxdegree = 0
    basis = MonomialBasis(maxdegree, 2)
    basisset = BasisSet(basis)

    for id in pu.indices:
        # set center and shifted point for patch
        node = pu.get_node(id)
        cn = node.center
        cn2 = cn.copy()
        cn2[0] += node.size[0] / 3
        cn2[1] += node.size[1] / 3
        print "node ", node, cn
        # get neighbours
        neighbours = pu.get_neighbours(id)
        print "neighbours", neighbours, "of node", id
        active_neighbours = pu.get_active_neighbours(id, cn)
        print "active neighbours at center", active_neighbours, len(active_neighbours) == 0, "(has to be empty for center of patches!)"
        # evaluate pu at center and other point
        pu.prepare_neighbours(id)
        y = pu(cn, gradient=False)
        Dy = pu(cn, gradient=True)
        print "center f(", cn, ") =", y
        print "center dx(", cn, ") =", Dy
        active_neighbours = pu.get_active_neighbours(id, cn2)
        print "active neighbours", active_neighbours, "(does not have to be empty!)"
        y = pu(cn2, gradient=False)
        Dy = pu(cn2, gradient=True)
        print "f(", cn2, ") =", y
        print "dx(", cn2, ") =", Dy
    

test_main()
