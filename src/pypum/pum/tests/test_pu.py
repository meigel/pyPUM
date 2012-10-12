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
    print "\n" + "*"*50
    print "TEST PU"
    print "*"*50
    bbox = Box([[0, 1], [0, 1]])
    tree = nTree(bbox=bbox)
    pu = PU(tree, weighttype='bspline1', scaling=1.25)
    pu.tree.refine(1)
    
    for id in pu.indices:
        print "\t", id, ":", pu.get_node(id)

    for id in pu.indices:        
        node = pu.get_node(id)
        cn = node.center
        print "\n", node
        neighbours = pu.get_neighbours(id)
        active_neighbours = pu.get_active_neighbours(id, cn)
        print "\tneighbours", neighbours
        print "\tactive neighbours", active_neighbours
        pu.prepare_neighbours(id)
        y = pu(cn, gradient=False)
        print "\tcenter f(", cn, ") =", y
        Dy = pu(cn, gradient=True)
        print "\tcenter Df(", cn, ") =", Dy


def test_monomials():
    print "\n" + "*"*50
    print "TEST MONOMIALS"
    print "*"*50
    # 1d
    degree = 2
    B1 = MonomialBasis(degree, 1)
    idx = B1.idx
    N = 5
    x = np.linspace(0, 1, N)
    y = np.zeros_like(x)
    Dy = np.zeros_like(x)
    print "1d for", x
    for i in range(B1.dim):
        print "\tbasis", i, str(idx[i])
        y = B1(x, i, gradient=False, y=y)
        Dy = B1(x, i, gradient=True, y=Dy)
        print "\ty =", y, "\tDy =", Dy

    #2d
    degree = 2
    B2 = MonomialBasis(degree, 2)
    idx2 = B2.idx
    x2 = np.array([[tx, 1] for tx in x])
    y2 = np.zeros_like(x2[:, 0])
    Dy2 = np.zeros_like(x2)
    print "2d for", x2
    for i in range(B2.dim):
        print "\tbasis", i, str(idx2[i])
        y = B2(x2, i, gradient=False, y=y2)
        Dy = B2(x2, i, gradient=True, y=Dy2)
        print "\ty2 =", y2, "\tDy2 =", Dy2


# NOTE: BasisSet (and thus this test) is deprecated!
def xtest_pu_basis():
    print "\n" + "*"*50
    print "TEST PU BASIS"
    print "*"*50
    # setup PU
    bbox = Box([[0, 1], [0, 1]])
    tree = nTree(bbox=bbox)
    pu = PU(tree, weighttype='bspline1', scaling=1.5)
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
