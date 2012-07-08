from pypum.utils.box import Box
from pypum.utils.ntree import nTree
from pypum.pum.pu import PU
from pypum.pum.pubasis import PUBasis
from pypum.pum.tensorproduct import TensorProduct
from pypum.pum.weightfunctions import Spline, Monomial
from pypum.utils.testing import *

import numpy as np
import logging
logger = logging.getLogger(__name__)


def test_pu():
    bbox = Box([[0, 1], [0, 1]])
    tree = nTree(bbox=bbox)
    weightfunc = TensorProduct([Spline(3)] * bbox.dim)
    pu = PU(tree, weightfunc=weightfunc, scaling=1.25)
    pu.tree.refine(1)
    pu.clear_cache()
    for id in pu.indices():
        node = pu.get_node(id)
        cn = node.center
        print "node ", node, cn
        neighbours = pu.get_neighbours(id)
        active_neighbours = pu.get_active_neighbours(id, cn)
        print "neighbours", neighbours
        print "active neighbours", active_neighbours
        print "center f(", cn, ") =", pu(cn, id)

def test_pu_basis():
    # setup PU
    bbox = Box([[0, 1], [0, 1]])
    tree = nTree(bbox=bbox)
    weightfunc = TensorProduct([Spline(3)] * bbox.dim)
    pu = PU(tree, weightfunc=weightfunc, scaling=1.25)
    pu.tree.refine(1)
    # setup monom basis
    maxdegree = 0
    basis1d = [Monomial(k) for k in range(maxdegree + 1)]
    basis = TensorProduct.create_basis(basis1d, bbox.dim)
    # setup PU basis
    pubasis = PUBasis(pu, basis)
    
    for id in pu.indices():
        # set center and shifted point for patch
        node = pu.get_node(id)
        cn = node.center
        cn2 = cn.copy()
        cn2[0] += node.size[0] / 3
        cn2[1] += node.size[1] / 3
        print "node ", node, cn
        # get neighbours
        neighbours = pu.get_neighbours(id)
        print "neighbours", neighbours
        active_neighbours = pu.get_active_neighbours(id, cn)
        print "active neighbours at center", active_neighbours, len(active_neighbours) == 0, "(has to be empty for center of patches!)"
        # evaluate pu at center and other point
        print "center f(", cn, ") =", pubasis(cn, id)
        print "center dx(", cn, ") =", pubasis.dx(cn, id)
        active_neighbours = pu.get_active_neighbours(id, cn2)
        print "active neighbours", active_neighbours, "(does not have to be empty!)"
        print "f(", cn2, ") =", pubasis(cn2, id)
        print "dx(", cn2, ") =", pubasis.dx(cn2, id)
    

test_main()
