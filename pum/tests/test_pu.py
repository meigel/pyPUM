from pypum.utils.box import Box
from pypum.utils.ntree import nTree
from pypum.pum.pu import PU
from pypum.pum.pubasis import PUBasis
from pypum.pum.tensorproduct import TensorProduct
from pypum.pum.weightfunctions import Spline
from pypum.utils.testing import *

import numpy as np
import logging
logger = logging.getLogger(__name__)


def test_pu():
    bbox = Box(((0, 1), (0, 1)))
    tree = nTree(bbox=bbox)
    weightfunc = TensorProduct([Spline(3)] * bbox.dim)
    pu = PU(tree, weightfunc=weightfunc, scaling=1.3, parentlevel=2)
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
    bbox = Box(((0, 1), (0, 1)))
    tree = nTree(bbox=bbox)
    weightfunc = TensorProduct([Spline(3)] * bbox.dim)
    pu = PU(tree, weightfunc=weightfunc, scaling=1.3, parentlevel=2)
    pu.tree.refine(1)
    # setup monom basis
    maxdegree = 4
    basis1d = [np.poly1d([1] + [0] * k) for k in range(maxdegree + 1)]
    basis = TensorProduct.create_basis(basis1d, 2)
    # setup PU basis
    pubasis = PUBasis(pu, basis)
    
    for id in pu.indices():
        node = pu.get_node(id)
        cn = node.center
        print "node ", node, cn
        neighbours = pu.get_neighbours(id)
        active_neighbours = pu.get_active_neighbours(id, cn)
        print "neighbours", neighbours
        print "active neighbours", active_neighbours
        print "center f(", cn, ") =", pu(cn, id)
    

test_main()
