from __future__ import division

from pypum.utils.box import Box
from pypum.utils.ntree import nTree
from pypum.pum.pu_cy import PU
from pypum.pum.monomialbasis import MonomialBasis
from pypum.pum.basis import BasisSet
from pypum.utils.plotter import Plotter
from pypum.utils.testing import *

import numpy as np
import logging
logger = logging.getLogger(__name__)

with_plot = True

def test_pu():
    print "\n" + "*"*50
    print "TEST PU"
    print "*"*50
    
    # 1d
    # ==================
    if False:
        print "======== 1d ========="
        bbox = Box([[0, 1]])
        tree = nTree(bbox=bbox)
        pu = PU(tree, weighttype='bspline3', scaling=1.8)
        pu.tree.refine(2)
        
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
            if with_plot:
#                pu.prepare_neighbours(id, onlyself=True)
                Plotter.plot(lambda x:pu(x, gradient=False), 1, [-1 / 4, 5 / 4], resolution=1 / 50)
    
    # 2d
    # ==================
    if True:
        print "======== 2d ========="
        bbox = Box([[0, 1], [0, 1]])
        tree = nTree(bbox=bbox)
        pu = PU(tree, weighttype='bspline3', scaling=1.8)
        pu.tree.refine(2)
        
        for id in pu.indices:
            print "\t", id, ":", pu.get_node(id)
    
        if with_plot:
            pc = 0
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
            if with_plot:
#                pu.prepare_neighbours(id, onlyself=True)
                if pc <= 5:     # don't plot too many functions...
                    pc += 1
                    Plotter.plot(lambda x:pu(x, gradient=False, only_weight=False), 2, [[-1 / 4, 5 / 4], [-1 / 4, 5 / 4]], resolution=1 / 50)


def test_monomials():
    print "\n" + "*"*50
    print "TEST MONOMIALS"
    print "*"*50
    # 1d
    # ==================
    degree = 2
    B1 = MonomialBasis(degree, 1)
    idx = B1.idx
    N = 5
    x = np.linspace(0, 1, N)
    y = np.zeros_like(x)
    Dy = np.zeros_like(x)
    print "==== 1d for", x
    for i in range(B1.dim):
        print "\tbasis", i, str(idx[i])
        B1(x, i, gradient=False, y=y)
        B1(x, i, gradient=True, y=Dy)
        print "\ty =", y, "\tDy =", Dy
        if False and with_plot:
            Plotter.plot(lambda x:B1(x, i, gradient=False), 1, [0, 1], resolution=1 / 50)

    #2d
    # ==================
    degree = 2
    B2 = MonomialBasis(degree, 2)
    idx2 = B2.idx
    x2 = np.array([[cx, 1] for cx in x])
#    x2v = x2.view()
#    x2v.shape = x2.shape[0] * x2.shape[1]
#    print "X2", x2v
    y2 = np.zeros_like(x2[:, 0])
    Dy2 = np.zeros_like(x2)
    print "==== 2d for", x2
    for i in range(B2.dim):
        print "\tbasis", i, str(idx2[i])
        B2(x2, i, gradient=False, y=y2)
        B2(x2, i, gradient=True, y=Dy2)
        print "\ty2 =", y2
        print "\tDy2 =", Dy2
        if with_plot:
            Plotter.plot(lambda x:B2(x, i, gradient=False), 2, [[0, 1], [0, 1]], resolution=1 / 50)


test_main()
