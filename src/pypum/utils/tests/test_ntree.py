from pypum.utils.box import Box
from pypum.utils.ntree import nTree
from pypum.utils.testing import *

def test_ntree():
    bbox = Box(((0, 1), (0, 1)))
    tree = nTree(bbox)
    tree.refine(2)
    print "tree has", tree.totalsize, " nodes and", tree.size, " leafs:"
    
    # nodes and leafs
    for id in tree.nodes():
        print "\tnode id =", id
    for id in tree.leafs():
        print "\tleaf id =", id

    # neighbour tests
    c = 0;
    f = 0;
    for id in tree.leafs():
        c += 1
        n1 = tree.find_neighbours(id, 1.2)
        n2 = tree.find_neighbours_exhaustive(id)
        ndiff = set(n1).symmetric_difference(set(n2))
        if len(ndiff) > 0:
            f += 1
            print "neighbours for node ", id, " differ by ", ndiff, "    ( n1 =", n1, ")"
    print "neighbour test successful for", c - f, "out of", c, "nodes"
test_main()
