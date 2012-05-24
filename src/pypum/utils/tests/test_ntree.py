from pypum.utils.box import Box
from pypum.utils.ntree import nTree
from pypum.utils.testing import *

def test_ntree():
    bbox = Box(((0,1),(0,1)))
    tree = nTree(bbox)
    tree.refine(2)
    print "tree has", tree.totalsize, " nodes and", tree.size, " leafs:"
    for id in tree.nodes():
        print "\tnode id =", id
    for id in tree.leafs():
        print "\tleaf id =", id

test_main()
