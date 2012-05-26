from pypum.utils.type_check import takes, anything, list_of, optional
from pypum.utils.box import Box

import matplotlib as mpl
import logging
logger = logging.getLogger(__name__)


class Node(object):
    """Node of nTree."""
    def __init__(self, bbox, id=None, parent=None):
        self._bbox = bbox
        self._id = id
        self._parent = parent
        self._kids = []
    
    def is_leaf(self):
        try:
            return not any(self._kids)
        except AttributeError:
            return True

    def split(self):
        for bbox in self.bbox.split():
            self._kids.append(Node(bbox, parent=self))
        
    @property
    def parent(self):
        return self._parent
    
    @property
    def kids(self):
        return self._kids
    
    @property
    def bbox(self):
        return self._bbox
    
    @property
    def center(self):
        return self.bbox.center
    
    @property
    def id(self):
        return self._id
    
    def __str__(self):
        return "Node " + str(self.id) + " " + str(self.bbox)


class nTree(object):
    """N-binary tree in arbitrary dimensions."""
    def __init__(self, bbox=None, rootnode=None):
        if bbox is not None:
            rootnode = Node(bbox)
        self.root = rootnode
        
    @property
    def root(self):
        return self._root
    
    @root.setter
    def root(self, rootnode):
        self._root = rootnode
        self._nodes = {}
        self._ids = 0
        self._init_data()
    
    @property
    def bbox(self):
        return self.root.bbox
    
    def _init_data(self, node=None):
        if node is None:
            node = self.root
        def traverse_nodes(root):
            if not root.is_leaf():
                for node in root.kids:
                    traverse_nodes(node)
            else:
                root._id = self._ids
                self._nodes[root.id] = root
                self._ids += 1
        traverse_nodes(node)

    def __iter__(self):
        """Iterator for leafs of tree."""
        for leaf in self.leafs():
            yield leaf
    
    def nodes(self):
        """Node generator."""
        for id in self._nodes.iterkeys():
            yield id
    
    def leafs(self):
        """Leaf generator."""
        for id in self._nodes.iterkeys():
            if not self[id].is_leaf():
                continue
            else:
                yield id

    def find_neighbours(self, id, scaling=1):
        bbox = self[id].bbox
        neighbours = []
        nodes = [self.root]
        while True:
            next_nodes = []
            for node in nodes:
                if node.is_leaf() and node.id != id:
                    # add leaf
                    neighbours.append(node.id)
                else:
                    # check kids
                    for k in node.kids:
                        if bbox.do_intersect(k.bbox, scaling=scaling):
                            next_nodes.append(k)
#                    [next_nodes.append(k) for k in node.kids if bbox.do_intersect(k.bbox, scaling=scaling)] 
            if not next_nodes:
                break
            else:
                nodes = next_nodes
        return neighbours
    
    def find_neighbours_old(self, id, parentlevel=2, scaling=1):
        """Find neighbours of node."""
        assert parentlevel > 0
        node = self[id]
        neighbours = [node]
        for _ in range(parentlevel):
            kids = []
            [map(lambda k: kids.append(k), node.parent.kids) for node in neighbours if node.parent is not None]
            neighbours = [k for k in kids if node.bbox.do_intersect(k.bbox, scaling=scaling)]
#            neighbours = [kid for kid in [node.parent.kids() if kid.intersect(node.bbox, scaling=scaling) for node in neighbours if not node.parent is None]
        return list(set([n.id for n in neighbours if n.id != node.id]))

    def find_neighbours_exhaustive(self, id, scaling=1):
        node = self[id]
        neighbours = [nid for nid in self.leafs() if nid != id and node.bbox.do_intersect(self[nid].bbox, scaling=scaling)]
        return neighbours

    def find_node(self, x):
        """Find node to which point x belongs."""
        assert (not self.root is None) and self.bbox.is_inside(x)
        node = self.root
        while not node.is_leaf():
            for kid in node.kids():
                if kid.bbox.is_inside(x):
                    node = kid
                    break
        return node.id

    def find_overlapping_nodes(self, x, parentlevel=2, scaling=1):
        """Find all nodes which contain point x. The first node returned is the one x belongs to."""
        node = self.find_node(x)
        neighbours = self.find_neighbours(node, parentlevel, scaling)
        neighbours = [nid for nid in neighbours if self[nid].bbox.is_inside(x, scaling=scaling)]
        return [node.id] + neighbours

    def refine(self, levels=1):
        for _ in range(levels):
            ids = [id for id in self.leafs()]
            for id in ids:
                self.split(id)

    def split(self, id):
        """Split node with id by bisection in all dimensions."""
        node = self[id] 
        node.split()
        # assign ids
        for kid in node.kids:
            kid._id = self._ids
            self._ids += 1
        # update node map
        self._init_data(node=node)

    @property
    def size(self):
        """Return number leafs."""
        return len([id for id in self.leafs()])

    @property
    def totalsize(self):
        """Return number nodes."""
        return len([id for id in self.nodes()])

    def __getitem__(self, id):
        """Return node with id."""
        return self._nodes[id]

    def __str__(self):
        return "nTree of size", self.totalsize, " has", self.size, " leafs"

    def plot(self):
        pass
        # plot rectangles
#        try:
#            from matplotlib.pyplot import figure, show, text
#            from matplotlib.patches import Rectangle
#            fig = figure()
#            ax = fig.add_subplot(111, aspect='equal')
#            ax.set_xlim(-1, 1)
#            ax.set_ylim(-1, 1)
#            for i, r in enumerate(R):
#        #        print 'current r ', r
#                if len(r) > 4 and r[4] == 1:  # leaf node
#                    rect = Rectangle((r[0], r[1]), r[2], r[3], facecolor='r', alpha=0.5, lw=3)
#                    if len(r) > 5:
#                        nodeid = int(r[5])
#                    else:
#                        nodeid = i
#                    text(r[0] + r[2] / 2, r[1] + r[3] / 2, nodeid,
#                         horizontalalignment='center',
#                         verticalalignment='center',
#                         fontsize=8)
#        
#                else:
#                    rect = Rectangle((r[0], r[1]), r[2], r[3], facecolor='w', alpha=0.1, hatch='o', lw=1)
#                ax.add_artist(rect)
#            show()
#        except Exception as ex:
#            print 'plotting not supported (probably missing matplotlib)'
#            print ex
