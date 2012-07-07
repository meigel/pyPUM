from functools import partial

from pypum.utils.box import Box

import logging
logger = logging.getLogger(__name__)


def _box_boundary(intbox, bbox):
    bbnd = []
    # check if box intersects with boundary box of tree
    if bbox.do_intersect(intbox):
        # test box with all sides of box
        sides = []
        for d in range(bbox.dim):
            for e in range(2):
                pos = bbox.pos
                pos[d][e] = pos[d][(e + 1) % 2] 
                sides.append(Box(pos))
        bbnd = [intbox.intersect(side) for side in sides if intbox.do_intersect(side)]
    return bbnd


class Assembler(object):
    """Assemble discrete problem."""
    def __init__(self, tree, basis, dof, quad, scaling, boundary=None):
        if boundary is None:
            boundary = partial(_box_boundary, bbox=tree.bbox)
        self._tree = tree
        self._basis = basis
        self._dof = dof
        self._quad = quad
        self._scaling = scaling
        self._boundary = boundary
    
    def assemble(self, A=None, b=None, lhs=None, rhs=None, ids=None, symmetric=True):
        assert lhs is None or A is not None
        assert rhs is None or b is not None
        if ids is None:
            ids = [id for id in self._dof.indices()]
        for id1 in ids:
            bbox1 = self._tree[id1].bbox
            nids = self._tree.find_neighbours(id1, scaling=self._scaling)
            nids = [nid for nid in nids if nid < id]   # assume symmetric operator
            nids.append(id1)
            for id2 in nids:
                bbox2 = self._tree[id2].bbox
                if self._tree.bbox.do_intersect(bbox2, scaling=self._scaling):
                    intbox = self._tree.bbox.intersect(bbox2, scaling=self._scaling)
                    # check for boundary patch
                    bndbox = self._boundary(intbox)
                    if bndbox:
                        logger.debug("assembling boundary patches " + str(id1) + " " + str(id2) + " with intersection box " + str(intbox) + "and " + str(len(bndbox)) + " boundaries")
                    else:
                        logger.debug("assembling patches " + str(id1) + " " + str(id2) + " with intersection box " + str(intbox))
                    idx1 = self._dof[id1]
                    idx2 = self._dof[id2]
                    logger.debug("\tindices are " + str(idx1) + " " + str(idx2))
                    if lhs is not None:
                        lhs(A, idx1, idx2, self._basis[id1], self._basis[id2], self._quad, intbox, bndbox)
                    if rhs is not None:
                        rhs(b, idx2, self._basis[id2], self._quad, intbox, bndbox)
        return A, b
