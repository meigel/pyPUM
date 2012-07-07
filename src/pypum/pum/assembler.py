import logging
logger = logging.getLogger(__name__)


def _noboundary(intbox):
    return []


class Assembler(object):
    """Assemble discrete problem."""
    def __init__(self, tree, basis, dof, quad, scaling, boundary=_noboundary):
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
                    if self._tree.bbox.do_intersect(bbox2, scaling=[1, self._scaling]):
                        bndbox = self._boundary(intbox)
                        logger.debug("assembling boundary patches " + str(id1) + " " + str(id2) + " with intersection box " + str(intbox) + "and " + str(len(bndbox)) * " boundaries")
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
