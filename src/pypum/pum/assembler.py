

class Assembler(object):
    """Assemble discrete problem."""
    def __init__(self, tree, basis, dof, quad, scaling):
        self._tree = tree
        self._basis = basis
        self._dof = dof
        self._quad = quad
        self._scaling = scaling
    
    def assemble_symmetric(self, A=None, b=None, lhs=None, rhs=None, ids=None):
        assert lhs is None or A is not None
        assert rhs is None or b is not None
        if ids is None:
            ids = [id for id in self._dof.indices()]
        for id1 in ids:
            bbox1 = self._tree[id1].bbox
            nids = self._tree.find_neighbours(id1, scaling=scaling)
            nids = [nid for nid in nids if nid < id]   # assume symmetric operator
            nids.append(id1)
            for id2 in nids:
                bbox2 = self._tree[id2].bbox
                if bbox.do_intersect(bbox2, scaling=scaling):
                    intbox = bbox.intersect(bbox2, scaling=scaling)
                    idx1 = self._dof[id1]
                    idx2 = self._dof[id2]
                    if lhs is not None:
                        lhs(A, idx1, idx2, self._basis[id1], self._basis[id2], self._quad, intbox)
                    if rhs is not None:
                        rhs(b, idx2, self._basis[id2], self._quad, intbox)
        return A, b
