from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve
import numpy as np

from pypum.pum.assembler import Assembler
from pypum.pum.dofmanager import DofManager
from pypum.pum.pubasis import PUBasis
from pypum.pum.pu import PU
from pypum.pum.tensorquadrature import TensorQuadrature
from pypum.pum.tensorproduct import TensorProduct
from pypum.pum.weightfunctions import Spline, Monomial
from pypum.utils.box import Box
from pypum.utils.ntree import nTree
from pypum.utils.testing import *

import logging
logger = logging.getLogger(__name__)


def lhs(A, idx1, idx2, basis1, basis2, quad, intbox):
    # NOTE: the quadrature degree should depend on the weight function, the basis degree, coefficients and the equation 
    basisdim1 = len(basis1)
    basisdim2 = len(basis2)
    tx, w = quad.transformed(intbox, basisdim1)
    # TODO
    for j in range(idx1, idx1 + basisdim1):
        for k in range(idx2, idx2 + basisdim2):
            A[j, k] = 1.0

def rhs(b, idx2, basis2, quad, intbox):
    basisdim2 = len(basis2)
    tx, w = quad.transformed(intbox, basisdim2)
    # TODO
    for k in range(idx2, idx2 + basisdim2):
        b[k] = k

def test_assembler():
    # setup discretisation
    # --------------------
    # setup PU
    scaling = 1.3
    bbox = Box(((0, 1), (0, 1)))
    tree = nTree(bbox=bbox)
    weightfunc = TensorProduct([Spline(3)] * bbox.dim)
    pu = PU(tree, weightfunc=weightfunc, scaling=scaling)
    pu.tree.refine(1)
    # setup monom basis
    maxdegree = 1
    basis1d = [Monomial(k) for k in range(maxdegree + 1)]
    basis = TensorProduct.create_basis(basis1d, bbox.dim)
    # setup PU basis
    basis = PUBasis(pu, basis)
    # setup dof manager
    ids = [id for id in tree.leafs()]
    dof = DofManager(ids, basis)
    # setup quadrature
    quad = TensorQuadrature()
    # setup assembler
    asm = Assembler(tree, basis, dof, quad, scaling)
    
    # assemble problem
    # ----------------
    N = dof.dim()
    logger.info("system has dimension " + str(N))
    A = lil_matrix((N, N))
    b = np.zeros(N)
    asm.assemble(A, b, lhs, rhs, symmetric=True)

    # solve system
    # ------------
    A = A.tocsr()
#    A = A.tocsc()
    x = spsolve(A, b)

    print A.todense()
    print b
    print x

    # test dense solve since sparse seems to have issues...
#    x = solve(A.todense(), b)
#    print x
    
    # plot solution
    # TODO

test_main()
