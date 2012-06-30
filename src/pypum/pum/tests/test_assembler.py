from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

from pypum.pum.assembler import Assembler
from pypum.pum.dofmanager import DofManager
from pypum.geom.tensorquadrature import TensorQuadrature
from pypum.pum.weightfunctions import Spline, Monomial
from pypum.utils.ntree import nTree
from pypum.utils.testing import *


def lhs(basis1, basis2, quad, intbox):
    degree = basis1.dim
    tx, w = quad.transformed(inbox, degree)
    TODO

def rhs(basis2, quad, intbox):
    degree = basis1.dim
    tx, w = quad.transformed(inbox, degree)
    TODO

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
    N = dof.dim
    A = lil_matrix((N, N))
    b = np.zeros(N)
    asm.assembly_symmetric(A, b, lhs, rhs)

    # solve system
    # ------------
    A = A.tocsr()
    x = spsolve(A, b)
    print x
    
    # plot solution
    # TODO

test_main()
