from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve, bicg
#from scipy.sparse.linalg.dsolve import linsolve
from numpy.linalg import solve
import numpy as np

from pypum.apps.discretisation import ReactionDiffusion
from pypum.pum.assembler import Assembler
from pypum.pum.dofmanager import DofManager
from pypum.pum.basis import BasisSet
from pypum.pum.pu import PU
from pypum.pum.tensorquadrature import TensorQuadrature
from pypum.pum.tensorproduct import TensorProduct
from pypum.pum.weightfunctions import Spline, Monomial
from pypum.utils.box import Box
from pypum.utils.ntree import nTree
from pypum.utils.testing import *

import logging
logger = logging.getLogger(__name__)

# setup logging
# log level and format configuration
LOG_LEVEL = logging.DEBUG
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=__file__[:-2] + 'log', level=LOG_LEVEL,
                    format=log_format)

# set initial cover refinements
refines = 1
# set maximal polynomial degree of patch basis
maxdegree = 1
# plotting flag
plot_patches = False


#class _testBasis(object):
#    """Basis implementation for performance testing."""
#    def __init__(self, f=lambda x:sum(x), dx=lambda x:np.ones(len(x))):
#        self._f = f
#        self._dx = dx
#        
#    def __call__(self, x):
#        if isinstance(x, (list, tuple)):
#            return np.array([self._f(cx) for cx in x])
#        else:
#            return self._f(x)
#        
#    def dx(self, x):
#        if isinstance(x, (list, tuple)):
#            return [self._dx(cx) for cx in x]
#        else:
#            return self._dx(x)


def test_assembler():
    # setup discretisation
    # --------------------
    # setup PU
    scaling = 1.25
    bbox = Box([[0, 1], [0, 1]])
    tree = nTree(bbox=bbox)
    weightfunc = TensorProduct([Spline(3)] * bbox.dim)
    pu = PU(tree, weightfunc=weightfunc, scaling=scaling)
    pu.tree.refine(refines)
    if plot_patches:
        pu.tree.plot2d()
    # setup monomial basis
    basis1d = [Monomial(k) for k in range(maxdegree + 1)]
    basis = TensorProduct.create_basis(basis1d, bbox.dim)
    # setup PU basis
    basisset = BasisSet(basis)
#    basisset = BasisSet([_testBasis() for _ in range(maxdegree + 1)])
    # setup dof manager
    ids = [id for id in tree.leafs()]
    dof = DofManager(ids, basisset)
    # setup quadrature
    quad = TensorQuadrature()
    # setup assembler
    asm = Assembler(tree, pu, basisset, dof, quad, scaling)
    
    # assemble problem
    # ----------------
    N = dof.dim()
    logger.info("system has dimension " + str(N))
    A = lil_matrix((N, N))
    b = np.zeros(N)
    PDE = ReactionDiffusion(D=1, r=1)
    asm.assemble(A, b, lhs=PDE.lhs, rhs=PDE.rhs, symmetric=True)

    # solve system
    # ------------
    A = A.tocsr()
#    A = A.tocsc()
    x = spsolve(A, b)
#    x = bicg(A, b)

    print A.todense()
    print b
    print x
    print x.shape

    # test dense solve since sparse seems to have issues...
#    x = solve(A.todense(), b)
#    print x
    
    # plot solution
    # TODO

test_main()
