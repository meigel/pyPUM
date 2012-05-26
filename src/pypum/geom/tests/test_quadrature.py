from pypum.pum.tensorquadrature import TensorQuadrature
from pypum.utils.testing import *


def test_quadrature():
    from pypum.utils.box import Box
    Q = TensorQuadrature()    
    b2 = Box(((0, 1), (0.5, 1)))
    x, w = Q.transformed(b2, 3)
    print x
    print sum(w), w

    b3 = Box(((0, 0.5), (0.5, 1), (-1, 1)))
    x, w = Q.transformed(b3, 2)
    print x
    print sum(w), w


test_main()
