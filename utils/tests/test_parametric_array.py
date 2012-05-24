from spuq.utils.parametric_array import ParametricArray
from spuq.utils.testing import *


def test_param_array():
    global no
    no = 0

    def myfunc(i):
        global no
        no = no + 1
        return (no - 1, i)

    pa = ParametricArray(myfunc)

    assert_equal(pa[2], (0, 2))
    assert_equal(pa[0], (1, 0))
    assert_equal(pa[2], (0, 2))
    assert_equal(pa[3], (2, 3))

    assert_equal(len(pa), 4)
    assert_equal(pa[17], (3, 17))
    assert_equal(pa[8], (4, 8))
    assert_equal(pa[17], (3, 17))
    assert_equal(len(pa), 18)


test_main()
