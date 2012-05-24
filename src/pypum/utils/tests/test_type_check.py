from spuq.utils.testing import *
from spuq.utils.type_check import *

def test_simple():
    @takes(int)
    def foo(a):
        pass

    foo(1)
    assert_raises(TypeError, foo, "hallo")


def test_optional():
    @takes(int, optional(float),optional(int))
    def foo(a,b=None,c=None):
        return b

    assert_equal(foo(1),  None)
    assert_equal(foo(1, 3.0), 3)
    assert_equal(foo(1, b=3.0), 3)
    assert_equal(foo(1, c=7, b=3.0), 3)
    assert_equal(foo(1, b=3.0, c=7), 3)
    assert_equal(foo(1, c=7), None)
    assert_raises(TypeError, foo, 1, 3)


def test_optional2():
    @takes(int, b=optional(float))
    def foo(a,b=None):
        return b

    assert_equal(foo(1),  None)
    assert_equal(foo(1, 3.0), 3)
    assert_equal(foo(1, b=3.0), 3)
    # can't raise for this case, better avoid named
    # parameters in type checker list
    #assert_raises(TypeError, foo, 1, 3)

