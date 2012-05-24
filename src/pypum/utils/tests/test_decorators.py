import numpy as np

from spuq.utils.testing import *
from spuq.utils.decorators import *
import inspect

class TestCopyDocs(TestCase):
    
    class A(object):
        def x(self):
            """A.x"""
            pass
        def y(self):
            pass
        def z(self):
            pass

    @copydocs
    class Bc(A):
        def x(self):
            pass
        def y(self):
            """B.y"""
            pass

    def test_copy_doc1(self):
        assert_equal(self.A.x.__doc__, "A.x")
        assert_equal(self.Bc.x.__doc__, "A.x")
        assert_equal(self.Bc.y.__doc__, "B.y")

    class B(A):
        def x(self):
            pass
        def y(self):
            """B.y"""
            pass

    @copydocs
    class C(B):
        def x(self):
            pass
        def z(self):
            """C.z"""
            pass

    def test_copy_doc2(self):
        assert_equal(self.C.x.__doc__, "A.x")
        assert_equal(self.C.y.__doc__, "B.y")
        assert_equal(self.C.z.__doc__, "C.z")

    @copydocs
    class Cc(Bc):
        def x(self):
            """C.x"""
            pass
        def z(self):
            """C.z"""
            pass

    def test_copy_doc3(self):
        assert_equal(self.Cc.x.__doc__, "C.x")
        assert_equal(self.Cc.y.__doc__, "B.y")
        assert_equal(self.Cc.z.__doc__, "C.z")


def test_int_cache():
    global call_count
    call_count = 0
    @simple_int_cache(10)
    def foo(n):
        """foo docu"""
        global call_count
        call_count = call_count + 1
        return 10*n

    @simple_int_cache(10)
    def foo2(n):
        return 20*n

    # first pass, evaluate some arguments within and without range
    assert_equal(foo(-1), -10)
    assert_equal(foo(0), 0)
    assert_equal(foo(2), 20)
    assert_equal(foo(9), 90)
    assert_equal(foo(10), 100)

    # second pass, call_count should only increase for values outside range
    assert_equal(call_count, 5)
    assert_equal(foo(-1), -10)
    assert_equal(call_count, 6)
    assert_equal(foo(0), 0)
    assert_equal(call_count, 6)
    assert_equal(foo(2), 20)
    assert_equal(call_count, 6)
    assert_equal(foo(9), 90)
    assert_equal(call_count, 6)
    assert_equal(foo(10), 100)
    assert_equal(call_count, 7)

    assert_equal(foo(5), 50)
    assert_equal(foo2(5), 100)
    assert_equal(foo2(6), 120)
    assert_equal(foo(6), 60)


    # make sure decorator copies the docu
    assert_equal( foo.__doc__, "foo docu")
    def foo2(n): pass
    assert_equal( inspect.getargspec(foo), inspect.getargspec(foo2))



def test_cache():
    global call_count
    call_count = 0
    @cache
    def foo(s, n, d):
        """foo docu"""
        global call_count
        call_count += 1
        return "%s %s %s" % (s, n, d)

    # first pass, evaluate some arguments within and without range
    assert_equal(foo("x", 1, 4.5), "x 1 4.5")
    assert_equal(foo("x", 2, 5.5), "x 2 5.5")

    # second pass, call_count should only increase for values outside range
    assert_equal(call_count, 2)
    assert_equal(foo("x", 1, 4.5), "x 1 4.5")
    assert_equal(call_count, 2)
    assert_equal(foo("x", 2, 5.5), "x 2 5.5")
    assert_equal(call_count, 2)
    assert_equal(foo("x", 3, 5.5), "x 3 5.5")
    assert_equal(call_count, 3)

    # make sure decorator copies the docu
    assert_equal( foo.__doc__, "foo docu")
    def foo2(s, n, d): pass
    assert_equal( inspect.getargspec(foo), inspect.getargspec(foo2))


def test_total_ordering():
    @total_ordering
    class Foo(object):
        def __init__(self,a):
            self.a = a
        def __le__(self, other):
            return self.a<=other.a
    foo_1a = Foo(1)
    foo_1b = Foo(1)
    foo_2 = Foo(2)
    assert_true(foo_1a <= foo_1b)
    assert_true(foo_1a >= foo_1b)
    assert_true(not foo_1a < foo_1b)
    assert_true(not foo_1a > foo_1b)
    assert_true(foo_1a <= foo_2)
    assert_true(not foo_1a >= foo_2)
    assert_true(foo_1a < foo_2)
    assert_true(not foo_1a > foo_2)

test_main()
