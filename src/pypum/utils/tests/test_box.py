from pypum.utils.box import Box
from pypum.utils.testing import *

def test_box():
    p1 = (0, 1)
    p2 = (-1, 0.5)
    p3 = (0, 1.5)
    p4 = (1,2)
    # 1d
    b1a = Box((p1,))
    b1b = Box((p2,))
    b1c = Box((p3,))
    b1d = Box((p4,))
    print b1a
    # intersection test
    print b1a.do_intersect(b1a)
    print b1a.do_intersect(b1b)
    print b1b.do_intersect(b1a)
    print b1a.do_intersect(b1c)
    print b1d.do_intersect(b1b)
    print b1d.do_intersect(b1b, scaling=2)
    # intersections
    print b1a.intersect(b1a)
    print b1a.intersect(b1b)
    print b1d.intersect(b1b, scaling=2)
    
    # 2d
    b2a = Box((p1, p1))
    b2b = Box((p1, p3))
    print b2a
    b2a.do_intersect(b2b)
    print b2a.intersect(b2b)
    
    # 3d
    b3a = Box((p1, p2, p3))

test_main()
