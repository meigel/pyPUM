#!/usr/bin/env python
# http://code.activestate.com/recipes/577065-type-checking-function-overloading-decorator/

from functools import wraps
from type_check import takes, returns

class InputParameterError(Exception): pass
def overloaded(func):
    @wraps(func)
    def overloaded_func(*args, **kwargs):
        for f in overloaded_func.overloads:
            try:
                return f(*args, **kwargs)
            except (InputParameterError, TypeError):
                pass
        else:
            raise TypeError("No compatible signatures")

    def overload_with(func):
        overloaded_func.overloads.append(func)
        return overloaded_func
    overloaded_func.overloads = [func]
    overloaded_func.overload_with = overload_with
    return overloaded_func

#############


if __name__ == '__main__':
    @overloaded
    def a():
        print 'no args a'
        pass
    @a.overload_with
    def a(n):
        print 'arged a'
        pass

    a()
    a(4)

    @overloaded
    @returns(int)
    @takes(int, int, float)
    def foo(a, b, c):
        return int(a * b * c)

    @foo.overload_with
    @returns(int)
    @takes(int, float, int, int)
    def foo(a, b, c, d):
        return int(a + b + c)

    @foo.overload_with
    @returns(int)
    @takes(int, float, c=int)
    def foo(a, b, c):
        return int(a + b + c)

    print foo(2, 3, 4.)
    print foo(10, 3., c=30)
    print foo(1, 9., 3, 3)
    print foo('string')
