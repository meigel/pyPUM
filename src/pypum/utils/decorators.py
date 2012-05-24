"""Common decorators for spuq."""  

from decorator import decorator, getattr_

def copydocs(cls):
    """A decorators that copies docstrings of functions from base
    classes.

    When base class method are overridden in derived classes, it is
    often not necessary to alter the docstring of those methods, as
    usually only the implementation has changed, but not the
    interface. How the implementatin has changed or is now
    implemented, can usually be inferred from the class
    description. Using `copydocs` then, saves time and removes
    duplication of docstrings.

    `copydocs` applies to a class and copies all the functions
    docstrings from base classes if none were specified. The lookup is
    depth first through the inheritance tree.
    """

    def _find_doc(cls, attrname):
        """Helper function for copydoc/copydocs"""
        attr = getattr(cls, attrname, None)
        func = getattr(attr, "__func__", None) 
        doc = getattr(func, "__doc__", None) 
        if doc is not None:
            return doc

        for b in cls.__bases__:
            doc = _find_doc(b, attrname)
            if doc is not None:
                return doc

        return doc

    for attrname in cls.__dict__:
        attr = getattr(cls, attrname)
        func = getattr(attr, "__func__", None) 
        if func is not None and getattr(func, "__doc__") is None:
            func.__doc__ = _find_doc(cls, attrname)
    return cls

@decorator
def cache(func, *args):
    cache_dict = getattr_(func, "cache_dict", dict) 
    # cache_dict is created at the first call
    if args in cache_dict:
        return cache_dict[args]
    else:
        result = func(*args)
        cache_dict[args] = result
        return result


class EmptySlot(object):
    pass
Empty = EmptySlot()
def simple_int_cache(size):
    def mk_int_cache():
        return size * [Empty]

    @decorator
    def cached_func(func, n):
        int_cache = getattr_(func, "int_cache", mk_int_cache) 
        if not (0 <= n < size):
            return func(n)
        if int_cache[n] is Empty:
            int_cache[n] = func(n)
        return int_cache[n]
    return cached_func


# Class decorator to fill 
# 
def total_ordering(cls):
    """Class decorator that fills-in missing ordering methods.

    Taken from http://code.activestate.com/recipes/576685/ with small modifications.

    This one still has problems: see
    http://regebro.wordpress.com/2010/12/13/python-implementing-rich-comparison-the-correct-way/
    """
    convert = {
        '__lt__': [('__gt__', lambda self, other: other < self),
                   ('__le__', lambda self, other: not other < self),
                   ('__ge__', lambda self, other: not self < other)],
        '__le__': [('__ge__', lambda self, other: other <= self),
                   ('__lt__', lambda self, other: not other <= self),
                   ('__gt__', lambda self, other: not self <= other)],
        '__gt__': [('__lt__', lambda self, other: other > self),
                   ('__ge__', lambda self, other: not other > self),
                   ('__le__', lambda self, other: not self > other)],
        '__ge__': [('__le__', lambda self, other: other >= self),
                   ('__gt__', lambda self, other: not other >= self),
                   ('__lt__', lambda self, other: not self >= other)]
    }
    if False and hasattr(object, '__lt__'):
        # this doesn't seem to work in Python 2.6
        roots = [op for op in convert if getattr(cls, op) is not getattr(object, op)]
    else:
        roots = set(dir(cls)) & set(convert)
    assert roots, 'must define at least one ordering operation: < > <= >='
    root = max(roots)       # prefer __lt __ to __le__ to __gt__ to __ge__
    for opname, opfunc in convert[root]:
        if opname not in roots:
            opfunc.__name__ = opname
            opfunc.__doc__ = getattr(int, opname).__doc__
            setattr(cls, opname, opfunc)
    return cls





# http://code.activestate.com/recipes/577689-enforce-__all__-outside-the-import-antipattern/
import sys
import types
import warnings

class EncapsulationWarning(RuntimeWarning): pass

class ModuleWrapper(types.ModuleType):
    def __init__(self, context):
        self.context = context
        super(ModuleWrapper, self).__init__(
                context.__name__,
                context.__doc__)

    def __getattribute__(self, key):
        context = object.__getattribute__(self, 'context')
        if key not in context.__all__:
            warnings.warn('%s not in %s.__all__' % (key, context.__name__),
                          EncapsulationWarning,
                          2)
        return context.__getattribute__(key)

#import example
#sys.modules['example'] = ModuleWrapper(example)



def dump_args(func):
    "This decorator dumps out the arguments passed to a function before calling it"
    argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
    fname = func.func_name
    def echo_func(*args,**kwargs):
        print fname, ":", ', '.join(
            '%s=%r' % entry
            for entry in zip(argnames,args) + kwargs.items())
        return func(*args, **kwargs)
    return echo_func
