import sys
import os.path as op

from numpy.testing import *
from numpy.testing.nosetester import import_nose

class TestCase(TestCase):

    def myAssertIsInstance(self, obj, cls, msg=None):
        """Same as self.assertTrue(isinstance(obj, cls)), with a nicer
        default message."""
        if not isinstance(obj, cls):
            #standardMsg = '%r is not an instance of %r' % (obj, cls)
            #self.fail(self._formatMessage(msg, standardMsg))
            # TODO: remove this ugly hack
            assert_true(type(obj) == cls)

    def failUnlessRaises(self, excClass, callableObj, *args, **kwargs):
        """Fail unless an exception of class excClass is thrown
           by callableObj when invoked with arguments args and keyword
           arguments kwargs. If a different type of exception is
           thrown, it will not be caught, and the test case will be
           deemed to have suffered an error, exactly as for an
           unexpected exception.
        """
        try:
            callableObj(*args, **kwargs)
        except excClass, e:
            return e
        else:
            if hasattr(excClass, '__name__'): excName = excClass.__name__
            else: excName = str(excClass)
            raise self.failureException, "%s not raised" % excName

    def dummy(self):
        pass

    assertRaises = failUnlessRaises

_tc = TestCase("dummy")

#assert_equal = _tc.assertEqual
assert_not_equal = _tc.assertNotEqual
assert_true = _tc.assertTrue
assert_false = _tc.assertFalse

if sys.hexversion >= 0x02070000:
    assert_is = _tc.assertIs
    assert_is_not = _tc.assertIsNot
    assert_is_none = _tc.assertIsNone
    assert_is_not_none = _tc.assertIsNotNone
    assert_in = _tc.assertIn
    assert_not_in = _tc.assertNotIn
    assert_is_instance = _tc.assertIsInstance
    assert_not_is_instance = _tc.assertNotIsInstance
else:
    assert_is_instance = _tc.myAssertIsInstance


assert_raises = _tc.assertRaises
if sys.hexversion >= 0x02070000:
    assert_raises_regexp = _tc.assertRaisesRegexp

#assert_almost_equal = _tc.assertAlmostEqual
#assert_not_almost_equal= _tc.assertNotAlmostEqual
if sys.hexversion >= 0x02070000:
    assert_greater = _tc.assertGreater
    assert_greater_equal = _tc.assertGreaterEqual
    assert_less = _tc.assertLess
    assert_less_equal = _tc.assertLessEqual
    assert_regexp_matches = _tc.assertRegexpMatches
    assert_not_regexp_matches = _tc.assertNotRegexpMatches
    assert_items_equal = _tc.assertItemsEqual

def get_base_module(fullfile):
    # get module name from filename
    (path, filename) = op.split(op.abspath(fullfile))
    if not (filename.startswith("test_") and
            filename.endswith(".py")):
        return None
    modname = filename[5:-3]

    (path, dirname) = op.split(path)
    if dirname != "tests":
        return None

    packlist = []
    while True:
        (path, dirname) = op.split(path)
        if dirname == "src":
            break
        if not dirname:
            return None
        packlist.insert(0, dirname)
    return ".".join(packlist) + "." + modname

@dec.setastest(False)
def test_main(with_coverage=False):
    """``main`` function for test modules"""
    frame = sys._getframe(1)
    mod_name = frame.f_locals.get('__name__', None)
    if mod_name == "__main__":
        file_to_run = frame.f_locals.get('__file__', None)
        if file_to_run is not None:
            argv = ['', file_to_run, '-vv', '-s']
            if with_coverage:
                module = get_base_module(file_to_run)
                if module is not None:
                    argv = argv + ['--with-coverage', '--cover-package', module]
                    # reload module so it can be instrumented by coverage
                    sys.modules.pop(module)
                    # reloading here 
                    #reload(sys.modules[module])
            import_nose().run(argv=argv)
            #run_module_suite(file_to_run)
        else:
            print "Could not determine the file name of the current file. Probably this function has been called by execfile."

skip_if = dec.skipif
slow = dec.slow
no_test = dec.setastest(False)
no_test = no_test(no_test)

