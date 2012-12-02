import marchingcube_py as mc
import numpy as np

def levelset_func(data):
    dim = len(data)
    print "\t levelset_func: ", dim, data
    return 10*sum(data)

# callback test
print "\n", "*"*80, "\nTEST callback\n"
r = mc.test_callback(levelset_func)
print "result =", r

# data transfer test
data = np.arange(100, dtype=np.float64)
print "\n", "*"*80, "\nTEST decompose\n"
r = mc.decompose(data)
print "result =", r
print "DATA", data
