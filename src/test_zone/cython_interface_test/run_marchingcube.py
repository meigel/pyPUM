import marchingcube_py as mc
import numpy as np

def test_cutcell(dim, order):
    from random import randint
    assert dim == 2 or dim == 3
    N = 4 if dim == 2 else 6
    
    # set an arbitrary number >0 of vertices to 1
    S = randint(1, N)
#    vertex_vals = np.arange(N, dtype=int)
    vertex_vals = np.zeros(N, dtype=int)
    while S > 0:
        i = randint(0, N - 1)
        if vertex_vals[i] == 0:
            vertex_vals[i] = 1
            S -= 1
    
    print "TEST decompose in", dim, "dimensions with vertices", vertex_vals, "\n"
    cell_data = np.arange(1000 * dim * order, dtype=np.float64)
    facet_data = np.arange(1000 * dim * order, dtype=np.float64)
    m, n = mc.decompose(dim, vertex_vals, order, cell_data, facet_data)
    print "decompose returned %i volume quadrature points and %i facet quadrature points" % (m, n)
    
    # parse data
    size_pw = dim + 1
    cell_pw = [(cell_data[i * size_pw:i * size_pw + dim], cell_data[i * size_pw + dim]) for i in range(m)]
    facet_pw = [(facet_data[i * size_pw:i * size_pw + dim], cell_data[i * size_pw + dim]) for i in range(n)]
    
    # print out
    print "\nCELL QUADRATURE:"
    for i, pw in enumerate(cell_pw):
        p, w = pw
        print "\t%i: point %s    weight %f" % (i, str(p), w)
    print "\nFACET QUADRATURE:"
    for i, pw in enumerate(facet_pw):
        p, w = pw
        print "\t%i: point %s    weight %f" % (i, str(p), w)
        
 
# test quadrature decomposition
# =============================
 
N = 1
sepstr = "\n" + "*"*50
for i in range(N):
    print  sepstr + "\n************* 2D test #%i" % (i + 1) + sepstr
    test_cutcell(dim=2, order=3)

for i in range(N):
    print sepstr + "\n************* 3D test #%i" % (i + 1) + sepstr
    test_cutcell(dim=3, order=2)
