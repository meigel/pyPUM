from __future__ import division
import itertools as iter
from pypum.utils.box import Box
from math import copysign

# define levelset for circle at (0.5,0.5) with radius r=0.5
def circle_levelset(x):
    d = (x[0]-0.5)**2 + (x[1]-0.5)**2 - 0.25
    return -1*copysign(1, d)

# number patches per dimension
N = 5
# create set of patches for [0,1]x[0,1]
dx = 1/N
idx = [i for i in iter.product(range(N-1), repeat=2)]
cover = [Box([ [xi*dx, (xi+1)*dx], [yi*dx, (yi+1)*dx] ]) for xi, yi in idx]

for i, p in enumerate(cover):
    print "patch",i,":",p

print "level set at (0.25,0.25) =", circle_levelset((0.25,0.25))
print "level set at (0.75,1.0) =", circle_levelset((0.75,1.0))

# TODO:
# * identification of patches intersected by implicit surface
# * removal of patches with small intersection area
# * projection of patch centers onto level set
# * appropriate enlargement of patches
# * visualisation of level set and patches (use matplotlib, also see pypum.utils.ntree)
