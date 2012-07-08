import numpy as np

def inner(v1, v2):
    """Vectorised scalar product."""
#    dim1 = len(v1)
#    dim2 = len(v2)
#    if dim1 > dim2:
#        assert dim2 == 1
#        v2 = [v2] * dim1
#    elif dim2 > dim1: 
#        assert dim1 == 1
#        v1 = [v1] * dim2
    val = np.array([np.inner(vec1, vec2) for vec1, vec2 in zip(v1, v2)])
    return val
