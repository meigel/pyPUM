import numpy as np

def inner(v1, v2):
    """Vectorised scalar product."""
    val = np.array([np.inner(vec1, vec2) for vec1, vec2 in zip(v1, v2)])
    return val
