import numpy as np
from math import*
def _validate_vector(u, dtype=None):
    # XXX Is order='c' really necessary?
    u = np.asarray(u, dtype=dtype, order='c').squeeze()
    # Ensure values such as u=1 and u=[1] still return 1-D arrays.
    u = np.atleast_1d(u)
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    return u

u = [3, 45, 7, 2]
v = [2, 54, 13, 15]
u = _validate_vector(u)
v = _validate_vector(v)
dist =  round(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)),3)
print dist