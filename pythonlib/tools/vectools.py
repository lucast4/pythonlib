""" working with vectors, generally 2d, generally realted to drawing stuff"""
import numpy as np

from pythonlib.tools.distfunctools import modHausdorffDistance

# def _convert_to_rel_units(x,y):
#     # converts from x,y to relative units from top-left corner
#     # top left = (0,0)
#     # bottom right = (1,1)
# #             print("---")
# #             print((x,y))
# #             print((xmin, xmax))
# #             print((ymin, ymax))
# #             print("---")
#     x = (x - xmin)/(xmax-xmin)
#     y = (ymax - y)/(ymax-ymin)
    
#     return (x,y)

# def get_lefttop_extreme(t):
#     """get x and y extreme that is most top left
#     """
#     x = min(t["x_extremes"])
#     y = max(t["y_extremes"])
#     x,y = _convert_to_rel_units(x,y)
    # return x,y


def get_vector_between_points(A,B):
    # vector from A --> B
    x = B[0] - A[0]
    y = B[1] - A[1]
    return (x,y)
    
# def get_vector_norm(x,y):
#     # norm of vector (x,y), scaled so that 1 is maximum (corner to corner)
#     dist = ((x**2 + y**2)**0.5)/(2**0.5) #
#     return dist

def get_dot_between_unit_vectors(u, v):
    # ignores length, just get angle.
    if not isinstance(u, np.ndarray) or not isinstance(v, np.ndarray):
        u = np.array(u)
        v = np.array(v)
    c = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v) 
    return c
    

# def get_projection_along_diagonal(x,y):
#     # if x,y is vector with origin at top left, with positive values pointing towards bottom right, then
#     # this gives projection along diagonal. 0 means is at top-left. 1 means is at bottom right.
    
#     proj = (x*1+y*1)/(2**0.5) # get projection
#     proj = proj/(2**0.5) # normalize
#     return proj

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_angle(v):
    from math import pi
    """given a vector, gets angle from (1,0), 
    in domain [0, 2pi)"""
    a = angle_between([1,0], v)
#     print(a)
    if v[1]<0:
        a = 2*pi-a
    return a

def angle_diff(a1, a2):
    """ get difference between two angles.
    will give smallest absolute difference.
    - a1 and a2 in radians. can be - or +
    """
    from math import pi
    
    a = abs(a1-a2)
    a = a%(2*pi)
    if a <= pi:
        return a
    elif a > pi:
        assert a <= 2*pi
        return 2*pi - a


