""" working with vectors, generally 2d, generally realted to drawing stuff"""
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.tools.distfunctools import modHausdorffDistance
from math import pi

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

def _convert_list_vector_to_array(vec_as_list):
    """
    Convert list of 2 numbers to np array, shape (2,), a standard
    format for vectors.
    PARAMS:
    - vec_as_list, either list of 2 nums, or vector or 2 nums, shape (1,2), (2,1), or (2,)
    RETURNS:
    - vector, (2,) array of c
    """
    # Convert to np arrays
    if isinstance(vec_as_list, (list, tuple)):
        vector = np.array(vec_as_list)
    else:
        vector = vec_as_list

    # Make its shape (2,)
    if vector.shape in [(1,2), (2,1)]:
        vector = vector.squeeze()

    assert vector.shape == (2,)

    return vector

def average_angle(angles, weights=None):
    """ Given angles in radians, return the mean angle by firrst converting
    all to vectors,. getting mean vector, then convering back to angle.
    NOTE: the mean can be very low norm....
    PARAMS:
    - angles, in radians (where (1,0) is defined to be 0)
    - weights, optional, to weight each angle
    """
    if weights is None:
        vectors = np.stack([get_vector_from_angle(a) for a in angles])
    else:
        vectors = np.stack([w * get_vector_from_angle(a) for a, w in zip(angles, weights)])
    
    assert vectors.shape[1]==2
    assert vectors.shape[0]==len(angles)

    mean_vec = np.mean(vectors, axis=0)

    angle_mean, norm_mean = cart_to_polar(mean_vec[0], mean_vec[1])
    return angle_mean, norm_mean

def average_vectors_wrapper(vectors, length_method):
    """
    Methods for computing average vector:
    - "sum": vector summation (actually returns the mean)
    - "dot": square root of the average dot product between all pairs of vectors.   
    vectors, array (n_data, 2)
    """

    assert vectors.shape[1]==2  

    # First, generic method using vector addition
    mean_vec = np.mean(vectors, axis=0)
    angle_mean, norm_mean = cart_to_polar(mean_vec[0], mean_vec[1])

    if length_method == "sum":
        # Vector summation
        length = norm_mean

    elif length_method == "dot":
        # Square root of the average dot product between all pairs of vectors.
        # - This is high if the vectors are similar angle. 
        # - This is like testing the hypotehsis -- all vectors are random samples from the
        # same distribution of vectors.

        if vectors.shape[0]==1:
            # Need at least 2 rows to do this
            length = np.nan
        
        squared_length_estimates =[]
        for i, vec1 in enumerate(vectors):
            for j, vec2 in enumerate(vectors):
                if j>i:
                    squared_length_estimates.append(np.dot(vec1, vec2))

        # Convert to eucl distance
        squared_length = np.mean(squared_length_estimates)
        length = np.sign(squared_length) * (np.abs(squared_length)**0.5)
    else:
        assert False

    return angle_mean, length
    
def get_vector_from_angle(angle):
    """ Returns unit vector given angle in rads, where
    0rad is (1,0) and goes CCw from there. modulo 2pi
    """
    from math import cos, sin, pi
    return unit_vector((cos(angle), sin(angle)))


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
    """ Returns the unit vector of the vector.  
    returns shape (2,)
    """
    vector = _convert_list_vector_to_array(vector)
    x = vector / np.linalg.norm(vector)
    return x

def cart_to_polar(x, y):
    """ convert from cartesiaon to polar coords
    RETURNS:
    - theta, norm, the anglea nd legngth of vector
    """

    if x==0. and y==0.:
        return 0, 0
    
    theta = get_angle((x,y))
    norm = np.linalg.norm((x,y))
    # print(norm)
    # norm = _convert_list_vector_to_array(norm)

    return theta, norm

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2',
    from v1 to v2, where counterclockwise is positive.
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
    """given a vector, gets angle from (1,0), 
    in domain [0, 2pi)"""
    from math import pi
    v = _convert_list_vector_to_array(v)
    a = angle_between([1,0], v)
    if np.sum(v**2)==0:
        print(v)
        assert False, "cannot compute angle -- length of vector is 0..."
    if v[1]<0:
        a = 2*pi-a
    return a

def angle_clamp_within_range_0_2pi(angle):
    """
    Convert any angle in (-inf, inf) to the same angle in [0, 2pi)

    PARAMS:
    - angle, float
    RETURNS:
    - angle, float

    EXMAPEL":
        x = np.linspace(-10, 10, 100)
        y = [angle_clamp_within_range_0_2pi(a) for a in x]
        fig, ax = plt.subplots()
        ax.plot(x, y, "-x")

    """
    return angle%(2*pi)

def angle_diff_ccw(a1, a2):
    """
    Get difference in angles, i.e., the angle to traverse to get from a1-->a2, 
    returning in range of (0, 2pi)
    PARAMS:
    - a1, a2, angles in radians, in domain (-inf, inf)
    """
    a = a2 - a1
    return angle_clamp_within_range_0_2pi(a)

def angle_diff(a1, a2):
    """ get difference between two angles.
    will give smallest absolute difference.
    - a1 and a2 in radians. can be - or +
    """
    from math import pi
    
    a = np.abs(a1-a2)
    a = a%(2*pi)
    if a <= pi:
        return a
    elif a > pi:
        assert a <= 2*pi
        return 2*pi - a

def angle_diff_vectorized_jax(a1, a2):
    """ get difference between two angles.
    will give smallest absolute difference.
    - a1 and a2 in radians. can be - or +
    - a1 and a2 are (N,1) arrays
    """
    from math import pi
    import jax.numpy as np
    assert False, "not finished coded."
    a = np.abs(a1-a2)
    a = a%(2*pi)
    a[a>pi] = 2*pi - a[a>pi]
    if not np.all(a<=2*pi):
        print(a)
        print(a<=2*pi)
    assert np.all(a<=2*pi)
    return a

def angle_diff_vectorized(a1, a2):
    """ get difference between two angles.
    will give smallest absolute difference.
    - a1 and a2 in radians. can be - or +
    - a1 and a2 are (N,1) arrays
    """
    from math import pi
    
    a = np.abs(a1-a2)
    a = a%(2*pi)
    a[a>pi] = 2*pi - a[a>pi]
    if not np.all(a<=2*pi):
        print(a)
        print(a<=2*pi)
    assert np.all(a<=2*pi)
    return a

# @param angles_all: array of angles (all data must be between [0, 2*pi])
# @param negative_angle: transform data equivalent to angles in [0, negative_angle] to be negative
#
# example: angles_all=[pi/2, pi, 3*pi/2, 7.1*pi/4, 7.2*pi/4], negative_angle= -pi/4
# - only [7.1*pi/4, 7.2*pi/4] are in between 0 and -pi/4
# - they are changed to [-0.9*pi/4, -0.8*pi/4]
#
# originally coded to help with binning arrowheads
# - arrowhead pointing right: angles between [-pi/4, pi/4]; but data is always between [0, 2*pi]
# - solution: transform data between [7*pi/4, 2*pi] to negative equivalents [-pi/4, 0]
def transform_data_to_negative_start(angles_all, negative_angle):
    assert negative_angle<=0, "negative_angle must be negative!"
    temp=np.array(angles_all)
    inds = (temp>=(2*pi+negative_angle)) & (temp<=2*pi) 
    temp[inds] = temp[inds]-2*pi
    return temp


# takes in angles_all: all your angles from data
# spits out a same-size array as angles_all, but with categorical info
# i.e. creates bins, and categorizes your angles
def bin_angle_by_direction(angles_all, starting_angle=0, num_angle_bins=4, 
    binnames = None, PLOT=False):
    """ bin angle, based on slicing circle into same-sized slices, with first bin
    always starting from (1,0), and going CCW.
    INPUTS:
    - angles_all, array of angles, with 0 at 1,0, and ccw. in rad.
    - binnames, replaces bins with this, dict. e.g., could call l-->r one value,
    and r-->l another.
    RETURNS:
    - angles_named (based on mapping using binnames)
    NOTE:
    - bins start from 1...
    - nans will be 
    """
    from math import pi

    if PLOT:
        import copy
        angles_all_orig = copy.copy(angles_all)

    if binnames is None:
        binnames = {i+1:i+1 for i in range(num_angle_bins)}
        binnames[num_angle_bins+1] = np.nan
        # print(binnames)
        # assert False
        # binnames = {1: 0, 2:1, 3:1, 4:0, 5:np.nan}
        
    # if not num_angle_bins==4:
    #     assert False, "bin_names will not be accurate. code this"

    assert starting_angle<=0 and starting_angle>=-2*pi, "starting_angle is not between [-2pi,0]"

    # Shift convert angles to [starting_angle, 2*pi+starting_angle], then
    # shift back so that starting angle starts bin 0.
    angles_all = np.asarray(angles_all)
    angles_all = transform_data_to_negative_start(angles_all, starting_angle) - starting_angle

    # bin angles
    # bins = np.linspace(starting_angle, starting_angle+2*pi, num_angle_bins+1)
    bins = np.linspace(0, 2*pi, num_angle_bins+1)
    angles_all_binned = [np.digitize(a, bins) for a in angles_all]

#     plt.figure()
#     plt.plot(angles_all, angles_all_binned, 'ok')

    # assign bin to a label
    # binnames = {1: "L->R", 2:"R->L", 3:"R->L", 4:"L->R", 5:"undefined"}
    angles_named = [binnames[b] for b in angles_all_binned]

    if PLOT:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(angles_all_orig, angles_named, "ok")
        ax.axvline(2*pi+starting_angle)
        ax.set_ylabel("bins")
        ax.set_xlabel("input angles (blue, vline is starting angle)")

    return angles_named

def linearize_angles_around_mean(angles, PLOT=False):
    """
    Given vector of angles, find its mean, and then linear data by subtracting this from 
    all angles, and then taking sine, so that all data is now approx centered at 0, and deviating
    neg and postiive.

    Throws AssertionError if you pass in angles where the max-min is greater than pi, beucase then it 
    doesnt make sense to linearize.

    RETURNS:
    - angles_diff, angles_diff_sin, anglemean, anglemean_norm
    """
    # Note: for angle, linearize angle by converting to difference from mean angle
    from math import pi
    
    angles = np.array(angles)

    # Get mean angle
    anglemean, anglemean_norm = average_angle(angles)

    # Convert all angles to difference from this angle.
    # angles_diff = angle_diff_vectorized(angles, anglemean * np.ones(angles.shape)) # This gives absolute...
    angles_diff = angles - anglemean

    # Take sine of this difference to deal with wraparound effects.
    # (sine beucase wnat this to be linear)
    angles_diff_sin = np.sin(angles_diff)

    # Sanity check, because linearizing doesnt make sense if the (max-min) of angles if >pi.
    if (max(angles_diff)-min(angles_diff))>pi:
        fig, ax = plt.subplots()
        ax.plot(angles, angles_diff, "xr")
        ax.plot(angles, angles_diff_sin, "xk")
        assert False

    if PLOT:
        fig, ax = plt.subplots()
        ax.plot(angles, angles_diff, "xr")
        ax.plot(angles, angles_diff_sin, "xk")

    return angles_diff, angles_diff_sin, anglemean, anglemean_norm

def projection_onto_axis_subspace(xmean_base1, xmean_base2, X, doplot=False,
    plot_color_labels=None):
    """
    Project all pts in X onto axis (subspace) between xmean_base1 and xmean_base2, 
    such that pts close to xmean_base1 are 0 and clsoe to xmean_base2 are 1

    PARAMS:
    - xmean_base1, (ndims,)
    - xmean_base2, (ndims,)
    - X, (ntrials, ndims), the data to project
    RETURNS:
    - X_proj_scal_norm, (ntrials,), projected data
    """
    import seaborn as sns

    assert X.shape[1] == xmean_base1.shape[0] == xmean_base2.shape[0]

    # - get axis.
    encoding_axis = xmean_base2 - xmean_base1
    encoding_axis_unit = unit_vector(encoding_axis)
    encoding_axis_length = np.linalg.norm(encoding_axis)

    # for any pt
    # indthis = 100
    # pt = X[indthis, :]

    # # - subtract base1.
    # projection_scalar = np.dot((pt - xmean_base1), encoding_axis_unit)
    # print(projection_scalar)
    # projection_scalar_norm = projection_scalar/encoding_axis_length

    # vectorized version
    X_proj_scal = np.dot((X - xmean_base1), encoding_axis_unit) # (ntrials,)
    X_proj_scal_norm = X_proj_scal/encoding_axis_length

    if doplot:
        fig, axes = plt.subplots(2,2, figsize=(8, 8), sharex=True, sharey=True)
        list_dims = [(0,1), (2,3), (4,5), (6,7)]
        for dims, ax in zip(list_dims, axes.flatten()):

            d1 = dims[0]
            d2 = dims[1]
            
            if d2 > len(xmean_base1)-1:
                break

            ax.plot(xmean_base1[d1], xmean_base1[d2], "sk")
            ax.text(xmean_base1[d1], xmean_base1[d2], "base1")
            ax.plot(xmean_base2[d1], xmean_base2[d2], "sr")
            ax.text(xmean_base2[d1], xmean_base2[d2], "base2")
            ax.plot([xmean_base1[d1], xmean_base2[d1]], [xmean_base1[d2], xmean_base2[d2]], "-")
            if plot_color_labels is None:
                h = ax.scatter(X[:,d1], X[:,d2], c=X_proj_scal_norm, cmap="plasma")
                plt.colorbar(h)
            else:
                from pythonlib.tools.listtools import sort_mixed_type
                hue_order = sort_mixed_type(set(plot_color_labels))
                # make categorical (string)
                plot_color_labels = [f"{v}" for v in plot_color_labels]
                hue_order = [f"{v}" for v in hue_order]
                sns.scatterplot(x=X[:,d1], y=X[:,d2], hue=plot_color_labels, hue_order=hue_order, ax=ax, alpha=0.7)
            ax.set_title("color=projection along axis between base1 and base2")
            ax.set_ylabel(f"dims={dims}")

    if doplot:
        return X_proj_scal_norm, fig
    else:
        return X_proj_scal_norm
    
def cosine_similarity(vector_a, vector_b):
    """
    Compute the cosine_similarity between two vectors.

    1 means parallel.
    0 means orthogonal
    -1 means antiparlle

    :param vector_a: 1D array-like
    :param vector_b: 1D array-like
    :return: A float representing the cosine_similarity.

    """
    # Convert inputs to NumPy arrays (in case they're Python lists)
    a = np.array(vector_a, dtype=float)
    b = np.array(vector_b, dtype=float)
    
    # Compute dot product
    dot_product = np.dot(a, b)
    
    # Compute norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Calculate cosine similarity
    cosine_similarity = dot_product / (norm_a * norm_b)
    
    # print(cosine_similarity)
    assert cosine_similarity<=1.0001
    assert cosine_similarity>=-1.0001

    # Return cosine distance
    return cosine_similarity
