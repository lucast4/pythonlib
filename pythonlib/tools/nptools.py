import numpy as np
import matplotlib.pyplot as plt

### 
# np.r_ is for quick contaneation.

def bin_values_categorical_factorize(vals):
    """ like pd.factorize.
    convert categorical value (strings) to 0,1,2,...
    """
    return np.unique(vals, return_inverse=True)[1]

def bin_values(vals, nbins=8, valmin = None, valmax=None, epsilon = 0.0001,
    assert_all_vals_within_bins=True):
    """ REturn vals, but binned, unofrmly from min to max
    Values binned between 1 and nbins.
    PARAMS;
    - nbins, num bbins.
    - epsilon, scalar, small number to pad, to make sure get all values.
    RETIURNS:
    - array, (n,)
    """
    if valmin is None:
        valmin = np.min(vals)
    if valmax is None:
        valmax = np.max(vals)

    bins = np.linspace(valmin-epsilon, valmax+epsilon, nbins+1)
    vals_binned = np.array([np.digitize(v, bins) for v in vals], dtype=np.int_)

    if assert_all_vals_within_bins:
        assert np.all((vals_binned>=1) | (vals_binned<=nbins))
    return vals_binned

def rankItems(arr):
    """ for each item, returns its
    rank out of all items (0 = smallest);
    if arr = [3,1,5,0], then
    returns [2, 1, 3, 0]. 
    - deals with ties by assigning lower label
    to the earlyier bvalue
    """
    array = np.array(arr)
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks


def sortPop(X, dim, filt):
    """ sort population, general purpose. 
    - X is array
    - dim, dimension to sort.
    - filt, function to apply to subarray, e..g, 
    vals_to_sort = [filt(X[:,0, ...]), filt(X[:,1, ...]), ...] 
    if dim is 1. will then sort in ascending order.
    """
    # 1) copy array, and move dim to be the new dim 0.
    Xcopy = np.copy(X)
    Xcopy = np.moveaxis(Xcopy, dim, 0)
    
    # 2) for each element in new dim 0, get scalar by applying filt
#     print(Xcopy[0].shape)
#     print([x.shape[0] for x in Xcopy])
    vals = [filt(x) for x in Xcopy]
    list_to_sort = [[v, x] for v, x in zip(vals, Xcopy)] # combine, so can sort
#     print([l[0] for l in list_to_sort])
    list_to_sort = sorted(list_to_sort, key=lambda x:x[0])
#     print([l[0] for l in list_to_sort])
    Xcopy = [l[1] for l in list_to_sort] # recreate
    Xcopy = np.array(Xcopy)
    
    # 3) return axes to original position
    Xcopy = np.moveaxis(Xcopy, 0, dim)
    return Xcopy   

def isin_close(a, v, atol=1e-06):
    """ returns True if a (scalar) is in v (vector) allowing
    for tolerace. essentially a for loop using np.isclose
    RETURNS:
    - bool, whether a is in v
    - idx, location of a in v.
    """
    x = [np.isclose(a, b, atol=atol) for b in v]
    return np.any(x), np.where(x)[0]

def isin_array(arr, list_arr, atol=1e-06):
    """ Returns True if arr (array) is equal to at least one of arrays in list_arr
    PARAMS;
    - arr, np array
    - list_arr, list of arrays
    """
    tmp = [(arr == x).all() for x in list_arr] # array of bools.
    tmp = [np.allclose(arr, x, atol) for x in list_arr] # array of bools.
    return any(tmp)

def stringify(arr, remove_decimels=True):
    """ Convert array to list of strings, 
    - each row in arr is a single item int he otuptu string.
    - cols are concatenated like col1|col2|, ...
    PARAMS:
    - remove_decimels, bool, if True, then uses only singles digits and up.
    RETURNS:
    - out, list of strings.
    """

    assert len(arr.shape)==2

    out = []
    for val in arr:
        s = ""
        for i, item in enumerate(val):
            if i>0:
                sep = "|"
            else:
                sep = ""
            if remove_decimels:
                s += f"{sep}{item:.0f}"
            else:
                s += f"{sep}{item}"
        out.append(s)
        # out.append(f"{val[0]:.0f}|{val[1]:.0f}")
    return out

def unique_tol(arr, decimels=4):
    """ get unique, alowoing for tolerance, based on roudning decimels
    """
    return np.unique(arr.round(decimels=decimels))


def sort_by_labels(X, labels, axis=0):
    """ Sort X by labels, in incresaing order.
    PARAMS:
    - X, ndat x ndim np array, rows will be osrted.
    - labels, list of ints, length ndat, to suport sorting in incresaing order
    - axis, dimeision to srot byt. if 0, then sorts rows...
    RETURNS:
    - X, labels, but sorted (copies)
    """
    
    inds = np.argsort(labels)

    if axis==0:
        X = X[inds,:]
    elif axis==1:
        X = X[:, inds]
    else:
        print(X.shape)
        print(axis)
        assert False

    labels = [labels[i] for i in inds]
    
    return X, labels

def find_peaks_troughs(vals, PLOT=False):
    """ Find indices of peaks and troughs in vals
    Does so by looking at diffs
    PARAMS:
    - vals, (1,N) or (N,) array of values.
    RETURNS:
    - inds_peaks, inds_troughs, list of ints, indices into vals.
    """
    
    # Peaks are where the diffs change sign from pos to neg. and similarly
    # for troughs
    
    if len(vals.shape)>1:
        assert vals.shape[1]==1
        vals = vals.squeeze()

    valsdiff = np.diff(vals, axis=0)
    assert len(valsdiff)==len(vals)-1, "prob shape issue fior vals"

    inds_troughs = list(np.where(np.diff(np.sign(np.diff(vals)))>0)[0]+1)
    inds_peaks = list(np.where(np.diff(np.sign(np.diff(vals)))<0)[0]+1)

    if PLOT:
        fig, ax = plt.subplots()
        ax.plot(range(len(vals)), vals, "ok")
        for idx in inds_troughs:
            ax.plot(idx, 0, "xr")
        for idx in inds_peaks:
            ax.plot(idx, 0, "xb")        

    return inds_peaks, inds_troughs