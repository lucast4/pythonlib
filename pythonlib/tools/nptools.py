import numpy as np

### 
# np.r_ is for quick contaneation.


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


def isin_close(a, v):
    """ returns True if a (number) is in v (vector) allowing
    for tolerace. essentially a for loop using np.isclose
    """
    return np.any([np.isclose(a, b) for b in v])
