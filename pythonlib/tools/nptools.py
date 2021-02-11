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


