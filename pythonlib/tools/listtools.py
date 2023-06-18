"""stuff to dow with lists"""
from operator import itemgetter
import numpy as np
# import torch

def stringify_list(li):
    """ 
    list --> list of str
    """
        
    out = []
    for x in li:
        if isinstance(x, list):
            out.extend(stringify_list(x))
#         elif isinstance(x, np.ndarray):
#             from pythonlib.tools.nptools import stringify
#             out.extend(stringify(x))
        else:
            out.append(str(x))
    return out


def sort_mixed_type(mylist):
    """ Sort, works even if elements in mylist are mixed type.
    PROBLEMS:
    if an element is a lsit of lsits. will then depend on input order.
    eg: these are different
        sort_mixed_type(['start', 'end', [[2]], [[None]], [1,2], (1,2), (99, 99)])
        sort_mixed_type(['start', 'end', [[None]], [[2]], [1,2], (1,2), (99, 99)])
    """

    def key(x):
        try:
            if isinstance(x, list):
                return (0, hash(tuple(x)))
            else:
                # NOTE: hash(None) is valid
                return (0, hash(x))
        except TypeError as err:
            # Place at back
            return (1, '')

    return sorted(mylist, key=lambda x: key(x))

def permuteRand(mylist, N, includeOrig=True, not_enough_ok=False):
    """gets N random permutations from muylist, no
    replacement. works by shuffling, taking first one, and
    checking if already have, and so on. can choose to enforce that 
    the entered mylist is included. can allow for N to be larger than
    maximum num permutations. will then just ouput p[ermutations."""

    from random import shuffle
    from math import factorial
   
    if not_enough_ok:
        if N>factorial(len(mylist)):
            from itertools import permutations
            print("not enough permutations, so getting {}".format(factorial(len(mylist))))
            return list(permutations(mylist))
    else:
        assert N <= factorial(len(mylist)), "not enough unique permutations..."
       
    # convert list to list of indices, so that can put in sets.
    mylist_inds = list(range(len(mylist)))
    myset = set()
    if includeOrig:
        myset.add(tuple(mylist_inds))
        
    while len(myset) < N:
        shuffle(mylist_inds)
        myset.add(tuple(mylist_inds))
   
#     print([list(x) for x in myset])
   
    # get back permuted lists
    permutedlists = []
    from operator import itemgetter
    for inds in myset:
        permutedlists.append(itemgetter(*inds)(mylist))
#     print(permutedlists)
   
    return permutedlists
if False:
    # mylist = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T']
    mylist = ["0","1","2","3","4","5", "6", "7", "8"]
    permuteRand(mylist, 7*720)



def concatStringsInList(mylist):
    """given list """
    pass


def tabulate_list(l, return_as_list=False):
    """outputs dict with unique entries as keys, and num
    occurances as entries
    like tabulate() in matlab
    - return_as_list, then returns as list of tuples, not as dict.
    """
    l_unique = set(l);
    outdict = {}
    for key in l_unique:
        outdict[key] = len([ll for ll in l if ll==key])
    if return_as_list:
        return [(k, v) for k, v in outdict.items()]
    return outdict


def partition(collection):
    """ from https://stackoverflow.com/questions/19368375/set-partitions-in-python
    gets all partitions of the list collection.
    e..g, one partition is [[1], [2,3,4]]
    """
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller

    
    # something = list(range(1,5))

    # for n, p in enumerate(partition(something), 1):
    #     print(n, sorted(p))


def get_counts(vals):
    """ returns dict, where keys are unique values in vals,
    and values are counts (frequenceis). vals can be anything
    hashable (?) by np.unique, e..g, strings, nums, etc.
    INPUTS:
    - vals, array-like,
    """
    x, y = np.unique(np.array(vals), return_counts=True)
    counts_dict = {xx:yy for xx, yy in zip(x, y)}
    return counts_dict


def counts_to_pdist(counts_dict, cats_in_order, dtype=None, 
    prior_counts=0., print_stuff=True):
    """
    Get an array of probabilities, where the
    indices are in order defined by cats_in_order.
    INPUT:
    - counts_dict, from get_counts.
    - cats_in_order, list of elements matching (superset) of unique
    values in vals. will use this to determine ordering in pdist output. 
    can also be subset, in which will ignore any keys not in it. will always
    normalize output to sum to 1.
    - dtype, dtype for torch tensor
    - prior_counts, adds this to counts before getting probs
    RETURNS:
    - pdist, torch tensor, len of cats_in_order.
    """
    import torch
    if dtype is None:
        dtype = torch.float32
    probs = []
    for c in cats_in_order:
        if c not in counts_dict:
            nelem = 0. + prior_counts
        else:
            nelem = counts_dict[c] + prior_counts
        probs.append(nelem)

    pdist = torch.tensor(probs, dtype=dtype)
    pdist = pdist/torch.sum(pdist)
    pdist = pdist.squeeze()
    
    if print_stuff:
        print("cats, this order:")
        print(cats_in_order)
        print("these probs:")
        print(pdist)
    
    return pdist


def rankinarray1_of_minofarray2(array1, array2, look_for_max=False):
    """ for the index of min in array2, what is the rank of the
    correspodning value in array1? e.g., if
    array1 = [1,40,8,32,32] and array2 = [3,2,4,5,1], then first say
    index of min array 2 is 4, then looks in array1[4]. 32 is the 3rd 
    rank (this accounts for ties by taking the best rank), so the output 
    is 3.
    - look_for_max, then same, but looks for max in both arrays.
    RETURNS:
    - rank, item1, item2, idx, where items are the original values in arrays 1 and 2.
    and idx is the index.
    NOTES:
    - e.g., useful is array2 is beh-task distance, and array1 is some measure of
    the task efficeincey(score), for scoring how efficient behvior was.
    - 0-indexed, so best rank is 0.

    """    
    import scipy.stats as ss
        
    if look_for_max:
        array1 = [-a for a in array1]
        array2 = [-a for a in array2]
        
    # convert array1 to ranks
    array1_ranks = ss.rankdata(array1, method="min") # e..g, [12 13 2 2] --> [3 4 1 1]
    array1_ranks = array1_ranks-1 # since is 1-indexed

    # find the rank of min array2
    idx = array2.index(min(array2)) # find the index of the task sequence that is most aligned to beh.

    return array1_ranks[idx], array1[idx], array2[idx], idx

def random_inds_uniformly_distributed(vals, ntoget):
    """ sorts vals incresaing, then returns unofmrly sampled values.
    PARAMS:
    - vals, array of numbers
    RETURNS:
    - list of indices into original vals [NOT THE VALUES THEMSELVES]
    """
    assert ntoget<=len(vals)
    inds = np.argsort(vals)
    idxs = np.linspace(0, len(inds)-1, ntoget)
    idxs = np.floor(idxs)
    idxs = np.array(idxs, dtype=int)
    return inds[idxs]


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def extract_novel_unique_items_in_order(listin):
    """ 
    Get change points, when the items in listin change.
    PARAMS:
    - listin, list of items with equals method
    RETURNS:
    - list_new_items, list of items at change points
    - list_positions, list of indices in listin wherre those change pts are fond (right side)
    e.g, listin=[0,2,2,1,3, 3, 2], returns:
    [0, 2, 1, 3, 2]
    [0, 1, 3, 4, 6]
    """
    list_new_items =[]
    list_positions = []
    curr_item = None
    for i, x in enumerate(listin):
        if x==curr_item:
            continue
        else:
            list_new_items.append(x)
            list_positions.append(i)
            curr_item = x

    return list_new_items, list_positions

def remove_values_refrac_period(times_in_order, refrac_period):
    """ Given list of times, return indices to slice, to remove times that
    occur too close in time to preceding (i.e. interval less than refrac_period)
    PARAMS:
    - times_in_order, list-like of times, assumed to be in order. will not check
    - refrac_period, duration.
    RETURNS:
    - inds_keep, list of ints indices to keep.
    - inds_remove, complement of inds_keep
    EXAMPLE:
        values_in_order = [0.1, 0.11, 0.2]
        refrac_period = 0.1
        --> ([0], [1, 2])
    """
    inds_remove = np.argwhere(np.diff(times_in_order)<refrac_period)+1
    inds_remove = [x[0].astype(int) for x in inds_remove]
    inds_keep = [i for i in range(len(times_in_order)) if i not in inds_remove]
    return inds_keep, inds_remove

def indices_into_original_list_after_concat(list_of_lists):
    """
    returns a list, of sum of lents of all innter lisrts, with index back
    into orignal list

    e.g., x = [[0,1], [3,4,5], [10], [], range(5), range(2)]
    RETURNS:
    0
    0
    1
    1
    1
    2
    4
    4
    4
    4
    4
    5
    5
    """

    ct = 0
    idxs = []
    for i, xx in enumerate(list_of_lists):
        for j in range(len(xx)):
            # print(i)
            idxs.append(i)
        ct = ct+len(xx)

    return idxs

def unique_input_order(mylist):
    """ list set(mylist) but returns
    in the order inputed.
    e.g., 
    mylist = [0,4,3,4, 2,4,0, 'cow'] -->
        [0, 4, 3, 2, 'cow']
    """
    
#     mylist_uniq = list(set(mylist))
    mylist_uniq = []
    for item in mylist:
        if item not in mylist_uniq:
            mylist_uniq.append(item)
    
    return mylist_uniq

def list_roll(mylist, shift):
    """ Like np.roll but for lists. does not spruiously
    convert list of tuples into list of lists.
    PARAMS:
    - mylist, list
    - shift, int, how much to shift to right. 0 means no shift. 
    e.g.,,:
        x = [1,2,3,4,5]
        list_roll(x, 2) --> [4, 5, 1, 2, 3]
        list_roll(x, 6) --> [5, 1, 2, 3, 4] (becuase 6%5=1)
        list_roll(x, -2) --> [3, 4, 5, 1, 2] 
        list_roll(x, -7) --> [3, 4, 5, 1, 2] 
    """
    assert isinstance(mylist, list)
    shift = shift%len(mylist)
    return mylist[-shift:] + mylist[:-shift]

