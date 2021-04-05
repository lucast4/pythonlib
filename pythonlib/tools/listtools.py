"""stuff to dow with lists"""
from operator import itemgetter

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