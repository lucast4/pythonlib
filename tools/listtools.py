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



def concatStringsInList(mylist):
    """given list """
    pass