""" working with and extracting chunks, and all related things, like hierarchies, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

def _check_is_proper_chunk(chunk):
    """ Must be list of lists.
    """
    assert isinstance(chunk, list)
    for c in chunk:
        assert isinstance(c, list)
        for cc in c:
            assert isinstance(cc, int)


def check_all_strokes_used(chunks, nstrokes):
    """ REturns true if each stroke is used once and only once across all ch in chunks
    PARAMS:
    - chunks, list of list, e.g, [[0,1], [2]]
    - nstrokes, int, how many strokes.
    RETURNS:
    - bool, indiciating whether all used.
    """
    
    _check_is_proper_chunk(chunks)

    inds_used = sorted([cc for c in chunks for cc in c])
    inds_check = list(range(nstrokes))
    if inds_used==inds_check:
        return True
    else:
        return False


def sample_a_group_list(list_groups, list_possible_inds):
    """ get all ways of sampling groups, given sets of groups. 
    IN:
    - list_groups, list of all n-tuples (circle, line), which are groups.
    e.g., [[1,2], [0,2], [1,2,3]]
    - list_possible_inds, list of inds for all objects in storkes.
    --- usually just range(len(strokes))
    OUT:
    - chunk, list of list of ints,a single  randomly sampled chunk that maximizes use of lollis.
    NOTE:
    - will use each obj (int) once and onlye once.
    - will be sorted, since assumes temporal order doesnt matter.
    """
    import random

    def _is_eligible(gp, items_selected):
        # Returns bool for whether this group is 
        # eligible (onky if its items are not in items_selected)
        # no items already selected
        # items_selected, flat list of inds
        for l in gp:
            if l in items_selected:
                return False
        return True

    items_selected = []
    groups_selected = []
    eligible_groups = [gp for gp in list_groups if _is_eligible(gp, items_selected)]
    eligible_singles = []
    
    # Keep sampling until no more eligible groups
    while len(eligible_groups)>0:

        # Pick one
        gp_picked = random.sample(eligible_groups, 1)[0]
        groups_selected.append(gp_picked)
        items_selected.extend(gp_picked)

        # update eligible lolli
        eligible_groups = [gp for gp in list_groups if _is_eligible(gp, items_selected)]

    groups_selected = sorted(groups_selected)    

    # conver to chunk, always list of lollis (sorted) and then remianing inds
    inds_extra = [i for i in list_possible_inds if i not in items_selected]
    chunk = sort_chunk(groups_selected + inds_extra)
    return chunk

def sample_all_possible_chunks(list_groups, list_possible_inds, nfailstoabort=10, maxsearch=1000):
    """
    Get list of possible chunks consisetent with input groupings. First samples groups from list_groups
    without replacement, until get a complete chunk (using each stroke once). Then repeats to find more chunks
    unitl cannot find any new ones nfailstoabort times in a row.
    IN:
    - list_groups, list of groups, where each group is a list of inds specifying objects that group together
    --- e.g., [[0,1], [0,3], [2,4]] could be 3 lollis.
    - list_possible_inds, usualyl just list(range(len(strokes)))
    OUT:
    - list_chunks, list of hierarchices/chunks, i.e, all unique ways of chunking.
    --- chunks with same structure but different order will be considered the same (ordering by sorting as tuples)
    """

    def _chunk_already_selected(list_chunks, chunk):
        if tuple(chunk) in [tuple(c) for c in list_chunks]:
            return True
        else:
            return False

    nfail_in_a_row = 0
    list_chunks = []
    ct = 0
    while nfail_in_a_row<nfailstoabort:
        
        chunk = sample_a_group_list(list_groups, list_possible_inds)
        
        if _chunk_already_selected(list_chunks, chunk):
            nfail_in_a_row+=1
        else:
            list_chunks.append(chunk)
            nfail_in_a_row=0
        ct+=1
        if ct>maxsearch:
            print("breaking while in sample_all_possible_chunks, > maxsearch: ",  maxsearch)
            break
    return list_chunks


################### LOW-LEVEL TOOLS WITH CHUNKS (WORKING WITH STROKES, USUALLY)
def chunks_are_identical(chunks1, chunks2):
    """ 
    Return True if chunks are identical. Deals with hierarhical issues, liek the following
    [[1], 2] identical to [1,2]
    NOTE:
    - Assuems that two chunks with diff order are actually different
    """

    # convert to list of lists
    a = [c if isinstance(c, list) else [c] for c in chunks1]
    b = [c if isinstance(c, list) else [c] for c in chunks2]
    return a==b

def chunks_are_identical_full(ch1, hi1, fi1, ch2, hi2, fi2):
    """ Entire features, chunks, hierarchy, and fixed order, are they identical
    over two items?
    NOTE: if order diff, then will call it diff..
    """
    return chunks_are_identical(ch1, ch2) and chunks_are_identical(hi1, hi2) and fi1==fi2
    

def clean_chunk(chunk):
    """input like this: [[1,3], 0], will output: [[0], [1, 3]]"""
    chunk_clean = []
    for c in chunk:
        if isinstance(c, list):
            chunk_clean.append(c)
        elif isinstance(c, int):
            chunk_clean.append([c])
        else:
            assert False
    return chunk_clean

def sort_chunk(chunk):
    """ converts to list of tuples, sorts, then converts back
    NOTE:
    - input like this: [[1,3], 0], will output: [[0], [1, 3]]. Effectively the same for all code
    that works with chunks.
    """
    chunk_tuple = []
    for c in chunk:
        if isinstance(c, list):
            chunk_tuple.append(tuple(c))
        elif isinstance(c, int):
            chunk_tuple.append(tuple([c]))
        else:
            assert False
    chunk_tuple = sorted(chunk_tuple)
    return [list(c) for c in chunk_tuple]

def chunk_strokes(strokes, chunks, use_each_traj_once=True,
    reorder=False, thresh=10, sanity_check=False, generic_list=False):
    """ [GOOD] Apply chunks to strokes
    INPUT:
    - strokes, list of np array. Use - generic list flag if this is generic list.
    - chunks, list of chunks, each can be int or another list. e..g,:
    --- [[1,2],0]; [0,1,2]; [[0], 1,2]; [[0], [1,2]]
    --- NOTE: cannot go deeper than that.
    - use_each_traj_once, then ensures that used each once and only once.
    - generic_list, then will not try to concatenate. This also means the output 
    WILL be list of lists.
    """
    from pythonlib.tools.stroketools import concatStrokes

    inds_used =[]
    strokes_chunked = []
    for ch in chunks:
        if isinstance(ch, int):
            if generic_list:
                strokes_chunked.append([strokes[ch]])
            else:
                strokes_chunked.append(strokes[ch])
            inds_used.append(ch)
        elif isinstance(ch, list):
            x = [strokes[i] for i in ch]
            if generic_list:
                pass
            else:
                x = concatStrokes(x, reorder=reorder, 
                    thresh=thresh, sanity_check=sanity_check) # list, len1
                x = np.concatenate(x, axis=0)
            strokes_chunked.append(x)
            inds_used.extend([i for i in ch])
        else:
            print(ch)
            print(type(ch))
            assert False, "wrong type"
    if use_each_traj_once==True:
        assert sorted(inds_used)==list(range(len(strokes))), "did not do one to one use of strokes"

    # Remove temporal infor, since not accurate anymore.
    if not generic_list:
        for i, s in enumerate(strokes_chunked):
            strokes_chunked[i] = s[:, :2]
        
    return strokes_chunked

def chunks2parses(list_chunks, strokes, reorder=False, thresh=10, sanity_check=False,
    use_each_traj_once=True):
    """ for this list_chunks (list of chunks) and model, extract
    each way to chunking strokes. e.g.
    [[0,1], 2] leads to chuning of strokes 0 and 1
    Returns one strokes object for each way of chunking. 
    (note, can do:
    chunklist = getTrialsTaskChunks(...))
    - NOTE: 3rd dim (time) might not make sense).
    - parses_list is len of num chunks, each element a strokes.
    --- eachstroke, then will treat each stroke as chunk,
    INPUT:
    - list_chunks, 
    --- e.g,, list_chunks = [
        [[0, 1], 2], 
        [[0, 2], 1]]
    OUT:
    - strokes, same structure as chunks, but instead of ints, have np arrays (Nx2)
    """
    # from pythonlib.tools.stroketools import concatStrokes

    parses_list = []
    for chunks in list_chunks:
        parses_list.append(
            chunk_strokes(strokes, chunks, use_each_traj_once=True,
                reorder=reorder, thresh=thresh, sanity_check=sanity_check)
            )
        # strokesnew = [concatStrokes([strokes[i] for i in s], 
        #     reorder=reorder, thresh=thresh, sanity_check=sanity_check)[0] for s in c]
        # parses_list.append(strokesnew)

    # # === remove temporal inforamtion from parses (since innacuarte)
    # parses_list = [[strok[:,[0,1]] for strok in strokes] for strokes in parses_list]

    return parses_list

def hier_is_flat(hier):
    """ 
    Check if this hierarhcy is flat
    PARAMS:
    - hier is 2-level, list of list or list of mixture of ints and lists (see below)
    e..g, [[1,2], 3] is not flat
    e.g., [[1], [2], [3]] is flat
    RETURNS:
    - True if hierarchy is flat. evem this [[1], 2, [0]] woudl be True
    """
    for h in hier:
        if isinstance(h, list):
            if len(h)>1:
                return False
    return True

def search_permutations_chunks(chunks_hier, fixed_order, max_perms=1000):
    """ Returns all permutatoins of chunks or hierarchy, while folliwng 
    fixed_order rules
    PARAMS:
    - chunks_hier, chunks or hierarhcy object, list of lists or ints
    RETURNS:
    - list_out, list of chunks, each of which has same shape as chunks_hier,
    modulo permutations. is all the ways of permuiting chunks hier
    EG:
        chunks = [[1,2], [3], [4, 5]]
        fo = {0:False, 1:[False, False, True]}
        search_permutations_chunks(chunks, fo)    
            [[[1, 2], [4, 5], [3]],
             [[2, 1], [4, 5], [3]],
             [[3], [4, 5], [1, 2]],
             [[3], [4, 5], [2, 1]],
             [[4, 5], [3], [1, 2]],
             [[4, 5], [3], [2, 1]],
             [[4, 5], [1, 2], [3]],
             [[4, 5], [2, 1], [3]],
             [[1, 2], [3], [4, 5]],
             [[2, 1], [3], [4, 5]],
             [[3], [1, 2], [4, 5]],
             [[3], [2, 1], [4, 5]]]
    """

    from pythonlib.chunks.chunks import hier_is_flat
    from pythonlib.tools.stroketools import getStrokePermutationsWrapper
    from itertools import product

    _check_is_proper_chunk(chunks_hier)

    def _get_permutations(inlist):
        """ 
        Also returns the input list
        Gets all permutations
        """
        return getStrokePermutationsWrapper(inlist, "all_orders", num_max=max_perms)

    def _get_permutations_hierarchical(list_of_list, list_of_fixed):
        """ 
        Will permute all the inner lists in list_of_list, if their correspoding fixed is False
        RETURNS list, with shape being list(shape(list_of_list))
        PARAMS:
        - list_of_list, 
        e.g., [[0,1], [2,3]]
        - list_of_fixed
        e.g., [True, False]
        RETURNS:
        - list of chunks
        e.g.,
            [[[0,1], [2,3]], [[0,1], [3,2]]
        """
        assert len(list_of_list)==len(list_of_fixed)
        assert isinstance(list_of_list, list) and isinstance(list_of_fixed, list)
        
        list_of_list_out_all = []
        for inlist, fo in zip(list_of_list, list_of_fixed):
            if not fo:
                list_of_list_out = _get_permutations(inlist)
                list_of_list_out_all.append(list_of_list_out)
            else:
                list_of_list_out_all.append([inlist])

        return [list(x) for x in product(*list_of_list_out_all)] # list of lists, where inner lists are same shape as inner lsit of list_of_list.


    hier = chunks_hier

    # First level
    # - combine hier and fixed order, so not broken apart after permutation
    hier_fo = [(h,f) for h, f in zip(hier, fixed_order[1])]
    if fixed_order[0]==False:
        out = _get_permutations(hier_fo)
    else:
        out = [hier_fo]

    # Second level
    list_out = []
    for X in out: # for each way of permuting the top level
        hier = [x[0] for x in X] # [[0,1], [2,3]]
        fo = [x[1] for x in X] # [False, True]
        out_this = _get_permutations_hierarchical(hier, fo)
        list_out.append(out_this)
        
    list_out = [xx for x in list_out for xx in x]

    for out in list_out:
        _check_is_proper_chunk(out)
    return list_out

# def search_permutations_chunks(chunks):
#     """ Returns all permutations of chunks. THink of this as "grouding" the abstract
#     chunking (grouping) into concrete sequence of tokens. 
#     NOTE:
#     - this is taken from parser.search_parse
#     """
#     from pythonlib.tools.stroketools import getStrokePermutationsWrapper
#     if direction_within_stroke_doesnt_matter:
#         # then just reorder. much quicker.
#         list_parse = getStrokePermutationsWrapper(parse, 
#             "all_orders", num_max=configs_per)
#     else:
#         from .search import search_parse as sp
#         list_parse, _ = sp(parse, 
#             configs_per=configs_per, trials_per=trials_per, 
#             max_configs=max_configs)
#     return list_parse

def fixed_order_for_this_hier(hier, top_level_allow_reorder=True, 
        bottom_level_allow_reorder=True):
    """ automaticlaly generate fixed_order for this hierarchy
    PARAMS:
    - hier, list of list of ints, e.g., [[1,2], [0]]
    - top_level_allow_reorder, bottom_level_allow_reorder, bools, for whether
    allow reorder.
    """
    # print(hier, top_level_allow_reorder, bottom_level_allow_reorder)
    fixed_order = {}
    fixed_order[0] = not top_level_allow_reorder
    fixed_order[1] = [not bottom_level_allow_reorder for _ in range(len(hier))]
    return fixed_order

