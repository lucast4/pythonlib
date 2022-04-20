""" working with and extracting chunks, and all related things, like hierarchies, etc.
"""

import numpy as np
import matplotlib.pyplot as plt

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


#### HIGH-LEVEL CODE FOR WORKING WITH TASKS 
def find_chunks_wrapper(Task, expt, rule, strokes=None, params = {},
    use_baseline_if_dont_find=False):
    """ [GOOD] General purpose to return chunks, hierarchies,
    and what kind of permutations allwoed, givne some kind of input
    model, etc, rules, etc.
    INPUT:
    - Task, TaskGeneral instance
    - strokes, generally is same as Task.Strokes, but if enter, then assesrts that
    OUT:
    - list_chunks, where each chunk is
    a list indictating a way to chunk the strokes in Task.Strokes
    - list_hier, similar to chunks, but used for hierarhical permutations, without needing
    to concat storkes.
    - list_fixed_order, dict, what allowed to permute, and what not, for list_hier 
    #TODO: 
    - should return empty chunks if there are not reasonable chunks? e..g lolli rule
    will still output all strokes even if there are no lollis at all.
    - NOTE: order within and across chunks will not matter. Identity is determined by
    sorting lists of tuples. 
    """

    objects = Task.Shapes
    if strokes is not None:
        from pythonlib.tools.stroketools import check_strokes_identical
        assert check_strokes_identical(Task.Strokes, strokes)
        assert len(objects)==len(strokes), "did you chunk the strokes already?"
    else:
        strokes = Task.Strokes
        
    def _inds_by_shape(shape):
        # Return list of inds with this shape
        return [i for i, x in enumerate(objects) if x[0]==shape]
    def _chunks_by_shapes_inorder(list_shapes):
        # Return [indsshape1, indsshape2, ...], where each inds is lsit of int
        x = [_inds_by_shape(shape) for shape in list_shapes]
        x = [xx for xx in x if len(xx)>0] # e.g, if this shape doesnt exist for this trial.
        return x
    
    # Find list_chunks and list_hier - note: they will be same lenght, 
    if expt in ["gridlinecircle"]:
        chunks = list(range(len(objects))) # never concat strokes
        list_chunks = [chunks] # only one way
        if rule =="baseline":
            # circle, line, but no order
            hier = chunks
            list_hier = [hier]
        elif rule == "circletoline":
            # first circles, then lines
            list_shapes = ["circle", "line"]
            hier = _chunks_by_shapes_inorder(list_shapes)

            list_hier = [hier]
        elif rule=="linetocircle":
            list_shapes = ["line", "circle"]
            hier = _chunks_by_shapes_inorder(list_shapes)
            list_hier = [hier]
        elif rule=="lolli":
            # Returns both (no chunking, all hier) and (chukjning, no hier), combined
            # into long list
            # 1) NOT concating
            # find all ways of exhausting the objects with combos of lollis.
            paramsthis = {
                "expt":expt,
                "rule":rule,
                "ver":"lolli"
            }
            list_lollis, left_over = find_object_groups_new(Task, paramsthis)


            if len(list_lollis)==0:
                # then no lollis. then just skip this, since is same as baseline
                list_hier = [chunks]
            else:

                # 1) Get all hierarchices, holding hcyunks constaint
                list_hier = sample_all_possible_chunks(list_lollis, list(range(len(strokes))))
                list_chunks = [chunks for _ in range(len(list_hier))]

                # 2) Get all chunks, concating (so replicating hierahcy)
                list_chunks_toadd = list_hier # concat strokes
                list_hier_toadd = [list(range(len(h))) for h in list_hier] # no hierarchy

                ## combine 1 and 2
                list_chunks += list_chunks_toadd
                list_hier += list_hier_toadd

        elif rule=="alternate":
            # Alternate between lines and circles
            # - pick a random circle, then 
            import random

            def _eligible_tokens(tokens_remain, tokens_taken):
                """ 
                all are lists of indices
                """
                # only those not taken and not identical shape to prev taken
                if len(tokens_taken)==0:
                    tokens_elegible = tokens_remain
                else:
                    tokens_elegible = [t for t in tokens_remain if list_tokens[t] != list_tokens[tokens_taken[-1]]]
                    if len(tokens_elegible)==0 and len(tokens_remain)>0:
                        # then open up eligibility to all tokens
                        tokens_elegible = tokens_remain
                return tokens_elegible

            def _sample_token(tokens_remain, tokens_taken):
                tokens_elig = _eligible_tokens(tokens_remain, tokens_taken)
                ind_tok = random.choice(tokens_elig)
                
                tokens_taken.append(ind_tok)
                tokens_remain = [t for t in tokens_remain if t!=ind_tok]
                return tokens_remain, tokens_taken
                
            def _sample_single_chunk(list_tokens):
                tokens_remain = range(len(list_tokens))
                tokens_taken = []
                while len(tokens_remain)>0:
                    tokens_remain, tokens_taken = _sample_token(tokens_remain, tokens_taken)
                return tokens_taken

            # list_tokens = ["line", "circle", "line", "circle"]
            list_tokens = [o[0] for o in objects]
            list_chunks = []
            list_hier = []
            Nsamp = 20
            for _ in range(Nsamp):
                list_hier.append(_sample_single_chunk(list_tokens))
                list_chunks.append(list(range(len(list_tokens))))

        else:
            assert False

        ##### Fixed order
        def _fixed_order_for_this_hier(hier):
            if rule in ["baseline", "lolli"]:
                # Order allows all reordering
                fixed_order = fixed_order_for_this_hier(True, True)
            elif rule in ["circletoline", "linetocircle"]:
                # Order only allows for reordering both hier levels.
                fixed_order = fixed_order_for_this_hier(False, True)
            elif rule in ["alternate"]:
                # becasue hier is a specific sequence for altenration
                # e..g, hier = [0, 1, 2, 3]
                fixed_order = fixed_order_for_this_hier(False, False)
            else:
                assert False
            return fixed_order
        list_fixed_order = [_fixed_order_for_this_hier(hier) for hier in list_hier]

        # Make sure no redundant ones (identical in chunk, hier, and fixed)
        def _print(l):
            print("--")
            for ll in l:
                print(ll)
        # print("_____")
        # _print(list_chunks)
        # _print(list_hier)

        list_chunks_good = [list_chunks[0]]
        list_hier_good = [list_hier[0]]
        list_fixed_order_good = [list_fixed_order[0]]

        for ch, hi, fi in zip(list_chunks[1:], list_hier[1:], list_fixed_order[1:]):
            # Check against all already gotten
            is_good = True
            for chB, hiB, fiB in zip(list_chunks_good, list_hier_good, list_fixed_order_good):
                if chunks_are_identical_full(ch, hi, fi, chB, hiB, fiB):
                    is_good = False
                    continue
            if is_good:
                list_chunks_good.append(ch)
                list_hier_good.append(hi)
                list_fixed_order_good.append(fi)
        list_chunks = list_chunks_good
        list_hier = list_hier_good
        list_fixed_order = list_fixed_order_good

        # list_chunks_good = [list_chunks[0]]
        # list_hier_good = [list_hier[0]]
        # list_fixed_order_good = [list_fixed_order[0]]
        # for ch1, hi1, in zip(list_chunks[1:], list_hier[1:]):
        #     # check if is in
        #     # if any([chunks_are_identical(ch, x) for x in list_chunks_good]):
        #     #     continue
        #     if any([chunks_are_identical_full(ch1, hi1, fi1, ch2, hi2, fi2) for ch2, hi2 in zip(list_chunks_good, list_hier_good)]):
        #         continue
        #     else:
        #         list_chunks_good.append(ch)
        #         list_hier_good.append(hi)
        # list_chunks = list_chunks_good
        # list_hier = list_hier_good
        # _print(list_chunks)
        # _print(list_hier)
        # print("_____")

    else:
        print(params)
        assert False
    
    # Remove anything that is just same as baseline
    if use_baseline_if_dont_find == False:
        if not rule=="baseline":
            list_chunks_baseline, list_hier_baseline, list_fixed_order_baseline = find_chunks_wrapper(Task, 
                expt, "baseline", strokes, params)
            
            list_chunks_good = []
            list_hier_good = []
            list_fixed_order_good = []

            for ch, hi, fi in zip(list_chunks, list_hier, list_fixed_order):
                for chB, hiB, fiB in zip(list_chunks_baseline, list_hier_baseline, list_fixed_order_baseline):
                    if chunks_are_identical_full(ch, hi, fi, chB, hiB, fiB):
                        continue
                    else:
                        list_chunks_good.append(ch)
                        list_hier_good.append(hi)
                        list_fixed_order_good.append(fi)
            list_chunks = list_chunks_good
            list_hier = list_hier_good
            list_fixed_order = list_fixed_order_good
    # Return as a list of possible chunkings.
    assert len(list_chunks)==len(list_hier)
    assert len(list_hier)==len(list_fixed_order)
    # _print(list_chunks)
    # _print(list_hier)
    # print(list_fixed_order)
    # assert False
    return list_chunks, list_hier, list_fixed_order
    

def find_object_groups_new(Task, params):
    """ return list of groups (list of objects/shapes) passing constraints.
    Uses tokens from Task.tokens_generate(), and assumes this is unordered list. 
    So assumes any ordering of tokens, which means this outputs different ways
    of chunking. This differs from BehClass methods for finding motifs, because
    the latter assumes ordered tokens.
    e..g, find list of lollis, where each lolli is a set of 2 inds in objects
    PARAMS:
    - Task, taskclass instnace
    - params, dict holding flexible params
    RETURNS:
    - list_groups, list of list, each inner list being a single group or chunk. Ordering
    will not matter (both across chunks and within chunks)
    - left_over, list of ints, taskstroke indices that were not used in any chunk
    NOTE:
    - a given object can be used more than once, or not at all,
    e.g., if it participates in multiple gourpings.
    """

    # Generate tokens
    tokens = Task.tokens_generate(params, track_order=False)
    objects = tokens # naming convention change.

    # Methods for defining tokens or relations between tokens in a pair
    def _shape(i):
        """ String name of shape of i"""
        return objects[i]["shape"]
    def _oshape(i):
        """ String name, shape+orientation"""
        return objects[i]["shape_oriented"]
    def _location_grid(i):
        """ Location on grid, in grid units (integers, centered at 0)
        Returns (x,y)
        """
        return objects[i]["gridloc"]
    def _posdiffs_grid(i, j):
        """ Difference in positions between i and j, in grid units
        Return xdiff, ydiff"""
        pos1 = _location_grid(i)
        pos2 = _location_grid(j)
        return pos2[0]-pos1[0], pos2[1] - pos1[1]
    def _direction_grid(i, j):
        """ String name, cardinal direction, only if adjacnet on grid.
        Uses grid locations. direction from i to j """
        xdiff, ydiff = _posdiffs_grid(i,j)
        if np.isclose(xdiff, 0.) and ydiff ==1.:
            return "up"
        elif np.isclose(xdiff, 0.) and ydiff ==-1.:
            return "down"
        elif xdiff ==-1. and np.isclose(ydiff, 0.):
            return "left"
        elif xdiff ==1. and np.isclose(ydiff, 0.):
            return "right"
        else:
            return "far"
    def _orient(i):
        """ Orientation, in string, {'horiz, 'vert', ...}
        """
        assert False, "first pull in theta to objects in TaskClass"""
        if np.isclose(objects[i][1]["theta"], 0.):
            return "horiz"
        elif np.isclose(objects[i][1]["theta"], pi):
            return "horiz"
        elif np.isclose(objects[i][1]["theta"], pi/2):
            return "vert"
        elif np.isclose(objects[i][1]["theta"], 3*pi/2):
            return "vert"
        else:
            print(objects[i])
            assert False


    # Given a rule for chunking, extract all the ways of chunking
    if params["expt"]=="gridlinecircle" and params["rule"]=="lolli":
        # get list of all lollis possible
        list_groups = [] # aleays [(circle, line), ...]

        for i in range(len(objects)):
            for j in range(i+1, len(objects)):
                if _oshape(i)=="circle" and _oshape(j)=="hline":
                    if _direction_grid(i, j) in ["left", "right"]:
                        list_groups.append([i,j])
                elif _oshape(i)=="circle" and _oshape(j)=="vline":
                    if _direction_grid(i, j) in ["up", "down"]:
                        list_groups.append([i,j])
                elif _oshape(i)=="hline" and _oshape(j)=="circle":
                    if _direction_grid(i, j) in ["left", "right"]:
                        list_groups.append([i,j])
                elif _oshape(i)=="vline" and _oshape(j)=="circle":
                    if _direction_grid(i, j) in ["up", "down"]:
                        list_groups.append([i,j])
        # Find what objects are left over
        list_groups_flat = [xx for x in list_groups for xx in x]
        left_over = [i for i in range(len(objects)) if i not in list_groups_flat]

    return list_groups, left_over

def find_object_groups(Task, params):
    """ return list of groups (list of obj) passing constraints.
    General-purpse, takes in objects (Task.Shapes).
    e..g, find list of lollis, where each lolli is a set of 2 inds in objects
    NOTE:
    - a given object can be used more than once, or not at all,
    e.g., if it participates in multiple gourpings.
    """
    from math import pi
    
    assert False, "use find_object_groups_new"
    expt = params["expt"]
    rule = params["rule"]
    if expt=="gridlinecircle":
        xgrid = np.linspace(-1.7, 1.7, 3)
        ygrid = np.linspace(-1.7, 1.7, 3)
    else:
        assert False
    
    # 1) assign each object a grid location
    objects = Task.Shapes
    locations = []
    for o in objects:
        xloc = o[1]["x"]
        yloc = o[1]["y"]
        xind = int(np.where(xgrid==xloc)[0])
        yind = int(np.where(ygrid==yloc)[0])
        locations.append((xind, yind))
        
    def _shape(i):
        # return string
        return objects[i][0]
    
    def _posdiffs(i, j):
        # return xdiff, ydiff, 
        # in grid units.
        pos1 = locations[i]
        pos2 = locations[j]
        return pos2[0]-pos1[0], pos2[1] - pos1[1]
        
    def _direction(i, j):
        # only if adjacnet on grid.
        xdiff, ydiff = _posdiffs(i,j)
        if np.isclose(xdiff, 0.) and ydiff ==1.:
            return "up"
        elif np.isclose(xdiff, 0.) and ydiff ==-1.:
            return "down"
        elif xdiff ==-1. and np.isclose(ydiff, 0.):
            return "left"
        elif xdiff ==1. and np.isclose(ydiff, 0.):
            return "right"
        else:
            return "far"

    def _orient(i):
        if np.isclose(objects[i][1]["theta"], 0.):
            return "horiz"
        elif np.isclose(objects[i][1]["theta"], pi):
            return "horiz"
        elif np.isclose(objects[i][1]["theta"], pi/2):
            return "vert"
        elif np.isclose(objects[i][1]["theta"], 3*pi/2):
            return "vert"
        else:
            print(objects[i])
            assert False


    # What to look for
    if params["ver"] == "lolli":
        # get list of all lollis possible
        list_lollis = [] # aleays [(circle, line), ...]
        for i, o1 in enumerate(objects):
            for j, o2 in enumerate(objects):
                if _shape(i)=="circle" and _shape(j)=="line":
                    if _direction(i, j)=="left" and _orient(j) == "horiz":
                        list_lollis.append([i, j])
                    elif _direction(i, j)=="up" and _orient(j) == "vert":
                        list_lollis.append([i, j])
                    elif _direction(i, j)=="right" and _orient(j) == "horiz":
                        list_lollis.append([i, j])
                    elif _direction(i, j)=="down" and _orient(j) == "vert":
                        list_lollis.append([i, j])
        
        # Find what objects are left over
        list_lollis_flat = [xx for x in list_lollis for xx in x]
        left_over = [i for i in range(len(objects)) if i not in list_lollis_flat]

        return list_lollis, left_over
    else:
        print(params)
        assert False
        
                    

def sample_a_lolli_list(list_lollis, list_possible_inds):
    """ Hacky, but good template for future mods.
    get all ways of sampling lollis, given sets of objects. 
    IN:
    - list_lollis, list of all 2-tuples (circle, line), which are lollis.
    - left_over, list of ints, those unused in list_lollis.
    - list_possible_inds, list of inds for all objects in storkes.
    --- usually just range(len(strokes))
    OUT:
    - sample randomly a single chunk that maximizes use of lollis.
    - will use each obj (int) once and onlye once.
    - will be sorted.
    """
    import random

    def _is_eligible(lolli, items_selected):
        # no items already selected
        # items_selected, flat list of inds
        for l in lolli:
            # if len(items_selected)>3:
            #     print(items_selected)
            #     assert False
            if l in items_selected:
                return False
        return True

    items_selected = []
    lollis_selected = []
    eligible_lollis = [lolli for lolli in list_lollis if _is_eligible(lolli, items_selected)]
    eligible_singles = []
    
    while len(eligible_lollis)>0:

        # Pick one
        lolli_picked = random.sample(eligible_lollis, 1)[0]
        lollis_selected.append(lolli_picked)
        items_selected.extend(lolli_picked)

        # update eligible lolli
        eligible_lollis = [lolli for lolli in list_lollis if _is_eligible(lolli, items_selected)]
        # print("---")
        # print(eligible_lollis)
        # print(lollis_selected)
        # print(items_selected)
        # eligible_singles = [it for it in left_over if _is_eligible([it], items_selected)]

    lollis_selected = sorted(lollis_selected)    

    # conver to chunk, always list of lollis (sorted) and then remianing inds
    inds_extra = [i for i in list_possible_inds if i not in items_selected]
    chunk = sort_chunk(lollis_selected + inds_extra)
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
        
        chunk = sample_a_lolli_list(list_groups, list_possible_inds)
        
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

def fixed_order_for_this_hier(hier, top_level_allow_reorder=True, 
        bottom_level_allow_reorder=True):
    """ automaticlaly generate fixed_order for this hierarchy
    PARAMS:
    - hier, list of list of ints, e.g., [[1,2], [0]]
    - top_level_allow_reorder, bottom_level_allow_reorder, bools, for whether
    allow reorder.
    """
    fixed_order = {}
    fixed_order[0] = not top_level_allow_reorder
    fixed_order[1] = [not bottom_level_allow_reorder for _ in range(len(hier))]
    return fixed_order


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

