"""
## models for discrete grammar expt
# Given list of rules, each a generative model, ask what each model woudl do,
# for each task. NOT comparing to behavior (except minor cases).
"""

import pandas as pd
import numpy as np
#### HIGH-LEVEL CODE FOR WORKING WITH TASKS 
##
## @param rule, either one word e.g. ['left', 'right'] or dashed e.g. ['rank-IVL', 'chain-LVI']
##
## @return [list_chunks, list_hier, list_fixed_order]
## - list_chunks, where each chunk is a list indicating a way to concat the strokes in Task.Strokes
## - list_hier, similar to list_chunks, but used for hierarchical permutations, without needing to concat strokes.
## - list_fixed_order, dict, what allowed to permute, and what not, for list_hier 
def find_chunks_hier(Task, expt, rulestring, strokes=None, params=None, 
    use_baseline_if_dont_find=False, DEBUG=False):
    """Find possible parses given this Task and rule. 
    PARAMS;
    - expt, often None
    - rulestring, string, usualyl format: <category>-<subcat>-<rule>
    """
    from pythonlib.chunks.chunks import sample_all_possible_chunks, clean_chunk, fixed_order_for_this_hier

    if params is None:
        params = {}
    objects = Task.Shapes
    tokens = Task.tokens_generate()
    # NOTE: objects is just
    # P = tokens[0]["Prim"]
    # objects[0] = P.extract_as("shape")

    if strokes is not None:
        from pythonlib.tools.stroketools import check_strokes_identical
        assert check_strokes_identical(Task.Strokes, strokes)
        assert len(objects)==len(strokes), "did you chunk the strokes already?"
    else:
        strokes = Task.Strokes

    def _fixed_order_for_this_hier(ruledict, hier):
        """Repository of params for deciding on fixed order
        for different levels of the hierarcy, depending on the
        rule
        """

        # NOTE: bools are NOT fixed order.
        map_rule_to_fixedorder = {
            ("ss", "rank"): [False, True], # [alow reorder, allow reorder.]
            ("ss", "chain"): [False, False],
            ("dir", "null"): [False, False],
            ("ch", "dir2"): [False, True],
            ("ch", "dir1"): [True, False],
            ("ch", "dirdir"): [False, False],
            ("rand", "null"): [True, True], # random, 
        }
        key = (ruledict["categ"], ruledict["subcat"])
        tmp = map_rule_to_fixedorder[key] # [False, True]
        return fixed_order_for_this_hier(hier, tmp[0], tmp[1])

    # @param objects, list of lists in format ['shape', {x:, y:}]
    # @param direction, one of [left,right,up,down]
    # @return objects, sorted in direction
    def _get_sequence_on_dir(objects, direction):
        """Find a single parse that reorders objects based on 
        a global spatial direction. 
        TODO: break ties. currently just takes their input order
        """
        # objects:  either:
        # [
        # ['Lcentered-3-0', {'x': -2.342, 'y': 0.05}],
        # ['V-4-0', {'x': -1.146, 'y': 0.05}],
        # ['line-3-0', {'x': 0.05, 'y': 0.05}]
        # ]
        # OR:
        # list of dict.
        def _getX(e):
            if isinstance(e, list) and len(e)==2:
                return e[1]['x']
            elif isinstance(e, dict):
                return e['x']
            else:
                print(e)

        def _getY(e):
            if isinstance(e, list) and len(e)==2:
                return e[1]['y']
            elif isinstance(e, dict):
                return e['y']
            else:
                print(e)

        # print("TODO: break ties")
        # print("OBJECTS:", objects)
        # assert False
        # NOTE: does not expect/handle ambiguous cases (i.e. two in row or col)
        if direction=='L':
            # sort by descending x
            return sorted(objects, reverse=True, key=_getX)
        elif direction=='R':
            # sort by ascending x
            return sorted(objects, key=_getX)
        elif direction=='U':
            # sort by ascending y
            return sorted(objects, key=_getY)
        elif direction=='D':
            # sort by descending y
            return sorted(objects, reverse=True, key=_getY)
        else:
            assert False, 'invalid direction'

    # @param objects, list of lists in format ['shape', {x:, y:}]
    # @param ordering_rule, one of [rank,chain]
    # @return objects, sorted in direction
    def _get_sequences_on_ordering_rule(objects, rule, shape_order):
        """Get set of parses for ordering by shapes, using
        specific strategy (e.g., rank or chain)
        """ 
        # objects: [
        # ['Lcentered-3-0', {'x': -2.342, 'y': 0.05}],
        # ['V-4-0', {'x': -1.146, 'y': 0.05}],
        # ['line-3-0', {'x': 0.05, 'y': 0.05}]
        # ]
        # print(objects)
        # print(rule)
        # print(shape_order)
        # assert False, "make sure names are correct"
        if rule=='rank':
            # AABBCC; so, need to specify A,B,C first
            return _chunks_by_shape_rank(objects, shape_order)
        elif rule=='chain':
            # ABCABC; so, need to specify A,B,C first
            return _chunks_by_shape_chain(objects, shape_order)
        else:
            print("RULETHIS", rule)
            assert False, 'invalid rule'

    def _inds_by_shape(objects, shape):
        # Return list of inds with this shape
        return [ind for ind, obj in enumerate(objects) if shape in obj[0]] # if obj[0] contains shape substring
    
    def _chunks_by_shape_rank(objects, shape_order):
        # Return [indsshape1, indsshape2, ...], where each inds is list of ints
        x = [_inds_by_shape(objects, shape) for shape in shape_order]
        x = [xx for xx in x if len(xx)>0] # e.g, if this shape exists for this trial.
        return [x]

    def _chunks_by_shape_chain(objects, shape_order):
        # takes in [['shape', {x,y}],...]

        # returns [
        # [[A1B1C],[A2B2]],
        # [[A1B2C],[A2B1]],
        # [[A2B1C],[A1B2]],
        # [[A2B2C],[A1B1]]]
        x = [_inds_by_shape(objects, shape) for shape in shape_order]
        #print("x", x)
        x_perms = _get_all_shape_ind_perms(x)
        result = []
        [result.append(_get_chain_for_single_x(xx)) for xx in x_perms]
        return result

    # gets the chain for a single x, taking the first element of each sub-list until all are exhausted
    # @param x: list of shape_ind lists, e.g. [[0], [2], [1, 3]]
    def _get_chain_for_single_x(x):
        import copy
        cp = copy.deepcopy(x) # NOTE: we don't want to change the original x..

        result = []
        while any(cp):
            subresult = []
            for shape_ind_list in cp:
                if shape_ind_list: #not-empty
                    shape_ind = shape_ind_list[0]
                    subresult.append(shape_ind)
                    shape_ind_list.remove(shape_ind) # should be unique..
            result.append(subresult)
        #print("chain for single cp", result)
        return result

    # gets all permutations of sub-lists within a list, preserving first-order
    #
    # e.g. [[0,1],[2,3],[4]] ->
    # -- [[0,1],[2,3],[4]],
    # -- [[0,1],[3,2],[4]],
    # -- [[1,0],[2,3],[4]],
    # -- [[1,0],[3,2],[4]]
    def _get_all_shape_ind_perms(shape_inds):
        result = []
        for l in shape_inds:
            subresult = itertools.permutations(l)
            subresult_list = [list(x) for x in subresult]
            result.append(subresult_list)
        #print("all_shape_ind_perms", [list(xx) for xx in itertools.product(*result)])
        return [list(xx) for xx in itertools.product(*result)]

    #### Define chunks. For now, concats are never done...
    chunks = list(range(len(objects))) # never concat strokes
    list_chunks = [chunks] # only one way

    #### Define hierarchies
    ruledict = rules_map_rulestring_to_ruledict(rulestring)
    if ruledict["categ"]=="dir":
    # if rule in ["left", "right", "up", "down"]:
        # print(objects)
        shape_order = _get_sequence_on_dir(objects, ruledict["params_good"])
        # print(shape_order)
        # assert False
        hier = [objects.index(x) for x in shape_order] # O(n^2), on short list tho so just go with it...
        list_hier = [hier]
    elif ruledict["categ"]=="ss": # shape sequence:
    # elif rule in ["rank", "chain"]:
        # assert expt in ["shapesequence2"], "code this rule up more generally, so dont need to pass in expt."

        # Map from shapeorderstring (e.g, LVl1) to list of shapes
        s_order = ruledict["params_good"]
        rank_or_chain = ruledict["subcat"]

        # if shape_order == 'IVL':
        #     s_order = ('line','V','Lcentered')
        # elif shape_order == 'LVI':
        #     s_order = ('Lcentered','V','line')
        # else:
        #     assert False, 'incorrect shape_order after rule-'
        list_hier = _get_sequences_on_ordering_rule(objects, rank_or_chain, s_order)
    elif ruledict["categ"] == "ch" and ruledict["subcat"] == "dir2":
        # Concrete chunk, with direction across chunks fixed, but
        # direction within variable (i.e., chunk_mask)
        # Dir across chunks is defined by param

        paramsthis = {
            # "expt":expt,
            "rule":"concrete_chunk",
            "shapes_in_order":ruledict["params_good"][0],
            "orientation":ruledict["params_good"][1]
        }
        list_groups, left_over = find_object_groups_new(Task, paramsthis)
        

        # 1) Get all hierarchices, holding hcyunks constaint
        list_possible_inds = list(range(len(tokens)))
        list_hier = sample_all_possible_chunks(list_groups, list_possible_inds)

        # Reorder each of the hier by direction in space
        direction = ruledict["params_good"][2]

        # concatenate each chunk into a temporary object, with mean positions
        def _mean_loc(inds_taskstrokes):
            tokens_this = [tokens[i] for i in inds_taskstrokes]
            tok = Task.tokens_concat(tokens_this)
            return {
                "x": tok["gridloc"][0],
                "y": tok["gridloc"][1],
                "inds_taskstrokes": inds_taskstrokes
            }

        list_hier_reordered = []
        for hier in list_hier:
            locations = [_mean_loc(inds_taskstrokes) for inds_taskstrokes in hier]
            locations = _get_sequence_on_dir(locations, direction)
            hier_reordered = [x["inds_taskstrokes"] for x in locations]
            list_hier_reordered.append(hier_reordered)
        list_hier = list_hier_reordered

    elif ruledict["categ"] == "alternation":
        assert False, "FILL THIS IN"

        #             # Alternate between lines and circles
        #             # - pick a random circle, then 
        #             import random

        #             def _eligible_tokens(tokens_remain, tokens_taken):
        #                 """ 
        #                 all are lists of indices
        #                 """
        #                 # only those not taken and not identical shape to prev taken
        #                 if len(tokens_taken)==0:
        #                     tokens_elegible = tokens_remain
        #                 else:
        #                     tokens_elegible = [t for t in tokens_remain if list_tokens[t] != list_tokens[tokens_taken[-1]]]
        #                     if len(tokens_elegible)==0 and len(tokens_remain)>0:
        #                         # then open up eligibility to all tokens
        #                         tokens_elegible = tokens_remain
        #                 return tokens_elegible

        #             def _sample_token(tokens_remain, tokens_taken):
        #                 tokens_elig = _eligible_tokens(tokens_remain, tokens_taken)
        #                 ind_tok = random.choice(tokens_elig)
                        
        #                 tokens_taken.append(ind_tok)
        #                 tokens_remain = [t for t in tokens_remain if t!=ind_tok]
        #                 return tokens_remain, tokens_taken
                        
        #             def _sample_single_chunk(list_tokens):
        #                 tokens_remain = range(len(list_tokens))
        #                 tokens_taken = []
        #                 while len(tokens_remain)>0:
        #                     tokens_remain, tokens_taken = _sample_token(tokens_remain, tokens_taken)
        #                 return tokens_taken

        #             # list_tokens = ["line", "circle", "line", "circle"]
        #             list_tokens = [o[0] for o in objects]
        #             list_chunks = []
        #             list_hier = []
        #             Nsamp = 20
        #             for _ in range(Nsamp):
        #                 list_hier.append(_sample_single_chunk(list_tokens))
        #                 list_chunks.append(list(range(len(list_tokens))))
    elif ruledict["categ"]=="rand":
        # Random sampling
        list_hier = [list(range(len(objects)))]

    else:
        print(rulestring)
        print(ruledict)
        assert False, "code up rule"

    ### Clean up the chunks
    list_hier = [clean_chunk(hier) for hier in list_hier]

    ### Expand chunks and fixed order to match number of identified hiers
    list_chunks = [chunks for i in range(len(list_hier))] # repeat chunks as many times as there are possible hiers
    list_fixed_order = [_fixed_order_for_this_hier(ruledict, hier) for hier in list_hier]
    
    if DEBUG:
        print("list_chunks")
        for chunk in list_chunks:
            print(chunk)
        print("list_hier")
        for hier in list_hier:
            print(hier)
        print("list_fixed_order")
        for fo in list_fixed_order:
            print(fo)

    # How to deal with hier that dont get all strokes.
    # Replace it entirely with a single random parse.
    def _random_parse(tokens):
        """ sampel a single ranodm parse, fixed order
        """
        import random
        tokens_indices = range(len(tokens))
        hier = [[i] for i in random.sample(tokens_indices, len(tokens))]
        chunks = [[i] for i in range(len(tokens))]
        fixed_order = fixed_order_for_this_hier(hier, False, False)
        return hier, chunks, fixed_order

    inds_all = list(range(len(tokens)))
    for i, hier in enumerate(list_hier):
        inds_used = [xx for x in hier for xx in x]
        inds_not_used = [ind for ind in inds_all if ind not in inds_used]
        if len(inds_not_used)>0:
            # replace this with a random parse
            h, c, f = _random_parse(tokens)
            list_hier[i] = h
            list_chunks[i] = c
            list_fixed_order[i] = f

    # Sanity checks: Should always be something
    for hier in list_hier:
        if len(hier)==0:
            print(list_hier)
            print(ruledict)
            Task.plotStrokes()
            assert False
    if len(list_hier)==0:
        print(ruledict)
        Task.plotStrokes()
        assert False
    
    # Return as a list of possible chunkings.
    assert len(list_chunks)==len(list_hier) # TODO: check, is this necesary?
    assert len(list_hier)==len(list_fixed_order)
    assert len(list_hier)>0, "why empty?"
    assert isinstance(list_hier[0], list)
    assert isinstance(list_hier[0][0], list)

    return list_chunks, list_hier, list_fixed_order 


# #### HIGH-LEVEL CODE FOR WORKING WITH TASKS 
# def find_chunks_wrapper(Task, expt, rule, strokes=None, params = None,
#     use_baseline_if_dont_find=False):
#     """ [OLD] General purpose to return chunks, hierarchies,
#     and what kind of permutations allwoed, givne some kind of input
#     model, etc, rules, etc.
#     INPUT:
#     - Task, TaskGeneral instance
#     - strokes, generally is same as Task.Strokes, but if enter, then assesrts that
#     OUT:
#     - list_chunks, where each chunk is
#     a list indictating a way to chunk the strokes in Task.Strokes
#     - list_hier, similar to chunks, but used for hierarhical permutations, without needing
#     to concat storkes.
#     - list_fixed_order, dict, what allowed to permute, and what not, for list_hier 
#     #TODO: 
#     - should return empty chunks if there are not reasonable chunks? e..g lolli rule
#     will still output all strokes even if there are no lollis at all.
#     - NOTE: order within and across chunks will not matter. Identity is determined by
#     sorting lists of tuples. 
#     """

#     assert False, "use new function find_chunks_hier"
#     objects = Task.Shapes
#     print(objects)
#     if strokes is not None:
#         from pythonlib.tools.stroketools import check_strokes_identical
#         assert check_strokes_identical(Task.Strokes, strokes)
#         assert len(objects)==len(strokes), "did you chunk the strokes already?"
#     else:
#         strokes = Task.Strokes
        
#     def _inds_by_shape(shape):
#         # Return list of inds with this shape
#         return [i for i, x in enumerate(objects) if x[0]==shape]
#     def _chunks_by_shapes_inorder(list_shapes):
#         # Return [indsshape1, indsshape2, ...], where each inds is lsit of int
#         x = [_inds_by_shape(shape) for shape in list_shapes]
#         x = [xx for xx in x if len(xx)>0] # e.g, if this shape doesnt exist for this trial.
#         return x
    
#     # print(Task)
#     # print(expt)
#     # print(rule)
#     # print(params)

#     # Find list_chunks and list_hier - note: they will be same lenght, 
#     if expt in ["gridlinecircle", "neuralbiasdir5c"]:
#         chunks = list(range(len(objects))) # never concat strokes
#         list_chunks = [chunks] # only one way

#         if rule =="baseline":
#             # circle, line, but no order
#             hier = chunks
#             list_hier = [hier]
#         elif rule == "circletoline":
#             # first circles, then lines
#             list_shapes = ["circle", "line"]
#             hier = _chunks_by_shapes_inorder(list_shapes)

#             list_hier = [hier]
#         elif rule=="linetocircle":
#             list_shapes = ["line", "circle"]
#             hier = _chunks_by_shapes_inorder(list_shapes)
#             list_hier = [hier]
#         elif rule=="lolli":
#             # Returns both (no chunking, all hier) and (chukjning, no hier), combined
#             # into long list
#             # 1) NOT concating
#             # find all ways of exhausting the objects with combos of lollis.
#             paramsthis = {
#                 "expt":expt,
#                 "rule":rule,
#                 "ver":"lolli"
#             }
#             list_lollis, left_over = find_object_groups_new(Task, paramsthis)

#             if len(list_lollis)==0:
#                 # then no lollis. then just skip this, since is same as baseline
#                 list_hier = [chunks]
#             else:

#                 # 1) Get all hierarchices, holding hcyunks constaint
#                 list_hier = sample_all_possible_chunks(list_lollis, list(range(len(strokes))))
#                 list_chunks = [chunks for _ in range(len(list_hier))]

#                 # 2) Get all chunks, concating (so replicating hierahcy)
#                 list_chunks_toadd = list_hier # concat strokes
#                 list_hier_toadd = [list(range(len(h))) for h in list_hier] # no hierarchy

#                 ## combine 1 and 2
#                 list_chunks += list_chunks_toadd
#                 list_hier += list_hier_toadd

#         elif rule=="alternate":
#             # Alternate between lines and circles
#             # - pick a random circle, then 
#             import random

#             def _eligible_tokens(tokens_remain, tokens_taken):
#                 """ 
#                 all are lists of indices
#                 """
#                 # only those not taken and not identical shape to prev taken
#                 if len(tokens_taken)==0:
#                     tokens_elegible = tokens_remain
#                 else:
#                     tokens_elegible = [t for t in tokens_remain if list_tokens[t] != list_tokens[tokens_taken[-1]]]
#                     if len(tokens_elegible)==0 and len(tokens_remain)>0:
#                         # then open up eligibility to all tokens
#                         tokens_elegible = tokens_remain
#                 return tokens_elegible

#             def _sample_token(tokens_remain, tokens_taken):
#                 tokens_elig = _eligible_tokens(tokens_remain, tokens_taken)
#                 ind_tok = random.choice(tokens_elig)
                
#                 tokens_taken.append(ind_tok)
#                 tokens_remain = [t for t in tokens_remain if t!=ind_tok]
#                 return tokens_remain, tokens_taken
                
#             def _sample_single_chunk(list_tokens):
#                 tokens_remain = range(len(list_tokens))
#                 tokens_taken = []
#                 while len(tokens_remain)>0:
#                     tokens_remain, tokens_taken = _sample_token(tokens_remain, tokens_taken)
#                 return tokens_taken

#             # list_tokens = ["line", "circle", "line", "circle"]
#             list_tokens = [o[0] for o in objects]
#             list_chunks = []
#             list_hier = []
#             Nsamp = 20
#             for _ in range(Nsamp):
#                 list_hier.append(_sample_single_chunk(list_tokens))
#                 list_chunks.append(list(range(len(list_tokens))))

#         else:
#             assert False

#         ##### Fixed order
#         def _fixed_order_for_this_hier(hier):
#             if rule in ["baseline", "lolli"]:
#                 # Order allows all reordering
#                 fixed_order = fixed_order_for_this_hier(hier, True, True)
#             elif rule in ["circletoline", "linetocircle"]:
#                 # Order only allows for reordering both hier levels.
#                 fixed_order = fixed_order_for_this_hier(hier, False, True)
#             elif rule in ["alternate"]:
#                 # becasue hier is a specific sequence for altenration
#                 # e..g, hier = [0, 1, 2, 3]
#                 fixed_order = fixed_order_for_this_hier(hier, False, False)
#             else:
#                 assert False
#             return fixed_order
#         list_fixed_order = [_fixed_order_for_this_hier(hier) for hier in list_hier]

#         # Make sure no redundant ones (identical in chunk, hier, and fixed)
#         def _print(l):
#             print("--")
#             for ll in l:
#                 print(ll)
#         # print("_____")
#         # _print(list_chunks)
#         # _print(list_hier)

#         list_chunks_good = [list_chunks[0]]
#         list_hier_good = [list_hier[0]]
#         list_fixed_order_good = [list_fixed_order[0]]

#         for ch, hi, fi in zip(list_chunks[1:], list_hier[1:], list_fixed_order[1:]):
#             # Check against all already gotten
#             is_good = True
#             for chB, hiB, fiB in zip(list_chunks_good, list_hier_good, list_fixed_order_good):
#                 if chunks_are_identical_full(ch, hi, fi, chB, hiB, fiB):
#                     is_good = False
#                     continue
#             if is_good:
#                 list_chunks_good.append(ch)
#                 list_hier_good.append(hi)
#                 list_fixed_order_good.append(fi)
#         list_chunks = list_chunks_good
#         list_hier = list_hier_good
#         list_fixed_order = list_fixed_order_good

#         # list_chunks_good = [list_chunks[0]]
#         # list_hier_good = [list_hier[0]]
#         # list_fixed_order_good = [list_fixed_order[0]]
#         # for ch1, hi1, in zip(list_chunks[1:], list_hier[1:]):
#         #     # check if is in
#         #     # if any([chunks_are_identical(ch, x) for x in list_chunks_good]):
#         #     #     continue
#         #     if any([chunks_are_identical_full(ch1, hi1, fi1, ch2, hi2, fi2) for ch2, hi2 in zip(list_chunks_good, list_hier_good)]):
#         #         continue
#         #     else:
#         #         list_chunks_good.append(ch)
#         #         list_hier_good.append(hi)
#         # list_chunks = list_chunks_good
#         # list_hier = list_hier_good
#         # _print(list_chunks)
#         # _print(list_hier)
#         # print("_____")

#     else:
#         print(params)
#         assert False, "code up expt"
    
#     # Remove anything that is just same as baseline
#     if use_baseline_if_dont_find == False:
#         if not rule=="baseline":
#             list_chunks_baseline, list_hier_baseline, list_fixed_order_baseline = find_chunks_wrapper(Task, 
#                 expt, "baseline", strokes, params)
            
#             list_chunks_good = []
#             list_hier_good = []
#             list_fixed_order_good = []

#             for ch, hi, fi in zip(list_chunks, list_hier, list_fixed_order):
#                 for chB, hiB, fiB in zip(list_chunks_baseline, list_hier_baseline, list_fixed_order_baseline):
#                     if chunks_are_identical_full(ch, hi, fi, chB, hiB, fiB):
#                         continue
#                     else:
#                         list_chunks_good.append(ch)
#                         list_hier_good.append(hi)
#                         list_fixed_order_good.append(fi)
#             list_chunks = list_chunks_good
#             list_hier = list_hier_good
#             list_fixed_order = list_fixed_order_good
#     # Return as a list of possible chunkings.
#     assert len(list_chunks)==len(list_hier)
#     assert len(list_hier)==len(list_fixed_order)
#     # _print(list_chunks)
#     # _print(list_hier)
#     # print(list_chunks)
#     # print(list_hier)
#     # print(list_fixed_order)
#     # assert False

#     return list_chunks, list_hier, list_fixed_order
    

def find_object_groups_new(Task, params):
    """ return list of groups (list of objects/shapes) passing constraints.
    Uses tokens from Task.tokens_generate(), and assumes this is unordered list. 
    So allows for any ordering of tokens, which means this outputs different ways
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
    tokens = Task.tokens_generate(assert_computed=True)
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
            return "U"
        elif np.isclose(xdiff, 0.) and ydiff ==-1.:
            return "D"
        elif xdiff ==-1. and np.isclose(ydiff, 0.):
            return "L"
        elif xdiff ==1. and np.isclose(ydiff, 0.):
            return "R"
        else:
            # FAR
            return "f"
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


    if params["rule"]=="concrete_chunk":
        # concrete chunk is grouping of defined shape/locations. spatially defined, (i.e,, any temporal order)
        shapes_in_order = params["shapes_in_order"]
        orientation = params["orientation"]
        # e.g {'rule': 'concrete_chunk', 'shapes_in_order': ['line-8-4-0', 'line-8-3-0'], 'orientation': 'U'}
        if len(shapes_in_order)>2:
            print(shapes_in_order)
            assert False, "# not coded yet, assuming length 2"
        
        list_groups = [] # 
        for i in range(len(objects)):
            for j in range(len(objects)):
                if _oshape(i)==shapes_in_order[0] and _oshape(j)==shapes_in_order[1] and _direction_grid(i, j)==orientation:
                    # Found a chunk, include it
                    if False:
                        print("FOUDN THIS")
                        print(i, j)
                        print(_oshape(i), _oshape(j), _direction_grid(i, j))
                    list_groups.append([i, j])

                # assert False
                # if _oshape(i)=="circle" and _oshape(j)=="hline":
                #     if _direction_grid(i, j) in ["left", "right"]:
                #         list_groups.append([i,j])
                # elif _oshape(i)=="circle" and _oshape(j)=="vline":
                #     if _direction_grid(i, j) in ["up", "down"]:
                #         list_groups.append([i,j])
                # elif _oshape(i)=="hline" and _oshape(j)=="circle":
                #     if _direction_grid(i, j) in ["left", "right"]:
                #         list_groups.append([i,j])
                # elif _oshape(i)=="vline" and _oshape(j)=="circle":
                #     if _direction_grid(i, j) in ["up", "down"]:
                #         list_groups.append([i,j])

        # Find what objects are left over
        list_groups_flat = [xx for x in list_groups for xx in x]
        left_over = [i for i in range(len(objects)) if i not in list_groups_flat]
    else:
        print(params)
        assert False, "code it"




    # # Given a rule for chunking, extract all the ways of chunking
    # if params["expt"]=="gridlinecircle" and params["rule"]=="lolli":
    #     # get list of all lollis possible
    #     list_groups = [] # aleays [(circle, line), ...]

    #     for i in range(len(objects)):
    #         for j in range(i+1, len(objects)):
    #             # print(i, j)
    #             # print(_oshape(i), _oshape(j), _direction_grid(i, j))
    #             if _oshape(i)=="circle" and _oshape(j)=="hline":
    #                 if _direction_grid(i, j) in ["left", "right"]:
    #                     list_groups.append([i,j])
    #             elif _oshape(i)=="circle" and _oshape(j)=="vline":
    #                 if _direction_grid(i, j) in ["up", "down"]:
    #                     list_groups.append([i,j])
    #             elif _oshape(i)=="hline" and _oshape(j)=="circle":
    #                 if _direction_grid(i, j) in ["left", "right"]:
    #                     list_groups.append([i,j])
    #             elif _oshape(i)=="vline" and _oshape(j)=="circle":
    #                 if _direction_grid(i, j) in ["up", "down"]:
    #                     list_groups.append([i,j])
    #     # Find what objects are left over
    #     list_groups_flat = [xx for x in list_groups for xx in x]
    #     left_over = [i for i in range(len(objects)) if i not in list_groups_flat]

    # # print(list_groups)
    # # assert False 
    return list_groups, left_over



# def find_object_groups(Task, params):
#     """ return list of groups (list of obj) passing constraints.
#     General-purpse, takes in objects (Task.Shapes). This does not care aobut the 
#     ordering in Task tokens, instead is about goruping in space.
#     e..g, find list of lollis, where each lolli is a set of 2 inds in objects
#     NOTE:
#     - a given object can be used more than once, or not at all,
#     e.g., if it participates in multiple gourpings.
#     """
#     from math import pi
    
#     assert False, "use find_object_groups_new"
#     expt = params["expt"]
#     rule = params["rule"]
#     if expt=="gridlinecircle":
#         xgrid = np.linspace(-1.7, 1.7, 3)
#         ygrid = np.linspace(-1.7, 1.7, 3)
#     else:
#         assert False
    
#     # 1) assign each object a grid location
#     objects = Task.Shapes
#     locations = []
#     for o in objects:
#         xloc = o[1]["x"]
#         yloc = o[1]["y"]
#         xind = int(np.where(xgrid==xloc)[0])
#         yind = int(np.where(ygrid==yloc)[0])
#         locations.append((xind, yind))
        
#     def _shape(i):
#         # return string
#         return objects[i][0]
    
#     def _posdiffs(i, j):
#         # return xdiff, ydiff, 
#         # in grid units.
#         pos1 = locations[i]
#         pos2 = locations[j]
#         return pos2[0]-pos1[0], pos2[1] - pos1[1]
        
#     def _direction(i, j):
#         # only if adjacnet on grid.
#         xdiff, ydiff = _posdiffs(i,j)
#         if np.isclose(xdiff, 0.) and ydiff ==1.:
#             return "up"
#         elif np.isclose(xdiff, 0.) and ydiff ==-1.:
#             return "down"
#         elif xdiff ==-1. and np.isclose(ydiff, 0.):
#             return "left"
#         elif xdiff ==1. and np.isclose(ydiff, 0.):
#             return "right"
#         else:
#             return "far"

#     def _orient(i):
#         if np.isclose(objects[i][1]["theta"], 0.):
#             return "horiz"
#         elif np.isclose(objects[i][1]["theta"], pi):
#             return "horiz"
#         elif np.isclose(objects[i][1]["theta"], pi/2):
#             return "vert"
#         elif np.isclose(objects[i][1]["theta"], 3*pi/2):
#             return "vert"
#         else:
#             print(objects[i])
#             assert False


#     # What to look for
#     if params["ver"] == "lolli":
#         # get list of all lollis possible
#         list_lollis = [] # aleays [(circle, line), ...]
#         for i, o1 in enumerate(objects):
#             for j, o2 in enumerate(objects):
#                 if _shape(i)=="circle" and _shape(j)=="line":
#                     if _direction(i, j)=="left" and _orient(j) == "horiz":
#                         list_lollis.append([i, j])
#                     elif _direction(i, j)=="up" and _orient(j) == "vert":
#                         list_lollis.append([i, j])
#                     elif _direction(i, j)=="right" and _orient(j) == "horiz":
#                         list_lollis.append([i, j])
#                     elif _direction(i, j)=="down" and _orient(j) == "vert":
#                         list_lollis.append([i, j])
        
#         # Find what objects are left over
#         list_lollis_flat = [xx for x in list_lollis for xx in x]
#         left_over = [i for i in range(len(objects)) if i not in list_lollis_flat]

#         return list_lollis, left_over
#     else:
#         print(params)
#         assert False
        
                    

def _get_default_grouping_map_tasksequencer_to_rule():
    """ Dict that maps tasksequencer params (which in matlab
    dictates the sequencing rule for each block) to a string name for the 
    rule. Hard code these, but they are general across expts
    """
    grouping_map_tasksequencer_to_rule = {}
    grouping_map_tasksequencer_to_rule[(None, None)] = "base"

    grouping_map_tasksequencer_to_rule[("direction", "3.14")] = "L"
    grouping_map_tasksequencer_to_rule[("direction", "0.00")] = "R"


    grouping_map_tasksequencer_to_rule[("directionv2", ("lr",))] = "R"
    grouping_map_tasksequencer_to_rule[("directionv2", ("rl",))] = "L"
    grouping_map_tasksequencer_to_rule[("directionv2", ("ud",))] = "D"
    grouping_map_tasksequencer_to_rule[("directionv2", ("du",))] = "U"

    grouping_map_tasksequencer_to_rule[("directionv2", ("right",))] = "R"
    grouping_map_tasksequencer_to_rule[("directionv2", ("left",))] = "L"
    grouping_map_tasksequencer_to_rule[("directionv2", ("down",))] = "D"
    grouping_map_tasksequencer_to_rule[("directionv2", ("up",))] = "U"
    grouping_map_tasksequencer_to_rule[("directionv2", ("topright",))] = "TR"

    grouping_map_tasksequencer_to_rule[("prot_prims_in_order", ('line-8-3', 'V-2-4', 'Lcentered-4-3'))] = "lVL1"
    grouping_map_tasksequencer_to_rule[("prot_prims_in_order", ('Lcentered-4-3', 'V-2-4', 'line-8-3'))] = "LVl1"
    grouping_map_tasksequencer_to_rule[("prot_prims_in_order", ('V-2-4', 'line-8-3', 'Lcentered-4-3'))] = "VlL1"
    
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('line-8-3', 'line-8-4', 'V-2-4'))] = "llV1"
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('line-9-3', 'line-9-4', 'Lcentered-6-8'))] = "llV1"
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('line-8-3', 'line-13-13', 'line-8-4', 'line-13-14', 'V-2-4', 'V2-2-4'))] = "llV1"
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('line-8-3', 'line-13-13', 'line-8-4', 'line-13-14', 'V-2-4', 'V2-2-4', 'V2-2-2'))] = "llV1"
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('V2-2-2', 'V2-2-4', 'V-2-4', 'line-13-14', 'line-8-4', 'line-13-13', 'line-8-3'))] = "llV1R"

    grouping_map_tasksequencer_to_rule[("prot_prims_chunks_in_order", ('line-8-4', 'line-8-3'))] = "AnBm1a"
    grouping_map_tasksequencer_to_rule[("prot_prims_chunks_in_order", ('line-8-1', 'line-8-2'))] = "AnBm2"
    grouping_map_tasksequencer_to_rule[("prot_prims_chunks_in_order", ('line-8-4', 'line-11-1', 'line-8-3', 'line-11-2'))] = "AnBm1b"
    grouping_map_tasksequencer_to_rule[("prot_prims_chunks_in_order", ('line-11-1', 'line-11-2'))] = "AnBmHV"
    grouping_map_tasksequencer_to_rule[("prot_prims_chunks_in_order", ('squiggle3-3-1', 'V-2-4'))] = "AnBm0"

    grouping_map_tasksequencer_to_rule[("hack_220829", tuple(["hack_220829"]))] = "(AB)n"

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('line-8-4', 'line-11-1', 'line-8-3', 'line-11-2', 'topright'))] = "AnBmTR"

    grouping_map_tasksequencer_to_rule[('randomize_strokes', tuple(["randomize_strokes"]))] = "rndstr"

    return grouping_map_tasksequencer_to_rule



# def map_epoch_rule_to_acceptable_rulestrings(list_epoch_rule):
#     return _rules_consistent_rulestrings_extract_auto(list_epoch_rule)

def rules_map_rulestring_to_ruledict(rulestring):
    """ Map from python string rule repreantation to matlab params:
    PARAMS:
    - rulestring, <rulecategory>-<subcategory>-<params>
    EG:
    - rulestring = ss-chain-LVl1
    Returns: 
        {'categ_matlab': 'prot_prims_in_order',
         'params_matlab': ('Lcentered-4-3', 'V-2-4', 'line-8-3'),
         'categ': 'ss',
         'subcat': 'chain',
         'params': 'LVl1'}
    """
    from pythonlib.tools.stringtools import decompose_string

    grouping_map_tasksequencer_to_rule = _get_default_grouping_map_tasksequencer_to_rule()

    # 1) decompose string
    substrings = decompose_string(rulestring)
    assert len(substrings)==3, "must be <rulecategory>-<subcategory>-<params>"
    categ = substrings[0]
    subcat = substrings[1]
    params = substrings[2]

    def _find_params_matlab(params):
        FOUND = False
        for key, val in grouping_map_tasksequencer_to_rule.items():
            if val==params:
                if FOUND:
                    print(categ_matlab, params_matlab)
                    print(key)
                    print(rulestring)
                    print(val, params)
                    assert False, "this val is in multiple items; use category to further refine"
                categ_matlab = key[0] # e.g., prot_prims_in_order
                params_matlab = key[1] # e.g., ('line-8-3', 'V-2-4', 'Lcentered-4-3')
                FOUND = True
        assert FOUND, "did not find this val"
        return categ_matlab, params_matlab

    # FInd the matlab params
    if categ=="ss":
        # shape orders are encoded in matlab parmas:
        # 2) find it in grouping_map_tasksequencer_to_rule
        categ_matlab, params_matlab = _find_params_matlab(params)
    # elif categ=="ch":
    #     # chunk, direction across chunks, and also within chunks --> i.e. only single correct sequewnce.
    #     assert False, "code it"

        # then the shape represntations may be wrong
        # num dashes
        def _convert_shape_string(s):
            # convert from Lcentered-4-3 to Lcentered-4-3-0
            substrings = decompose_string(s)
            if not len(substrings)==3:
                print(s)
                assert False, "expect like Lcentered-4-3"
            else:
                # then is liek: Lcentered-4-3, which has scale-rotation. assumes reflect is 0.
                # convert to: shape-rotation-reflect
                shape = substrings[0]
                scale = substrings[1]
                rot = substrings[2]
                refl = 0
                return f"{shape}-{scale}-{rot}-0"
        list_shapestring_good = [_convert_shape_string(shapestring) for shapestring in params_matlab]
        params_good = list_shapestring_good

    elif categ=="ch" and subcat=="dir2":
        # Concrete chunk, with direction across chunks fixed, but
        # direction within variable (i.e., chunk_mask)
        categ_matlab, params_matlab = _find_params_matlab(params)
        if categ_matlab=="hack_220829":
            # categ_matlab = "shape_chunk_concrete"
            # params_matlab = ("")
            # # e..g, ('lolli', {'D', 'R'}).
            shapes_in_order = ["line-8-4-0", "line-8-3-0"]
            rel_shapes = "U"
            direction = "R" # chunk to chunk.
        params_good = (shapes_in_order, rel_shapes, direction)
    elif categ=="dir":
        # Directions using string keys, no need to look at matlab params
        categ_matlab = None
        params_matlab = None
        params_good = params
        assert False, "check that params_ggood is string like R"
    elif categ=="rand" and subcat=="null":
        # Random beh
        categ_matlab = None
        params_matlab = None
        params_good = params
    else:
        print(categ)
        assert False, "code it"

    # # 3) Clean up the shapes
    # if categ_matlab in ["prot_prims_in_order", "prot_prims_chunks_in_order", 
    #         "prot_prims_in_order_AND_directionv2"]:
    # else:
    #     params_good = params
    #     # print(categ_matlab)
    #     # assert False, "code it"

    # 3) return as params.
    out = {
        "categ_matlab":categ_matlab,
        "params_matlab":params_matlab,
        "params_good":params_good,
        "categ":categ,
        "subcat":subcat,
        "params":params,
        "rulestring":rulestring}

    return out

def _get_rank_and_chain_variations(list_shape_orders):
    list_shape_orders_rankchain = []
    for order in list_shape_orders:
        list_shape_orders_rankchain.append(f"ss-rank-{order}")
        list_shape_orders_rankchain.append(f"ss-chain-{order}")
    return list_shape_orders_rankchain
def _get_direction_variations(list_dir):
    # e..g, list_dir = ["D", "U", "R", "L"]
    return [f"dir-null-{x}" for x in list_dir]
def _get_chunk_dirdir_variations(list_rule):
    """ chunk with direction both within and across, only one correct sequence"""
    return [f"ch-dirdir-{x}" for x in list_rule]
def _get_chunk_dir2_variations(list_rule):
    """ specific direction across chunks, but not within
    aka. chunk_mask in matlab.
    """
    return [f"ch-dir2-{x}" for x in list_rule]

def _rules_consistent_rulestrings_extract_auto(list_rules, debug=False, return_as_dict=False):
    """ 
     
    Find rulestrings that, if beh matches any of these, would lead to 
    behavior being called a correct trial.
    PARAMS:
    - list_epoch_rule, list of str, such as "(AB)n"
    - list_rules_related, relatied (alt hypothes) rulestrings.
    e.g,, bm.DictMclassToRules[mclass]
    RETURNS:
    - list of list fo str, where inner lists are lists of rules accepatable for each rule.
    
    list rules, list of str, i.e, epoch, such as "R"
    DICT_RULESTRINGS_CONSISTENT = {
        # ("LVl1", "lVL1", "VlL1"):_get_rank_and_chain_variations(("LVl1", "lVL1", "VlL1")),
        ("(AB)n", "AnBm1a"):_get_chunk_dir2_variations(["(AB)n"]) + ["ss-rank-AnBm1a"], # grammar1
        ("(AB)n", "AnBm1a"):_get_chunk_dir2_variations(["(AB)n"]) + ["ss-rank-AnBm1a"], # grammar1
        ("AnBm2"):["ss-rank-AnBm2"] # grammar2
        # ("AnBm2"):["ss-rank-AnBm2", "ss-rank-AnBm1a"] # grammar2
    }
    RETURNS:
    - list of list fo str, where inner lists are lists of rules accepatable for each rule.
    """
    assert isinstance(list_rules, list)

    DICT_RULESTRINGS_CONSISTENT = {}
    for r in ["D", "U", "R", "L"]:
        DICT_RULESTRINGS_CONSISTENT[r] = _get_direction_variations([r])
    for r in ["LVl1", "lVL1", "VlL1"]:
        DICT_RULESTRINGS_CONSISTENT[r] = _get_rank_and_chain_variations([r])
    for r in ["AnBm2", "AnBm1a"]:
        DICT_RULESTRINGS_CONSISTENT[r] = [f"ss-rank-{r}"]
    for r in ["(AB)n"]:
        DICT_RULESTRINGS_CONSISTENT[r] = _get_chunk_dir2_variations(["(AB)n"])

    if debug:
        for k, v in DICT_RULESTRINGS_CONSISTENT.items():
            print(k, ' -- ', v)
    for r in list_rules:
        if r not in DICT_RULESTRINGS_CONSISTENT.keys():
            print(r)
            print(DICT_RULESTRINGS_CONSISTENT)
            assert False, "add it."

    x = [DICT_RULESTRINGS_CONSISTENT[r] for r in list_rules]
    if return_as_dict:
        return [[rules_map_rulestring_to_ruledict(xxx) for xxx in xx] for xx in x]
    else:
        return x



def rules_map_rule_to_ruledict_extract_auto(D):
    """for each rule, get its ruledict
    RETURNS:
    - dicst, rule --> ruledict
    """
  
    list_rulestring = rules_related_rulestrings_extract_auto(D)

    map_rule_to_ruledict = {}
    for rs in list_rulestring:
        rule_dict = rules_map_rulestring_to_ruledict(rs)
        rule = rule_dict["params"]
        map_rule_to_ruledict[rule] = rule_dict
        
    return map_rule_to_ruledict


def rules_related_rulestrings_extract_auto(D):
    """ Helper to try to extract all relevant rules, based on:
    (i) the groundt truth rules in D< and (ii) related rules that
    are alternative huypotjeses to those rules
    """
    list_rules = D.Dat["epoch_rule_tasksequencer"].unique().tolist()
    return _rules_related_rulestrings_extract_auto(list_rules)

def _rules_related_rulestrings_extract_auto(list_rules):
    """
    Helper to get rulestrings that are related (i.e, altnerative hypotheses) to these
    rules
    list_rules, list of str, i.e, epochs, etc:
    list_rules = sorted(D.Dat["epoch_rule_tasksequencer"].unique().tolist())
    RETURNS: 
    - list of rulestrings which are considered related to any of the input rules
    (concatnated).
    """
    from pythonlib.tools.stringtools import decompose_string


    # # Get the consistent rulestrings for this rule
    # for rule in list_rules:
    #     DICT_RELATED_RULES[rule]
    # assert False

    DICT_RELATED_RULES = {
        # ("LVl1", "lVL1", "VlL1"):_get_rank_and_chain_variations(("LVl1", "lVL1", "VlL1")),
        ("LVl1", "lVL1", "VlL1"):_get_rank_and_chain_variations(("LVl1", "lVL1", "VlL1")),
        ("D", "U", "R", "L"):_get_direction_variations(["D", "U", "R", "L"]),
        ("(AB)n", "AnBm1a"):_get_chunk_dir2_variations(["(AB)n"]) + ["ss-rank-AnBm1a"], # grammar1
        ("AnBm2"):["ss-rank-AnBm2"] # grammar2
        # ("AnBm2"):["ss-rank-AnBm2", "ss-rank-AnBm1a"] # grammar2
    }
    RULES_IGNORE = ["base", "baseline"] # rules to ignore. assumed that other rules int he same day will
    # bring in all the rules.

    # 1) list of rules present in D
    # list_rules_dat = sorted(D.Dat["epoch_rule_tasksequencer"].unique().tolist())

    # 2) for each rule there, get "related" rules from a database
    def _find_related_rules(rulethis):
        related_rules = []
        FOUND = False
        for rule_keys, rule_set in DICT_RELATED_RULES.items():
            if rulethis in RULES_IGNORE:
                return []
            elif rulethis in rule_keys:
                FOUND = True
                related_rules.extend(rule_set)
        assert FOUND, f"didnt find this rule in any sets: {rulethis}"
        return list(set(related_rules))

    list_rules_related =[]
    for rulethis in list_rules:
        list_rules_related.extend(_find_related_rules(rulethis))

    # 3) combine
    # list_rules_all = list_rules + list_rules_related
    list_rules_all = list_rules_related

    # sanity check
    for rule in list_rules_all:
        assert len(decompose_string(rule))==3, "needs to be cat-subcat-rulename"
    return sorted(list(set(list_rules_all)))






#################### CATEGORIZE TASKS BASED ON SEQUENCE FEATURES
# e..g, ngram (AABBB)

def tasks_categorize_based_on_rule_mult(D, HACK=True):
    """
    """
    # Extract for each rule each tasks' categroyes
    from pythonlib.dataset.modeling.discrete import tasks_categorize_based_on_rule, rules_map_rule_to_ruledict_extract_auto
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    list_rule = []
    for rule, ruledict in rules_map_rule_to_ruledict_extract_auto(D).items():
        if ruledict["categ"]=="ss":
            # keep it. a shape sequence
            # list_rulestring.append(ruledict["rulestring"])
            list_rule.append(rule)

    if HACK:
        # just take the first ruel..
        rule = list_rule[0]          
    # for rule in list_rule:
    #     # rule = "AnBm1a"
        OUT, LIST_COLNAMES = tasks_categorize_based_on_rule(D, rule)
        # print("ASDASD", list_rule)

        for col in LIST_COLNAMES:
            D.Dat[col] = pd.DataFrame(OUT)[col]
        # print(rule)
        # print(OUT)
        # assert False

    # Assign mew col
    def F(x):
        return tuple([x[colname] for colname in LIST_COLNAMES])
    D.Dat = applyFunctionToAllRows(D.Dat, F, "taskcat_by_rule")
    print("New col: taskcat_by_rule")


def tasks_categorize_based_on_rule(D, rule):
    """ fore ach task, categorize it based on a given rule and on its
    features, such as what shapes are invovled. Is liek a more detaield 
    (and rule-dependent) version of taskgorups. e.g, if
    rule == AnBm, then each task is an ngram, and could be (3,2) meaning
    it is A3B2. 
    The kinds of categories will depend on the rule (hard coded).
    PARAMS:
    - rule, string.
    RETURNS:
    - list of dict, matching each trial in D.
    """

    # prepare the dicts
    # OUT = []
    OUT = [{} for _ in range(len(D.Dat))]

    # for a given trial, get what shapes it should be mapped to.
    def _extract_shapes_pool(ruledict):
        if ruledict["categ"]=="ss": # shape sequence
            if ruledict["subcat"]=="rank":
                shapes_pool = ruledict["params_good"]
            else:
                print(rd)
                assert False
        else:
            print(ruledict)
            assert False
        return shapes_pool

    # Get ruledict, to decide what features are relevant
    map_rule_ruledict = rules_map_rule_to_ruledict_extract_auto(D)
    rd = map_rule_ruledict[rule]
    LIST_COLNAMES = []
    if rd["categ"]=="ss":
        # Shape sequence.

        shapes_pool = _extract_shapes_pool(rd)

        ## 1) ngrams, e.g, (4,3, 1) means category A4B3 and 1 left over (unidentified)
        list_ns = []
        for ind in range(len(D.Dat)):
            tokens = D.taskclass_tokens_extract_wrapper(ind, "task")
            shapes = [t["shape"] for t in tokens]
            
            # ignore order. just count how many A, B, C, ... etc
            nshapes = []
            inds_used = []
            for sh in shapes_pool:
                n = sum([sh==x for x in shapes])
                nshapes.append(n)
            # shapes left over?
            n_left_over = len([x for x in shapes if x not in shapes_pool])
            nshapes.append(n_left_over)

            # list_ns.append(tuple(nshapes))
            colname = "ss-shapes_ngram"
            OUT[ind][colname] = tuple(nshapes)
        LIST_COLNAMES.append(colname)

    else:
        print(rule, rd)
        assert False, "not coded"

    return OUT, LIST_COLNAMES


