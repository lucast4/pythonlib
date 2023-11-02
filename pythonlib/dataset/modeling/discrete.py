"""
## models for discrete grammar expt
# Given list of rules, each a generative model, ask what each model woudl do,
# for each task. NOT comparing to behavior (except minor cases).
"""
from pythonlib.tools.stringtools import decompose_string
import pandas as pd
import numpy as np
from pythonlib.dataset.dataset_analy.motifs_search_unordered import find_object_groups_new
from pythonlib.tools.exceptions import NotEnoughDataException

####### 
MAP_EPOCHKIND_EPOCH = {
    "direction":["D", "U", "R", "L", "TR"],
    "shape":["LVl1", "lVL1", "VlL1", "llV1", "ZlA1"],
    "(AB)n":["(AB)n", "LolDR"],
    "AnBm":["AnBm1a", "AnBm2", "AnBmHV", "AnBm1b", "AnBm0"],
    "AnBmDir":["LCr2", "CLr2", "AnBmTR", "LCr1", "CLr1", "LCr3"],
    "rowcol":["rowsDR", "rowsUL", "colsRD", "colsLU"],
    "ranksup":["rndstr", "rank"],
    "baseline":["base", "baseline"]
}

RULES_IGNORE = ["base", "baseline"] # rules to ignore. assumed that other rules int he same day will
# bring in all the rules.

MAP_EPOCH_EPOCHKIND = {}
for epochkind, list_epoch in MAP_EPOCHKIND_EPOCH.items():
    for epoch in list_epoch:
        assert epoch not in MAP_EPOCH_EPOCHKIND.keys()
        MAP_EPOCH_EPOCHKIND[epoch] = epochkind
        
        # also, any of these with |0 appended are the same
        MAP_EPOCH_EPOCHKIND[f"{epoch}|0"] = epochkind
        # but a 1 means is color rank supervision
        MAP_EPOCH_EPOCHKIND[f"{epoch}|1"] = "ranksup"



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
    grouping_map_tasksequencer_to_rule[("directionv2", ("topleft",))] = "TL"
    grouping_map_tasksequencer_to_rule[("directionv2", ("UL",))] = "UL"

    grouping_map_tasksequencer_to_rule[("prot_prims_in_order", ('line-8-3', 'V-2-4', 'Lcentered-4-3'))] = "lVL1"
    grouping_map_tasksequencer_to_rule[("prot_prims_in_order", ('Lcentered-4-3', 'V-2-4', 'line-8-3'))] = "LVl1"
    grouping_map_tasksequencer_to_rule[("prot_prims_in_order", ('V-2-4', 'line-8-3', 'Lcentered-4-3'))] = "VlL1"
    
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('line-8-3', 'line-8-4', 'V-2-4'))] = "llV1"
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('line-9-3', 'line-9-4', 'Lcentered-6-8'))] = "llV1"
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('line-8-3', 'line-13-13', 'line-8-4', 'line-13-14', 'V-2-4', 'V2-2-4'))] = "llV1"
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('line-8-3', 'line-13-13', 'line-8-4', 'line-13-14', 'V-2-4', 'V2-2-4', 'V2-2-2'))] = "llV1"
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('V2-2-2', 'V2-2-4', 'V-2-4', 'line-13-14', 'line-8-4', 'line-13-13', 'line-8-3'))] = "llV1R"
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('zigzagSq-1-1', 'line-8-1', 'arcdeep-4-3'))] = "ZlA1" # diego, dirshapediego1
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('zigzagSq-1-1', 'line-8-1', 'line-9-1', 'arcdeep-4-3'))] = "ZlA1" # dirshapediego1
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('zigzagSq-1-1', 'line-8-1', 'line-9-1', 'line-6-1', 'arcdeep-4-3'))] = "ZlA1" # dirshapediego1
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('zigzagSq-1-1', 'line-8-1', 'line-9-1', 'line-6-1', 'arcdeep-4-3'))] = "ZlA1" # dirshapediego1
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('zigzagSq-1-1', 'Lcentered-4-4', 'line-8-1', 'line-9-1', 'line-6-1', 'arcdeep-4-3'))] = "ZlA1" # dirshapediego1
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('zigzagSq-1-1', 'Lcentered-4-4', 'line-6-2', 'line-8-1', 'line-9-1', 'line-6-1', 'arcdeep-4-3'))] = "ZlA1" # dirshapediego1
    grouping_map_tasksequencer_to_rule[('prot_prims_in_order', ('zigzagSq-1-1', 'Lcentered-4-4', 'line-6-2', 'line-8-1', 'line-9-1', 'line-6-1', 'arcdeep-4-3', 'V-2-4'))] = "ZlA1" # dirshapediego1

    grouping_map_tasksequencer_to_rule[("prot_prims_chunks_in_order", ('line-8-4', 'line-8-3'))] = "AnBm1a"
    grouping_map_tasksequencer_to_rule[("prot_prims_chunks_in_order", ('line-8-1', 'line-8-2'))] = "AnBm2"
    grouping_map_tasksequencer_to_rule[("prot_prims_chunks_in_order", ('line-8-4', 'line-11-1', 'line-8-3', 'line-11-2'))] = "AnBm1b"
    grouping_map_tasksequencer_to_rule[("prot_prims_chunks_in_order", ('line-11-1', 'line-11-2'))] = "AnBmHV"
    grouping_map_tasksequencer_to_rule[("prot_prims_chunks_in_order", ('squiggle3-3-1', 'V-2-4'))] = "AnBm0"

    grouping_map_tasksequencer_to_rule[("hack_220829", tuple(["hack_220829"]))] = "(AB)n"

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('line-8-4', 'line-11-1', 'line-8-3', 'line-11-2', 'topright'))] = "AnBmTR"

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('line', 'circle', 'right'))] = "LCr1" # gridlinecircleGOOD (diego)
    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('circle', 'line', 'right'))] = "CLr1" # gridlinecircleGOOD (diego)
    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('line', 'arcdeep', 'circle', 'right'))] = "LCr2" # gridlinecircleGOOD (diego)
    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('circle', 'arcdeep', 'line', 'right'))] = "CLr2" # gridlinecircleGOOD (diego)

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('line', 'arcdeep', 'circle', 'downright'))] = "LCr3" # linecirclerow1/2 (diego)

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('zigzagSq-1-1', 'Lcentered-4-4', 'line-6-2', 'line-8-1', 'line-9-1', 'line-6-1', 'arcdeep-4-3', 'V-2-4', 'topleft'))] = "llCV1" # Diego, dirgrammardiego1 (7/20/23)
    
    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('zigzagSq-1-1', 'Lcentered-4-4', 'line-6-2', 'line-8-1', 'line-9-1', 'line-6-1', 'arcdeep-4-3', 'V-2-4', 'left'))] = "llCV2" # Diego, dirgrammardiego1 (7/20/23)

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2_FIRSTSTROKEONLY', 
        ('zigzagSq-1-1', 'Lcentered-4-4', 'line-6-2', 'line-8-1', 'line-9-1', 'line-6-1', 'arcdeep-4-3', 'V-2-4', 'left'))] = "llCV2FstStk" # Diego, dirgrammardiego1 (7/20/23)

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('zigzagSq-1-1', 'Lcentered-4-4', 'line-6-2', 'line-8-1', 'line-9-1', 'line-6-1', 'arcdeep-4-3', 'V-2-4', 'UL'))] = "llCV3" # Diego, dirgrammardiego5 (8/14/23)

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('zigzagSq-1-1', 'Lcentered-4-4', 'line-6-2', 'line-14-2', 'line-8-1', 'line-9-1', 'line-6-1', 'line-14-1', 'arcdeep-4-3', 'V-2-4', 'UL'))] = "llCV3b" # Diego, gramstimdiego (11/1/23)
    
    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2_FIRSTSTROKEONLY', 
        ('zigzagSq-1-1', 'Lcentered-4-4', 'line-6-2', 'line-8-1', 'line-9-1', 'line-6-1', 'arcdeep-4-3', 'V-2-4', 'UL'))] = "llCV3FstStk" # Diego, dirgrammardiego5 (8/14/23)

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('line-8-3', 'line-13-13', 'line-8-4', 'line-13-14', 'V-2-4', 'V2-2-4', 'V2-2-2', 'left'))] = "AnBmCk1a" # Pancho, dirgrammarPancho1 (8/2023)

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('line-8-3', 'line-13-13', 'line-8-4', 'line-13-14', 'V-2-4', 'V2-2-4', 'V2-2-2', 'line-8-1', 'left'))] = "AnBmCk1b" # Pancho, dirgrammarPancho1 (8/2023)

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('line-8-3', 'line-13-13', 'line-8-4', 'line-13-14', 'V-2-4', 'V2-2-4', 'V2-2-2', 'line-8-1', 'line-8-2', 'left'))] = "AnBmCk1c" # Pancho, dirgrammarPancho1 (8/2023)

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2', 
        ('line-8-3', 'line-6-3', 'line-13-13', 'line-8-4', 'line-6-4', 'line-13-14', 'V-2-4', 'V2-2-4', 'V2-2-2', 'line-8-1', 'line-8-2', 'line-6-1', 'line-6-2', 'left'))] = "AnBmCk2" # Pancho, dirgrammarPancho1 (8/2023)

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2_FIRSTSTROKEONLY', 
        ('line-8-3', 'line-6-3', 'line-13-13', 'line-8-4', 'line-6-4', 'line-13-14', 'V-2-4', 'V2-2-4', 'V2-2-2', 'line-8-1', 'line-8-2', 'line-6-1', 'line-6-2', 'left'))] = "AnBmCk2FstStk" # Pancho, dirgrammarPancho1 (8/2023)

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2_FIRSTSTROKEEXCLUDE', 
        ('line-8-3', 'line-6-3', 'line-13-13', 'line-8-4', 'line-6-4', 'line-13-14', 'V-2-4', 'V2-2-4', 'V2-2-2', 'line-8-1', 'line-8-2', 'line-6-1', 'line-6-2', 'left'))] = "AnBmCk2NOFstStk" # Pancho, dirgrammarPancho1 (8/2023)


    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2_FLEXSTROKES', 
        ('zigzagSq-1-1', 'Lcentered-4-4', 'line-6-2', 'line-8-1', 'line-9-1', 'line-6-1', 'arcdeep-4-3', 'V-2-4', 'UL', (1,), (2, 3, 4, 5)))] = "llCV3RndFlx1" #

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2_FLEXSTROKES', 
        ('zigzagSq-1-1', 'Lcentered-4-4', 'line-6-2', 'line-8-1', 'line-9-1', 'line-6-1', 'arcdeep-4-3', 'V-2-4', 'UL', (1, 2), (3, 4, 5)))] = "llCV3RndFlx12" 

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2_FLEXSTROKES', 
        ('zigzagSq-1-1', 'Lcentered-4-4', 'line-6-2', 'line-8-1', 'line-9-1', 'line-6-1', 'arcdeep-4-3', 'V-2-4', 'UL', (1, 2, 3), (4, 5)))] = "llCV3RndFlx123"




    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2_FLEXSTROKES', 
        ('line-8-3', 'line-6-3', 'line-13-13', 'line-8-4', 'line-6-4', 'line-13-14', 'V-2-4', 'V2-2-4', 'V2-2-2', 'line-8-1', 'line-8-2', 'line-6-1', 'line-6-2', 'left', (), (1, 2, 3, 4)))] = "AnBmCk2RndFlx0" # Pancho

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2_FLEXSTROKES', 
        ('line-8-3', 'line-6-3', 'line-13-13', 'line-8-4', 'line-6-4', 'line-13-14', 'V-2-4', 'V2-2-4', 'V2-2-2', 'line-8-1', 'line-8-2', 'line-6-1', 'line-6-2', 'left', (1,), (2, 3, 4)))] = "AnBmCk2RndFlx1" # Pancho

    grouping_map_tasksequencer_to_rule[(
        'prot_prims_in_order_AND_directionv2_FLEXSTROKES', 
        ('line-8-3', 'line-6-3', 'line-13-13', 'line-8-4', 'line-6-4', 'line-13-14', 'V-2-4', 'V2-2-4', 'V2-2-2', 'line-8-1', 'line-8-2', 'line-6-1', 'line-6-2', 'left', (1, 2), (3, 4)))] = "AnBmCk2RndFlx12" # Pancho

    grouping_map_tasksequencer_to_rule[(
        'shape_chunk_concrete', 
        ('lolli', ('D', 'R')))] = "LolDR" # gridlinecircleGOOD (diego), lollis (circles down and right)

    grouping_map_tasksequencer_to_rule[(
        'rows_direction', 
        ('down','right')
        )] = "rowsDR" # rowcol1, Pancho, rows (down across, right within)
    
    grouping_map_tasksequencer_to_rule[(
        'rows_direction', 
        ('up','left')
        )] = "rowsUL" # 

    grouping_map_tasksequencer_to_rule[(
        'rows_direction', 
        ('up','snkLR')
        )] = "rowsUsnkLR" # 8/21/23 - Diego, dirgrammardiego6

    grouping_map_tasksequencer_to_rule[(
        'cols_direction', 
        ('right','down')
        )] = "colsRD" # 

    grouping_map_tasksequencer_to_rule[(
        'cols_direction', 
        ('left', 'up')
        )] = "colsLU" # 

    grouping_map_tasksequencer_to_rule[('randomize_strokes', tuple(["randomize_strokes"]))] = "rndstr"

    grouping_map_tasksequencer_to_rule[('specific_order', tuple(["indices"]))] = "SpcOrd1"

    return grouping_map_tasksequencer_to_rule



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
    list_fixed_order = None # will get fixed_order automatically if this stays NOne.

    # print(objects)
    # print(tokens)
    # assert False
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
        rule. 
        - levels: across chunks, within chunks.
        """

        # NOTE: bools are NOT fixed order.
        map_rule_to_notfixedorder = {
            ("ss", "rank"): [False, True], # [alow reorder, allow reorder.]
            ("ss", "rankdir"): [False, False], # line-circe, where forced to do specific order within line and cirlc.e
            ("ss", "chain"): [False, False],
            ("dir", "null"): [False, False],
            ("ch", "dir2"): [False, True],
            ("ch", "dir1"): [True, False],
            ("ch", "dirdir"): [False, False],
            ("chorient", "dirdir"): [False, False],
            ("chmult", "dirdir"): [False, False],
            ("rand", "null"): [True, True], # random, 
        }
        key = (ruledict["categ"], ruledict["subcat"])
        tmp = map_rule_to_notfixedorder[key] # [False, True]

        # print("HERERER")
        # print(key)
        # assert False
        return fixed_order_for_this_hier(hier, tmp[0], tmp[1])


    def direction_this_hier(Task, hier, direction, dir_version='across_chunks'):
        """ Get specific versions of this hier undergoing some directio rule
        PARAMS:
        - hier, list of list, e.g., [[2,1], [3]]
        - dir_version, at which level to apply direction rule,
        --- "across_chunks", then reorders the chunks without changing the within-chunk order.
        e.g., [[3], [1,2]]
        --- "within_chunks", reordered within each chunk, leaving order across chunks intact.
        RETURNS:
        - hier, but reorderd, e.g, [[3], [1,2]] if across, or only changing within, if within.
        PROBLEM: doesn't deal with ties. always returns a single solution.
        """

        tokens = Task.tokens_generate()

        # concatenate each chunk into a temporary object, with mean positions
        def _convert_tok_to_orderitem(tok):
            return {
                "x": tok["gridloc"][0],
                "y": tok["gridloc"][1],
            }

        def _mean_loc(inds_taskstrokes):
            # inds_taskstrokes, list of ints
            # Returns dict, with x, y, inds_taskstrokes keys.
            tokens_this = [tokens[i] for i in inds_taskstrokes]
            tok = Task.tokens_concat(tokens_this)
            item = _convert_tok_to_orderitem(tok)
            item["inds_taskstrokes"] = inds_taskstrokes
            return item

        def _reorder_within_chunk(ch):
            """ ch = [1,2] list of ints
            returns list of ints
            """
            locations = []
            for i in ch:
                item = _convert_tok_to_orderitem(tokens[i])
                item["ind_task"] = i
                locations.append(item)
            locations = _get_sequence_on_dir(locations, direction)
            return [x["ind_task"] for x in locations]
            
        if dir_version=="across_chunks":
            locations = [_mean_loc(inds_taskstrokes) for inds_taskstrokes in hier]
            locations = _get_sequence_on_dir(locations, direction)
            hier_reordered = [x["inds_taskstrokes"] for x in locations]
        elif dir_version=="within_chunks":
            hier_reordered = []
            for ch in hier: # list of int
                hier_reordered.append(_reorder_within_chunk(ch))
        else:   
            print(dir_version)
            assert False, "code it"

        return hier_reordered


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
                assert False

        def _getY(e):
            if isinstance(e, list) and len(e)==2:
                return e[1]['y']
            elif isinstance(e, dict):
                return e['y']
            else:
                print(e)
                assert False

        def _getXY(e):
            """ on diagoanl, where tr is positive
            """
            if isinstance(e, list) and len(e)==2:
                # e.g., ['line-8-4-0', {'x': -1.7, 'y': -1.7, 'sx': None, 'sy': None, 'theta': None, 'order': None}]
                x = e[1]['x']
                y = e[1]['y']
                return x + y # projection onto (1,1)
            elif isinstance(e, dict):
                return e['x'] + e['y']
            else:
                print(e)
                assert False

        def _getXYrev(e):
            """ on diagoanl, where tl is positive
            """
            if isinstance(e, list) and len(e)==2:
                # e.g., ['line-8-4-0', {'x': -1.7, 'y': -1.7, 'sx': None, 'sy': None, 'theta': None, 'order': None}]
                x = -e[1]['x']
                y = e[1]['y']
                return x + y # projection onto (1,1)
            elif isinstance(e, dict):
                return -e['x'] + e['y']
            else:
                print(e)
                assert False

        def _getLeftThenUp(e):
            """ left, but break ties using up.
            """
            if isinstance(e, list) and len(e)==2:
                # e.g., ['line-8-4-0', {'x': -1.7, 'y': -1.7, 'sx': None, 'sy': None, 'theta': None, 'order': None}]
                x = -e[1]['x']
                y = e[1]['y']
                return x + 0.1*y # projection onto (1,1)
            elif isinstance(e, dict):
                return -e['x'] + 0.1*e['y']
            else:
                print(e)
                assert False

        def _getRightThenUp(e):
            """ left, but break ties using up.
            """
            if isinstance(e, list) and len(e)==2:
                # e.g., ['line-8-4-0', {'x': -1.7, 'y': -1.7, 'sx': None, 'sy': None, 'theta': None, 'order': None}]
                x = e[1]['x']
                y = e[1]['y']
                return x + 0.1*y # projection onto (1,1)
            elif isinstance(e, dict):
                return e['x'] + 0.1*e['y']
            else:
                print(e)
                assert False

        
        # print("TODO: break ties")
        # print("OBJECTS:", objects)
        # assert False
        # NOTE: does not expect/handle ambiguous cases (i.e. two in row or col)
        if direction in ['rl', 'left', 'L']:
            # sort by descending x
            return sorted(objects, reverse=True, key=_getX)
        elif direction in ['lr', 'right', 'R']:
            # sort by ascending x
            return sorted(objects, key=_getX)
        elif direction in ['up', 'U']:
            # sort by ascending y
            return sorted(objects, key=_getY)
        elif direction in ['down', 'D']:
            # sort by descending y
            return sorted(objects, reverse=True, key=_getY)
        elif direction in ["TR"]:
            # top right
            return sorted(objects, key=_getXY)
        elif direction in ["TL", "topleft"]:
            # top left
            return sorted(objects, key=_getXYrev)
        elif direction in ["UL"]:
            # left, then break ties with up
            return sorted(objects, key=_getLeftThenUp)
        elif direction in ["topright"]:
            # typewriter: first right, ythrn break ties by up.
            return sorted(objects, key=_getRightThenUp)
        else:
            print(direction)
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
        import itertools
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
    HOW_DEAL_FLAT="mult_chunk"
    ruledict = rules_map_rulestring_to_ruledict(rulestring)
    if ruledict["categ"]=="dir":
    # if rule in ["left", "right", "up", "down"]:
        # print(objects)
        shape_order = _get_sequence_on_dir(objects, ruledict["params_good"])
        # print(shape_order)
        # assert False
        hier = [objects.index(x) for x in shape_order] # O(n^2), on short list tho so just go with it...
        list_hier = [hier]
        HOW_DEAL_FLAT = "single_chunk"

    elif ruledict["categ"]=="ss" and ruledict["subcat"] in ["rankdir"]:
        # shape sequence, e.g., lines to circles, where the order within lines is fixed,
        # and determined by direction
        # Returns a single hier in list hier, since currnetly doesnt break ties for direction.
        # e..g, list_hier = [[[3, 1, 5], [4, 2, 0]]]
        # if max_n_repeats=2, then list_hier = [[[3, 1], [4, 2], [5,0]]], with fixed order for last one being False.

        shape_order = ruledict["params_good"][0] # list of shape strings.
        direction_within_shape = ruledict["params_good"][1] # string, direction.
        max_n_repeats = ruledict["params_good"][2] # either None (no cap) or int.

        list_hier = _get_sequences_on_ordering_rule(objects, "rank", shape_order)

        assert len(list_hier)==1, "should onlye be a single" # e.g., list_hier[0] = [[1,2], [3,4]]
        hier = list_hier[0]

        # reorder
        hier = direction_this_hier(Task, hier, direction_within_shape,
            dir_version="within_chunks")

        if max_n_repeats:
            # prune each chunk to max n. take the first n. THis retains the ordering
            # in the input...
            hier = [hi[:max_n_repeats] for hi in hier]
        
        list_hier = [hier]

    elif ruledict["categ"]=="ss" and ruledict["subcat"] in ["rank", "chain"]: 
        # shape sequence, e.g., lines to circles, where the order within lines is not fixed

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
    
    elif ruledict["categ"] == "chmult" and ruledict["subcat"] == "dirdir":
        # GOOD, for lollis in gridlinecircle3. 
        # Appends extra prims in random oder.
        # e.g.,:
        # a single hier: [[2, 3], [0, 4], [1]]
        # a single FO: {0: True, 1: [True, True, False]}
        # Extract params
        list_shapes_in_order = ruledict["params_good"][0] # list of list of str
        # e.g,  'params_good': [['line-8-2-0', 'D', 'circle-6-1-0'],
                    # ['line-8-1-0', 'R', 'circle-6-1-0']],
        direction_across = ruledict["params_good"][1] # str

        ### 1) Find groups
        paramsthis = {
            # "expt":expt,
            "rule":"concrete_chunks_set",
            "list_shapes_in_order":list_shapes_in_order,
        }
        list_groups, left_over = find_object_groups_new(Task, paramsthis)

        # first reorder the groups, before appending the extra stuff.
        list_groups = direction_this_hier(Task, list_groups, direction_across, 
            dir_version="across_chunks")            

        # 1) Get all hierarchices, holding hcyunks constaint. Append extra prims.
        list_possible_inds = list(range(len(tokens)))
        list_hier, list_is_grp = sample_all_possible_chunks(list_groups, 
            list_possible_inds, append_unused_strokes_as_single_group=True)

        list_fixed_order = []
        for is_grp in list_is_grp: # list of bool
            list_fixed_order.append({0:True, 1:[this for this in is_grp]})

        # print(list_hier)
        # print(list_fixed_order)
        # assert False

    elif ruledict["categ"] == "chorient" and ruledict["subcat"] == "dirdir":
        # chunk (concrete shapes, but any orietnation, such as lollies
        # right and down), with direction across chunks fixed, and within
        # too, both defined by direction rules

        # Extract params
        shapes_in_order = ruledict["params_good"][0] # list of str
        allowed_orientations = ruledict["params_good"][1] # list of str
        direction_across = ruledict["params_good"][2] # list of str

        ### 1) Find groups
        paramsthis = {
            # "expt":expt,
            "rule":"concrete_chunk_any_orient",
            "shapes_in_order":shapes_in_order,
            "allowed_orientations":allowed_orientations # list of str
        }
        list_groups, left_over = find_object_groups_new(Task, paramsthis)
        # print(list_groups)
        # print(left_over)
        # assert False
        # 1) Get all hierarchices, holding hcyunks constaint
        list_possible_inds = list(range(len(tokens)))
        list_hier, list_is_grp = sample_all_possible_chunks(list_groups, list_possible_inds)

        # Reorder each of the hier by direction in space
        list_hier_reordered = []
        for hier in list_hier:
            # 1) reorder across groups in hier
            hier_reordered = direction_this_hier(Task, hier, direction_across, 
                dir_version="across_chunks")            
            # # 2) reorder within groups in heir
            # hier_reordered = direction_this_hier(Task, hier_reordered, direction_within, dir_version="within_chunks")
            # assert False, 'Code it'
            list_hier_reordered.append(hier_reordered)
        list_hier = list_hier_reordered

        assert False, "check that it works."

    elif ruledict["categ"] == "ch" and ruledict["subcat"] == "dir2":
        # Concrete chunk, with direction across chunks fixed, but
        # direction within being variable (i.e., chunk_mask)
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
        list_hier, list_is_grp = sample_all_possible_chunks(list_groups, 
            list_possible_inds, append_unused_strokes_as_single_group=False)
 
        # Reorder each of the hier by direction in space
        direction = ruledict["params_good"][2] # across chunks

        list_hier_reordered = []
        for hier in list_hier:
            hier_reordered = direction_this_hier(Task, hier, direction)
            # locations = [_mean_loc(inds_taskstrokes) for inds_taskstrokes in hier]
            # locations = _get_sequence_on_dir(locations, direction)
            # hier_reordered = [x["inds_taskstrokes"] for x in locations]
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
        HOW_DEAL_FLAT="single_chunk"

    elif ruledict["categ"]=="preset":
        # Each task has a defined sequence in its matlab code (tasksequencer).
        # e..g, rndstr was this, where a random sequence was sampled for eash task.
        # I.e. only a single specific sequence
        
        # This is the sequenver params!
        # Task.ml2_tasksequencer_params_extract()

        # print(ruledict)
        # print(Task.Params["input_params"])

        TT = Task.Params["input_params"]
        # print(1, TT)
        # print(2, TT.get_tasknew()["Grammar"])
        # print(3, TT.get_tasknew()["Grammar"]["Tasksequencer"])
        # assert False, "try to figure out specific_order mode. Q: does this include the specific indices?"

        C = TT.objectclass_extract_active_chunk()
        if C is not None:
            taskstroke_inds_correct_order = C.extract_strokeinds_as("flat")
        else:
            taskstroke_inds_correct_order = None
        list_hier = [taskstroke_inds_correct_order] # e..g, [[1, 0, 3, 2]]
        list_fixed_order = [
            {0:True, 1:[True for _ in range(len(taskstroke_inds_correct_order))]}
        ]
        # list_hier 
        # print(taskstroke_inds_correct_order)
        # print([tok["ind_taskstroke_orig"] for tok in tokens])
        # print(Task)
        # print(tokens)
        # assert False

    else:
        print(rulestring)
        print(ruledict)
        assert False, "code up rule"

    ### Clean up the chunks
    list_hier = [clean_chunk(hier, how_to_deal_with_flat=HOW_DEAL_FLAT) for hier in list_hier] 
    
    ### Expand chunks and fixed order to match number of identified hiers
    list_chunks = [chunks for i in range(len(list_hier))] # repeat chunks as many times as there are possible hiers
    # print("list_fixed_order")
    # print(list_fixed_order)
    # print(ruledict, hier)
    if list_fixed_order is None:
        # Get autoamtically
        list_fixed_order = [_fixed_order_for_this_hier(ruledict, hier) for hier in list_hier]   
    # print(list_fixed_order)

    # sanity check
    for hier, fo in zip(list_hier, list_fixed_order):
        # print(hier)
        # print(fo)
        # print(fo[1])
        assert len(hier) == len(fo[1])
    
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
    for i, (hier, fo) in enumerate(zip(list_hier, list_fixed_order)):
        inds_used = [xx for x in hier for xx in x]
        inds_not_used = [ind for ind in inds_all if ind not in inds_used]
        if len(inds_not_used)>0:
            # print("----")
            # print(hier)
            # print(inds_used)
            # print(inds_not_used)
            # assert False
            if False:
                # replace this with a random parse
                h, c, f = _random_parse(tokens)
                list_hier[i] = h
                list_chunks[i] = c
                list_fixed_order[i] = f
            else:
                # This better, just append the remaining on the end, and randomize the order
                # within them
                from pythonlib.chunks.chunks import hier_append_unused_strokes
                hier_new, fo_new = hier_append_unused_strokes(hier, fo, len(tokens))

                # # APpend the extra tokens
                # # e..g, if input:
                # # list_hier
                # #     [[1, 3], [0, 2]]
                # # list_fixed_order
                #     # {0: True, 1: [True, True]}
                # # OUTPUT:
                # # list_hier [[[1, 3], [0, 2], [4, 5]]], where 4 and 5 where left out.
                # # list_fixed_order [{0: True, 1: [True, True, False]}]                
                # lh, lg = sample_all_possible_chunks(hier, inds_all, 
                #     append_unused_strokes_as_single_group=True)
                # # print(lh, len(lh))
                # assert len(lh)==1
                # assert len(fo[1])==len(lg[0])-1
                # assert lg[0][-1]==False
                list_hier[i] = hier_new
                list_fixed_order[i] = fo_new
                # fo[1].append(lg[0][-1]) # False
    # print(list_fixed_order)
    # assert False


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
        if not FOUND:
            for key, val in grouping_map_tasksequencer_to_rule.items():
                print(key, val)
            print(params)
            print(rulestring)
            assert FOUND, "did not find this val"
        return categ_matlab, params_matlab
    
    def _convert_shape_string(s):
        """ Standardiztaion of string representing a shape
        # Flexible, depending on the input format of the shape string
        # if input is Lcentered-4-3-0, returns the same
        # if input is Lcentered-4-3, converts to Lcentered-4-3-0 [i.e. assuems reflecting is 0]
        # if input is Lcentered, return the same (this is abstract prim)
        # if input is Lcentered-4, then fails, as doesnt kknow what this is
        """

        substrings = decompose_string(s)
        if len(substrings)==3:
            # then is liek: Lcentered-4-3, which has scale-rotation. assumes reflect is 0.
            # convert to: shape-rotation-reflect
            shape = substrings[0]
            scale = substrings[1]
            rot = substrings[2]
            refl = 0
            return f"{shape}-{scale}-{rot}-0"
        elif len(substrings)==1:
            return s
        elif len(substrings)==4:
            # then is like Lcentered-4-3-0
            return s
        else:
            print(s)
            assert False, "is this mnistake?"

    # FInd the matlab params
    if categ=="ss" and subcat in ["rank", "chain"]:
        # Shape sequence, where the N shapes is not fixed.
        # - rank means does all A, then all B, etc, with order within As not fixed.
        # - chain means does ABC, then ABC, then ABC, each time sampling any of the pool of A, B, etc.

        # Exmaples:
        # rulestring = "ss-rank-AnBm1a"
        #     {'categ_matlab': 'prot_prims_chunks_in_order',
        #      'params_matlab': ('line-8-4', 'line-8-3'),
        #      'params_good': ['line-8-4-0', 'line-8-3-0'],
        #      'categ': 'ss',
        #      'subcat': 'rank',
        #      'params': 'AnBm1a',
        #      'rulestring': 'ss-rank-AnBm1a'}
        # rulestring = "ss-chain-AnBm1a"
        # {'categ_matlab': 'prot_prims_chunks_in_order',
        #  'params_matlab': ('line-8-4', 'line-8-3'),
        #  'params_good': ['line-8-4-0', 'line-8-3-0'],
        #  'categ': 'ss',
        #  'subcat': 'chain',
        #  'params': 'AnBm1a',
        #  'rulestring': 'ss-chain-AnBm1a'}   


        # shape orders are encoded in matlab parmas:
        # 2) find it in grouping_map_tasksequencer_to_rule
        categ_matlab, params_matlab = _find_params_matlab(params)
        list_shapestring_good = [_convert_shape_string(shapestring) for shapestring in params_matlab]
        params_good = list_shapestring_good

    elif categ=="ss" and subcat in ["rankdir", "rankdir_nmax2"]:
        # Sequence of sahpes, with direction fixed within a shape, e.g, line to circle in
        # gridlinecircled3, where first do lines to right, then circles to right.

        categ_matlab, params_matlab = _find_params_matlab(params)
        # params_matlab = ('line', 'arcdeep', 'circle', 'right'), where the last item is the direction

        shapes = params_matlab[:-1]
        direction = params_matlab[-1]
        if subcat == "rankdir":
            # no cap on repeats, et.g,, N lines, N circles
            max_n_repeats = None
        elif subcat=="rankdir_nmax2":
            # max 2 lines, 2 cirlces. 
            max_n_repeats = 2
            subcat = "rankdir"
        else:
            assert False

        list_shapestring_good = [_convert_shape_string(shapestring) for shapestring in shapes]

        params_good = (list_shapestring_good, direction, max_n_repeats)

    elif categ=="ch" and subcat=="dir2":
        # Concrete chunk (e.g. lolli), with direction across chunks fixed, but
        # direction within not fixed (i.e., chunk_mask). the relative locations of 
        # items in chunks are fixed.
        # e.g., lolli, with line-circle locations fixed, and must draw
        # lollis to right, but within lolli can do line or circle first.
        categ_matlab, params_matlab = _find_params_matlab(params)
        if categ_matlab=="hack_220829":
            # categ_matlab = "shape_chunk_concrete"
            # params_matlab = ("")
            # # e..g, ('lolli', {'D', 'R'}).
            shapes_in_order = ["line-8-4-0", "line-8-3-0"]
            rel_shapes = "U"
            direction = "R" # chunk to chunk.
        params_good = (shapes_in_order, rel_shapes, direction)
    elif categ=="chorient" and subcat=="dirdir":
        # chunk (e.g. lolli), with concreete shapes but any orientation
        # direction both within and across chunks are fixed
        categ_matlab, params_matlab = _find_params_matlab(params)

        # print(categ_matlab)
        # print(params_matlab)
        # print(params)
        # assert False

        chunkname  = params_matlab[0] # e.g., lolli
        if chunkname=="lolli":
            shapes_in_order = ["line", "circle"]
        else:
            print(shapes_in_order)
            assert False, "code it"
        allowed_orientations = params_matlab[1] # .e.g, [D, R]
        if len(params_matlab)==2:
            # then this was gridliencircle3. 
            assert params_matlab==('lolli', ('D', 'R'))
            direction_across = 'lr'
        else:
            direction_across = params_matlab[3]
        params_good = (shapes_in_order, allowed_orientations, direction_across)

    elif categ=="chmult" and subcat=="dirdir":
        # chunks (e.g. lolli), with concreete shapes, uses a list of 
        # possible concrete chunks, each a list of oriented shapes.
        # order within chunks defined by their order, and across chunks defined by
        # diorections,. e..g, lolli in gridlinecircle#
        categ_matlab, params_matlab = _find_params_matlab(params)

        # print(categ_matlab)
        # print(params_matlab)
        # print(params)
        # assert False

        if params_matlab==('lolli', ('D', 'R')):
            # gridline circle. hand code this as seuqence of oroetanted shapes.
            list_shapes_in_order = []
            list_shapes_in_order.append(["line-8-2-0", "D", "circle-6-1-0"]) # vert
            list_shapes_in_order.append(["line-8-1-0", "R", "circle-6-1-0"]) # horiz
            direction_across = 'R'
        else:
            print(params_matlab)
            assert False, 'code it'

        params_good = [list_shapes_in_order, direction_across]

    elif categ=="dir":
        # Directions using string keys, no need to look at matlab params
        categ_matlab = None
        params_matlab = None
        params_good = params
        if not isinstance(params_good, str):
            print(params)
            print(substrings)
            assert False, "check that params_ggood is string like R"
    elif categ=="rand" and subcat=="null":
        # Random beh
        categ_matlab = None
        params_matlab = None
        params_good = params
    elif categ=="preset" and subcat=="null":
        categ_matlab = None
        params_matlab = None
        params_good = params
    else:
        print(categ, subcat)
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
def _get_rankdir_variations(list_shape_orders):
    list_shape_orders_rankchain = []
    for order in list_shape_orders:
        list_shape_orders_rankchain.append(f"ss-rankdir-{order}")
    return list_shape_orders_rankchain
def _get_direction_variations(list_dir):
    # e..g, list_dir = ["D", "U", "R", "L"]
    return [f"dir-null-{x}" for x in list_dir]
def _get_chunk_dirdir_variations(list_rule):
    """ chunk with direction both within and across, only one correct sequence.
    e..g, Line-circle for gridlinecircle3"""
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
    for r in ["D", "U", "R", "L", "TR", "UL"]:
        DICT_RULESTRINGS_CONSISTENT[r] = _get_direction_variations([r])
    for r in ["LVl1", "lVL1", "VlL1"]:
        DICT_RULESTRINGS_CONSISTENT[r] = _get_rank_and_chain_variations([r])
    for r in ["AnBm2", "AnBm1a", "AnBmHV", "AnBm1b"]:
        # repeat A, repeat B, e... 
        # Can do any order within A...
        DICT_RULESTRINGS_CONSISTENT[r] = [f"ss-rank-{r}"]
    for r in ["(AB)n"]:
        DICT_RULESTRINGS_CONSISTENT[r] = _get_chunk_dir2_variations([r])
    for r in ["LCr2", "CLr2", "AnBmTR", "AnBmCk1a", "AnBmCk1b", "AnBmCk1c", "AnBmCk2", "llCV1", "llCV2", "llCV3", "llCV3b"]:
        # e.g., gridlinecircle3, lines to circles, and within lines is to right.
        DICT_RULESTRINGS_CONSISTENT[r] = _get_rankdir_variations([r])
    for r in ["LolDR"]:
        # e.g., gridlinecircle3, lollis, with fixed order (defined by hand), and fixed
        # order across lollis (direction) and appending extra prims (any order)
        DICT_RULESTRINGS_CONSISTENT[r] = [f"chmult-dirdir-{r}"]
    for r in ["rndstr", "llCV2FstStk", "llCV3FstStk", "AnBmCk2FstStk", "AnBmCk2NOFstStk", "llCV3RndFlx1", 
        "llCV3RndFlx12", "llCV3RndFlx123", "AnBmCk2RndFlx0", "AnBmCk2RndFlx1", "AnBmCk2RndFlx12", "SpcOrd1"]:
        # Each task has a defined sequence in its matlab code (tasksequencer).
        # e..g, rndstr was this, where a random sequence was sampled for eash task.
        # I.e. only a single specific sequence
        DICT_RULESTRINGS_CONSISTENT[r] = [f"preset-null-{r}"]


    # print(DICT_RULESTRINGS_CONSISTENT[r])
    # assert False
    
    if debug:
        for k, v in DICT_RULESTRINGS_CONSISTENT.items():
            print(k, ' -- ', v)
    for r in list_rules:
        if r in RULES_IGNORE:
            print(r)
            print(RULES_IGNORE)
            assert False, "remove this datapt first, this has no defined rule."
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
    """for each related to the the data in this D, get its ruledict
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
    # list_rules = D.Dat["epoch_rule_tasksequencer"].unique().tolist()
    # try:
    list_rules = D.Dat["epoch_orig"].unique().tolist()
    return _rules_related_rulestrings_extract_auto(list_rules)
    # except AssertionError as err:
    #     # Fails soemtimes if you have merged epochs..
    #     list_rules = D.Dat["epoch"].unique().tolist()  
    #     return _rules_related_rulestrings_extract_auto(list_rules)

def _rules_related_rulestrings_extract_auto(list_rules, DEBUG=False):
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
        ("LVl1", "lVL1", "VlL1", "llV1"):_get_rank_and_chain_variations(("LVl1", "lVL1", "VlL1", "llV1")),
        ("D", "U", "R", "L"):_get_direction_variations(["D", "U", "R", "L"]),
        ("TR",):_get_direction_variations(["TR"]),
        ("UL",):_get_direction_variations(["UL"]),
        ("(AB)n", "AnBm1a"):_get_chunk_dir2_variations(["(AB)n"]) + ["ss-rank-AnBm1a"], # grammar1
        ("AnBm2", "AnBmHV"):["ss-rank-AnBm2", "ss-rank-AnBmHV"], # grammar2, diag and hv lines
        ("AnBm1b",):["ss-rank-AnBm1b"], # grammar2b, diag and hv lines, both within a single rule
        ("LCr2", "CLr2", "LolDR"):_get_rankdir_variations(["LCr2", "CLr2"]) + [f"chmult-dirdir-LolDR"], #  gridlinecircle3
        ("AnBmCk1a",):_get_rankdir_variations(["AnBmCk1a"]) + _get_direction_variations(["L"]), #  dirgrammarPancho1
        ("AnBmCk1b",):_get_rankdir_variations(["AnBmCk1b"]) + _get_direction_variations(["L"]), #  dirgrammarPancho1
        ("AnBmCk1c",):_get_rankdir_variations(["AnBmCk1c"]) + _get_direction_variations(["L"]), #  dirgrammarPancho1
        ("llCV1",):_get_rankdir_variations(["llCV1"]) + _get_direction_variations(["L"]), #  dirgrammardiego1
        ("AnBmCk2",):_get_rankdir_variations(["AnBmCk2"]) + _get_direction_variations(["L"]), #  dirgrammarPancho1
        ("llCV2",):_get_rankdir_variations(["llCV2"]) + _get_direction_variations(["L"]), #  dirgrammardiego4
        ("llCV3",):_get_rankdir_variations(["llCV3"]), #  dirgrammardiego5
        ("llCV3b",):_get_rankdir_variations(["llCV3b"]), #  dirgrammardiego5
        ("AnBmTR",):_get_rankdir_variations(["AnBmTR"]) + _get_direction_variations(["TR"]), #  grammardir2
        ("rndstr",): ["preset-null-rndstr"], #  
        ("SpcOrd1",): ["preset-null-SpcOrd1"], #  
        ("llCV2FstStk",): ["preset-null-llCV2FstStk"], # colorgrammardiego1??, where first stroke is like llCV2, then the others are random.
        ("AnBmCk2FstStk",): ["preset-null-AnBmCk2FstStk"],
        ("AnBmCk2NOFstStk",): ["preset-null-AnBmCk2NOFstStk"],
        ("llCV3FstStk",): ["preset-null-llCV3FstStk"],
        ("llCV3RndFlx123",): ["preset-null-llCV3RndFlx123"],
        ("llCV3RndFlx12",): ["preset-null-llCV3RndFlx12"],
        ("llCV3RndFlx1",): ["preset-null-llCV3RndFlx1"],
        ("AnBmCk2RndFlx12",): ["preset-null-AnBmCk2RndFlx12"],
        ("AnBmCk2RndFlx1",): ["preset-null-AnBmCk2RndFlx1"],
        ("AnBmCk2RndFlx0",): ["preset-null-AnBmCk2RndFlx0"],
        # ("AnBm2"):["ss-rank-AnBm2", "ss-rank-AnBm1a"] # grammar2
    }

    for key, values in DICT_RELATED_RULES.items():
        assert isinstance(key, tuple), "you prob forgot comma in the parantheses to make this a tuple"
        assert isinstance(values, list)



    # 1) list of rules present in D
    # list_rules_dat = sorted(D.Dat["epoch_rule_tasksequencer"].unique().tolist())

    # 2) for each rule there, get "related" rules from a database
    def _find_related_rules(rulethis):
        related_rules = []
        FOUND = False
        for rule_keys, rule_set in DICT_RELATED_RULES.items():
            # print(rule_keys, type(rule_keys))
            if rulethis in RULES_IGNORE:
                return []
            elif rulethis in rule_keys:
                if DEBUG:
                    print("Found rule!!")
                    print("Rule:", rulethis)
                    print("Key that contains that rule:", rule_keys)
                    print("Rule strings of related rules:", rule_set)
                FOUND = True
                related_rules.extend(rule_set)
        if FOUND==False:
            # assert FOUND, f"didnt find this rule in any sets: {rulethis}"
            print(f"didnt find this rule in any sets: {rulethis}")
            raise NotEnoughDataException
        return list(set(related_rules))

    list_rules_related =[]
    for rulethis in list_rules:
        rules_related = _find_related_rules(rulethis)
        list_rules_related.extend(rules_related)
    #     print(rulethis, rules_related)
    # assert FAlse


    # 3) combine
    # list_rules_all = list_rules + list_rules_related
    list_rules_all = list_rules_related

    # sanity check
    for rule in list_rules_all:
        assert len(decompose_string(rule))==3, "needs to be cat-subcat-rulename"
    return sorted(list(set(list_rules_all)))






#################### CATEGORIZE TASKS BASED ON SEQUENCE FEATURES
# e..g, ngram (AABBB)
def tasks_categorize_based_on_rule_mult(D):
    """
    Assign each task to a category, where cats are defined (hard coded) based on rules, 
    e.g,,m for repeat tasks, categorize tasks based on n repeats. 
    """
    # Extract for each rule each tasks' categroyes
    from pythonlib.dataset.modeling.discrete import tasks_categorize_based_on_rule, rules_map_rule_to_ruledict_extract_auto
    from pythonlib.tools.pandastools import applyFunctionToAllRows

    # Get list of rules
    list_rule = D.grammarparses_rules_extract_info()["list_rules_exist"]
    # ALL_OUT = []
    # ALL_LIST_COL = []

    all_combined = [[] for _ in range(len(D.Dat))]
    for rule in list_rule:

        list_vals = tasks_categorize_based_on_rule(D, rule)
        # print("ASDASD", list_rule)
        # print(rule)
        # print(list_vals)
        # ALL_OUT.append(OUT)
        # ALL_LIST_COL.append(LIST_COLNAMES)

        assert len(all_combined)==len(list_vals)
        for ac, val in zip(all_combined, list_vals):
            if val is not None:
                ac.append(val)

    # convert all to tuples
    all_combined = [tuple(x) for x in all_combined]

    if True:
        D.Dat["taskcat_by_rule"] = all_combined
        # print(all_combined)
        # assert False
        # Assign mew col
    else:
        def F(x):
            return tuple([x[colname] for colname in LIST_COLNAMES])
        D.Dat = applyFunctionToAllRows(D.Dat, F, "taskcat_by_rule")
    print("New col: taskcat_by_rule")


def tasks_categorize_based_on_rule(D, rule, HACK=True):
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
    OUT = []
    # OUT = [{} for _ in range(len(D.Dat))]

    # for a given trial, get what shapes it should be mapped to.
    def _extract_shapes_pool(ruledict):
        if ruledict["categ"]=="ss": # shape sequence
            if ruledict["subcat"]=="rank":
                shapes_pool = ruledict["params_good"]
            elif ruledict["subcat"]=="rankdir":
                shapes_pool = ruledict["params_good"][0]
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
    if rd["rulestring"]=="chmult-dirdir-LolDR":
        # grid line circle, count the number of lollis.
        from pythonlib.chunks.chunks import sample_all_possible_chunks
        list_shapes_in_order = rd["params_good"][0] # list of list of str
        # e.g,  'params_good': [['line-8-2-0', 'D', 'circle-6-1-0'],
                    # ['line-8-1-0', 'R', 'circle-6-1-0']],
        direction_across = rd["params_good"][1] # str

        ### 1) Find groups
        paramsthis = {
            # "expt":expt,
            "rule":"concrete_chunks_set",
            "list_shapes_in_order":list_shapes_in_order,
        }

        list_n = []
        for ind in range(len(D.Dat)):
            Task = D.Dat.iloc[ind]["Task"]
            tokens = Task.tokens_generate()
            list_groups, left_over = find_object_groups_new(Task, paramsthis)

            # 1) Get all hierarchices, holding hcyunks constaint. Append extra prims.
            list_possible_inds = list(range(len(tokens)))
            list_hier, list_is_grp = sample_all_possible_chunks(list_groups, 
                list_possible_inds, append_unused_strokes_as_single_group=False)

            nlollis = int(np.max([sum(this) for this in list_is_grp])) # sum of list of bool.
            # print(list_groups)
            # print(list_hier)
            # print(list_is_grp)
            # print(nlollis)
            OUT.append(nlollis)
            # assert False

    elif rd["categ"]=="ss":
        # Shape sequence.
        from pythonlib.drawmodel.task_features import shapes_n_each_extract

        shapes_pool = _extract_shapes_pool(rd)
        # print(decompose_string(shapes_pool[0]))
        # assert False
        if len(decompose_string(shapes_pool[0]))==1:
            shape_key = "shapeabstract"
        elif len(decompose_string(shapes_pool[0]))==4:
            shape_key = "shape"
        else:
            print(shapes_pool)
            assert False

        ## 1) ngrams, e.g, (4,3, 1) means category A4B3 and 1 left over (unidentified)
        list_ns = []
        for ind in range(len(D.Dat)):
            Task = D.Dat.iloc[ind]["Task"]
            nshapes, n_left_over = shapes_n_each_extract(Task, shapes_pool, shape_key)
            nshapes.append(n_left_over)

            # tokens = D.taskclass_tokens_extract_wrapper(ind, "task")
            # shapes = [t[shape_key] for t in tokens]
            
            # # ignore order. just count how many A, B, C, ... etc
            # nshapes = []
            # inds_used = []
            # for sh in shapes_pool:
            #     n = sum([sh==x for x in shapes])
            #     nshapes.append(n)
            # # shapes left over?
            # n_left_over = len([x for x in shapes if x not in shapes_pool])
            # nshapes.append(n_left_over)

            # list_ns.append(tuple(nshapes))
            OUT.append(tuple(nshapes))
            # colname = "ss-shapes_ngram"
            # OUT[ind][colname] = 
    elif rd["categ"] == "ch":
        # Count the chunks
        if HACK:
            OUT = [None for _ in range(len(D.Dat))]
    elif rd["rulestring"] == "preset-null-rndstr":
        # Nothing to categorize them by. this is arbitrary pretermined sequence
        OUT = [None for _ in range(len(D.Dat))]
    else:
        print(rule, rd)
        assert False, "not coded"

    return OUT