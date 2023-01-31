## models for discrete grammar expt
# Given list of rules, each a generative model, ask what each model woudl do
# for each task, and compare this to behavior.

import pandas as pd

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


def map_from_rulestring_to_ruleparams(rulestring):
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
        "params":params}

    return out

def rules_extract_auto(D):
    """ Helper to try to extract all relevant rules, based on:
    (i) the groundt truth rules in D< and (ii) related rules that
    are alternative huypotjeses to those rules
    """

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


    DICT_RELATED_RULES = {
        # ("LVl1", "lVL1", "VlL1"):_get_rank_and_chain_variations(("LVl1", "lVL1", "VlL1")),
        ("LVl1", "lVL1", "VlL1"):_get_rank_and_chain_variations(("LVl1", "lVL1", "VlL1")),
        ("D", "U", "R", "L"):_get_direction_variations(["D", "U", "R", "L"]),
        ("(AB)n", "AnBm1a"):_get_chunk_dir2_variations(["(AB)n"]) + ["ss-rank-AnBm1a"], # grammar1
        ("AnBm2"):["ss-rank-AnBm2"] # grammar2
        # ("AnBm2"):["ss-rank-AnBm2", "ss-rank-AnBm1a"] # grammar2
    }
    RULES_IGNORE = ["base"] # rules to ignore. assumed that other rules int he same day will
    # bring in all the rules.

    # 1) list of rules present in D
    list_rules_dat = sorted(D.Dat["epoch_rule_tasksequencer"].unique().tolist())

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
    for rulethis in list_rules_dat:
        list_rules_related.extend(_find_related_rules(rulethis))

    # 3) combine
    # list_rules_all = list_rules_dat + list_rules_related
    list_rules_all = list_rules_related

    # sanity check
    from pythonlib.tools.stringtools import decompose_string
    for rule in list_rules_all:
        assert len(decompose_string(rule))==3, "needs to be cat-subcat-rulename"

    return sorted(list(set(list_rules_all)))

# return new dataframe object, with trialnum, epoch, character, trialcode, and rule booleans
def generate_scored_beh_model_data(D, list_rules, 
    dict_map_epoch_to_rulenames=None, binary_rule=False, how_define_correct_order="matlab",
    modelclass="default", DEBUG=False, return_as_bmh_object=True):
    """High-level extraction for each trial its parses under each given rule, and evaluate whether this
    trials' beahvhiro was correct/incorrect.
    PARAMS:
    - list_rules, list of str
    - dict_map_epoch_to_rulenames, dict mapping from epoch to rule. Useful to know for each
    trials whats its ground truth rule. If None, assumes identity.
    - binary_rule, bool, whether to get for eahc trial a binary code for bollean rules
    - how_define_correct_order, str, in {'matlab', 'epoch'}, whether to definde each trials' 
    correct sequence using the saved sequen inematlab or recomputed based on rules\
    """

    if dict_map_epoch_to_rulenames is None:
        dict_map_epoch_to_rulenames = {}
    ################## Generate behclass
    D.behclass_preprocess_wrapper()

    results = [] #dataframe to return
    for ind in range(len(D.Dat)):
        if ind%100==0:
            print("trial #", ind)
        
        # rename value in 'epoch' to rulename, if applicable
        epoch_i = D.Dat.iloc[ind]['epoch']
        if epoch_i in dict_map_epoch_to_rulenames.keys():
            epoch_i = dict_map_epoch_to_rulenames[epoch_i]
            # dfGramScore.at[i, "epoch_rule"] = epoch_i

        # # add necessary columns to results
        # results.append({'trial_num':i,'epoch':epoch_i,'character':D.Dat.iloc[i]['character'],'trialcode':D.Dat.iloc[i]['trialcode']})

        # generate all parses for this trial
        gramdict = D.sequence_extract_beh_and_task(ind)
        taskstroke_inds_beh_order = gramdict["taskstroke_inds_beh_order"]
        parsesdict = D.grammar_parses_extract(ind, list_rules)       

        # Score this trial as correct/incorrect
        if how_define_correct_order=="matlab":
            # correct order specified in matlab code
            order_correct = gramdict["taskstroke_inds_correct_order"]
            assert order_correct is not None, "not defined in matlab ObjectClass.. you must recompute, useing grammar_recompute_parses"
            list_order_correct = [order_correct] # only a single correct order
        elif how_define_correct_order=="epoch":
            # use the rule
            list_order_correct = parsesdict[epoch_i]
        else:
            assert False
        success_binary = taskstroke_inds_beh_order in list_order_correct

        ########## CONSIDER EXCEPTIONs, i..e, to say didn't actally fail the sequence...
        # note cases where did not complete the trial, but sequence was correct so far (exclude those)
        # NOTE: if multipel correct possible parses, considers each criterion one by one, instread of
        # considering only if any possible parses passes all criteria.. This is OK?
        def beh_good_ignore_length(order_beh, order_correct):
            """ True if beh is good up unitl the time beh fails.
            e.g., if beh is [0, 3, 2], and correct is [0, 3, 2, 1], then
            this is True
            """
            for x, y in zip(order_beh, order_correct):
                if x!=y:
                    return False
            return True

        list_beh_sequence_wrong = []
        list_beh_too_short = []
        list_beh_got_first_stroke = []
        for order_correct in list_order_correct:
            beh_sequence_wrong = not beh_good_ignore_length(taskstroke_inds_beh_order, order_correct)
            beh_too_short = len(taskstroke_inds_beh_order) < len(order_correct)
            beh_got_first_stroke = False # whether the first beh stroke was correct.
            if len(taskstroke_inds_beh_order)>0:
                if taskstroke_inds_beh_order[0]==order_correct[0]:
                    beh_got_first_stroke = True

            list_beh_sequence_wrong.append(beh_sequence_wrong)
            list_beh_too_short.append(beh_too_short)
            list_beh_got_first_stroke.append(beh_got_first_stroke)

        beh_too_short = all(list_beh_too_short)
        beh_got_first_stroke = any(list_beh_got_first_stroke)
        if any([not x for x in list_beh_sequence_wrong]):
            # Then for at least one parse, the beh is correct up until beh 
            # seq ends, implying failure due to online abort (stroke quality)
            beh_sequence_wrong = False
        else:
            # all are wrong..
            beh_sequence_wrong = True

        # exclude cases where beh was too short, but order was correct
        exclude_because_online_abort = beh_too_short and not beh_sequence_wrong

        if DEBUG:
        # if success_binary==False and beh_sequence_wrong==False:
            print(ind)
            print(taskstroke_inds_beh_order)
            print(list_order_correct)
            print(success_binary, beh_too_short, beh_got_first_stroke, beh_sequence_wrong)
        ########## [END] CONSIDER EXCEPTIONs

        ############ OTHER INFO
        # COLLECT
        results.append({
            "epoch_rule":epoch_i,
            "isprobe":D.Dat.iloc[ind]["probe"],
            "epoch_superv":D.Dat.iloc[ind]["epoch_superv"],
            "success_binary":success_binary,
            "beh_sequence_wrong":beh_sequence_wrong,
            "beh_too_short":beh_too_short,
            "beh_got_first_stroke":beh_got_first_stroke,
            "exclude_because_online_abort":exclude_because_online_abort,
            "epoch":D.Dat.iloc[ind]['epoch'],
            "block": D.Dat.iloc[ind]["block"],
            "datind":ind,
            "trialcode":D.Dat.iloc[ind]["trialcode"],
            "taskgroup":D.Dat.iloc[ind]["taskgroup"],
            "character":D.Dat.iloc[ind]["character"],
            "parsesdict":parsesdict
        })

        for rule in list_rules:
            parses = parsesdict[rule]
            # dfGramScore.at[i, f"behmodpost_{rule}_default"] = taskstroke_inds_beh_order in parses
            results[-1][f"behmodpost_{rule}_{modelclass}"] = taskstroke_inds_beh_order in parses

    dfGramScore = pd.DataFrame(results)
    # df = pd.DataFrame(results)

    if binary_rule:
        _add_binary_rule_tuple_col(dfGramScore, list_rules)

    # Append things to dfgramscore
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    cols_to_copy = ["success_binary", "beh_sequence_wrong", "beh_too_short", "exclude_because_online_abort"]
    for col in cols_to_copy:
        list_vals = []
        for i in range(len(D.Dat)):
            vals = dfGramScore[dfGramScore["datind"] == i][col].tolist()
            assert len(vals)==1
            list_vals.append(vals[0])
        if col in D.Dat.columns:
            assert list_vals==D.Dat[col].tolist()
        else:
            D.Dat[col] = list_vals
        print("Added this columnt to D.Dat: ", col)

    # Return as a BehModelHolder instance
    if return_as_bmh_object:
        from pythonlib.dataset.modeling.beh_model_holder import BehModelHolder
        dict_modelclass_to_rules = {modelclass:list_rules} # leav this hard coded for now. since not working with mult model classes.
        bm = BehModelHolder(dfGramScore, dict_modelclass_to_rules)
        return bm
    else:
        return dfGramScore

# for BOOLEAN rules only—adds column for each rule with True/False values.
# @noReturn; modifies existing D in-place, adding column 'binary_rule_tuple'
def _add_binary_rule_tuple_col(df, rule_cols):
    tuple_col_name = 'binary_rule_tuple'
    df[tuple_col_name] = ''

    # NOTE: uses "behmodpost_RULE_default" col format—may need to adapt
    for i in range(len(df)):
        df.at[i, tuple_col_name] = str(tuple([int(df.at[i, f"behmodpost_{x}_default"]) for x in rule_cols]))

