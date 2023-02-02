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

def _rules_consistent_rulestrings_extract_auto(list_rules, debug=False):
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
    return [DICT_RULESTRINGS_CONSISTENT[r] for r in list_rules]

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


# return new dataframe object, with trialnum, epoch, character, trialcode, and rule booleans
def generate_scored_beh_model_data(D, list_rules, 
    dict_map_epoch_to_rulenames=None, binary_rule=False, how_define_correct_order="matlab",
    modelclass="default", DEBUG=False, return_as_bmh_object=True,
    recompute_success_using_acceptable_scorers=True):
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
            "probe":D.Dat.iloc[ind]["probe"],
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

    ############# Convert to bm object -- useful method below for colnames
    from pythonlib.dataset.modeling.beh_model_holder import BehModelHolder
    dict_modelclass_to_rules = {modelclass:list_rules} # leav this hard coded for now. since not working with mult model classes.
    bm = BehModelHolder(dfGramScore, dict_modelclass_to_rules)

    if recompute_success_using_acceptable_scorers:
        ##### RECOMPUTE CORRECT/FAILURE
        # correct means matches at least one sequence for at least one of the related models
        # for the rule
        # map from current rule to "acceptable model rules"
        def F(x):
            rulethis = x["epoch_rule"]
            if rulethis=="base":
                return True
            else:
                list_rulestr = _rules_consistent_rulestrings_extract_auto([rulethis])[0]
                list_sname = bm.colnames_extract_scores(list_rule_get=list_rulestr)
                if len(list_sname)==0:
                    print(rulethis, list_rulestr, list_sname)
                    assert False, f"this rule has not acceptable rule scoreres: {rulethis}"
                successes = [x[name] for name in list_sname] # list of bool
                return any(successes)
        bm.Dat = applyFunctionToAllRows(bm.Dat, F, "success_binary")
        
    # Return as a BehModelHolder instance
    if return_as_bmh_object:
        return bm
    else:
        return bm.Dat



# for BOOLEAN rules only—adds column for each rule with True/False values.
# @noReturn; modifies existing D in-place, adding column 'binary_rule_tuple'
def _add_binary_rule_tuple_col(df, rule_cols):
    tuple_col_name = 'binary_rule_tuple'
    df[tuple_col_name] = ''

    # NOTE: uses "behmodpost_RULE_default" col format—may need to adapt
    for i in range(len(df)):
        df.at[i, tuple_col_name] = str(tuple([int(df.at[i, f"behmodpost_{x}_default"]) for x in rule_cols]))



#################### CATEGORIZE TASKS BASED ON SEQUENCE FEATURES
# e..g, ngram (AABBB)

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
            OUT[ind]["ss-shapes_ngram"] = tuple(nshapes)

    else:
        print(rule, rd)
        assert False, "not coded"
            
    return OUT