## models for discrete grammar expt
# Given list of rules, each a generative model, ask what each model woudl do
# for each task, and compare this to behavior.

import pandas as pd


# return new dataframe object, with trialnum, epoch, character, trialcode, and rule booleans
def generate_scored_beh_model_data(D, list_rules, 
    dict_map_epoch_to_rulenames=None, binary_rule=False, how_define_correct_order="matlab",
    modelclass="default", DEBUG=False):
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

    return dfGramScore

# for BOOLEAN rules only—adds column for each rule with True/False values.
# @noReturn; modifies existing D in-place, adding column 'binary_rule_tuple'
def _add_binary_rule_tuple_col(df, rule_cols):
    tuple_col_name = 'binary_rule_tuple'
    df[tuple_col_name] = ''

    # NOTE: uses "behmodpost_RULE_default" col format—may need to adapt
    for i in range(len(df)):
        df.at[i, tuple_col_name] = str(tuple([int(df.at[i, f"behmodpost_{x}_default"]) for x in rule_cols]))

