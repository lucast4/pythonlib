""" To study learning of rules/grammars.
Here assumes there is a single ground-truth sequence for each grammar, which is
saved in the ObjectClass (matlab task definition). Does not deal with model-based
analysis, e.g., parsing.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pythonlib.tools.snstools import rotateLabel
import pandas as pd
from pythonlib.tools.expttools import checkIfDirExistsAndHasFiles
from matplotlib import rcParams
from .learning import print_useful_things, plot_performance_all, plot_performance_timecourse, plot_performance_static_summary, plot_counts_heatmap, plot_performance_each_char, plot_performance_trial_by_trial


rcParams.update({'figure.autolayout': True})

def preprocess_dataset(D, grammar_recompute_parses = False, grammar_correct_rule=None,
        DEBUG = False, how_define_correct_order="matlab"):
    """ Preprocess Dataset as basis for all subsetquence grammar/learning analyses.
    PARAMS:
    - grammar_recompute_parses, bool, if False, then uses the saved "active chunk" used 
    during the task. This works for training tasks with single correct parse,
    but not (in general) for testing, which may have multiple correct parses. Deal with latter
    by recomputing parses using D.grammar_parses_generate. 
    - grammar_correct_rule, string, must be inputed if you use grammar_recompute_parses.
    This defined what will be the set of correct orders to compare with beh.
    RETURNS:
    - dfGramScore, dataframe holding each trial, whether is success (beh sequence matches task sequebce,
    where latter is from the active chunk, and alignemnet is done with alignment matrix.
    - list_blocksets_with_contiguous_probes, list of list of ints, where inner lists hold
    blocks that are continusous and which all have probe tasks. these are useful for making
    separate plots for each. 
    - SDIR, string path to directory for all saving of grammar.
    """
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    from .learning import preprocess_dataset as learn_preprocess
    from pythonlib.dataset.modeling.discrete import generate_scored_beh_model_data
    
    ################## Create save directiory
    SDIR = D.make_savedir_for_analysis_figures("grammar")
    savedir= f"{SDIR}/summary"
    os.makedirs(savedir, exist_ok=True) 

    # 1) Get learning metaparams
    list_blocksets_with_contiguous_probes = learn_preprocess(D)

    # 2) Get grammar scores.
    # list_rules = D.Dat["epoch"].unique().tolist()
    list_rules = []
    dfGramScore = generate_scored_beh_model_data(D, list_rules = list_rules,
        how_define_correct_order=how_define_correct_order)
    dfGramScore["which_probe_blockset"] = D.Dat["which_probe_blockset"]


    # ################## Generate behclass
    # D.behclass_preprocess_wrapper()

    # ################## Extract dataframe computing matches of beh sequence to task sequence
    # # 2) For each trial, determine whether (to waht extent) beh matches task inds sequence
    # gramscoredict = []
    # # def nmatch_(order_beh, order_correct):
    # print("frac strokes gotten in progress -> use a string edit distance")

    # for ind in range(len(D.Dat)):
    #     gramdict = D.sequence_extract_beh_and_task(ind)

    #     # frac strokes gotten
    #     order_beh = gramdict["taskstroke_inds_beh_order"]

    #     # What is the correct order?
    #     if grammar_recompute_parses:
    #         # Generate from stratch, based on defined rules.
    #         # This returns a list of orders, not just a single order
    #         assert grammar_correct_rule is not None
    #         this = D.grammar_parses_extract(ind, [grammar_correct_rule])
    #         list_order_correct = this[grammar_correct_rule]
    #     else:
    #         # Saved in matlab ObjectClass
    #         order_correct = gramdict["taskstroke_inds_correct_order"]
    #         assert order_correct is not None, "not defined in matlab ObjectClass.. you must recompute, useing grammar_recompute_parses"
    #         list_order_correct = [order_correct] # only a single correct order

    #     # binary fail/success
    #     success_binary = order_beh in list_order_correct
    #     # success_binary = order_beh==order_correct
        
    #     ########## CONSIDER EXCEPTIONs, i..e, to say didn't actally fail the sequence...
    #     # note cases where did not complete the trial, but sequence was correct so far (exclude those)
    #     # NOTE: if multipel correct possible parses, considers each criterion one by one, instread of
    #     # considering only if any possible parses passes all criteria.. This is OK?
    #     def beh_good_ignore_length(order_beh, order_correct):
    #         """ True if beh is good up unitl the time beh fails.
    #         e.g., if beh is [0, 3, 2], and correct is [0, 3, 2, 1], then
    #         this is True
    #         """
    #         for x, y in zip(order_beh, order_correct):
    #             if x!=y:
    #                 return False
    #         return True

    #     list_beh_sequence_wrong = []
    #     list_beh_too_short = []
    #     list_beh_got_first_stroke = []
    #     for order_correct in list_order_correct:
    #         beh_sequence_wrong = not beh_good_ignore_length(order_beh, order_correct)
    #         beh_too_short = len(order_beh) < len(order_correct)
    #         beh_got_first_stroke = False # whether the first beh stroke was correct.
    #         if len(order_beh)>0:
    #             if order_beh[0]==order_correct[0]:
    #                 beh_got_first_stroke = True

    #         list_beh_sequence_wrong.append(beh_sequence_wrong)
    #         list_beh_too_short.append(beh_too_short)
    #         list_beh_got_first_stroke.append(beh_got_first_stroke)

    #     beh_too_short = all(list_beh_too_short)
    #     beh_got_first_stroke = any(list_beh_got_first_stroke)
    #     if any([not x for x in list_beh_sequence_wrong]):
    #         # Then for at least one parse, the beh is correct up until beh 
    #         # seq ends, implying failure due to online abort (stroke quality)
    #         beh_sequence_wrong = False
    #     else:
    #         # all are wrong..
    #         beh_sequence_wrong = True

    #     # exclude cases where beh was too short, but order was correct
    #     exclude_because_online_abort = beh_too_short and not beh_sequence_wrong

    #     if DEBUG:
    #     # if success_binary==False and beh_sequence_wrong==False:
    #         print(ind)
    #         print(order_beh)
    #         print(list_order_correct)
    #         print(success_binary, beh_too_short, beh_got_first_stroke, beh_sequence_wrong)
    #     ######################################

    #     # combination of (epoch, supervision_tuple)
    #     epoch_superv = (gramdict["epoch"], gramdict["supervision_tuple"])

    #     # # epoch, converting from epoch --> rule name
    #     # epoch_i = D.Dat.iloc[ind]['epoch']
    #     # if epoch_i in dict_map_epoch_to_rulenames.keys():
    #     #     epoch_i = dict_map_epoch_to_rulenames[epoch_i]

    #     # block (sanity check)
    #     block = D.Dat.iloc[ind]["block"]
        
    #     # taskgroup
    #     taskgroup = D.Dat.iloc[ind]["taskgroup"]
        
    #     # character
    #     char = D.Dat.iloc[ind]["character"]
        
    #     # COLLECT
    #     gramscoredict.append({
    #         "success_binary":success_binary,
    #         "beh_sequence_wrong":beh_sequence_wrong,
    #         "beh_too_short":beh_too_short,
    #         "beh_got_first_stroke":beh_got_first_stroke,
    #         "exclude_because_online_abort":exclude_because_online_abort,
    #         "epoch_superv":epoch_superv,
    #         "epoch":D.Dat.iloc[ind]['epoch'],
    #         "block": block,
    #         "datind":ind,
    #         "taskgroup":taskgroup,
    #         "character":char
    #     })

    # dfGramScore = pd.DataFrame(gramscoredict)

    # # Append things to dfgramscore
    # from pythonlib.tools.pandastools import applyFunctionToAllRows
    # cols_to_copy = ["success_binary", "beh_sequence_wrong", "beh_too_short", "exclude_because_online_abort", "epoch_superv", 
    #                 "which_probe_blockset"]
    # for col in cols_to_copy:
    #     list_vals = []
    #     for i in range(len(D.Dat)):
    #         vals = dfGramScore[dfGramScore["datind"] == i][col].tolist()
    #         assert len(vals)==1
    #         list_vals.append(vals[0])
    #     if col in D.Dat.columns:
    #         assert list_vals==D.Dat[col].tolist()
    #     else:
    #         D.Dat[col] = list_vals
    #     print("Added this columnt to D.Dat: ", col)

    return dfGramScore, list_blocksets_with_contiguous_probes, SDIR

