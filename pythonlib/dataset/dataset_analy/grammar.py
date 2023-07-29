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
from .learning import preprocess_dataset as learn_preprocess


rcParams.update({'figure.autolayout': True})

## OLD, before changed it to make sure it only works with  matlab rules (not new parses)
def preprocess_dataset_recomputeparses(D, DEBUG=False):
    """ Preprocess Dataset, extracting score by looking at parsese of rules for each epoch,
    and asking if beh is compatible with any of them.
    NOTE: dataset length will be multiplied by however many rules there are...
    """
    from pythonlib.behmodelholder.preprocess import generate_scored_beh_model_data_long
    from pythonlib.dataset.modeling.discrete import rules_related_rulestrings_extract_auto
    
    # get epochsets
    D.epochset_apply_sequence_wrapper()

    # 2) Get grammar scores.
    # - get rules autoamticlaly.
    list_rules = rules_related_rulestrings_extract_auto(D)
    bm = generate_scored_beh_model_data_long(D, list_rules = list_rules, DEBUG=DEBUG)

    return bm

def preprocess_dataset_matlabrule(D):
    """ Preprocess Dataset using matlab rules (NOT all parses)
    Each trial is success/failure based on ObjectClass
    """
    from pythonlib.behmodelholder.preprocess import generate_scored_beh_model_data_matlabrule
        
    # get epochsets
    D.epochset_apply_sequence_wrapper()

    # 2) Get grammar scores.
    bm = generate_scored_beh_model_data_matlabrule(D)

    return bm

def pipeline_generate_and_plot_all(D, which_rules="matlab", 
    reset_grammar_dat=False, doplots=True, remove_repeated_trials=True):
    """ Entire pipeline to extract data and plot, for 
    a single dataset
    PARAMS:
    - which_rules, str, either to use ObjectClass matlab rule, or to regenreate
    parsesa nd ask if beh is compativle iwth any of the "same-rule" parses.
    """
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    from pythonlib.dataset.modeling.discrete import rules_related_rulestrings_extract_auto

    if reset_grammar_dat:
        D.GrammarDict = {}

    # 1) Get learning metaparams
    list_blocksets_with_contiguous_probes = learn_preprocess(D, remove_repeated_trials=remove_repeated_trials)

    ################## Create save directiory
    SDIR = D.make_savedir_for_analysis_figures("grammar")
    savedir= f"{SDIR}/summary"
    os.makedirs(savedir, exist_ok=True) 

    # grammar_recompute_parses = False # just use the matlab ground truth
    if which_rules=="matlab":
        # use the ground truth objectclass
        bmh  = preprocess_dataset_matlabrule(D)
    elif which_rules=="recompute_parses":
        bmh  = preprocess_dataset_recomputeparses(D)
        assert False, "aggregate bmh.DatLong so that there is only one ind per trialcode. this should work since success_binary_quick should be identical for all instance for a given trialcode. confirm this"
    else:
        print(which_rules)
        assert False

    ####### 1) COmpare beh to all hypotheses (rules, discrete)
    # Also make plots for rule-based analysis
    savedir= f"{SDIR}/discrete_rules"
    os.makedirs(savedir, exist_ok=True) 

    # combine in single plot (all taskgroups)
    sdir = f"{savedir}/score_epoch_x_rule_splitby"
    os.makedirs(sdir, exist_ok=True)

    if "epochset" in bmh.columns:
        LIST_SPLIT_BY = ["taskgroup", "probe", "epochset"]
    else:
        ["taskgroup", "probe"]
    if doplots:
        for split_by in LIST_SPLIT_BY:
            # Old plots
            fig1, fig2 = bmh.plot_score_cross_prior_model_splitby(split_by=split_by)
            fig1.savefig(f"{sdir}/splitby_{split_by}-trialdat-1.pdf")
            fig2.savefig(f"{sdir}/splitby_{split_by}-trialdat-2.pdf")

            # New plots
            bmh.plot_score_cross_prior_model_splitby_v2(split_by=split_by, savedir=sdir)

        ######### 2) Plot summary
        dfGramScore = bmh.DatLong  
        if not checkIfDirExistsAndHasFiles(f"{SDIR}/summary")[1]:
            plot_performance_all(dfGramScore, list_blocksets_with_contiguous_probes, SDIR)
            plot_performance_timecourse(dfGramScore, list_blocksets_with_contiguous_probes, SDIR)
            plot_performance_static_summary(dfGramScore, list_blocksets_with_contiguous_probes, SDIR, False)
            plot_performance_static_summary(dfGramScore, list_blocksets_with_contiguous_probes, SDIR, True)
            plot_counts_heatmap(dfGramScore, SDIR)
            plot_performance_trial_by_trial(dfGramScore, D, SDIR)
            plot_performance_each_char(dfGramScore, D, SDIR)
            # 1) print all the taskgroups
            D.taskgroup_char_ntrials_print_save(SDIR)
        else:
            print("[SKIPPING, since SDIR exists and has contents: ", SDIR)

    return bmh, SDIR
