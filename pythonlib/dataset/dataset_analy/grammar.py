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
        DEBUG = False, how_define_correct_order="matlab",
        reset_grammar_dat = False):
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
    from pythonlib.dataset.modeling.discrete import generate_scored_beh_model_data, rules_related_rulestrings_extract_auto
    
    if reset_grammar_dat:
        D.GrammarDict = {}
    ################## Create save directiory
    SDIR = D.make_savedir_for_analysis_figures("grammar")
    savedir= f"{SDIR}/summary"
    os.makedirs(savedir, exist_ok=True) 

    # 1) Get learning metaparams
    list_blocksets_with_contiguous_probes = learn_preprocess(D)

    # 2) Get grammar scores.
    # list_rules = D.Dat["epoch"].unique().tolist()
    if False:
        list_rules = []
    else:
        # 2) get additional rules (hypotheses)
        ## Test each beh against hypothetical rules (discrete models)
        list_rules = rules_related_rulestrings_extract_auto(D)

    bm = generate_scored_beh_model_data(D, list_rules = list_rules,
        how_define_correct_order=how_define_correct_order, binary_rule=True)
    bm.Dat["which_probe_blockset"] = D.Dat["which_probe_blockset"]

    return bm, list_blocksets_with_contiguous_probes, SDIR


def pipeline_generate_and_plot_all(D):
    """ Entire pipeline to extract data and plot, for 
    a single dataset
    """

    bmh, list_blockset, SDIR = preprocess_dataset(D)

    ####### 1) COmpare beh to all hypotheses (rules, discrete)
    # Also make plots for rule-based analysis
    savedir= f"{SDIR}/discrete_rules"
    os.makedirs(savedir, exist_ok=True) 

    # combine in single plot (all taskgroups)
    sdir = f"{savedir}/score_epoch_x_rule_splitby"
    os.makedirs(sdir, exist_ok=True)

    for split_by in ["taskgroup", "isprobe"]:
        fig = bmh.plot_score_cross_prior_model_splitby(split_by=split_by)
        fig.savefig(f"{sdir}/splitby_{split_by}-trialdat.pdf")
    
    ######### 2) Plot summary
    dfGramScore = bmh.Dat
    if not checkIfDirExistsAndHasFiles(f"{SDIR}/summary")[1]:
        plot_performance_all(dfGramScore, list_blockset, SDIR)
        plot_performance_timecourse(dfGramScore, list_blockset, SDIR)
        plot_performance_static_summary(dfGramScore, list_blockset, SDIR, False)
        plot_performance_static_summary(dfGramScore, list_blockset, SDIR, True)
        plot_counts_heatmap(dfGramScore, SDIR)
        plot_performance_trial_by_trial(dfGramScore, D, SDIR)
        plot_performance_each_char(dfGramScore, D, SDIR)
        # 1) print all the taskgroups
        D.taskgroup_char_ntrials_print_save(SDIR)
    else:
        print("[SKIPPING, since SDIR exists and has contents: ", SDIR)
