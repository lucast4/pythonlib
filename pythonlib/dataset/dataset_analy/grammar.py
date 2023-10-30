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
from pythonlib.tools.plottools import savefig
import pandas as pd
from pythonlib.tools.expttools import checkIfDirExistsAndHasFiles
from matplotlib import rcParams
from .learning import print_useful_things, plot_performance_all, plot_performance_timecourse, plot_performance_static_summary, plot_counts_heatmap, plot_performance_each_char, plot_performance_trial_by_trial
from .learning import preprocess_dataset as learn_preprocess


rcParams.update({'figure.autolayout': True})

## OLD, before changed it to make sure it only works with  matlab rules (not new parses)
def preprocess_dataset_recomputeparses(D, DEBUG=False, exclude_because_online_abort=False):
    """ Preprocess Dataset, extracting score by looking at parsese of rules for each epoch,
    and asking if beh is compatible with any of them.
    NOTE: dataset length will be multiplied by however many rules there are...
    """
    from pythonlib.behmodelholder.preprocess import generate_scored_beh_model_data_long
    from pythonlib.dataset.modeling.discrete import rules_related_rulestrings_extract_auto
    
    # get epochsets
    D.epochset_apply_sequence_wrapper()

    # get taskfeatures (e.g., repeat circle, separated by line)
    D.taskfeatures_category_by(method="shape_repeat", colname="taskfeat_cat")

    # 2) Get grammar scores.
    # - get rules autoamticlaly.
    list_rules = rules_related_rulestrings_extract_auto(D)
    bm = generate_scored_beh_model_data_long(D, list_rules = list_rules, DEBUG=DEBUG)

    if exclude_because_online_abort:
        # remove the rows from bm that have good sequence, but online abort.
        bm.DatLong = bm.DatLong[~bm.DatLong["exclude_because_online_abort"]].reset_index(drop=True)

    return bm

def preprocess_dataset_matlabrule(D, exclude_because_online_abort=False):
    """ Preprocess Dataset using matlab rules (NOT all parses)
    Each trial is success/failure based on ObjectClass
    """
    from pythonlib.behmodelholder.preprocess import generate_scored_beh_model_data_matlabrule
        
    # get epochsets
    D.epochset_apply_sequence_wrapper()

    # get taskfeatures (e.g., repeat circle, separated by line)
    D.taskfeatures_category_by(method="shape_repeat", colname="taskfeat_cat")

    # 2) Get grammar scores.
    bm = generate_scored_beh_model_data_matlabrule(D)

    if exclude_because_online_abort:
        # remove the rows from bm that have good sequence, but online abort.
        bm.DatLong = bm.DatLong[~bm.DatLong["exclude_because_online_abort"]].reset_index(drop=True)

    return bm

def pipeline_generate_and_plot_all(D, which_rules="matlab", 
    reset_grammar_dat=False, doplots=True, remove_repeated_trials=True):
    """ Entire pipeline to extract data and plot, for 
    a single dataset
    PARAMS:
    - which_rules, str, either to use ObjectClass matlab rule, or to regenreate
    parsesa nd ask if beh is compativle iwth any of the "same-rule" parses.
    RETURNS:
    - bmh
    - SDIR
    OR (None, None) if not enough data 
    """
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    from pythonlib.dataset.modeling.discrete import rules_related_rulestrings_extract_auto

    ################## Create save directiory
    SDIR = D.make_savedir_for_analysis_figures("grammar")
    savedir= f"{SDIR}/summary"
    os.makedirs(savedir, exist_ok=True) 

    if reset_grammar_dat:
        D.GrammarDict = {}

    # 1) Get learning metaparams
    list_blocksets_with_contiguous_probes = learn_preprocess(D, remove_repeated_trials=remove_repeated_trials)

    # grammar_recompute_parses = False # just use the matlab ground truth
    if which_rules=="matlab":
        # use the ground truth objectclass
        bmh  = preprocess_dataset_matlabrule(D)
    elif which_rules=="recompute_parses":
        print("******** len Dat:", len(D.Dat))
        bmh  = preprocess_dataset_recomputeparses(D)
        assert False, "aggregate bmh.DatLong so that there is only one ind per trialcode. this should work since success_binary_quick should be identical for all instance for a given trialcode. confirm this"
    else:
        print(which_rules)
        assert False

    if len(D.Dat)==0:
        return None, None

    if doplots:
        ####### 1) COmpare beh to all hypotheses (rules, discrete)
        # Also make plots for rule-based analysis
        savedir= f"{SDIR}/discrete_rules"
        os.makedirs(savedir, exist_ok=True) 

        # combine in single plot (all taskgroups)
        sdir = f"{savedir}/score_epoch_x_rule_splitby"
        os.makedirs(sdir, exist_ok=True)

        LIST_SPLIT_BY = ["taskgroup", "probe"]
        if "epochset" in bmh.DatLong.columns:
            LIST_SPLIT_BY.append("epochset")
        if "taskfeat_cat" in bmh.DatLong.columns:
            LIST_SPLIT_BY.append("taskfeat_cat")          
      
        # Use only no-sup data for these
        dfthis = bmh.DatLong[bmh.DatLong["superv_SEQUENCE_SUP"]=="off"]
        dfthis = dfthis[dfthis["exclude_because_online_abort"]==False]
        for split_by in LIST_SPLIT_BY:
            
            # Old plots
            bmh.plot_score_cross_prior_model_splitby(df=dfthis, split_by=split_by,
                sdir=sdir, suffix="trialdat") 

            # fig1, fig2 = bmh.plot_score_cross_prior_model_splitby(df=dfthis, split_by=split_by) 
            # savefig(fig1, f"{sdir}/splitby_{split_by}-trialdat-1.pdf") 
            # savefig(fig2, f"{sdir}/splitby_{split_by}-trialdat-2.pdf")

            # agg version of old plots
            bmh.plot_score_cross_prior_model_splitby_agg(split_by=split_by,
                sdir=sdir, suffix="aggdat") 
            # savefig(fig1, f"{sdir}/splitby_{split_by}-aggdat-1.pdf") 
            # savefig(fig2, f"{sdir}/splitby_{split_by}-aggdat-2.pdf")
            
            # New plots
            bmh.plot_score_cross_prior_model_splitby_v2(df=dfthis, split_by=split_by, savedir=sdir)

            plt.close("all")

            # except Exception as err:
            #     pass

            # Plot timecourse, one plot for each epoch
            list_levels = D.Dat[split_by].unique().tolist()
            for lev in list_levels:
                df = D.Dat[(D.Dat["exclude_because_online_abort"]==False) & (D.Dat[split_by]==lev)]
                fig=sns.relplot(data=df, x="tvalfake", col="epoch", col_wrap=3, y="success_binary_quick", 
                            hue="session",
                            height=3, aspect=3, alpha=0.25)
                savefig(fig, f"{sdir}/timecourse-splitby_{split_by}-lev_{lev}.pdf")   

                plt.close("all") 

        # Do permutation test of whether score is significant across epochs
        # (e.g., if using microstim to perturb baehavior)
        if not np.all(bmh.DatLong["epoch"] == bmh.DatLong["epoch_orig"]):
            # Then epoch is different from epoch_orig
            bmh.stats_score_permutation_test(split_plots_by="epoch_orig", 
                savedir=sdir)        
        else:
            bmh.stats_score_permutation_test(split_plots_by="epoch", 
                savedir=sdir)        

        ######### 2) Plot summary
        dfGramScore = bmh.DatLong  
        dfGramScore = dfGramScore[dfGramScore["exclude_because_online_abort"]==False]
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

            # plot counts, only success triuals
            df = bmh.DatLong[bmh.DatLong["success_binary_quick"]==True].reset_index(drop=True)
            plot_counts_heatmap(df, SDIR, suffix="SUCCESS")
        else:
            print("[SKIPPING, since SDIR exists and has contents: ", SDIR)

        ######## CONJUNCTIONS PLOTS
        DS, dataset_pruned_for_trial_analysis, params_anova, params_anova_extraction = conjunctions_preprocess(D)
        if DS is not None:
            savedir = f"{SDIR}"
            conjunctions_plot(D, DS, savedir, params_anova)

    return bmh, SDIR


def conjunctions_preprocess(D):
    """
    Online sequenbce, plot all conjucntions for grammar neural analyses
    NOTE: returns None if D is empty after preprocessing.
    """
    from neuralmonkey.metadat.analy.anova_params import dataset_apply_params
    from neuralmonkey.classes.snippets import datasetstrokes_extract

    # remove baseline
    # D.grammarmatlab_successbinary_score()

    # First remove baseline
    # print(1, D.Dat["epoch"].unique())    
    D.preprocessGood(params=["remove_baseline", "one_to_one_beh_task_strokes"])
    # print(2, D.Dat["epoch"].unique())
    # assert False
    
    if len(D.Dat)==0:
        return (None for _ in range(4))

    # Second get parses.
    D.grammarparses_successbinary_score()
    D.preprocessGood(params=["correct_sequencing_binary_score"])

    if len(D.Dat)==0:
        return (None for _ in range(4))

    # Assign chunks info to tokens
    for ind in range(len(D.Dat)):
        D.grammarparses_taskclass_tokens_assign_chunk_state_each_stroke(ind)

    # Clean and extract dataset as would in neural analy
    ListD = [D]
    animal = D.animals(force_single=True)[0]
    tmp = D.Dat["date"].unique()
    assert len(tmp)==1
    DATE = int(tmp[0])
    which_level = "stroke"
    ANALY_VER = "seqcontext"
    anova_interaction = False

    Dall, dataset_pruned_for_trial_analysis, TRIALCODES_KEEP, params_anova, params_anova_extraction = \
        dataset_apply_params(ListD, animal, DATE, which_level, ANALY_VER, anova_interaction)

    # list_features = ["chunk_rank", "chunk_within_rank", "chunk_within_rank", "chunk_n_in_chunk"]
    list_features = []
    DS = datasetstrokes_extract(dataset_pruned_for_trial_analysis, list_features=list_features)

    return DS, dataset_pruned_for_trial_analysis, params_anova, params_anova_extraction


def conjunctions_plot(D, DS, savedir, params_anova):
    """ Make all plots and printed text for conjuctuions relevant for grammar, at level of
    strokes during drawing
    PARAMS:
    - D, DS, savedir, params_anova, see output of conjunctions_preprocess
    """

    from neuralmonkey.metadat.analy.anova_params import _conjunctions_print_plot_all
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
    
    # LIST_VAR = params_anova["LIST_VAR"]
    # LIST_VARS_CONJUNCTION = params_anova["LIST_VARS_CONJUNCTION"]           

    DS.context_define_local_context_motif(1,1)
    DS.context_chunks_assign_columns()

    ########## Rank within chunk
    sdir = f"{savedir}/conjunctions_strokes/chunk_within_rank"
    LIST_VAR = [
        "chunk_within_rank",
        "chunk_within_rank_fromlast",
        ]
    LIST_VARS_CONJUNCTION = [
        ["CTXT_prev_this_next", "chunk_rank", "epoch"],
        ["CTXT_prev_this_next", "chunk_rank", "epoch"],
    ]
    DF = DS.Dat
    _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, params_anova["globals_nmin"], D)

    ########## N prims in upcoming chunk
    sdir = f"{savedir}/conjunctions_strokes/chunk_n_in_chunk"
    LIST_VAR = [
        "chunk_n_in_chunk",
        "chunk_n_in_chunk",
    ]
    LIST_VARS_CONJUNCTION = [
        ["CTXT_prev_this_next", "chunk_diff_from_prev", "chunk_n_in_chunk_prev", "epoch"],
        ["CTXT_prev_this_next", "chunk_diff_from_prev", "epoch"],
    ]
    DF = DS.Dat[DS.Dat["chunk_diff_from_prev"]==True]
    if len(DF)>0:
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, params_anova["globals_nmin"], D)

    ########## Rule switching, chunk switch, whether dissociate rule switch from shape switch
    sdir = f"{savedir}/conjunctions_strokes/chunk_diff_from_prev"
    LIST_VAR = [
        "chunk_diff_from_prev",
    ]
    LIST_VARS_CONJUNCTION = [
        ["CTXT_prev_this_next"],
    ]
    DF = DS.Dat
    if len(DF)>0:
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, params_anova["globals_nmin"], D)

    ######## epoch encoding, context same
    sdir = f"{savedir}/conjunctions_strokes/epoch_context1"
    LIST_VAR = [
        "epoch",
    ]
    LIST_VARS_CONJUNCTION = [
        ["CTXT_prev_this_next"],
    ]
    # DS.context_define_local_context_motif(1,1)
    # DS.context_chunks_assign_columns()
    DF = DS.Dat
    if len(DF)>0:
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, params_anova["globals_nmin"], D)

    ######## epoch encoding, context same
    sdir = f"{savedir}/conjunctions_strokes/epoch_context2"
    LIST_VAR = [
        "epoch",
    ]
    LIST_VARS_CONJUNCTION = [
        ["CTXT_prev_this_next"],
    ]
    DS.context_define_local_context_motif(2,1)
    DS.context_chunks_assign_columns()
    DF = DS.Dat
    if len(DF)>0:
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, params_anova["globals_nmin"], D)
    # Restore to defaults
    DS.context_define_local_context_motif(1,1)
    DS.context_chunks_assign_columns()

    ######## encoding of shape (otherwise context matched), 
    sdir = f"{savedir}/conjunctions_strokes/shape_byepoch"
    LIST_VAR = [
        "shape",
    ]
    LIST_VARS_CONJUNCTION = [
        ["CTXT_prev_next", "gridloc", "epoch"],
    ]
    DF, dictdf = extract_with_levels_of_conjunction_vars(DS.Dat, var="epoch", 
                                                             vars_others=["CTXT_prev_next", "gridloc"], 
                                                             n_min=2, lenient_allow_data_if_has_n_levels=2)
    if len(DF)>0:
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, params_anova["globals_nmin"], D)
