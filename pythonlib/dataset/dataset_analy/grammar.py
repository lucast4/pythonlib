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
from pythonlib.tools.exceptions import NotEnoughDataException


rcParams.update({'figure.autolayout': True})

## OLD, before changed it to make sure it only works with  matlab rules (not new parses)
def preprocess_dataset_recomputeparses(D, DEBUG=False,
                                       exclude_because_online_abort=False,
                                       ONLY_ACTUAL_RULE=True,
                                       replace_agent_with_epoch=True,
                                       include_alternative_rules=False,
                                       recompute_taskfeat_cat=True):
    """ Preprocess Dataset, extracting score by looking at parsese of rules for each epoch,
    and asking if beh is compatible with any of them.
    NOTE: dataset length will be multiplied by however many rules there are...
    PARAMS:
    - include_alternative_rules, bool, if True, then includes all rules related to the rules in this experiment,
    which can be time consuming and also fail if tasks are incompaible with rules which assert that tasks are...
    """
    from pythonlib.behmodelholder.preprocess import generate_scored_beh_model_data_long
    from pythonlib.dataset.modeling.discrete import rules_related_rulestrings_extract_auto
    
    # get epochsets
    D.epochset_apply_sequence_wrapper()

    if recompute_taskfeat_cat:
        # get taskfeatures (e.g., repeat circle, separated by line)
        D.taskfeatures_category_classify_by(method="shape_repeat", colname="taskfeat_cat")

    # 2) Get grammar scores.
    # - get rules autoamticlaly.
    if include_alternative_rules:
        # Then include alternative hypotheses. This can lead to many rules. Can also fail if it is a rule that is incompatibvle
        # with data (e.g., sequence of shapes, but this sequence doesnt include all of the sahpes in beh)
        list_rules = rules_related_rulestrings_extract_auto(D)
    else:
        list_rules = D.grammarparses_rulestrings_exist_in_dataset()
    bm = generate_scored_beh_model_data_long(D, list_rules = list_rules, DEBUG=DEBUG,
                                             ONLY_ACTUAL_RULE=ONLY_ACTUAL_RULE)

    if bm is None:
        return None

    if ONLY_ACTUAL_RULE:
        # Then confirm that max one trial per trialcode.
        from pythonlib.tools.pandastools import grouping_get_inner_items
        groupdict = grouping_get_inner_items(bm.DatLong, "trialcode")
        if max([len(x) for x in groupdict.values()])>1:
            print(bm.DatLong["trialcode"].value_counts())
            print(groupdict)
            assert False, "prob multiple agents per trialcode?"

    if replace_agent_with_epoch:
        # This so that plots use epoch, not agent (which is rulestring).
        # This is the standard if do matlab version, but not for parses version
        bm.DatLong["agent_rule_orig"] = bm.DatLong["agent_rule"]
        bm.DatLong["agent_rule"] = bm.DatLong["epoch"]
        bm._initialize_agent()

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
    D.taskfeatures_category_classify_by(method="shape_repeat", colname="taskfeat_cat")

    # 2) Get grammar scores.
    bm = generate_scored_beh_model_data_matlabrule(D)

    if exclude_because_online_abort:
        # remove the rows from bm that have good sequence, but online abort.
        bm.DatLong = bm.DatLong[~bm.DatLong["exclude_because_online_abort"]].reset_index(drop=True)

    return bm

def pipeline_generate_and_plot_all(D,
                                   # which_rules="matlab", # 11/28/23 - beause parses are required for somet hings, like stepwise.py
                                   which_rules="recompute_parses",
                reset_grammar_dat=False, doplots=True, remove_repeated_trials=True,
                save_suffix=None, run_inner_loop_only=False):
    """ Entire pipeline to extract data and plot, for 
    a single dataset
    PARAMS:
    - which_rules, str, either to use ObjectClass matlab rule, or to regenreate
    parsesa nd ask if beh is compativle iwth any of the "same-rule" parses.
    RETURNS:
    - bmh, with removed trials that are "exclude_because_online_abort"
    - SDIR
    OR (None, None) if not enough data 
    """
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    from pythonlib.dataset.modeling.discrete import rules_related_rulestrings_extract_auto

    # Remove baselione
    D.preprocessGood(params=["remove_baseline"])

    # Remove single prim tasks that may be labeled as "grammar" and within grammar blocks.
    D.preprocessGood(params=["task_strokes_more_than_one"])

    # Get shape code seuqence
    D.grammar_correct_sequence_tasksequencer_shapeseq_assign()

    ################# AD HOC SPLITTING OF DATASETS
    # if D.animals()==["Diego"] and D.Dat["date"].unique().tolist()==['231025'] and not run_inner_loop_only:
    if False and not run_inner_loop_only:
        # Then split sess
        list_sess = sorted(D.Dat["session"].unique().tolist())
        for sess in list_sess:
            Dcopy = D.copy()
            Dcopy.Dat = Dcopy.Dat[Dcopy.Dat["session"]==sess].reset_index(drop=True)
            bmh, SDIR = pipeline_generate_and_plot_all(Dcopy, which_rules, 
                reset_grammar_dat, doplots, remove_repeated_trials, 
                save_suffix=f"sess_{sess}", run_inner_loop_only=True)
    else:
        ################## Create save directiory
        SDIR = D.make_savedir_for_analysis_figures("grammar")
        if save_suffix is not None:
            SDIR = f"{SDIR}/{save_suffix}"

        # savedir= f"{SDIR}/summary"
        # os.makedirs(savedir, exist_ok=True) 

        if reset_grammar_dat:
            D.GrammarDict = {}

        # 1) Get learning metaparams
        list_blocksets_with_contiguous_probes = learn_preprocess(D, remove_repeated_trials=remove_repeated_trials)

        # Epochsets
        # print(D.Dat)
        # print(len(D.Dat))
        # print(D.Dat["epoch"])
        # assert False
        D.sequence_strokes_compute_01_sameness_status() # strokes01_sameness

        # grammar_recompute_parses = False # just use the matlab ground truth
        if which_rules=="matlab":
            # use the ground truth objectclass
            bmh  = preprocess_dataset_matlabrule(D)
        elif which_rules=="recompute_parses":
            print("******** len Dat:", len(D.Dat))
            bmh  = preprocess_dataset_recomputeparses(D, ONLY_ACTUAL_RULE=True)
            # assert False, "aggregate bmh.DatLong so that there is only one ind per trialcode. this should work since success_binary_quick should be identical for all instance for a given trialcode. confirm this"
            # Problem sovled with ONLY_ACTUAL_RULE
            assert bmh is not None
        else:
            print(which_rules)
            assert False

        if len(D.Dat)==0:
            print("NO DATA!!")
            return None, None
        
        # Use only no-sup data for these
        # dfthis = bmh.DatLong[bmh.DatLong["superv_SEQUENCE_SUP"]=="off"]
        # dfthis = dfthis[dfthis["exclude_because_online_abort"]==False]
        bmh.DatLong = bmh.DatLong[bmh.DatLong["superv_SEQUENCE_SUP"]=="off"]
        bmh.DatLong = bmh.DatLong[bmh.DatLong["exclude_because_online_abort"]==False].reset_index(drop=True)

        # make grouping var
        if "epochset" in bmh.DatLong.columns:
            from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
            _, bmh.DatLong = grouping_append_and_return_inner_items(bmh.DatLong, ["epochset", "epoch_orig"], new_col_name="epochset_epoch", return_df=True)

        ### SAVE A GOOD DATAFRAME (for connecting across expts)
        # First, preprocess to score/extract all grammar-related stuff
        path = f"{SDIR}/df_trial.pkl"
        bmh.DatLong.to_pickle(path)

        if doplots:

            ######### 2) Plot summary
            # dfGramScore = bmh.DatLong  
            # dfGramScore = dfGramScore[dfGramScore["exclude_because_online_abort"]==False]

            from pythonlib.tools.pandastools import stringify_values
            if False:
                DF = stringify_values(bmh.DatLong)
            else:
                DF = stringify_values(D.Dat)

            # if not checkIfDirExistsAndHasFiles(f"{SDIR}/summary")[1]:
            plot_performance_all(DF, list_blocksets_with_contiguous_probes, SDIR)
            plot_performance_timecourse(DF, list_blocksets_with_contiguous_probes, SDIR)
            plot_performance_static_summary(DF, list_blocksets_with_contiguous_probes, SDIR, False)
            if False: # I never hceck this anymore...
                plot_performance_static_summary(DF, list_blocksets_with_contiguous_probes, SDIR, True)
            plot_counts_heatmap(DF, SDIR)
            plot_performance_trial_by_trial(None, D, SDIR)
            if False: # takes too long, not runing for now
                plot_performance_each_char(DF, D, SDIR)
            # 1) print all the taskgroups
            D.taskgroup_char_ntrials_print_save(SDIR)

            # plot counts, only success triuals
            df = DF[DF["success_binary_quick"]==True].reset_index(drop=True)
            plot_counts_heatmap(df, SDIR, suffix="SUCCESS")
            # else:
            #     print("[SKIPPING, since SDIR exists and has contents: ", SDIR)

            ####### 1) COmpare beh to all hypotheses (rules, discrete)
            # Also make plots for rule-based analysis
            savedir= f"{SDIR}/discrete_rules"
            os.makedirs(savedir, exist_ok=True) 

            ####### Transitive inference, shape seq, concrete syntax
            # Shape sequence, split by what shapes are presnet.
            from pythonlib.dataset.modeling.discrete import _tasks_categorize_based_on_rule_shape_sequence_TI
            from pythonlib.tools.pandastools import stringify_values

            # combine in single plot (all taskgroups)
            # -- only do this for shapes rules
            #######
            plot_syntax_TI(D, savedir)

            ####### Combine in single plot (all taskgroups)
            sdir = f"{savedir}/score_epoch_x_rule_splitby"
            os.makedirs(sdir, exist_ok=True)

            LIST_SPLIT_BY = ["taskconfig_shp_code", "taskgroup", "probe", "strokes01_sameness"]
            if "epochset" in bmh.DatLong.columns:
                LIST_SPLIT_BY.append("epochset")
            if "taskfeat_cat" in bmh.DatLong.columns:
                LIST_SPLIT_BY.append("taskfeat_cat")          
          
            # # Use only no-sup data for these
            # # dfthis = bmh.DatLong[bmh.DatLong["superv_SEQUENCE_SUP"]=="off"]
            # # dfthis = dfthis[dfthis["exclude_because_online_abort"]==False]
            # bmh.DatLong = bmh.DatLong[bmh.DatLong["superv_SEQUENCE_SUP"]=="off"]
            # bmh.DatLong = bmh.DatLong[bmh.DatLong["exclude_because_online_abort"]==False].reset_index(drop=True)

            for split_by in LIST_SPLIT_BY:
                try:
                    # Old plots
                    bmh.plot_score_cross_prior_model_splitby(df=bmh.DatLong, split_by=split_by,
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
                    bmh.plot_score_cross_prior_model_splitby_v2(df=bmh.DatLong, split_by=split_by, savedir=sdir)

                    plt.close("all")

                    # except Exception as err:
                    #     pass

                    # Plot timecourse, one plot for each epoch
                    list_levels = D.Dat[split_by].unique().tolist()
                    for lev in list_levels:
                        df = D.Dat[(D.Dat["exclude_because_online_abort"]==False) & (D.Dat[split_by]==lev)]
                        fig=sns.relplot(data=df, x="tvalfake", col="epoch", col_wrap=2, y="success_binary_quick", 
                                    hue="session",
                                    height=3, aspect=3, alpha=0.25)
                        savefig(fig, f"{sdir}/timecourse-splitby_{split_by}-lev_{lev}.pdf")   

                        plt.close("all") 
                except Exception as err:
                    print("[grammar] Skipping for split_by=", split_by)

            # Do permutation test of whether score is significant across epochs
            # (e.g., if using microstim to perturb baehavior)
            if not np.all(bmh.DatLong["epoch"] == bmh.DatLong["epoch_orig"]):
                # Then epoch is different from epoch_orig
                bmh.stats_score_permutation_test(split_plots_by="epoch_orig", 
                    savedir=sdir)        
                bmh.stats_score_permutation_test(split_plots_by="epoch_orig", 
                    savedir=sdir, suffix="flat")        
            try:
                bmh.stats_score_permutation_test(split_plots_by=None, savedir=sdir)
            except NotEnoughDataException as err:
                pass
            except Exception as err:
                raise err
            bmh.stats_score_permutation_test(split_plots_by=None, savedir=sdir, suffix="flat")

            #### Separate p-vals for each epochset.

            ### MICROSTIM PLOTS
            if "microstim_epoch_code" in bmh.DatLong.columns:
                CODE_OFF = "off"
                sdir = f"{savedir}/MICROSTIM-score_epoch_x_rule_splitby"
                os.makedirs(sdir, exist_ok=True)

                for split_by in LIST_SPLIT_BY:
                    bmh.plot_score_cross_prior_model_splitby(df=bmh.DatLong, split_by=split_by,
                        sdir=sdir, suffix="trialdat", var="microstim_epoch_code") 
                    if False:
                        # afgg doesnt have the variables.
                        bmh.plot_score_cross_prior_model_splitby_agg(split_by=split_by,
                            sdir=sdir, suffix="aggdat", var="microstim_epoch_code") 
                
                bmh.stats_score_permutation_test(var="microstim_epoch_code", savedir=sdir, split_plots_by="epoch_orig")
                bmh.stats_score_permutation_test(var="microstim_epoch_code", savedir=sdir, split_plots_by=None)

                bmh.stats_score_permutation_test(var="microstim_epoch_code", savedir=sdir, split_plots_by="epoch_orig", suffix="flat")
                bmh.stats_score_permutation_test(var="microstim_epoch_code", savedir=sdir, split_plots_by=None, suffix="flat")

                # compare each one to off
                list_code = bmh.DatLong["microstim_epoch_code"].unique().tolist()
                for code in list_code:
                    if code==CODE_OFF:
                        continue
                    dfthis = bmh.DatLong[bmh.DatLong["microstim_epoch_code"].isin([CODE_OFF, code])].reset_index(drop=True)

                    bmh.stats_score_permutation_test(df=dfthis, var="microstim_epoch_code", savedir=sdir, split_plots_by="epoch_orig", suffix=f"code_{code}")
                    bmh.stats_score_permutation_test(df=dfthis, var="microstim_epoch_code", savedir=sdir, split_plots_by=None, suffix=f"code_{code}")

                    bmh.stats_score_permutation_test(df=dfthis, var="microstim_epoch_code", savedir=sdir, split_plots_by="epoch_orig", suffix=f"code_{code}-flat")
                    bmh.stats_score_permutation_test(df=dfthis, var="microstim_epoch_code", savedir=sdir, split_plots_by=None, suffix=f"code_{code}-flat")

            ############# time-dependence
            sdir = f"{savedir}/by_time"
            os.makedirs(sdir, exist_ok=True)
            plot_binned_by_time(D, sdir)
            if "microstim_epoch_code" in bmh.DatLong.columns:
                plot_trial_by_trial(D, sdir)

            ######## CONJUNCTIONS PLOTS
            # DS, dataset_pruned_for_trial_analysis, params_anova, params_anova_extraction = conjunctions_preprocess(D)
            DS, D, params_anova = conjunctions_preprocess(D)
            if DS is not None:
                savedir = f"{SDIR}"
                conjunctions_plot(D, DS, savedir, params_anova)

            ## STEPWISE action plots (e..g, classify seuqence errors)
            # try:
            plot_stepwise_actions(D)
            # except Exception as err:
            #     # skip, since I usualyy dont use this.
            #     print("-- Skipping stepwise, caught this error: ", err)
            #     pass
            # from pythonlib.grammar.stepwise import preprocess_plot_actions
            # preprocess_plot_actions(D)

        plt.close("all")
        return bmh, SDIR

def plot_syntax_TI(D, savedir):
    """
    Plots related to syntax concrete, e..g, (1,0,2,...) for ACC... sequence.
    Breaks out scores split by syntax concrete.
    Also plots microstim, if that exists.
    """
    from pythonlib.dataset.modeling.discrete import _tasks_categorize_based_on_rule_shape_sequence_TI
    from pythonlib.tools.pandastools import stringify_values

    Dc = D.copy()
    rules_shapes = Dc.grammarparses_rules_involving_shapes(return_as_epoch_orig=True)
    print("These are the epoch_orig that are shapes rules for today: ", rules_shapes)
    print("These are the epoch_orig that exist today:",  Dc.Dat["epoch_orig"].unique())
    Dc.Dat = Dc.Dat[Dc.Dat["epoch_orig"].isin(rules_shapes)].reset_index(drop=True)

    if "FEAT_num_strokes_beh" not in Dc.Dat:
        Dc.extract_beh_features()

    sdir = f"{savedir}/syntax_concrete_TI"
    os.makedirs(sdir, exist_ok=True)
    # Dc = D.copy()
    Dc.preprocessGood(params=["remove_baseline"])

    # version = "shape_indices"
    version = "endpoints"
    list_taskcat = [_tasks_categorize_based_on_rule_shape_sequence_TI(Dc, ind, version) for ind in range(len(Dc.Dat))]
    Dc.Dat["taskfeat_cat_TI"] = list_taskcat

    Dc.grammarparses_syntax_concrete_append_column()
    Dc.sequence_extract_shapes_drawn()

    # Plot
    df = stringify_values(Dc.Dat)

    # Plot counts
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    for nstrokes in df["FEAT_num_strokes_task"].unique():
        dfthis = df[df["FEAT_num_strokes_task"]==nstrokes].reset_index(drop=True)
        dfthis_success = df[(df["FEAT_num_strokes_task"] == df["FEAT_num_strokes_beh"]) & (df["FEAT_num_strokes_task"] == nstrokes)].reset_index(drop=True)
        
        # Plot counts of syntax concrete vs. epoch. Including by locaiton config. 
        # Useful if epoch designates different shape sets.
        fig = grouping_plot_n_samples_conjunction_heatmap(dfthis_success, "epoch", "syntax_concrete", ["FEAT_num_strokes_task"])
        savefig(fig, f"{sdir}/counts-SUCCESS-epoch-vs-syntax_concrete-nstrokes={nstrokes}.pdf")

        fig = grouping_plot_n_samples_conjunction_heatmap(dfthis_success, "locs_drawn_x", "syntax_concrete", ["epoch", "FEAT_num_strokes_task"]);
        savefig(fig, f"{sdir}/counts-SUCCESS-locs_drawn_x-vs-syntax_concrete-nstrokes={nstrokes}.pdf")
        
        fig = grouping_plot_n_samples_conjunction_heatmap(dfthis_success, "locs_drawn_x", "epoch", ["syntax_concrete", "FEAT_num_strokes_task"]);
        savefig(fig, f"{sdir}/counts-SUCCESS-locs_drawn_x-vs-epoch-nstrokes={nstrokes}.pdf")

        # Older plots.
        fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, "syntax_concrete", "probe", ["epoch", "FEAT_num_strokes_task"])
        savefig(fig, f"{sdir}/counts-syntax_concrete-vs-probe-nstrokes={nstrokes}.pdf")
        
        fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, "syntax_concrete", "taskfeat_cat_TI", ["epoch", "FEAT_num_strokes_task"])
        savefig(fig, f"{sdir}/counts-syntax_concrete-vs-taskfeat_cat_TI-nstrokes={nstrokes}.pdf")

        fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, "syntax_concrete", "supervision_online", ["epoch", "FEAT_num_strokes_task"])
        savefig(fig, f"{sdir}/counts-syntax_concrete-vs-supervision_online-nstrokes={nstrokes}.pdf")

        from pythonlib.tools.pandastools import plot_subplots_heatmap
        fig, axes = plot_subplots_heatmap(dfthis, "syntax_concrete", "probe", "success_binary_quick", "supervision_online", 
                                            share_zlim=True, annotate_heatmap=False, ZLIMS=[0,1])
        savefig(fig, f"{sdir}/heatmap-syntax_concrete-vs-probe-nstrokes={nstrokes}.pdf")
    plt.close("all")

    # Plot scores
    order = sorted(df["syntax_concrete"].unique().tolist())
    fig = sns.catplot(data=df, x="syntax_concrete", y="success_binary_quick", kind="point", col="FEAT_num_strokes_task", 
                        aspect=2, row="supervision_online", hue="probe", order=order)
    rotateLabel(fig, 90)
    for ax in fig.axes.flatten():
        ax.set_ylim([0, 1])
    savefig(fig, f"{sdir}/syntax_concrete-1.pdf")

    fig = sns.catplot(data=df, y="syntax_concrete", x="success_binary_quick", kind="point", col="FEAT_num_strokes_task", 
                        aspect=0.5, row="supervision_online", hue="probe", order=order)
    rotateLabel(fig, 90)
    for ax in fig.axes.flatten():
        ax.set_ylim([0, 1])
    savefig(fig, f"{sdir}/syntax_concrete-1b.pdf")

    fig = sns.catplot(data=df, x="syntax_concrete", y="success_binary_quick", alpha=0.4, jitter=True, 
                        col="FEAT_num_strokes_task", aspect=2, row="supervision_online",
                    order=order)
    rotateLabel(fig, 90)
    savefig(fig, f"{sdir}/syntax_concrete-2.pdf")

    fig = sns.catplot(data=df, y="syntax_concrete", x="success_binary_quick", alpha=0.4, jitter=True, 
                        col="FEAT_num_strokes_task", aspect=2, row="supervision_online",
                    order=order)
    rotateLabel(fig, 90)
    savefig(fig, f"{sdir}/syntax_concrete-2b.pdf")

    # Separate by epoch
    fig = sns.catplot(data=df, x="syntax_concrete", y="success_binary_quick", kind="bar", 
                    hue="epoch", col="FEAT_num_strokes_task", aspect=2, row="supervision_online",
                    order=order)
    rotateLabel(fig, 90)
    savefig(fig, f"{sdir}/syntax_concrete-epoch-1.pdf")

    fig = sns.catplot(data=df, y="syntax_concrete", x="success_binary_quick", kind="bar", 
                    hue="epoch", col="FEAT_num_strokes_task", aspect=2, row="supervision_online",
                    order=order)
    rotateLabel(fig, 90)
    savefig(fig, f"{sdir}/syntax_concrete-epoch-1b.pdf")
    
    order = sorted(df["taskfeat_cat_TI"].unique())
    fig = sns.catplot(data=df, x="taskfeat_cat_TI", y="success_binary_quick", kind="bar", col="FEAT_num_strokes_task", 
                        aspect=2, row="supervision_online", hue="probe", order=order)
    rotateLabel(fig, 90)
    savefig(fig, f"{sdir}/taskfeat_cat_TI-1.pdf")

    fig = sns.catplot(data=df, x="taskfeat_cat_TI", y="success_binary_quick", alpha=0.4, jitter=True, col="FEAT_num_strokes_task", aspect=2, row="supervision_online",
                    order=order)
    rotateLabel(fig, 90)
    savefig(fig, f"{sdir}/taskfeat_cat_TI-2.pdf")

    # Separate by epoch
    fig = sns.catplot(data=df, x="taskfeat_cat_TI", y="success_binary_quick", kind="bar", 
                        hue="epoch",
                        col="FEAT_num_strokes_task", aspect=2, row="supervision_online",
                    order=order)
    rotateLabel(fig, 90)
    savefig(fig, f"{sdir}/taskfeat_cat_TI-epoch-1.pdf")

    plt.close("all")

    if "microstim_epoch_code" in df.columns:
        order = sorted(df["syntax_concrete"].unique().tolist())
        fig = sns.catplot(data=df, x="syntax_concrete", y="success_binary_quick", kind="bar", 
                        hue="microstim_epoch_code", col="FEAT_num_strokes_task", aspect=2, row="supervision_online",
                        order=order)
        rotateLabel(fig, 90)
        savefig(fig, f"{sdir}/syntax_concrete-microstim_epoch_code-1.pdf")

        order = sorted(df["taskfeat_cat_TI"].unique())
        fig = sns.catplot(data=df, x="taskfeat_cat_TI", y="success_binary_quick", kind="bar", col="FEAT_num_strokes_task", aspect=2, row="supervision_online",
                        hue="microstim_epoch_code", order=order)
        rotateLabel(fig, 90)
        savefig(fig, f"{sdir}/taskfeat_cat_TI-microstim_epoch_code-1.pdf")

        stim_codes = [code for code in df["microstim_epoch_code"].unique() if code!="off"]
        from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
        for code in stim_codes:
            for plot_text in [False, True]:
                _, fig = plot_45scatter_means_flexible_grouping(df, "microstim_epoch_code", "off", code, ("supervision_online", "FEAT_num_strokes_task"), 
                                                    "success_binary_quick", "syntax_concrete", shareaxes=True, plot_text=plot_text);            
                savefig(fig, f"{sdir}/scatter45-syntax_concrete-microstim_epoch_code-{code}-plot_text={plot_text}.pdf")

                _, fig = plot_45scatter_means_flexible_grouping(df, "microstim_epoch_code", "off", code, ("supervision_online", "FEAT_num_strokes_task", "taskfeat_cat_TI"), 
                                                    "success_binary_quick", "syntax_concrete", shareaxes=True, plot_text=plot_text);            
                savefig(fig, f"{sdir}/scatter45-syntax_concrete-microstim_epoch_code-splitby_taskfeat_cat_TI-{code}-plot_text={plot_text}.pdf")


def plot_stepwise_actions(D):
    """ All plots related to probs of actions at each stroke step.
    """
    from pythonlib.grammar.stepwise import preprocess_plot_actions

    # 1) All data
    preprocess_plot_actions(D, suffix="alldata")

    # 2) Separate for each epochset
    list_epochset = D.Dat["epochset"].unique().tolist()
    nmin = 10
    for es in list_epochset:
        Dc = D.copy()
        Dc.Dat = Dc.Dat[Dc.Dat["epochset"]==es].reset_index(drop=True)
        if len(Dc.Dat)>nmin:
            preprocess_plot_actions(Dc, suffix=f"epochset-{es}")

    # 3) split by epochset_char
    D.epochset_extract_common_epoch_sets("character", epochset_col_name="epochsetchar")
    list_epochset = D.Dat["epochsetchar"].unique().tolist()
    nmin = 10
    for es in list_epochset:
        Dc = D.copy()
        Dc.Dat = Dc.Dat[Dc.Dat["epochsetchar"]==es].reset_index(drop=True)
        if len(Dc.Dat)>nmin:
            preprocess_plot_actions(Dc, suffix=f"epochsetchar-{es}")

    # 3) Split into first and last half of data
    D1, D2 = D.splitdataset_by_trial()
    preprocess_plot_actions(D1, suffix=f"splitbytime_half1")
    preprocess_plot_actions(D2, suffix=f"splitbytime_half2")

def plot_binned_by_time(D, sdir):
    """ bin trials by their times. change over session?"""

    LIST_SPLIT_BY = ["probe", "epochset"]

    Dc = D.copy()
    Dc.preprocessGood(params=["remove_baseline", "no_supervision"])
    # print(len(Dc.Dat))

    ############### SPLIT BY TIME (TIME BINS)
    # bin trials and assing to dataframe
    from pythonlib.tools.nptools import bin_values
    list_sess = Dc.Dat["session"].unique().tolist()
    for sess in list_sess:
        dfthissess = Dc.Dat[Dc.Dat["session"]==sess].reset_index(drop=True)
        # sort by trial
        dfthissess = dfthissess.sort_values(by="trial").reset_index(drop=True)
        for nbins in [2,4,8]:
            vals = dfthissess["trial"].tolist()
            timebins = bin_values(vals, nbins)
            dfthissess["trial_binned"] = timebins

            y = "success_binary_quick"
            for split_by in LIST_SPLIT_BY:
                fig = sns.catplot(data=dfthissess, x="trial_binned", y=y, hue="epoch", row=split_by, kind="point", errorbar=("ci", 68))
                path = f"{sdir}/binned_time-nbins_{nbins}-sess_{sess}-splitby_{split_by}.pdf"
                savefig(fig, path)
                print(path)
                plt.close("all")



def plot_trial_by_trial(D, sdir):
    """ to lookk closely at whether trials post stim have lasting effets
    """

    Dc = D.copy()
    Dc.preprocessGood(params=["remove_baseline", "no_supervision"])
    dfthis = Dc.Dat
    print(len(dfthis))

    # Plot timecourse and visualize
    # sns.relplot(data=dfthis, x="trial", y="success_binary_quick", hue="microstim_epoch_code",
    #            aspect=8, alpha=0.5, style="block", row="session")

    # sns.relplot(data=dfthis, x="trial", hue="success_binary_quick", y="microstim_epoch_code",
    #            aspect=8, alpha=0.5, style="block", row="session")

    fig = sns.relplot(data=dfthis, x="trial", hue="microstim_epoch_code", y="block",
               aspect=10, alpha=0.5, style="success_binary_quick", row="session")
    savefig(fig, f"{sdir}/TIMECOURSE.pdf")   

    ####### ALINGED TO STIM TRIALS.
    for n in [1,2,3,4]:
        # y = "success_seq_nmatch"
        y = "success_binary_quick"
        EXCLUDE_IF_ABORT=False
        PRINT = False

        # run a sliding window over trials
        list_sess = dfthis["session"].unique().tolist()
        for sess in list_sess:
            dfthissess = dfthis[dfthis["session"]==sess].reset_index(drop=True)

            # sort by trial
            dfthissess = dfthissess.sort_values(by="trial").reset_index(drop=True)
            
            list_dfwindows = []
            for i_center in range(n, len(dfthissess)-n):
                dfwind = dfthissess.iloc[i_center-n:i_center+n+1].reset_index(drop=True)
                
                ## CRITERIA FOR EXCLUSION
                if EXCLUDE_IF_ABORT:
                    if np.any(dfwind["exclude_because_online_abort"]):
                        continue
                
                if not np.all(dfwind["microstim_epoch_code"][:n]=="off"):
        #             print(dfwind["microstim_epoch_code"][:n])
                    continue
                
                if not np.all(dfwind["microstim_epoch_code"][n+1:]=="off"):
                    continue
                    
                assert np.all(np.diff(dfwind["trial"])>0)
                
                list_dfwindows.append(dfwind) # get n flanking each side.

                if PRINT:
                    print(i_center, "----", dfwind["trial"].tolist())

            DICT_LISTS_VALUES = {}

            for dfwind in list_dfwindows:

                # only keep if middle index is stim
                STIM_CODE = dfwind.iloc[n]["microstim_epoch_code"]

                # get timecoures of variable.
                succs = 1*dfwind[y].values

                if STIM_CODE not in DICT_LISTS_VALUES.keys():
                    DICT_LISTS_VALUES[STIM_CODE] = [succs]
                else:
                    DICT_LISTS_VALUES[STIM_CODE].append(succs)        

                if False:
                    if STIM_CODE=="off":
                        ax = axes.flatten()[0]
                    elif STIM_CODE=="TTL3-fgon":
                        ax = axes.flatten()[1]
                    elif STIM_CODE=="TTL4-fgon":
                        ax = axes.flatten()[2]
                    else:
                        print(STIM_CODE)
                        assert False

                    ax.plot(xs, succs, "ok", alpha=0.005)

            for k,v in DICT_LISTS_VALUES.items():
                DICT_LISTS_VALUES[k] = np.stack(v, axis=0) # (ndat, ntimes)

            # plot
            nstimcodes = len(DICT_LISTS_VALUES.keys())
            if nstimcodes==0:
                continue
            elif nstimcodes==1:
                nstimcodes=2 # for unsqueezing
            fig, axes = plt.subplots(1, nstimcodes, figsize=(6,3))

            xs = np.arange(2*n+1)

            for ax, (stim_code, values_mat) in zip(axes.flatten(), DICT_LISTS_VALUES.items()):
                values_mean = np.mean(values_mat, axis=0)    
                ax.plot(xs, values_mean, "-ok")
                ax.set_title(stim_code)
                
                ax.set_ylabel(f"n={values_mat.shape[0]}")
                ax.set_xlabel(f"trials, aligned at middle trial")

                if y=="success_seq_nmatch":
                    ax.set_ylim(0,6)
                elif y=="success_binary_quick":
                    ax.set_ylim(0,1)
                else:
                    assert False

            savefig(fig, f"{sdir}/ALIGNED_TRIALS-n_{n}-sess_{sess}.pdf")   
            plt.close("all")

def conjunctions_preprocess(D):
    """
    Online sequenbce, plot all conjucntions for grammar neural analyses
    NOTE: returns None if D is empty after preprocessing.
    """
    from neuralmonkey.metadat.analy.anova_params import dataset_apply_params, params_getter_dataset_preprocess
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
    D.grammarparses_successbinary_score_wrapper()
    D.preprocessGood(params=["correct_sequencing_binary_score"])

    if len(D.Dat)==0:
        return (None for _ in range(4))

    # Assign chunks info to tokens
    for ind in range(len(D.Dat)):
        D.grammarparses_taskclass_tokens_assign_chunk_state_each_stroke(ind)

    # Clean and extract dataset as would in neural analy
    # ListD = [D]
    animal = D.animals(force_single=True)[0]
    tmp = D.Dat["date"].unique()
    assert len(tmp)==1
    DATE = int(tmp[0])
    # which_level = "stroke"
    ANALY_VER = "seqcontext"
    # anova_interaction = False

    # Dall, dataset_pruned_for_trial_analysis, TRIALCODES_KEEP, params_anova, params_anova_extraction = \
    #     dataset_apply_params(ListD, None, ANALY_VER, animal, DATE)
    D, _, params_anova = dataset_apply_params(D, None, ANALY_VER, animal, DATE)

    from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
    DS = preprocess_dataset_to_datstrokes(D, "all_no_clean")

    # # list_features = ["chunk_rank", "chunk_within_rank", "chunk_within_rank", "chunk_n_in_chunk"]
    # list_features = []
    # DS = datasetstrokes_extract(dataset_pruned_for_trial_analysis, list_features=list_features)

    # # Legacy code -- get params anova.
    # params_anova = params_getter_dataset_preprocess(ANALY_VER, animal, DATE)

    return DS, D, params_anova


def conjunctions_plot(D, DS, savedir, params_anova):
    """ Make all plots and printed text for conjuctuions relevant for grammar, at level of
    strokes during drawing
    PARAMS:
    - D, DS, savedir, params_anova, see output of conjunctions_preprocess
    """

    globals_nmin = 5

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
    _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, globals_nmin, D)

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
    DF = DS.Dat[DS.Dat["chunk_diff_from_prev"]==True].reset_index(drop=True)
    if len(DF)>0:
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, globals_nmin, D)

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
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, globals_nmin, D)

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
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, globals_nmin, D)

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
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, globals_nmin, D)
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
    DF, dictdf = extract_with_levels_of_conjunction_vars(DS.Dat, var="epoch", vars_others=["CTXT_prev_next", "gridloc"],
                                                         n_min_across_all_levs_var=2,
                                                         lenient_allow_data_if_has_n_levels=2)
    if len(DF)>0:
        _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, globals_nmin, D)

def chunk_rank_global_extract(df, check_low_freq_second_shape=True, shape_ratio_max = 0.75):
    """
    For each (date, epoch) map each shape to a global chunk_rank, determined by checking frequency of 
    presentation of each shape at each rank.
    
    PARAMS:
    - df, holds data. must have columns: date, epoch, chunk_rank, shape.

    RETURNS:
    - dfchunkrankmap, each row is a diff "chunk_rank_global", mapped to a (date, epoch, shape).
    - NOTE: also modifies input to have a new column: chunk_rank_global
    """
    
    assert len(df[(df["chunk_rank"]=="none")])==0, "all cr must be numbers, or else will fail"
    
    ### (1) Get the mapping from chunk_rank to shape by taking the most freqeunt shape at each chunkrank
    def F(x):
        shapes_ordered = x["shape"].value_counts().index.tolist()
        counts_ordered = x["shape"].value_counts().values
        if len(shapes_ordered)>1: # multipel shapes at this cr...
            assert counts_ordered[1] <= counts_ordered[0]
            if check_low_freq_second_shape:
                if counts_ordered[1]/counts_ordered[0]>shape_ratio_max:
                    print(shapes_ordered, counts_ordered)
                    print(x)
                    print(x["shape"].value_counts())
                    assert False, "do you expect this many secondary shapes in this cr? if so, then skip this failure/"
        return shapes_ordered[0] # Return the most frequent shape.
    dfchunkrankmap = df.groupby(["date", "epoch", "chunk_rank"]).apply(F).reset_index(name="shape")

    ### Sanity checks
    # (1) Each shape is mapped to just one cr.
    def F(x):    
        unique_cr_shapes = set(tuple(this) for this in x.loc[:, ["chunk_rank", "shape"]].values.tolist())
        if not len(unique_cr_shapes) == len(x["chunk_rank"].unique()):
            print(unique_cr_shapes)
            print(x["chunk_rank"].unique())
            display(x)
            assert False, "this is major problem. simple - need to create better map"   
    dfchunkrankmap.groupby(["date", "epoch", "chunk_rank"]).apply(F)
    
    # (2) Check that all shapes are used. Do this by checking that there are not more shapes than chunk ranks.
    # Combined with above check that shapes are not reused across cr, this guarantees all shapes are used
    def F(x):
        # display(x)
        if len(x["chunk_rank"].unique()) < len(x["shape"].unique()):
            print("Why are there more shapes than chunk ranks?")
            print("All unique chunk ranks: ", x["chunk_rank"].unique())
            print("All unique shapes: ", x["shape"].unique())
            assert False
    # df[~(df["chunk_rank"]=="none")].groupby(["date", "epoch"]).apply(F) # if none, then this is single prims...
    df.groupby(["date", "epoch"]).apply(F)

    # Rename to crglobal
    dfchunkrankmap = dfchunkrankmap.rename({"chunk_rank":"chunk_rank_global"}, axis=1)

    # Assign to inout df
    map_DaEpSh_to_crglob = {}
    for _, row in dfchunkrankmap.iterrows():
        k = (row["date"], row["epoch"], row["shape"])
        v = row["chunk_rank_global"]
        map_DaEpSh_to_crglob[k] = v
    for k, v in map_DaEpSh_to_crglob.items():
        print(k, " -- ", v)
    df[f"chunk_rank_global"] = [map_DaEpSh_to_crglob[(row["date"], row[f"epoch"], row[f"shape"])] for _, row in df.iterrows()]

    # Also add a conjnctive variable
    from pythonlib.tools.pandastools import append_col_with_grp_index
    df = append_col_with_grp_index(df, ["chunk_rank_global", "shape"], "crg_shape")

    return dfchunkrankmap


########## STUFF RELATED TO extracting information for each token, regardless of what you did in behavior. ie 
# using any inputed toekns for this trial. useful e.g, for eye tracking, where you want to get information about
# fixation events.
def syntaxconcrete_extract_more_info(syntax_concrete, index_gap, append_99_to_mean_done_button=False):
    """
    Given syntax_concrete and a gap index, get more useful information about
    the current state (i.e, like the state of the drawing agent)

    PARAMS:
    - gap_index, where 0 means the gap between stroke 0 and 1.

    NOTE: If you pass in index_gap that asks about the last stroke, then this will fail unless
    you also put append_99_to_mean_done_button==True
    """

    s_tuple = []
    for chunk_rank, n_in_chunk in enumerate(syntax_concrete):
        s_tuple.extend([chunk_rank for _ in range(n_in_chunk)])
    
    if index_gap == len(s_tuple)-1:
        assert append_99_to_mean_done_button==True, "you need to turn on append_99_to_mean_done_button for this to work."

    # To allow indexing into the last gap (ie. between lsat stroke and end)
    if append_99_to_mean_done_button:
    # if index_gap == len(s_tuple)-1:
        s_tuple.append(99)
    #     do_remove_99=True
    # else:
        # do_remove_99 = False

    if index_gap < 0:
        # Then this is any time before first stroke
        pre_chunk_rank_global = -1
        s_post = s_tuple
        s_pre = tuple([])
    else:
        assert index_gap>-1
        s_post = s_tuple[(index_gap+1):]
        s_pre = s_tuple[:(index_gap+1)]
        pre_chunk_rank_global = s_pre[-1]
    post_chunk_rank_global = s_post[0]

    n_remain_in_chunk = sum([x==post_chunk_rank_global for x in s_post])
    n_completed_in_chunk = sum([x==post_chunk_rank_global for x in s_pre])

    n_in_chunk = n_completed_in_chunk + n_remain_in_chunk

    post_chunk_within_rank = n_completed_in_chunk

    # Remaining chunk ranks
    s_post_not_yet_started = [cr for cr in s_post if not cr==pre_chunk_rank_global]
    chunk_ranks_remain_not_yet_started = list(set(s_post_not_yet_started))

    # Mapping: rank --> chunk_rank
    map_rank_to_chunk_rank = {}
    for rank, chunk_rank in enumerate(s_tuple):
        map_rank_to_chunk_rank[rank] = chunk_rank    

    # Mapping: rank --> rank_within_chunk
    cr_current = -1
    map_rank_to_rank_within_chunk = {}
    for rank, cr in enumerate(s_tuple):
        if cr!=cr_current:
            # A new chunk_rank. Do reset.
            cr_current = cr
            counter_within_chunk_rank = 0
        else:
            counter_within_chunk_rank += 1
        map_rank_to_rank_within_chunk[rank] = counter_within_chunk_rank
    map_rank_to_rank_within_chunk.values()    

    # Get the indices where transitions occur (ie.. AAABBC would return [2, 4])
    n_strokes = len(s_tuple)
    inds_transition = []
    counter = -1
    for x0 in syntax_concrete:
        counter+=x0
        if counter>-1 and counter<n_strokes-1:
            inds_transition.append(counter)
    inds_transition = sorted(set(inds_transition))

    info = {
        "syntax_concrete":syntax_concrete,
        "index_gap":index_gap,
        "index_gap_is_chunk_switch":post_chunk_rank_global>pre_chunk_rank_global,
        "s_tuple":s_tuple,
        "s_tuple_remain":s_post,
        "chunk_ranks_remain_not_yet_started":chunk_ranks_remain_not_yet_started,
        "map_rank_to_chunk_rank":map_rank_to_chunk_rank,
        "map_rank_to_rank_within_chunk":map_rank_to_rank_within_chunk,
        "pre_chunk_rank_global":pre_chunk_rank_global,
        "post_chunk_rank_global":post_chunk_rank_global,
        "n_remain_in_chunk":n_remain_in_chunk,
        "n_completed_in_chunk":n_completed_in_chunk,
        "n_in_chunk":n_in_chunk,
        "post_chunk_within_rank":post_chunk_within_rank,
        "inds_stroke_before_chunk_transition":inds_transition,
    }
    return info
 
def syntaxconcrete_extract_more_info_eye_fixation(syntax_concrete, index_gap, tokens_correct_order, list_fixed_idx_task):
                                                #   dfgaps, trialcode):
    """
    Given syntax_concrete and a gap index, get more useful information about
    the current state of the eye fixation, ie., the task shape that is currnetly
    being looked at (fixated).

    PARAMS:
    - syntax_concrete, e.g, (0, 2, 3) or (2,2,1)
    - index_gap, int, state of drwing, e.g, where 0 means just finished stroke 0. Note: -1 means the gap before first stroke, which
    means any/all time before first stroke.
    - tokens_correct_order, the tokens, in correct order of drwaing. This is required to map from idx_task to 
    the actual token
    - fixed_idx_task, int, where you are looking at, ie the index_task. This is the same as tok["ind_taskstroke_orig"]
    where tok is an item in tokens_correct_order
    """

    assert index_gap>=-2
    if index_gap==-2:
        # This is the code for (between samp -- go)
        index_gap = -1
    
    ### Prep -- get information about this gap
    # First, get gap information
    if False:
        dfrow = dfgaps[(dfgaps["trialcode"]==trialcode) & (dfgaps["index_gap"]==index_gap)]
        assert len(dfrow)==1

        # gap_chunk_rank_global
        # diff_chunk_rank_global
        chunk_pre = dfrow["pre_chunk_rank_global"].values[0]
        chunk_post = dfrow["post_chunk_rank_global"].values[0]
        assert dfrow["syntax_concrete"].values[0] == syntax_concrete
        syntax_concrete = dfrow["syntax_concrete"].values[0]
    
    info_gap = syntaxconcrete_extract_more_info(syntax_concrete, index_gap)
    # assert chunk_pre == info_gap["pre_chunk_rank_global"]
    # assert chunk_post == info_gap["post_chunk_rank_global"]
    chunk_pre = info_gap["pre_chunk_rank_global"]
    chunk_post = info_gap["post_chunk_rank_global"]
    map_rank_to_chunk_rank = info_gap["map_rank_to_chunk_rank"]
    map_rank_to_rank_within_chunk = info_gap["map_rank_to_rank_within_chunk"]

    rank_pre = index_gap # ie index_gap = 0 means the gap between rank 0 and rank 1.
    rank_post = index_gap + 1

    ### For each fixation, get its information, within this gap
    # Convert task index to rank, this is easier (map from token index, to its rank (in correct sequence))
    map_taskidx_to_rank = {}
    for rank, tok in enumerate(tokens_correct_order):
        idx_orig = tok["ind_taskstroke_orig"]
        map_taskidx_to_rank[idx_orig] = rank

    list_info_fixation = []
    for fixed_idx_task in list_fixed_idx_task:
        info_fixation = {}

        fixed_rank = map_taskidx_to_rank[fixed_idx_task] # the rank in sequence of strokes, for this fixated shape

        # (1) Info related to the fixated chunk_rank
        fixed_chunk_rank = map_rank_to_chunk_rank[fixed_rank]

        # Number, summarizing
        info_fixation["chunkrank_fixed"] = fixed_chunk_rank
        info_fixation["chunkrank_fixed_minus_post"] = fixed_chunk_rank - chunk_post
        info_fixation["chunkrank_fixed_minus_pre"] = fixed_chunk_rank - chunk_pre

        # Semantic label, related to donness of chunk
        if fixed_chunk_rank < chunk_pre :
            assert fixed_chunk_rank < chunk_post, "logically not possible"
            info_fixation["chunkrank_fixed_status"] = "completed_before_last_stroke"
        elif (fixed_chunk_rank == chunk_pre) and (fixed_chunk_rank < chunk_post):
            info_fixation["chunkrank_fixed_status"] = "completed_by_last_stroke"
        elif (fixed_chunk_rank == chunk_pre) and (fixed_chunk_rank == chunk_post):
            info_fixation["chunkrank_fixed_status"] = "ongoing"
        elif (fixed_chunk_rank > chunk_pre) and (fixed_chunk_rank == chunk_post):
            info_fixation["chunkrank_fixed_status"] = "start_in_next_stroke"
        elif (fixed_chunk_rank > chunk_pre) and (fixed_chunk_rank > chunk_post):
            info_fixation["chunkrank_fixed_status"] = "start_after_next_stroke"
        else:
            print(fixed_chunk_rank, chunk_pre, chunk_post)
            assert False

        # Another semantic label, differentiating based on inner rank within chunk.
        # This is differnet from above, in that it defines "next chunk" to be the one
        # different from current, even when currently within a chunk. In contrast, above,
        # next chunk would be the current chunk.
        current_chunk = chunk_pre

        # Get the actual chunk ranks rthat are upcoming on this trial
        # Note: if this trial skips a chunk, then next chunk jumps also.
        # next_chunk = chunk_pre + 1
        next_chunk, nextnext_chunk, nextnextnext_chunk = None, None, None
        chunk_ranks_remain_not_yet_started = info_gap["chunk_ranks_remain_not_yet_started"]
        if len(chunk_ranks_remain_not_yet_started)>0:
            next_chunk = chunk_ranks_remain_not_yet_started[0] # e.g, [1,3] means these are the following hcunk ranks
        if len(chunk_ranks_remain_not_yet_started)>1:
            nextnext_chunk = chunk_ranks_remain_not_yet_started[1] # e.g, [1,3] means these are the following hcunk ranks
        if len(chunk_ranks_remain_not_yet_started)>2:
            nextnextnext_chunk = chunk_ranks_remain_not_yet_started[2] # e.g, [1,3] means these are the following hcunk ranks
        
        fixed_rank_within_chunk = map_rank_to_rank_within_chunk[fixed_rank]
        info_fixation["rank_within_chunk_fixed"] = fixed_rank_within_chunk
        assert fixed_chunk_rank == info_fixation["chunkrank_fixed"]

        # if (fixed_chunk_rank == nextnextnext_chunk) and (fixed_rank_within_chunk==0):
        #     info_fixation["chunkrank_fixed_status_v2"] = "nxtnxtnxt_chk_first_stk"

        # elif (fixed_chunk_rank == nextnextnext_chunk) and (fixed_rank_within_chunk>0):
        #     info_fixation["chunkrank_fixed_status_v2"] = "nxtnxtnxt_chk_inner_stk"

        # elif (fixed_chunk_rank == nextnext_chunk) and (fixed_rank_within_chunk==0):
        #     info_fixation["chunkrank_fixed_status_v2"] = "nxtnxt_chk_first_stk"

        # elif (fixed_chunk_rank == nextnext_chunk) and (fixed_rank_within_chunk>0):
        #     info_fixation["chunkrank_fixed_status_v2"] = "nxtnxt_chk_inner_stk"

        # elif (fixed_chunk_rank == next_chunk) and (fixed_rank_within_chunk==0):
        #     info_fixation["chunkrank_fixed_status_v2"] = "nxt_chnk_first_stk"

        # elif (fixed_chunk_rank == next_chunk) and (fixed_rank_within_chunk>0):
        #     info_fixation["chunkrank_fixed_status_v2"] = "nxt_chnk_inner_stk"

        # elif (fixed_chunk_rank == current_chunk):
        #     info_fixation["chunkrank_fixed_status_v2"] = "chk_prev_stk"

        # elif (fixed_chunk_rank < current_chunk):
        #     info_fixation["chunkrank_fixed_status_v2"] = "any_chk_fini_b4_prev_stk"

        # else:
        #     print(fixed_chunk_rank, fixed_rank_within_chunk)
        #     print(current_chunk, next_chunk)
        #     print(info_fixation)
        #     assert False

        if (fixed_rank == rank_post):
            # Then looking at next strokes. Important to have this (even thuogh it is the
            # only one that cares about rank, not chunkrank) beucase it is "null model", what we
            # expect during non-cr-transitioning gaps)
            info_fixation["chunkrank_fixed_status_v2"] = "ntx_rank"

        elif (fixed_chunk_rank == nextnextnext_chunk) and (fixed_rank_within_chunk==0):
            info_fixation["chunkrank_fixed_status_v2"] = "nxtnxtnxt_chk_first_stk"

        elif (fixed_chunk_rank == nextnextnext_chunk) and (fixed_rank_within_chunk>0):
            info_fixation["chunkrank_fixed_status_v2"] = "nxtnxtnxt_chk_inner_stk"

        elif (fixed_chunk_rank == nextnext_chunk) and (fixed_rank_within_chunk==0):
            info_fixation["chunkrank_fixed_status_v2"] = "nxtnxt_chk_first_stk"

        elif (fixed_chunk_rank == nextnext_chunk) and (fixed_rank_within_chunk>0):
            info_fixation["chunkrank_fixed_status_v2"] = "nxtnxt_chk_inner_stk"

        elif (fixed_chunk_rank == next_chunk) and (fixed_rank_within_chunk==0):
            info_fixation["chunkrank_fixed_status_v2"] = "nxt_chnk_first_stk"

        elif (fixed_chunk_rank == next_chunk) and (fixed_rank_within_chunk>0):
            info_fixation["chunkrank_fixed_status_v2"] = "nxt_chnk_inner_stk"

        elif (fixed_chunk_rank == current_chunk) and (fixed_rank < rank_post):
            info_fixation["chunkrank_fixed_status_v2"] = "chk_prev_stk_past"

        elif (fixed_chunk_rank == current_chunk) and (fixed_rank >= rank_post):
            info_fixation["chunkrank_fixed_status_v2"] = "chk_prev_stk_future"

        # elif (fixed_chunk_rank == current_chunk):
        #     info_fixation["chunkrank_fixed_status_v2"] = "chk_prev_stk"

        elif (fixed_chunk_rank < current_chunk):
            info_fixation["chunkrank_fixed_status_v2"] = "any_chk_fini_b4_prev_stk"

        else:
            print(fixed_chunk_rank, fixed_rank_within_chunk)
            print(current_chunk, next_chunk)
            print(info_fixation)
            assert False


        # (2) Info related to the rank (ignoring chunk)
        assert rank_pre == rank_post-1
        # res["rank_fixed_minus_post"] = fixed_rank - rank_post
        info_fixation["rank_fixed"] = fixed_rank
        info_fixation["rank_fixed_minus_pre"] = fixed_rank - rank_pre # only need this, beucase rank_pre == rank_post-1
        # # Is this stroke already done

        # if fixed_rank < rank_pre :
        #     res["rank_fixed_status"] = "completed_before_last_stroke"
        # elif (fixed_rank == rank_pre):
        #     res["rank_fixed_status"] = "completed_by_last_stroke"
        # elif (fixed_rank == rank_post):
        #     res["rank_fixed_status"] = "start_in_next_stroke"
        # elif fixed_rank > rank_post:
        #     res["rank_fixed_status"] = "start_after_next_stroke"
        # else:
        #     print(fixed_rank, rank_pre, rank_post)
        #     assert False

        list_info_fixation.append(info_fixation)

    # convert to dataframe
    df_fixations = pd.DataFrame(list_info_fixation)

    return df_fixations, info_gap

def syntaxconcrete_dfmod_postprocess(D, dfthis_long):
    """
    MOdifies df, where each row is trialcode, index_gap, index_shape
    """
    # Add more data to dflong
    from neuralmonkey.scripts.analy_syntax_good_gap_durations import syntaxconcrete_extract_more_info

    assert "syntax_concrete" in dfthis_long
    assert "index_gap" in dfthis_long

    map_shape_to_chunk_rank_global, map_chunk_rank_global_to_shape = D.grammarparses_rules_shape_AnBmCk_get_map_shape_to_chunk_rank()

    tmp = []
    list_pre_shape = []
    list_post_shape = []
    list_n_remain_in_chunk = []
    for _, row in dfthis_long.iterrows():
        info = syntaxconcrete_extract_more_info(row["syntax_concrete"], row["index_gap"])
        tmp.append(info["index_gap_is_chunk_switch"])

        if info["pre_chunk_rank_global"]>-1:
            # This is during drwaing
            list_pre_shape.append(map_chunk_rank_global_to_shape[info["pre_chunk_rank_global"]])
        else:
            list_pre_shape.append("none")

        list_post_shape.append(map_chunk_rank_global_to_shape[info["post_chunk_rank_global"]])

        list_n_remain_in_chunk.append(info["n_remain_in_chunk"])

    dfthis_long["index_gap_is_chunk_switch"] = tmp
    dfthis_long["pre_shape"] = list_pre_shape
    dfthis_long["post_shape"] = list_post_shape
    dfthis_long["n_remain_in_chunk"] = list_n_remain_in_chunk # remaining in the post chunk.

    # Also recode rank_fixed to be relative to current index_gap
    dfthis_long["rankfixed_min_idxgap"] = dfthis_long["rank_fixed"] - dfthis_long["index_gap"] # 1 means that you are looking at the next shape

    # Note if looking at something already done
    dfthis_long["looking_at_already_drawn"] = dfthis_long["rankfixed_min_idxgap"]<=0
    dfthis_long["looking_at_already_drawn_earlier"] = dfthis_long["rankfixed_min_idxgap"]<=1


def syntaxconcrete_extract_wrapper_chunks_future_errors_info(D):
    """
    For each beh stroke, get useful info regarding its chunk status, as well as the status of upcoming strokes,
    including both what subject did (including failures) as well as what he should have done (correct). The main
    difference from other code (above) is the distinction between whta did and what shoudl have done.

    NOTE: focuses more on the future strokes rather than ucrrect stroke, which is already gotten elsewhere.

    RETURNS:
    - df, where each row is a beh stroke
    """

    res = []
    for ind, row in D.Dat.iterrows():
        TkBeh = D.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data", return_as_tokensclass=True) # what did
        TkCorrect = D.grammarparses_task_tokens_correct_order_sequence(ind, return_as_tokensclass=True) # what should have done
        nstrokes_beh = len(TkBeh.Tokens)
        nstrokes_correct = len(TkCorrect.Tokens)

        # Trial-level info
        syntax_concrete = row["syntax_concrete"]
        trialcode = row["trialcode"]
        # Was this trial a success?
        
        # For each beh stroke, get whether it was success.
        corrects = D.grammarparses_syntax_each_stroke_error_failure(ind)
        # corrects = []
        # for i, tokbeh in enumerate(TkBeh.Tokens):
        #     tokcorr = TkCorrect.Tokens[i]
        #     if tokbeh["ind_taskstroke_orig"] == tokcorr["ind_taskstroke_orig"]:
        #         corrects.append(1)
        #     else:
        #         corrects.append(0)
        # corrects = np.array(corrects)

        # What is the failure mode
        stroke_quality_abort = row["exclude_because_online_abort"]
        if False:
            # Previous method, which works fine, except for when he makes an 
            # extra stroke at end. This is scored as success==True, but 
            # not all corrects.
            success = row["success_binary_quick"]
            if success:
                assert np.all(corrects)
            elif stroke_quality_abort:
                # Then correct squence, just bad storke
                assert np.all(corrects)
            else:
                # Then is bad sequence
                assert not np.all(corrects)          
        else:
            # Redefine success based on each beh stroke being correct.
            # This is better, since it solves the above problem by redefining success
            # based on corrects. 
            success = [tok["ind_taskstroke_orig"] for tok in TkBeh.Tokens] == [tok["ind_taskstroke_orig"] for tok in TkCorrect.Tokens]
            # success = np.all(corrects) &

        # What stroke was the first to fail?
        if success:
            first_indstroke_that_fails_grammar = "none"
        elif stroke_quality_abort:
            # Then no stroke failed sequence
            first_indstroke_that_fails_grammar = "none"
        else:
            # Find the first instance of 0.
            first_indstroke_that_fails_grammar = int(np.argwhere(corrects==0)[0][0])


        # Classify trial 
        if True:
            if success==True and stroke_quality_abort==False:
                sequence_error_string = "allgreat"
            elif success==False and stroke_quality_abort==True:
                sequence_error_string = "goodseq_failstroke"
            elif success==False and stroke_quality_abort==False:
                sequence_error_string = "badseq"
            else:
                print(success, stroke_quality_abort)
                print("This trial: ", ind)
                assert False, "Not possible, unless I misunderstand these variables."
        else:
            sequence_error_string = D.grammarparses_classify_sequence_error(ind)

        ### For each beh stroke, get inforamtion.
        for indstroke in range(nstrokes_beh):

            # Did this stroke fail on stroke quality abort?
            if stroke_quality_abort and indstroke == nstrokes_beh-1:
                AbortStrokeQuality_ThisStrk_Draw = True
            else:
                AbortStrokeQuality_ThisStrk_Draw = False

            # Then this is before the last stroke
            # from pythonlib.dataset.dataset_analy.grammar import syntaxconcrete_extract_more_info
            info = syntaxconcrete_extract_more_info(syntax_concrete, indstroke, append_99_to_mean_done_button=True)

            if False: # Ignore, since this is allowed to happen if this is a failed storke
                print(info)
                print(indstroke)
                print(TkBeh.Tokens[indstroke]["chunk_rank"])
                assert info["chunk_ranks_remain_not_yet_started"][0] > TkBeh.Tokens[indstroke]["chunk_rank"]

            # Get shape and location, as sanity check for linking to other data
            shape = TkBeh.Tokens[indstroke]["shape"]
            gridloc = TkBeh.Tokens[indstroke]["gridloc"]

            if indstroke < nstrokes_beh-1:
                # Then there is at least one more beh stroke
                    
                # Next stroke that will be drawn
                Cr_NextStrk_Draw = TkBeh.Tokens[indstroke+1]["chunk_rank"]
                RnkWthn_NextStrk_Draw = TkBeh.Tokens[indstroke+1]["chunk_within_rank"]
                RnkWthnLast_NextStrk_Draw = TkBeh.Tokens[indstroke+1]["chunk_within_rank_fromlast"]
                
                # is next stroke correct equence?
                Success_NextStrk_Draw = corrects[indstroke+1]
            else:
                # This is the last beh stroke
                Cr_NextStrk_Draw = "none"
                RnkWthn_NextStrk_Draw = "none"
                RnkWthnLast_NextStrk_Draw = "none"

                Success_NextStrk_Draw = 1 # just give it a dummy
            
            # What bout the next correct stroke?
            if indstroke < nstrokes_correct-1:
                
                # Next stroke that will be drawn
                Cr_NextStrk_Corr = TkCorrect.Tokens[indstroke+1]["chunk_rank"]
                RnkWthn_NextStrk_Corr = TkCorrect.Tokens[indstroke+1]["chunk_within_rank"]
                RnkWthnLast_NextStrk_Corr = TkCorrect.Tokens[indstroke+1]["chunk_within_rank_fromlast"]

            else:
                Cr_NextStrk_Corr = "none"
                RnkWthn_NextStrk_Corr = "none"
                RnkWthnLast_NextStrk_Corr = "none"

            dat = {
                # Info
                "ind_dat":ind,
                "ind_stroke":indstroke,
                "trialcode":trialcode,
                "shape":shape,
                "gridloc":gridloc,

                # Success-related info about this trial
                "trial_stroke_quality_abort":stroke_quality_abort,
                "trial_first_indstroke_that_fails":first_indstroke_that_fails_grammar,
                "trial_success":success,
                "trial_syntax_concrete":syntax_concrete,
                "trial_corrects":corrects,
                "trial_sequence_error_string":sequence_error_string,

                # This stroke chunk info
                "chunk_rank":TkBeh.Tokens[indstroke]["chunk_rank"],
                "chunk_within_rank":TkBeh.Tokens[indstroke]["chunk_within_rank"],
                "chunk_within_rank_fromlast":TkBeh.Tokens[indstroke]["chunk_within_rank_fromlast"],
                "chunk_rank_correct":TkCorrect.Tokens[indstroke]["chunk_rank"],
                "chunk_within_rank_correct":TkCorrect.Tokens[indstroke]["chunk_within_rank"],

                # Next stroke that will be drawn
                "Cr_NextStrk_Draw": Cr_NextStrk_Draw,
                "RnkWthn_NextStrk_Draw": RnkWthn_NextStrk_Draw,
                "RnkWthnLast_NextStrk_Draw": RnkWthnLast_NextStrk_Draw,

                # Next stroke that is correct
                "Cr_NextStrk_Corr": Cr_NextStrk_Corr,
                "RnkWthn_NextStrk_Corr": RnkWthn_NextStrk_Corr,
                "RnkWthnLast_NextStrk_Corr": RnkWthnLast_NextStrk_Corr,

                # Next chunk
                "Cr_NextChunk_Corr": info["chunk_ranks_remain_not_yet_started"][0],

                # Was this stroke a failure?
                "Success_ThisStrk_Draw": corrects[indstroke],
                "AbortStrokeQuality_ThisStrk_Draw": AbortStrokeQuality_ThisStrk_Draw,
                # Was this the last stroke bfore a failure?
                "Success_NextStrk_Draw": Success_NextStrk_Draw,
            }

            res.append(dat)

    return pd.DataFrame(res)
