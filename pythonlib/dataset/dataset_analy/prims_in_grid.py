""" Analysis of stroke quality for prims in grid.
Useful for daily plots
2/22/23 - derived from 220710_analy_spatial_timecourse_exploration(prims in grid)
e.g., for ecah shape and location, plot example strokes ...
Effect of location? shape?
rank?

Also, which shapes he avoids? etc

NOTE: currently uses beh variation of DS.

"""
import matplotlib.pyplot as plt
import numpy as np
import os
from pythonlib.tools.plottools import savefig
import seaborn as sns
from pythonlib.tools.snstools import rotateLabel

def preprocess_dataset(D, doplots=False):

    from pythonlib.dataset.dataset_strokes import DatStrokes
    SAVEDIR = D.make_savedir_for_analysis_figures("prims_in_grid")
    # USE tHIS!!!

    D = D.copy()
    D.Dat = D.Dat[D.Dat["task_kind"] == "prims_on_grid"].reset_index(drop=True)

    D.preprocessGood(params=["beh_strokes_at_least_one"])

    # Determine if aborts were befaucs sequence error...
    D.grammarmatlab_successbinary_score() # Quickly score using matlab sequence...
    D.grammarparsesmatlab_score_wrapper_append()
    assert len(D.Dat)>0

    # get DatStrokes
    DS = DatStrokes(D)

    D.seqcontext_preprocess()
    D.taskclass_shapes_loc_configuration_assign_column()
 
    # Some params and metadat to save

    #############################
    if doplots:


        # Sequential context of strokes
        savedir = f"{SAVEDIR}/stroke_sequential_contexts"
        os.makedirs(savedir, exist_ok=True)
        plot_sequential_context_strokes(DS, savedir)


        # SAVE the conjunctions of shape and loc that were gotten
        path = f"{SAVEDIR}/shape_loc_grouping-by_epoch_block.txt"
        D.grouping_print_n_samples(["aborted", "epoch", "block", "seqc_0_loc", "seqc_0_shape"], savepath=path, save_as="txt")    

        path = f"{SAVEDIR}/shape_loc_grouping-by_epoch.txt"
        D.grouping_print_n_samples(["aborted", "epoch", "seqc_0_loc", "seqc_0_shape"], savepath=path, save_as="txt")    

        path = f"{SAVEDIR}/shape_loc_grouping-by_character.txt"
        D.grouping_print_n_samples(["aborted", "character", "seqc_0_loc", "seqc_0_shape"], savepath=path, save_as="txt")    

        path = f"{SAVEDIR}/shape_loc_grouping-by_epoch_character.txt"
        D.grouping_print_n_samples(["aborted", "epoch", "character", "seqc_0_loc", "seqc_0_shape"], savepath=path, save_as="txt")    

        path = f"{SAVEDIR}/shape_loc_grouping-by_shape_loc.txt"
        D.grouping_print_n_samples(["aborted", "seqc_0_loc", "seqc_0_shape", "epoch", "block"], savepath=path, save_as="txt")    

        ####### rew as function of n strokes in task
        savedir = f"{SAVEDIR}/rew_vs_nstrokestask"
        os.makedirs(savedir, exist_ok=True)
        list_blocks = D.Dat["block"].unique().tolist()
        D.Dat["aborted_int"] = np.array(D.Dat["aborted"], dtype=int) # or else plot will error.
        for block in list_blocks:
            dfthis = D.Dat[D.Dat["block"]==block]
            if len(dfthis)>20:
                fig = sns.pairplot(data=dfthis, vars=["beh_multiplier", "rew_total", "aborted_int"], 
                             hue="seqc_nstrokes_task", plot_kws={"alpha":0.4}, height=3, aspect=1.5)
                savefig(fig, f"{savedir}/rew_vs_nstrokestask-bk_{block}.pdf")        

        #### Plot score/rew vs. location config.
        savedir = f"{SAVEDIR}/locationconfig"
        # D.taskclass_shapes_loc_configuration_assign_column()
        os.makedirs(savedir, exist_ok=True)
        for y in ["beh_multiplier", "rew_total"]:
            fig = sns.catplot(data=D.Dat, x="taskconfig_loc", y=y, jitter=True, alpha=0.4, aspect=1.5)
            rotateLabel(fig)
            savefig(fig, f"{savedir}/{y}-vs-taskconfig_loc-1.pdf")        

            fig = sns.catplot(data=D.Dat, x="taskconfig_loc", y=y, kind="point", aspect=1.5)
            rotateLabel(fig)
            savefig(fig, f"{savedir}/{y}-vs-taskconfig_loc-2.pdf")        

        ######## LOOK FOR CONJUCNTIONS
        if False:
            # obsolete...
            D.taskclass_shapes_loc_configuration_assign_column()
            from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
            LIST_VAR = [
                "seqc_3_loc_shape", # same n strokes, just diff sequence
                "seqc_3_loc_shape", # same stim entirely
                "seqc_3_loc_shape", # same loc config
                "seqc_3_loc_shape", # same shape config

                "seqc_2_loc_shape",
                "seqc_2_loc_shape",
                "seqc_2_loc_shape",
                "seqc_2_loc_shape",

                "seqc_1_loc_shape",
                "seqc_1_loc_shape",
                "seqc_1_loc_shape",
                "seqc_1_loc_shape",

                "seqc_nstrokes_beh", # diff  n strokes
                "seqc_nstrokes_beh",
                "seqc_nstrokes_beh",
                ]
            LIST_VARS_CONJUNCTION = [
                ["seqc_nstrokes_beh", "seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc_shape"], 
                ["taskconfig_shploc", "seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc_shape"],
                ["taskconfig_loc", "seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc_shape"],
                ["taskconfig_shp", "seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc_shape"],

                ["seqc_nstrokes_beh", "seqc_0_loc_shape", "seqc_1_loc_shape"], 
                ["taskconfig_shploc", "seqc_0_loc_shape", "seqc_1_loc_shape"], 
                ["taskconfig_loc", "seqc_0_loc_shape", "seqc_1_loc_shape"], 
                ["taskconfig_shp", "seqc_0_loc_shape", "seqc_1_loc_shape"], 

                ["seqc_nstrokes_beh", "seqc_0_loc_shape"],
                ["taskconfig_shploc", "seqc_0_loc_shape"],
                ["taskconfig_loc", "seqc_0_loc_shape"],
                ["taskconfig_shp", "seqc_0_loc_shape"],

                ["seqc_0_loc_shape"], # diff n strokes.
                ["seqc_0_loc_shape", "seqc_1_loc_shape"],
                ["seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc_shape"],
            ]           
            for var, vars_others in zip(LIST_VAR, LIST_VARS_CONJUNCTION):
                sdir = f"{SAVEDIR}/list_seqc_conjunctions"
                os.makedirs(sdir, exist_ok=True)
                path = f"{sdir}/{var}|vs|{'-'.join(vars_others)}.txt"    
                vars_others = ["aborted"] + vars_others
                D.grouping_conjunctions_print_variables_save(var, vars_others, path)
        else:
            from neuralmonkey.metadat.analy.anova_params import conjunctions_print_plot_all
            # which_level="trial"
            # ANALY_VER = "seqcontext"
            # animal = D.animals()[0]
            conjunctions_print_plot_all([D], SAVEDIR, ANALY_VER="seqcontext")
            plt.close("all")

        # Pltoi cause of abort
        LIST_OUTCOMES = [
            ["online_abort_but_sequence_correct_so_far", "online_abort_but_sequence_correct_complete"],
            ["sequence_incorrect_online_abort"]
        ]
        LIST_OUTCOMES_CODE = [
            "grammarcorrect",
            "grammarwrong"
        ]

        # sUCCESSES shoudl always use all trials
        Dthis = D.copy()
        Dthis.Dat = Dthis.Dat[Dthis.Dat["task_kind"] == "prims_on_grid"].reset_index(drop=True)
        if len(Dthis.Dat)>0:
            DSthis = DatStrokes(Dthis)
            savedir = f"{SAVEDIR}/ABORTS-ALLDATA"
            os.makedirs(savedir, exist_ok=True)
            dfabort, dfheat_abort = plot_abort_cause(Dthis, DSthis, savedir, "abort") 
            dfsucc, dfheat_succ = plot_abort_cause(Dthis, DSthis, savedir, "success")
            
            # Plto fraction of cases aborted
            sdir = f"{savedir}/cause_of_abort_frac_of_success"
            os.makedirs(sdir, exist_ok=True)
            
            from pythonlib.tools.snstools import heatmap
            from pythonlib.tools.pandastools import convert_to_2d_dataframe

            assert dfheat_abort.columns.tolist() == dfheat_succ.columns.tolist()
            assert dfheat_abort.index.tolist() == dfheat_succ.index.tolist()

            dfheat_abort_frac = dfheat_abort / (dfheat_succ + dfheat_abort)
            dfheat_ntrials = dfheat_abort + dfheat_succ

            fig = heatmap(dfheat_abort_frac)[0]
            savefig(fig, f"{sdir}/heatmap-frac_abort.pdf")

            fig = heatmap(dfheat_ntrials)[0]
            savefig(fig, f"{sdir}/heatmap-ntrials_total.pdf")

            for OUTCOMES, OUTCOMES_CODE in zip(LIST_OUTCOMES, LIST_OUTCOMES_CODE):
                Dthis = D.copy()
                Dthis.Dat = Dthis.Dat[Dthis.Dat["task_kind"] == "prims_on_grid"].reset_index(drop=True)
                Dthis.Dat = Dthis.Dat[Dthis.Dat["grammar_score_string"].isin(OUTCOMES)].reset_index(drop=True) 
                DSthis = DatStrokes(Dthis)

                savedir = f"{SAVEDIR}/ABORTS-{OUTCOMES_CODE}"
                os.makedirs(savedir, exist_ok=True)

                dfabort, dfheat_abort = plot_abort_cause(Dthis, DSthis, savedir, "abort") 
                if dfabort is None:
                    # no data
                    continue
                # dfsucc, dfheat_succ = plot_abort_cause(Dthis, DSthis, savedir, "success")
         
                # Plto fraction of cases aborted
                sdir = f"{savedir}/cause_of_abort_frac_of_success"
                os.makedirs(sdir, exist_ok=True)
                
                from pythonlib.tools.snstools import heatmap
                from pythonlib.tools.pandastools import convert_to_2d_dataframe

                fig = heatmap(dfheat_abort)[0]
                savefig(fig, f"{sdir}/heatmap-dfheat_abort.pdf")

                if not dfheat_abort.index.tolist() == dfheat_succ.index.tolist():
                    # Is probably becusae lack any trials of some shape for one of them, liek this:
                    # ['line-8-3-0', 'line-8-4-0']
                    # ['V-2-4-0', 'line-8-3-0', 'line-8-4-0']
                    # - just skip for now.
                    if False:
                        print(dfheat_abort.index.tolist())
                        print(dfheat_succ.index.tolist())
                        assert False
                    else:
                        continue

                dfheat_abort_frac = dfheat_abort / (dfheat_succ + dfheat_abort)
                dfheat_ntrials = dfheat_abort + dfheat_succ

                fig = heatmap(dfheat_abort_frac)[0]
                savefig(fig, f"{sdir}/heatmap-frac_abort.pdf")

                fig = heatmap(dfheat_ntrials)[0]
                savefig(fig, f"{sdir}/heatmap-ntrials_total.pdf")

        plotscore_all(DS, SAVEDIR)
        plotdrawings_all(DS, SAVEDIR)


    return DS, SAVEDIR

def plot_abort_cause(D, DS, SAVEDIR, abort_or_success="abort"):
    """ Find cases of online abort, and plot reason for that abort, in terms of
    the stroke shape, location,m and eindex.
    """ 
    print("TODO: Rewrite This! it throws out the good inds on abort trials (for success)")
    import pandas as pd
    import seaborn as sns
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import convert_to_2d_dataframe

    sdir = f"{SAVEDIR}/cause_of_{abort_or_success}"
    os.makedirs(sdir, exist_ok=True)

    ### Collect data across all abort trials
    if abort_or_success=="abort":
        inds_abort = D.Dat[D.Dat["aborted"]==True].index.tolist()
    elif abort_or_success=="success":
        inds_abort = D.Dat[D.Dat["aborted"]==False].index.tolist()
    else:
        assert False

    res = []
    for ind in inds_abort:
        tokens = D.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data")
        if len(tokens)>0:
            if abort_or_success=="abort":
                # Only take the last index
                tok_last = tokens[-1]
                res.append({
                    "inddat":ind,
                    "trialcode":D.Dat.iloc[ind]["trialcode"],
                    "tok_last":tok_last,
                    "shape_last":tok_last["shape"],
                    "loc_last":tok_last["gridloc"],
                    "strokind_last":int(len(tokens))
                })
            elif abort_or_success=="success":
                for j, tok in enumerate(tokens):
                    res.append({
                        "inddat":ind,
                        "trialcode":D.Dat.iloc[ind]["trialcode"],
                        "tok_last":tok,
                        "shape_last":tok["shape"],
                        "loc_last":tok["gridloc"],
                        "strokind_last":j
                    })
            else:
                assert False

    dfres = pd.DataFrame(res)

    if len(dfres)>0:
        ### MAKE PLOTS
        # # sns.catplot(data=dfres, x="loc_last", y="strokind_last", hue="shape_last", jitter=True, alpha=0.2)
        # # sns.catplot(data=dfres, x="loc_last", y="strokind_last", hue="shape_last", kind="swarm", alpha=0.2)
        # sns.pairplot(data=dfres, vars=["loc_last", "strokind_last","shape_last"])
        fig = sns.displot(data=dfres, x="shape_last", y="strokind_last", col="loc_last")
        rotateLabel(fig)
        savefig(fig, f"{sdir}/displot-aborted_on_this_stroke.pdf")

        ### HEATMAPS of counts
        list_shape = sorted(DS.Dat["shape"].unique().tolist())
        list_loc = sorted(DS.Dat["gridloc"].unique().tolist())

        # def convert_to_2d_dataframe(df, col1, col2, plot_heatmap=False, 
        #     agg_method = "counts", val_name = "val", ax=None, 
        #     norm_method=None,
        #     annotate_heatmap=True, zlims=(None, None),
        #     diverge=False, dosort_colnames=True,
        #     list_cat_1 = None, list_cat_2 = None):

        # Heatmap
        dfheat, fig, _, _ = convert_to_2d_dataframe(dfres, "shape_last", "loc_last", plot_heatmap=True, list_cat_1 = list_shape, list_cat_2 = list_loc);
        savefig(fig, f"{sdir}/heatmap-aborted_on_this_stroke.pdf")

        # Heatmap, separating by stroke index
        list_strokind_last = dfres["strokind_last"].unique().tolist()
        for strokind_last in list_strokind_last:
            dfthis = dfres[dfres["strokind_last"] == strokind_last]
            
            _, fig, _, _ = convert_to_2d_dataframe(dfthis, "shape_last", "loc_last", plot_heatmap=True, list_cat_1 = list_shape, list_cat_2 = list_loc);
            savefig(fig, f"{sdir}/heatmap-aborted_on_this_stroke-strokeind_{strokind_last}.pdf")

        plt.close("all")
    else:
        return None, None

    return dfres, dfheat


def plotscore_all(DS, SAVEDIR):
    """
    Plots of beh-task similarity scores, broken into location, shapes, etc.
    """
    from pythonlib.tools.snstools import rotateLabel
    import os 
    import seaborn as sns
    from pythonlib.tools.pandastools import convert_to_2d_dataframe


    # Compute similarity for all data
    ##### Extract scores of stroke quality (how well beh stroke matches task stroke)
    DS.distgood_compute_beh_task_strok_distances()

    if False:
        print("Starting length, before remove nan rows:", len(DS.Dat))
        DS.Dat = DS.Dat.dropna(axis=0)
        print("After removing:", len(DS.Dat))

    list_taskkind = DS.Dat["task_kind"].unique().tolist()
    for tk in list_taskkind:
        
        # Prep, for this taskkind
        dfthis = DS.Dat[DS.Dat["task_kind"]==tk]
        savedir = f"{SAVEDIR}/beh_task_dist_scores/taskkind_{tk}"
        os.makedirs(savedir, exist_ok=True)

        # Make plots
        fig = sns.catplot(data=dfthis, x="shape_oriented", y="dist_beh_task_strok", aspect=2.5, row="gridloc")
        rotateLabel(fig)
        fig.savefig(f"{savedir}/scatter_shape_loc_1.pdf")

        try:
            fig = sns.catplot(data=dfthis, x="shape_oriented", hue="gridloc", y="dist_beh_task_strok", aspect=2.5, kind="bar")
            rotateLabel(fig)
            fig.savefig(f"{savedir}/bars_shape_loc_1.pdf")
        except Exception as err:
            pass

        # Does score depend onlocation?
        # Plot distributions of scores for each prim
        fig = sns.catplot(data=dfthis, x="gridloc", y="dist_beh_task_strok", aspect=2.5)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/scatter_location_1.pdf")

        fig = sns.catplot(data=dfthis, x="gridloc", y="dist_beh_task_strok", kind="bar", aspect=2.5)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/bars_location_1.pdf")

        fig = sns.catplot(data=dfthis, x="gridloc", hue="shape_oriented", y="dist_beh_task_strok", aspect=2.5, kind="bar")
        rotateLabel(fig)
        fig.savefig(f"{savedir}/bars_location_shape_1.pdf")

        plt.close("all")

        ##### Score, as function of shape and location
        # Heatmap of score
        _, fig, _, _ = convert_to_2d_dataframe(dfthis, "shape", "gridloc", True, agg_method="mean", val_name="dist_beh_task_strok", annotate_heatmap=False);
        fig.savefig(f"{savedir}/heat2d_shape_location_meanscore.pdf")

        # Heatmap of n trials
        _, fig, _, _ = convert_to_2d_dataframe(dfthis, "shape", "gridloc", True, annotate_heatmap=False);
        fig.savefig(f"{savedir}/heat2d_shape_location_ncounts.pdf")

        # Plot relating score to stroke index/rankd
        _, fig, _, _ = convert_to_2d_dataframe(dfthis, col1="shape_oriented", col2="stroke_index", plot_heatmap=True);
        fig.savefig(f"{savedir}/heat2d_shape_rank_counts.pdf")

        _, fig, _, _ = convert_to_2d_dataframe(dfthis, "shape", "stroke_index", True, agg_method="mean", val_name="dist_beh_task_strok", annotate_heatmap=False);
        fig.savefig(f"{savedir}/heat2d_shape_rank_meanscore.pdf")

        _, fig, _, _ = convert_to_2d_dataframe(dfthis, "gridloc", "stroke_index", True, agg_method="mean", val_name="dist_beh_task_strok", annotate_heatmap=False);
        fig.savefig(f"{savedir}/heat2d_location_rank_meanscore.pdf")

        plt.close("all")

        fig = sns.catplot(data=dfthis, x="stroke_index", y = "dist_beh_task_strok", 
            hue="shape_oriented", kind="point", aspect = 1.5)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/lines_rank_score_1.pdf")

        fig = sns.catplot(data=dfthis, x="stroke_index", y = "dist_beh_task_strok", 
            hue="gridloc", kind="point", aspect = 1.5)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/lines_rank_score_2.pdf")

        fig = sns.catplot(data=dfthis, x="gridloc", y="stroke_index", hue="shape_oriented", kind="point", aspect = 1.5)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/lines_shape_rank_1.pdf")

        sns.catplot(data=dfthis, x="gridloc", y="stroke_index", kind="violin", aspect = 1.5)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/violin_location_rank_1.pdf")
        
        sns.catplot(data=dfthis, x="shape", y="stroke_index", kind="boxen", aspect = 1.5)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/boxen_shape_rank_1.pdf")

        plt.close("all")

def plotdrawings_all(DS, SAVEDIR, n_examples = 3):
    """ Summary plots that are drawings, eg., xample 
    strokes for each location 
    """
    

    savedir = f"{SAVEDIR}/drawings"
    import os 
    os.makedirs(savedir, exist_ok=True)

    if False:
        DS.plotshape_multshapes_trials_grid(nrows=2);
    
    # Plot n exmaples for each shape/location combo
    list_taskkind = DS.Dat["task_kind"].unique().tolist()

    key_to_extract_stroke_variations_in_single_subplot = "gridloc"
    n_iter = 3
    for tk in list_taskkind:
        for i in range(n_iter):
            figholder = DS.plotshape_multshapes_egstrokes_grouped_in_subplots(task_kind = tk, n_examples=n_examples,
                                                                 key_to_extract_stroke_variations_in_single_subplot=key_to_extract_stroke_variations_in_single_subplot);
            for j, this in enumerate(figholder):
                fig = this[0]
                fig.savefig(f"{savedir}/egstrokes_shape_location--taskkind_{tk}-iter_{i}_{j}.pdf")

            plt.close("all")

    # Plot velocities
    savedir = f"{SAVEDIR}/velocities"
    os.makedirs(savedir, exist_ok=True)
    DS.plotwrap_timecourse_vels_grouped_by_shape(5, savedir=savedir, also_plot_example_strokes=True)

    if False:
        # Condition on a given shape
        nplot = 5
        import random
        DS.Dat["shape_oriented"].value_counts()
        shape = "circle-6-1-0"
        inds = DS.Dat[DS.Dat["shape_oriented"]==shape].index.tolist()
        inds = sorted(random.sample(inds, nplot))
        DS.plot_beh_and_aligned_task_strokes(inds, True)

def plot_sequential_context_strokes(DS, savedir):
    """ plot conjucjitions that exist for conv and divergents seuqneces
    """
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars

    LIST_N_MIN = [0, 4, 7]

    def plot_context(df, VER, savedir, suffix):
        
        if VER=="divergent":
            var = "CTXT_locshape_next"
            vars_others = ["CTXT_loc_prev", "CTXT_shape_prev", "gridloc", "shape"]
        elif VER=="convergent":
            var = "CTXT_locshape_prev"
            vars_others = ["CTXT_loc_next", "CTXT_shape_next", "gridloc", "shape"]
        else:
            assert False

        for n_min in LIST_N_MIN:

            path_text = f"{savedir}/divergent_locshape-n_min_{n_min}-{suffix}.txt"
            path_fig = f"{savedir}/divergent_locshape-n_min_{n_min}-{suffix}.pdf"

            extract_with_levels_of_conjunction_vars(df, var, vars_others, n_min_across_all_levs_var=n_min,
                                                    lenient_allow_data_if_has_n_levels=2, PRINT_AND_SAVE_TO=path_text,
                                                    plot_counts_heatmap_savepath=path_fig)

    for VER in ["divergent", "convergent"]:

        # 1) All data
        df = DS.Dat
        suffix = "ALLDATA"
        plot_context(df, VER, savedir, suffix)

        # 2) Each epoch
        list_epoch = DS.Dat["epoch"].unique().tolist()
        if len(list_epoch)>0:
            for epoch in list_epoch:
                df = DS.Dat[DS.Dat["epoch"]==epoch]
                suffix = f"epoch_{epoch}"
                plot_context(df, VER, savedir, suffix)
