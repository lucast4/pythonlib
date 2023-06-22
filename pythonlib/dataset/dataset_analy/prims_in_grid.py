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

def preprocess_dataset(D, doplots=False):

    from pythonlib.dataset.dataset_strokes import DatStrokes
    SAVEDIR = D.make_savedir_for_analysis_figures("prims_in_grid")
    # USE tHIS!!!

    DS = DatStrokes(D)

    # Some params and metadat to save

    # SAVE the conjunctions of shape and loc that were gotten
    D.seqcontext_preprocess()
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
    dfabort, dfheat_abort = plot_abort_cause(D, DS, SAVEDIR, "abort")
    dfsucc, dfheat_succ = plot_abort_cause(D, DS, SAVEDIR, "success")

    # Plto fraction of cases aborted
    sdir = f"{SAVEDIR}/cause_of_abort_frac_of_success"
    os.makedirs(sdir, exist_ok=True)
    
    from pythonlib.tools.snstools import heatmap
    from pythonlib.tools.pandastools import convert_to_2d_dataframe

    # list_shape = sorted(DS.Dat["shape"].unique().tolist())
    # list_loc = sorted(DS.Dat["gridloc"].unique().tolist())

    # dfheat_succ, _, _, _ = convert_to_2d_dataframe(dfsucc, "shape_last", "loc_last", plot_heatmap=True, list_cat_1 = list_shape, list_cat_2 = list_loc);
    # dfheat_abort, _, _, _ = convert_to_2d_dataframe(dfabort, "shape_last", "loc_last", plot_heatmap=True, list_cat_1 = list_shape, list_cat_2 = list_loc);

    assert dfheat_abort.columns.tolist() == dfheat_succ.columns.tolist()
    assert dfheat_abort.index.tolist() == dfheat_succ.index.tolist()

    dfheat_abort_frac = dfheat_abort / (dfheat_succ + dfheat_abort)
    dfheat_ntrials = dfheat_abort + dfheat_succ

    fig = heatmap(dfheat_abort_frac)[0]
    savefig(fig, f"{sdir}/heatmap-frac_abort.pdf")

    fig = heatmap(dfheat_ntrials)[0]
    savefig(fig, f"{sdir}/heatmap-ntrials_total.pdf")

    #############################
    if doplots:
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
        tokens = D.taskclass_tokens_extract_wrapper(ind, "beh")
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


# def conjunctions_print_plot_all(D, SAVEDIR):
#     """
#     Wrapper for all printing and plotting (saving) related to conjuicntions of varaibles that matter for PIG>
#     Think of these as the conjucntiosn that care about for neural analysis. Here help assess each beahvior quickly.
#     """
#     from neuralmonkey.metadat.analy.anova_params import dataset_apply_params, params_getter_plots
#     from pythonlib.tools.expttools import writeStringsToFile
#     from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap

#     sdir = f"{SAVEDIR}/conjunctions"
#     os.makedirs(sdir, exist_ok=True)


#     ListD = [D]
#     which_level="trial"
#     ANALY_VER = "seqcontext"
#     animal = D.animals()[0]
#     _dates = D.Dat["date"].unique()
#     assert len(_dates)==1
#     DATE = int(_dates[0])

#     ### Prep dataset, and extract params
#     Dall, Dpruned, TRIALCODES_KEEP, params, params_extraction = dataset_apply_params(ListD, 
#         animal, DATE, which_level, ANALY_VER)

#     ### Print and plot all conjucntions
#     LIST_VAR = params["LIST_VAR"]
#     LIST_VARS_CONJUNCTION = params["LIST_VARS_CONJUNCTION"]           
#     list_n = []
#     for var, vars_others in zip(LIST_VAR, LIST_VARS_CONJUNCTION):
        
#         print(var, "vs", vars_others)
        
#         # All data
#         path = f"{sdir}/{var}|vs|{'-'.join(vars_others)}.txt"
#         plot_counts_heatmap_savedir = f"{sdir}/heatmap-{var}|vs|{'-'.join(vars_others)}.pdf"
#         Dpruned.grouping_conjunctions_print_variables_save(var, vars_others, path, n_min=0, 
#                                                           plot_counts_heatmap_savedir=plot_counts_heatmap_savedir)
#         # Passing nmin
#         path = f"{sdir}/goodPassNmin-{var}|vs|{'-'.join(vars_others)}.txt"
#         plot_counts_heatmap_savedir = f"{sdir}/goodPassNmin-heatmap-{var}|vs|{'-'.join(vars_others)}.pdf"
#         dfout, dict_dfs = Dpruned.grouping_conjunctions_print_variables_save(var, vars_others, path, n_min=params["globals_nmin"], 
#                                                           plot_counts_heatmap_savedir=plot_counts_heatmap_savedir)
#         plt.close("all")
        
#         # Count
#         list_n.append(len(dict_dfs))
        
#     ### Print summary across conjucntions
#     strings = []
#     strings.append("n good levels of othervar | var |vs| othervars")
#     for var, vars_others, n in zip(LIST_VAR, LIST_VARS_CONJUNCTION, list_n):
#         s = f"{n} -- {var}|vs|{'-'.join(vars_others)}"
#         strings.append(s)
#     path = f"{sdir}/summary_n_levels_of_othervar_with_min_data.txt"
#     writeStringsToFile(path, strings)  

#     ### STROKE LEVEL - heatmaps of (shape, location) vs. index
#     from pythonlib.dataset.dataset_strokes import DatStrokes
#     DS = DatStrokes(Dpruned)
#     for task_kind in ["prims_single", "prims_on_grid"]:
#         dfthis = DS.Dat[DS.Dat["task_kind"]==task_kind]
        
#         if len(dfthis)>0:
#             fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1="shape", var2="gridloc", vars_others=["stroke_index"])
#             path = f"{sdir}/STROKELEVEL-conjunctions_shape_gridloc-task_kind_{task_kind}.pdf"
#             savefig(fig, path)

#             # Dissociate stroke index from remaining num strokes.
#             fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1="stroke_index", 
#                                                               var2="stroke_index_fromlast", vars_others=["shape", "gridloc"])
#             path = f"{sdir}/STROKELEVEL-conjunctions_stroke_index-task_kind_{task_kind}.pdf"
#             savefig(fig, path)

#             plt.close("all")
