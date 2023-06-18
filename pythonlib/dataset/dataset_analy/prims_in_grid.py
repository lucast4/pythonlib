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
        D.grouping_conjunctions_print_variables_save(var, vars_others, path)

    #############################
    if doplots:
        plotscore_all(DS, SAVEDIR)
        plotdrawings_all(DS, SAVEDIR)

    return DS, SAVEDIR

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