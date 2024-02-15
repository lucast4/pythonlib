"""
Good, to ask how well prims are draw, and prim statsitics, etc
useful for things like
- categorization expts,.
- microstim.

Works with obth single primsa nd pig

10/15/23 - Written for microstim.
Subsumes code in notebook for Luca novel vs. old prims:
230508_analy_primitiveness_primsingrid
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.tools.plottools import savefig
import seaborn as sns
import os

def preprocess_plot_pipeline(D, PLOT=True, microstim_version=False):
    """
    Wrapper to Preprocess and plot
    """
    # from pythonlib.dataset.dataset_analy.primitivenessv2 import plot_timecourse_results, plot_drawings_results, preprocess, extract_grouplevel_motor_stats, plot_grouplevel_results, plot_triallevel_results

    # grouping = ["shape", "gridloc", "epoch"]
    grouping = ["locshape_pre_this", "epoch", "block"] 

    ############### Extract data
    DS, SAVEDIR = preprocess(D, True, microstim_version=microstim_version)
    dfres, grouping = extract_grouplevel_motor_stats(DS, grouping, 
        microstim_version=microstim_version)

    if PLOT:
        ############### PLOTS

        # Plot, comparing mean across levels of contrast variable.
        # Each datapt is a single level of grouping.
        savedir = f"{SAVEDIR}/grouplevel"
        os.makedirs(savedir, exist_ok=True)
        contrast = "epoch"
        plot_grouplevel_results(dfres, DS, D, grouping, contrast, savedir)

        # Plot, each datapt a single trial.
        savedir = f"{SAVEDIR}/triallevel"
        os.makedirs(savedir, exist_ok=True)
        contrast = "epoch"
        plot_triallevel_results(DS, contrast, savedir)

        # Plot drawings
        savedir = f"{SAVEDIR}/drawings"
        os.makedirs(savedir, exist_ok=True)
        plot_drawings_results(DS, savedir)

        # Plot timecourses
        savedir = f"{SAVEDIR}/timecourse"
        os.makedirs(savedir, exist_ok=True)
        plot_timecourse_results(DS, savedir, "epoch")

    return DS, SAVEDIR, dfres, grouping

def preprocess(D, prune_strokes=True, microstim_version=False):
    """ Entire pipeline to extract motor, image, and other stats
    """
        
    params_preprocess = ["remove_baseline", "no_supervision", "only_blocks_with_n_min_trials"]

    if False:
        from pythonlib.dataset.dataset_analy.prims_in_grid import preprocess_dataset
        DS, _ = preprocess_dataset(D)
    else:
        from pythonlib.dataset.dataset_analy.motortiming import gapstrokes_preprocess_extract_strokes_gaps
        DS, _ = gapstrokes_preprocess_extract_strokes_gaps(D,
            microstim_version=microstim_version, prune_strokes=prune_strokes,
            params_preprocess=params_preprocess)

    SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("primitivenessv2")

    return DS, SAVEDIR


def extract_triallevel_motor_stats(DS, D, map_shape_to_newold=None):
    """ 
    Just add things to DS that are that trial(strokes') motor stats
    """

    ################################################
    ## Shapes are new or old? if comparing new vs. old.
    if map_shape_to_newold is not None:
        # Info about whether is new or old shape category
        list_shape = DS.Dat["shape"].tolist()
        list_new_old = []
        for sh in list_shape:
            list_new_old.append(map_shape_to_newold[sh])
            
        DS.Dat["new_or_old"] = list_new_old    


def extract_grouplevel_motor_stats(DS, grouping=None, PLOT = False,
    map_shape_to_newold=None, microstim_version=False):
    """
    Extract stats that use a distribution of strokes/trials, such as 
    average pairwise similarity across trials of the same shape (i.e., consistency)
    Here each row is a singe sahpe (or whatever goruping variable you want) 
    with stats.
    i.e, grouplevel could be "shape", or combo of shape and loc
    PARAMS:
    - grouping, list of str, grouping varaibale. each level will be a column in output.
    - map_shape_to_newold, dict from shape(str) to str ("new" or "old") whch is 
    used for expts comparing new vs. old shapes (categorization expts).
    """
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars, grouping_append_and_return_inner_items

    # First group trials into (shape, loc) or whatever you want
    if grouping is None:
        grouping = ["shape", "gridloc"]
    grpdict = DS.grouping_append_and_return_inner_items(grouping)

    # Collect for each grp level.
    res = []
    for grpname, inds in grpdict.items():
        print("TODO: use dtw of velocities (separately for x and y) instead of euclidian of diffs")
        # 1) Consistency of motor

    #     list_shape = DS.Dat["shape"].unique().tolist()
        # list_shape = ["arcdeep-4-5-0"] # remove the outliers
        # list_shape = ["Lcentered-4-6-0"] 

    #     for shape in list_shape:
    #         # extract velocity
    #         inds = DS.Dat[DS.Dat["shape"]==shape].index.tolist()

    #     print(shape, ", n=", len(inds))
        if len(inds)<2:
            print("SKIPPING (N too low)")
            continue

        # get the abstract shape
        tmp = DS.Dat.iloc[inds]["shapeabstract"].unique().tolist()
        assert len(tmp)==1
        shapeabstract=tmp[0]

        # shape 
        tmp = DS.Dat.iloc[inds]["shape"].unique().tolist()
        assert len(tmp)==1
        shape=tmp[0]

        # shape 
        tmp = DS.Dat.iloc[inds]["epoch_orig"].unique().tolist()
        assert len(tmp)==1
        epoch_orig=tmp[0]


        ########## COLLECT PAIRWISE MOTOR DIST
    #     strokes_vel = DS.extract_strokes_as_velocity(inds)
        strokes = DS.Dat.iloc[inds]["strok"].tolist()

        # get pairwise dtw of strokes vel
        # distancever = "euclidian_diffs"
        distancever = "dtw_vels_2d"
        simmat = DS._cluster_compute_sim_matrix(strokes, strokes, distancever=distancever)
        if PLOT:
            Cl = DS._cluster_compute_sim_matrix(strokes, strokes, distancever=distancever, return_as_Clusters=True)
            Cl.plot_heatmap_data(SIZE=5);

            # plot 
            DS.plot_multiple(inds);

            assert False

        # Get all off diagonal sims (all pairwise)
        score = np.mean(simmat[np.where(~np.eye(simmat.shape[0],dtype=bool))])

        res.append({
            "shapeabstract":shapeabstract,
            "shape":shape,
            "grp":grpname,
            "mean_sim_score":score,
            "distancever":distancever,
            "inds_DS":inds,
            "epoch_orig":epoch_orig,
        })

        if "microstim_epoch_code" in DS.Dat.columns:
            tmp = DS.Dat.iloc[inds]["microstim_epoch_code"].unique().tolist()
            assert len(tmp)==1
            microstim_epoch_code=tmp[0]
            res[-1]["microstim_epoch_code"] = microstim_epoch_code

        # for any key in grouping, also get it
        for g in grouping:
            if g not in res[-1].keys():
                tmp = DS.Dat.iloc[inds][g].unique().tolist()
                assert len(tmp)==1
                res[-1][g] = tmp[0]

    dfres = pd.DataFrame(res)


    ################################################
    ### Get features at single trial level.
    list_grp = dfres["grp"].tolist()
    list_inds = dfres["inds_DS"].tolist()

    for key in ["dist_beh_task_strok", "time_duration", "velocity", "distcum"]:
    #     key = "velocity"

        list_vals = []
        for grp, inds in zip(list_grp, list_inds):

            val = np.mean(DS.Dat.iloc[inds][key])
            list_vals.append(val)

        dfres[key] = list_vals

    ################################################
    ## Shapes are new or old? if comparing new vs. old.
    if map_shape_to_newold is not None:
        # Info about whether is new or old shape category
        list_shape = dfres["shape"].tolist()
        list_new_old = []
        for sh in list_shape:
            list_new_old.append(map_shape_to_newold[sh])
            
        dfres["new_or_old"] = list_new_old    
            
    # Prune so dfres is fully balanced.
    if microstim_version:
        assert grouping == ["locshape_pre_this", "epoch", "block"], "assumign this...change otheriwse."
        print("Before and after pruning dfres to balance [microstim_version]:")
        print(len(dfres))
        dfres, _ = extract_with_levels_of_conjunction_vars(dfres, "microstim_epoch_code",
                                                           ["epoch_orig", "locshape_pre_this", "block"],
                                                           n_min_across_all_levs_var=1)
        print(len(dfres))

    return dfres, grouping


def plot_triallevel_results(DS, contrast, savedir, context=None,
                            yvars=None):
    """
    """
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.stats.lme import lme_categorical_fit_plot
    from pythonlib.tools.pandastools import datamod_normalize_row_after_grouping

    list_block = DS.Dat["block"].unique().tolist()
    list_epoch_orig = DS.Dat["epoch_orig"].unique().tolist()

    if False:
        # Was previously in microstim.plot_motortiming
        DS.Dat = append_col_with_grp_index(DS.Dat, ["stroke_index", "locshape_pre_this"], "strk_idx_ctxt", use_strings=False)
        CONTEXT_VAR = "strk_idx_ctxt"
    else:
        if context is None:
            CONTEXT_VAR = "locshape_pre_this"
        else:
            CONTEXT_VAR = context
    if yvars is None:
        yvars = ["dist_beh_task_strok", "time_duration", "velocity", "distcum", "gap_from_prev_dur",
        "gap_from_prev_dist", "gap_from_prev_vel"]

    N_MIN_PER_BLOCK = 10
    # LIST_SHAPE = ["locshape_pre_this", "shape", "shapeabstract"]
    # LIST_SHAPE = ["locshape_pre_this"]
    LIST_SHAPE = [CONTEXT_VAR]
    for y in yvars:
        for bk in list_block:
            for sh in LIST_SHAPE:
                for epoch_orig in list_epoch_orig:

                    dfthis = DS.Dat[(DS.Dat["block"]==bk) & (DS.Dat["epoch_orig"]==epoch_orig)].reset_index(drop=True)

                    if len(dfthis)>N_MIN_PER_BLOCK:

                        print("primitivenessv2.plot_triallevel_results()", y, sh, bk, epoch_orig)

                        fig = sns.catplot(data=dfthis, x=contrast, col=sh, 
                            col_wrap=4, y=y, alpha=0.4)
                        rotateLabel(fig)
                        savefig(fig, f"{savedir}/triallevel-{contrast}-{sh}-{y}-epochorig_{epoch_orig}-bk_{bk}-1.pdf")

                        fig = sns.catplot(data=dfthis, x=sh, hue=contrast, y=y, kind="point", ci=68, aspect=3)
                        # fig = sns.catplot(data=dfthis, x=contrast, col=sh, 
                        #     col_wrap=4, y=y, kind="point", ci=68)
                        rotateLabel(fig)
                        savefig(fig, f"{savedir}/triallevel-{contrast}-{sh}-{y}-epochorig_{epoch_orig}-bk_{bk}-2.pdf")

                        plt.close("all")

                        ####### PLOTS OF CONTRAST ACROSS LEVELS.
                        # dfthis = DS.Dat[(DS.Dat["block"]==bk)]
                        INDEX = [sh, "epoch_orig", "block"]
                        if "microstim_epoch_code" in dfthis.columns:
                            fixed_treat = "microstim_epoch_code"
                            lev_treat_default = "off"
                        else:
                            fixed_treat = "epoch"
                            lev_treat_default = None

                        # Linear mixed effects
                        RES, fig, axes = lme_categorical_fit_plot(dfthis, y, fixed_treat, 
                            lev_treat_default, rand_grp_list=INDEX, PLOT = True)
                        savefig(fig, f"{savedir}/LME-{fixed_treat}-{sh}-{y}-epochorig_{epoch_orig}-bk_{bk}.pdf")

                        # Plot normalized to the default level.
                        _, _, _, _, fig = datamod_normalize_row_after_grouping(dfthis, 
                                                                              fixed_treat, 
                                                                              INDEX, 
                                                                              y,
                                                                              lev_treat_default,
                                                                              PLOT=True
                                                                             )
                        savefig(fig, f"{savedir}/NORM-{fixed_treat}-{sh}-{y}-epochorig_{epoch_orig}-bk_{bk}.pdf")

                        plt.close("all")

        ############### PLOTS ACROSS ALL DATA
        # fig = sns.catplot(data=DS.Dat, x=contrast, y=y, hue="shape", col="block", col_wrap=4,
        #     kind="point", ci=68)
        fig = sns.catplot(data=DS.Dat, x="block", y=y, hue=contrast, row="shape", kind="point", ci=68, aspect=1.5)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/all-{contrast}-{y}.pdf")

        plt.close("all")

        ###############
        fig = sns.catplot(data=DS.Dat, x=contrast, y=y, col=CONTEXT_VAR,
                          col_wrap=4, jitter=True, alpha=0.4)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/{CONTEXT_VAR}-triallevel-{contrast}-{y}-1.pdf")

        # sns.catplot(data=DS.Dat, x="epoch", y=contrast, hue="sh_loc_idx", col="sh_loc_idx", kind="point")
        # fig = sns.catplot(data=DS.Dat, x=contrast, y=y, col="sh_loc_idx",
        #                   col_wrap=4, kind="point")
        fig = sns.catplot(data=DS.Dat, x=CONTEXT_VAR, y=y, hue=contrast, kind="point", ci=68, aspect=2.5)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/{CONTEXT_VAR}-triallevel-{contrast}-{y}-2.pdf")

        plt.close("all")

def plot_grouplevel_results(dfres, DS, D, grouping, contrast, savedir):
    """ Plot and same summary of results
    """
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.stats.lme import lme_categorical_fit_plot
    from pythonlib.tools.pandastools import datamod_normalize_row_after_grouping
    
    for y in ["mean_sim_score", "dist_beh_task_strok", "time_duration", "velocity", "distcum"]:
        fig = sns.catplot(data=dfres, x=contrast, y=y)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/grplevel-{contrast}-{y}-1.pdf")

        fig = sns.catplot(data=dfres, x=contrast, y=y, kind="point", ci=68)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/grplevel-{contrast}-{y}-2.pdf")

        plt.close("all")
    
        fig = sns.catplot(data=dfres, x="locshape_pre_this", y=y, row="epoch_orig", hue=contrast, 
                    kind="point", aspect=3, alpha=0.5)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/grplevel-each_locshape_pre_this-{contrast}-{y}.pdf")


        from pythonlib.tools.snstools import plotgood_lineplot
        fig = plotgood_lineplot(dfres, contrast, y, "locshape_pre_this",
                         lines_add_ci=True, rowvar="epoch_orig", include_mean=True)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/grplevel-lineplot-{contrast}-{y}.pdf")

        ####### PLOTS OF CONTRAST ACROSS LEVELS.
        INDEX = ["locshape_pre_this", "epoch_orig", "block"]
        if "microstim_epoch_code" in dfres.columns:
            fixed_treat = "microstim_epoch_code"
            lev_treat_default = "off"
        else:
            fixed_treat = "epoch"
            lev_treat_default = None

        if True:
            # Linear mixed effects
            # This doesnt make sense, since there is only one datapt per group
            RES, fig, ax = lme_categorical_fit_plot(dfres, y=y, fixed_treat=fixed_treat, 
                    lev_treat_default=lev_treat_default, 
                    rand_grp_list=INDEX, PLOT=True)
            savefig(fig, f"{savedir}/LME-{fixed_treat}-{y}.pdf")

        # Plot normalized to the default level.
        _, _, _, _, fig = datamod_normalize_row_after_grouping(dfres, 
                                                              fixed_treat, 
                                                              INDEX, 
                                                              y,
                                                              lev_treat_default,
                                                              PLOT=True
                                                             )
        savefig(fig, f"{savedir}/NORM-{fixed_treat}-{y}.pdf")

    plt.close("all")

def plot_drawings_results(DS, savedir):
    """
    Results that relate to drawings.
    """ 

    DS.dataset_append_column("epoch")
    list_epoch = DS.Dat["epoch"].unique().tolist()
    list_shape = DS.Dat["shape"].unique().tolist()

    for sh in list_shape:
        for y in ["dist_beh_task_strok", "time_duration", "velocity", "distcum"]:
            fig, axes = DS.plotshape_egtrials_sorted_by_feature(sh, y)
            savefig(fig, f"{savedir}/egtrials-{y}-{sh}.pdf")
        plt.close("all")

    for sh in list_shape:
        for ep in list_epoch:
            for y in ["dist_beh_task_strok", "time_duration", "velocity", "distcum"]:
                fig, axes = DS.plotshape_egtrials_sorted_by_feature(sh, y, epoch=ep)
                if fig is not None:
                    savefig(fig, f"{savedir}/egtrials-{y}-{sh}-epoch_{ep}.pdf")
            plt.close("all")


def plot_timecourse_results(DS, savedir, contrast="epoch"):
    """ Plot timecourse (trial by trial)
    """

    DS.dataset_append_column("tvalfake")
    DS.dataset_append_column("session")
    list_epoch_orig = DS.Dat["epoch_orig"].unique().tolist()
    for epoch_orig in list_epoch_orig:
        dfthis = DS.Dat[DS.Dat["epoch_orig"]==epoch_orig]
        for y in ["dist_beh_task_strok", "time_duration", "velocity", "distcum"]:
            for sh in ["shape", "shapeabstract"]:

                fig = sns.relplot(data=dfthis, x="tvalfake", y=y, hue=contrast, 
                            row=sh, style="session", aspect=2, alpha=0.7)    

                savefig(fig, f"{savedir}/timecourse-{epoch_orig}-{contrast}-{sh}-{y}.pdf")

                plt.close("all")


