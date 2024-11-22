"""
All behaviroal figures in manuscript
Fig 1 - 3
Linked to notebook: 240912_MANUSCRIPT_FIGURES_1_shapes.

"""

# %cd ..
# from tools.utils import * 
# from tools.plots import *
# from tools.analy import *
# from tools.calc import *
# from tools.analyplot import *
# from tools.preprocess import *
# from tools.dayanalysis import *

from pythonlib.drawmodel.analysis import *
from pythonlib.tools.stroketools import *
import pythonlib
from pythonlib.dataset.dataset import load_dataset_notdaily_helper, load_dataset_daily_helper
import pickle
import seaborn as sns
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pythonlib.tools.plottools import savefig
from pythonlib.tools.snstools import rotateLabel
import pandas as pd


def load_pair_of_datasets(animal1, date1, animal2, date2):
    from pythonlib.tools.plottools import savefig
    from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    
    # Load two datasets (each animal)
    D1 = load_dataset_daily_helper(animal1, date1)

    # Load two datasets (each animal)
    D2 = load_dataset_daily_helper(animal2, date2)
    
    SAVEDIR = f"/lemur2/lucas/analyses/manuscripts/1_action_symbols/fig1_learned_primitives/{animal1}_{date1}-{animal2}_{date2}"
    os.makedirs(SAVEDIR, exist_ok=True)
    
    # Convert to strokes, and compare their strokes.
    DS1 = preprocess_dataset_to_datstrokes(D1, "singleprim") # get all strokes.
    DS2 = preprocess_dataset_to_datstrokes(D2, "singleprim") # get all strokes.

    # (prune to basis set prims)
    DS1.stroke_database_prune_to_basis_prims()
    DS2.stroke_database_prune_to_basis_prims()

    ##### For each animal, get its trial-by-trial distance between all trials.
    savedir = f"{SAVEDIR}/pairwise_beh_dists"
    os.makedirs(savedir, exist_ok=True)
    fig = grouping_plot_n_samples_conjunction_heatmap(DS1.Dat, "shape", "gridloc", ["gridsize", "task_kind"]);
    savefig(fig, f"{savedir}/counts-1.pdf")
    fig = grouping_plot_n_samples_conjunction_heatmap(DS2.Dat, "shape", "gridloc", ["gridsize", "task_kind"]);
    savefig(fig, f"{savedir}/counts-2.pdf")

    return D1, D2, DS1, DS2, SAVEDIR


    

def convert_cl_to_dfdists_and_plot(LIST_TRIAL_CL, SAVEDIR=None, PLOT=False, cl_var="Cl"):
    """
    """

    savedir = f"{SAVEDIR}/quantify"
    os.makedirs(savedir, exist_ok=True)
    print("Collecting for quantification...")
    print(savedir)

    if PLOT==False or SAVEDIR is None:
        savedir = None

    # Quantify
    list_dfdist = []
    list_dfaccuracy = []
    for res in LIST_TRIAL_CL:
        Cl = res[cl_var]
        animal_pair = res["animal_pair"]
        dist_ver = res["dist_ver"]

        ### (1) Distnace for same vs. different 
        dfdist = Cl.rsa_distmat_score_all_pairs_of_label_groups_datapts()
        dfdist["animal_pair"] = [animal_pair for _ in range(len(dfdist))]
        dfdist["dist_ver"] = dist_ver
        list_dfdist.append(dfdist)

        if PLOT:
            # Plots
            fig = sns.catplot(data=dfdist, x="shape_1", y="dist_mean", hue="shape_same", aspect=1.5, alpha=0.5, jitter=True)
            rotateLabel(fig)
            savefig(fig, f"{savedir}/dists-{dist_ver}-anipair={animal_pair}-1.pdf")

            fig = sns.catplot(data=dfdist, x="shape_2", y="dist_mean", col="shape_1", col_wrap=6, aspect=1.5, hue="shape_same")
            rotateLabel(fig)
            savefig(fig, f"{savedir}/dists-{dist_ver}-anipair={animal_pair}-2.pdf")

            fig = sns.catplot(data=dfdist, x="shape_1", y="dist_mean", hue="shape_same", aspect=1.5, kind="bar")
            rotateLabel(fig)
            savefig(fig, f"{savedir}/dists-{dist_ver}-anipair={animal_pair}-3.pdf")


        ### (2) Classification accuracy (what frac of trials for animal 1 would be classified correctly based on distnaces to animal 2?)
        if dist_ver in ["image", "image_beh_task"]:
            higher_score_better = False
        elif dist_ver == "motor":
            higher_score_better = True
        else:
            assert False
        score, score_adjusted, dfclasses, dfaccuracy = Cl.scalar_score_convert_to_classification_accuracy(dfdist, 
                                                                                    higher_score_better=higher_score_better, 
                                                                                    plot_savedir=savedir)
        dfaccuracy["animal_pair"] = [animal_pair for _ in range(len(dfaccuracy))]
        dfaccuracy["dist_ver"] = dist_ver

        list_dfaccuracy.append(dfaccuracy)

        if PLOT:
            # Plots
            fig = sns.catplot(data=dfaccuracy, x="label_actual", y="accuracy", aspect=1.5, kind="bar")
            rotateLabel(fig)
            savefig(fig, f"{savedir}/classaccuracy-{dist_ver}-anipair={animal_pair}.pdf")

        plt.close("all")

    ##################################
    ### Agg plots across animal comparisons
    savedir = f"{SAVEDIR}/quantify_agg"
    os.makedirs(savedir, exist_ok=True)
    print(savedir)

    DFDIST = pd.concat(list_dfdist).reset_index(drop=True)
    DFACCURACY = pd.concat(list_dfaccuracy).reset_index(drop=True)

    
    # Normalize the distances, to allow comparison across distance kinds
    # TODO: use interpretable normalizatoins:
    # -- image: max = across task-image, min = closest trace.
    # -- motor: correlation coefficient.
    # For now, kind of hacky, using distribution itself to determine normalization range

    map_distver_to_min = {}
    map_distver_to_max = {}
    for dist_ver in ["image", "motor", "image_beh_task"]:
        minmax = np.percentile(DFDIST[DFDIST["dist_ver"]==dist_ver]["dist_mean"], [0, 100])
        map_distver_to_min[dist_ver] = minmax[0]
        map_distver_to_max[dist_ver] = minmax[1]

    DFDIST["norm_min"] = [map_distver_to_min[dist_ver] for dist_ver in DFDIST["dist_ver"]]
    DFDIST["norm_max"] = [map_distver_to_max[dist_ver] for dist_ver in DFDIST["dist_ver"]]
    DFDIST["dist_mean_norm"] = (DFDIST["dist_mean"] - DFDIST["norm_min"])/(DFDIST["norm_max"] - DFDIST["norm_min"])

    dists = []
    for i, row in DFDIST.iterrows():
        if row["dist_ver"] == "motor":
            dists.append(1 - row["dist_mean_norm"])
        else:
            dists.append(row["dist_mean_norm"])
    DFDIST["dist_mean_norm"] = dists

    from pythonlib.tools.pandastools import stringify_values
    DFDIST = stringify_values(DFDIST)

    from pythonlib.tools.pandastools import append_col_with_grp_index
    print("HERE", len(DFDIST))
    DFDIST["same_animal"] = DFDIST["animal_pair"].isin(["0|0", "1|1"])
    DFDIST["same_shape"] = DFDIST["shape_1"] == DFDIST["shape_2"]
    # DFACCURACY["same_animal"] = DFACCURACY["animal_pair"].isin(["0|0", "1|1"])
    # DFACCURACY["same_shape"] = DFACCURACY["shape_same"]

    if PLOT:
        for dist, sharey in [
            ("dist_mean", False),
            ("dist_mean_norm", True),
            ]:
            fig = sns.catplot(data=DFDIST, x="animal_pair", y=dist, hue="shape_same", kind="bar", col="dist_ver", sharey=sharey)
            savefig(fig, f"{savedir}/combined-dfdist-dist={dist}-1.pdf")

            fig = sns.catplot(data=DFDIST, x="animal_pair", y=dist, hue="shape_same", alpha=0.4, jitter=True, col="dist_ver", sharey=sharey)
            savefig(fig, f"{savedir}/combined-dfdist={dist}-2.pdf")
            
            fig = sns.catplot(data=DFDIST, x="animal_pair", y=dist, hue="shape_same", kind="point", col="dist_ver", sharey=sharey)
            savefig(fig, f"{savedir}/combined-dfdist={dist}-3.pdf")
            
            fig =sns.catplot(data=DFDIST, x="animal_pair", y=dist, hue="shape_same", kind="violin", col="dist_ver", sharey=sharey)
            savefig(fig, f"{savedir}/combined-dfdist={dist}-4.pdf")

            plt.close("all")

        # Categories
        fig = sns.catplot(data=DFACCURACY, x="animal_pair", y="accuracy", kind="bar", errorbar=("ci", 68), col="dist_ver")
        savefig(fig, f"{savedir}/combined-accuracy.pdf")

        plt.close("all")

    return DFDIST, DFACCURACY

def score_motor_distances(DSthis, strokes1, strokes2, shapes1, shapes2, savedir, savesuff,
                          label_var="shape"):
    """
    """
    print("Scoring motor distance....")

    # Which distance score
    # if False:
    # Before 1/15/24
    # list_distance_ver  =("euclidian_diffs", "euclidian", "hausdorff_alignedonset")
    # 1/15/24 - This is much better
    # list_distance_ver  = ["dtw_vels_2d"]
    # list_distance_ver  = ["euclidian"]

    # Compute similarity
    Cl = DSthis.distgood_compute_beh_beh_strok_distances(strokes1, strokes2, labels_rows_dat=shapes1, 
                                                         labels_cols_feats=shapes2, label_var=label_var, 
                                                         clustclass_rsa_mode=True,
                                                         PLOT=True, savedir=savedir, savesuff=savesuff)
    # if len(strokes1)<200 and len(strokes2)<200:
    #     fig, _ = Cl.rsa_plot_heatmap()
    #     if fig is not None:
    #         savefig(fig, f"{savedir}/TRIAL_DISTS-motor-{savesuff}.pdf")  

    # ### Aggregate to get distance between groups (shapes)
    # _, Clagg = Cl.rsa_distmat_score_all_pairs_of_label_groups(return_as_clustclass=True, 
    #                                                           return_as_clustclass_which_var_score="dist_mean")

    # # Cl.rsa_distmat_score_all_pairs_of_label_groups_datapts
    # fig, _ = Clagg.rsa_plot_heatmap()
    # savefig(fig, f"{savedir}/TRIAL_DISTS_mean_over_shapes-motor-{savesuff}.pdf")  

    return Cl

def fig1_learned_prims_wrapper(animal1="Diego", date1=230616, animal2="Pancho", date2=240509):
    """
    Entire pipeline
    """
    ### Load dataset
    # animal1="Diego"
    # date1=230616
    # animal2="Pancho"
    # date2=240509
    D1, D2, DS1, DS2, SAVEDIR = load_pair_of_datasets(animal1, date1, animal2, date2)

    DSthis = DS1
    savedir = f"{SAVEDIR}/raw_heatmaps"
    os.makedirs(savedir, exist_ok=True)
    def _score_image_distances(strokes1, strokes2, shapes1, shapes2, savesuff):
        """
        """
        print("Scoring image distance....")

        Cl = DSthis.distgood_compute_image_strok_distances(strokes1, strokes2, shapes1, shapes2, do_centerize=True,
                                                            clustclass_rsa_mode=True, PLOT=True, savedir=savedir, savesuff=savesuff)
        # Cl = Cl.convert_copy_to_rsa_dist_version("shape", "image")


        # if N_TRIALS<12:
        #     fig, _ = Cl.rsa_plot_heatmap()
        #     if fig is not None:
        #         savefig(fig, f"{savedir}/TRIAL_DISTS-image-{savesuff}.pdf")  

        # ### Aggregate to get distance between groups (shapes)
        # _, Clagg = Cl.rsa_distmat_score_all_pairs_of_label_groups(return_as_clustclass=True, return_as_clustclass_which_var_score="dist_mean")

        # # Cl.rsa_distmat_score_all_pairs_of_label_groups_datapts
        # fig, _ = Clagg.rsa_plot_heatmap()
        # savefig(fig, f"{savedir}/TRIAL_DISTS_mean_over_shapes-image-{savesuff}.pdf")  

        return Cl

    def _score_motor_distances(strokes1, strokes2, shapes1, shapes2, savesuff):
        """
        """
        print("Scoring motor distance....")
        Cl = score_motor_distances(DSthis, strokes1, strokes2, shapes1, shapes2, savedir, savesuff)

        return Cl


    ### Compute all pairwise distances between strokes.
    from pythonlib.tools.pandastools import extract_trials_spanning_variable

    N_TRIALS = 10
    LIST_TRIAL_CL = []

    ### Also get distnace across animals
    # First, get common shapes
    shapes_common = sorted(set(DS1.Dat[DS1.Dat["shape"].isin(DS2.Dat["shape"].tolist())]["shape"]))

    inds, _ = extract_trials_spanning_variable(DS1.Dat, "shape", varlevels=shapes_common, n_examples=N_TRIALS)
    strokes1 = DS1.Dat.iloc[inds]["strok"].tolist()
    shapes1 = DS1.Dat.iloc[inds]["shape"].tolist()

    inds, _ = extract_trials_spanning_variable(DS2.Dat, "shape", varlevels=shapes_common, n_examples=N_TRIALS)
    strokes2 = DS2.Dat.iloc[inds]["strok"].tolist()
    shapes2 = DS2.Dat.iloc[inds]["shape"].tolist()

    assert shapes1 == shapes2, "then below will fail -- this is hacky, converting to rsa, which expects lables row and col to be identical. solve by updating code to also work with assymetric"

    ### (1) Image distance
    Cl = _score_image_distances(strokes1, strokes2, shapes1, shapes2, "01")
    LIST_TRIAL_CL.append({
        "animal_pair":(0,1),
        "dist_ver":"image",
        "Cl":Cl
        })    
        
    ### (2) Motor distance
    Cl = _score_motor_distances(strokes1, strokes2, shapes1, shapes2, "01")
    LIST_TRIAL_CL.append({
        "animal_pair":(0,1),
        "dist_ver":"motor",
        "Cl":Cl
        })

    # Get distances within each animal
    for i, DSthis in enumerate([DS1, DS2]):

        inds, _ = extract_trials_spanning_variable(DSthis.Dat, "shape", n_examples=N_TRIALS)
        strokes = DSthis.Dat.iloc[inds]["strok"].tolist()
        strokes_task = DSthis.extract_strokes(inds=inds, ver_behtask="task")
        shapes = DSthis.Dat.iloc[inds]["shape"].tolist()

        ### (1) Image distance (beh-beh)
        Cl = _score_image_distances(strokes, strokes, shapes, shapes, i)
        LIST_TRIAL_CL.append({
            "animal_pair":(i,i),
            "dist_ver":"image",
            "Cl":Cl
            })    

        ### (2) Motor distance (beh-beh)
        Cl = _score_motor_distances(strokes, strokes, shapes, shapes, i)
        LIST_TRIAL_CL.append({
            "animal_pair":(i,i),
            "dist_ver":"motor",
            "Cl":Cl
            })
        
        ### (3) Image distance (beh-task)
        Cl = _score_image_distances(strokes, strokes_task, shapes, shapes, f"beh_task-{i}")
        LIST_TRIAL_CL.append({
            "animal_pair":(i,i),
            "dist_ver":"image_beh_task",
            "Cl":Cl
            })    

        plt.close("all")

    ## Save the data 
    import pickle
    with open(f"{SAVEDIR}/LIST_TRIAL_CL.pkl", "wb") as f:
        pickle.dump(LIST_TRIAL_CL, f)
    # Collect all pairwise distances, across all conditions, into a single dataframe
    import seaborn as sns

    ##################################### QUANTIFICATIN
    DFDIST, DFACCURACY = convert_cl_to_dfdists_and_plot(LIST_TRIAL_CL, SAVEDIR, PLOT=True)

    ######################################
    # Specifically ask if beh is closer to itself than to the image (both using image distance)
    # DFDIST[""]
    savedir = f"{SAVEDIR}/image-vs-image_beh_task"
    os.makedirs(savedir, exist_ok=True)
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    for animal_pair in ["0|0", "1|1"]:
        dfthis = DFDIST[DFDIST["animal_pair"] == animal_pair].reset_index(drop=True)
        dfres, fig = plot_45scatter_means_flexible_grouping(dfthis, "dist_ver", "image", "image_beh_task", 
                                            "shape_same", "dist_mean", "shape_1", shareaxes=True,
                                            SIZE=4, plot_text=False)
        savefig(fig, f"{savedir}/anipair={animal_pair}.pdf")
   
    from pythonlib.tools.pandastools import pivot_table 
    a = DFDIST["animal_pair"].isin(["0|0", "1|1"])
    b = DFDIST["dist_ver"].isin(["image", "image_beh_task"])
    c = DFDIST["shape_same"]==True
    dfthis = DFDIST[a & b & c].reset_index(drop=True)
    dfpivot = pivot_table(dfthis, ["shape_1", "animal_pair"], ["dist_ver"], ["dist_mean"], flatten_col_names=True)

    from pythonlib.tools.statstools import signrank_wilcoxon, plotmod_pvalues
    res = signrank_wilcoxon(dfpivot["dist_mean-image"], dfpivot["dist_mean-image_beh_task"])

    fig, ax = plt.subplots()
    for i, row in dfpivot.iterrows():
        ax.plot([0, 1], [row["dist_mean-image"], row["dist_mean-image_beh_task"]], "-ok", alpha=0.5)
    ax.set_ylim(0)
    ax.set_xlim([-0.5, 1.5])
    plotmod_pvalues(ax, [0.5], [res.pvalue])
    ax.set_xlabel("dist_mean-image vs. dist_mean-image_beh_task")
    ax.set_ylabel("visual distance")
    savefig(fig, f"{savedir}/signrank.pdf")

    plt.close("all")

    ######################################
    ##### Plot all heatmaps on same heat scale
    def _get_cl(animal_pair, dist_ver):
        for res in LIST_TRIAL_CL:
            if (res["animal_pair"] == animal_pair) and (res["dist_ver"]==dist_ver):
                return res["Cl"]
        print(animal_pair, dist_ver)
        assert False, "didnt find it"

    map_animal_index_to_DSname = {
        0:(DS1, "Diego"),
        1:(DS2, "Pancho"),
    }
    for restrict_to_common_shapes in [False, True]:
        SIZE = 5
        if restrict_to_common_shapes:
            a = DFDIST["shape_1"].isin(shapes_common)
            b = DFDIST["shape_2"].isin(shapes_common)
            dfdist = DFDIST[a & b].reset_index(drop=True)
            labels_get = [(sh,) for sh in shapes_common]
        else:
            dfdist = DFDIST
            labels_get = None

        map_distver_to_zlims = {}
        for dist_ver in dfdist["dist_ver"].unique():
            minmax = np.percentile(dfdist[dfdist["dist_ver"]==dist_ver]["dist_mean"], [1, 99])
            map_distver_to_zlims[dist_ver] = minmax
        
        for dist_ver in ["image", "motor"]:
            zlims_this = map_distver_to_zlims[dist_ver]
            for zlims in [zlims_this, None]:
                savedir = f"{SAVEDIR}/heatmaps_combined/dist_ver={dist_ver}-zlims={zlims}-commonshapes={restrict_to_common_shapes}"
                os.makedirs(savedir, exist_ok=True)
                print(savedir)
                
                fig_heat_trial, axes_trial = plt.subplots(1,3, figsize=(3*SIZE, 1*SIZE))
                fig_heat_mean, axes_mean = plt.subplots(1,3, figsize=(3*SIZE, 1*SIZE))

                for ax_trial, ax_mean, animal_pair in zip(axes_trial.flatten(), axes_mean.flatten(), [(0,0), (1,1), (0,1)]):
                    Cl = _get_cl(animal_pair, dist_ver)
                    _, Clagg = Cl.rsa_distmat_score_all_pairs_of_label_groups(return_as_clustclass=True, 
                        return_as_clustclass_which_var_score="dist_mean",  labels_get=labels_get)

                    # Plot each trial
                    if N_TRIALS<10:
                        Cl.rsa_plot_heatmap(ax=ax_trial, zlims=zlims)
                        ax_trial.set_title(f"animal_pair={animal_pair}")

                    # Plot means
                    Clagg.rsa_plot_heatmap(ax=ax_mean, zlims=zlims)
                    ax_mean.set_title(f"animal_pair={animal_pair}")

                    # Plot the axis
                    for _i, animal_index in enumerate(animal_pair):
                        if _i==0:
                            shapes = [lab[0] for lab in Clagg.Labels]
                            axis="x"
                        elif _i==1:
                            shapes = [lab[0] for lab in Clagg.LabelsCols]
                            axis="y"
                        else:
                            assert False
                        ds, animal = map_animal_index_to_DSname[animal_index]
                        _, list_strok_basis, list_shape_basis = ds.stroke_shape_cluster_database_load_helper(which_basis_set=animal, which_shapes=shapes)
                        for title_shapes in [list_shape_basis, None]:
                            fig = ds.plotshape_row_figure_axis(list_strok_basis, axis=axis, title_shapes=title_shapes)
                            savefig(fig, f"{savedir}/axisshapes-anipair={animal_pair}-axis={axis}-ani={animal}-titles={title_shapes is not None}.pdf")
                savefig(fig_heat_trial, f"{savedir}/HEATMAPS-trial.pdf")
                savefig(fig_heat_mean, f"{savedir}/HEATMAPS-mean.pdf")
        plt.close("all")


def fig1_motor_invariance(DS, N_TRIALS, savedir, DEBUG=False):
    """
    Entire pipeline, to analyze invariance of motor parameters due to loation and size.
    Expt is assumed to have both loc and size vairation.

    Does: Make drawing plots, Compute pairwise dsitances, plot scores.

    TODO: Stat significance of violin plots. Should do shuffle (same as did for euclidian dist for neural).
    """
    grpvars = ["shape", "gridloc", "gridsize"]


    # Just to make quicker
    if DEBUG:
        shapes_keep = DS.Dat["shape"].unique().tolist()[:2]
        locs_keep = DS.Dat["gridloc"].unique().tolist()[:2]
        sizes_keep = DS.Dat["gridsize"].unique().tolist()[:2]

        a = DS.Dat["shape"].isin(shapes_keep)
        b = DS.Dat["gridloc"].isin(locs_keep)
        c = DS.Dat["gridsize"].isin(sizes_keep)

        DS.Dat = DS.Dat[(a & b & c)].reset_index(drop=True)

    # Clean up
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good
    from pythonlib.tools.pandastools import append_col_with_grp_index
    dfthis, inds_keep = extract_with_levels_of_var_good(DS.Dat, grpvars, min([4, N_TRIALS]))
    DS.Dat = dfthis

    DS.Dat = append_col_with_grp_index(DS.Dat, ["gridloc", "gridsize"], "locsize")
    DS.Dat = append_col_with_grp_index(DS.Dat, grpvars, "shape_loc_size")

    # Drawings
    n_iter_drawing = 3
    for i in range(n_iter_drawing):
        fig_beh, fig_task = DS.plotshape_row_col_size_loc()
        savefig(fig_beh, f"{savedir}/drawing-1-beh-iter{i}.pdf")
        savefig(fig_task, f"{savedir}/drawing-1-task-iter{i}.pdf")

        fig_beh, fig_task = DS.plotshape_row_col_vs_othervar("shape", "locsize", plot_task=True, ver_behtask="task")
        savefig(fig_beh, f"{savedir}/drawing-2-beh-iter{i}.pdf")
        savefig(fig_task, f"{savedir}/drawing-2-task-iter{i}.pdf")

        fig_beh, fig_task = DS.plotshape_row_col_vs_othervar("locsize", plot_task=True, ver_behtask="task")
        savefig(fig_beh, f"{savedir}/drawing-3-beh-iter{i}.pdf")
        savefig(fig_task, f"{savedir}/drawing-3-task-iter{i}.pdf")

    plt.close("all")

    # get pairwise distances between all strokes
    # OLD VERSION - obsolete -- uses mean strokes to compute distance
    if False:

        # Devos
        if False:
            WHICH_LEVEL = "trial"
            WHICH_BASIS_SET 
            ClustDict, ParamsDict, ParamsGeneral, dfdat = DS.features_wrapper_generate_all_features(WHICH_LEVEL,
                                                                                                    which_basis_set=WHICH_BASIS_SET)
            plt.close("all")

            # Get all strokes
            strokes = DS.Dat["strok"].tolist()
            shapes = DS.Dat["shape"].tolist()
        # Get mean stroke for each (shape, size, loc)
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
        grpdict = grouping_append_and_return_inner_items_good(DS.Dat, vars_grp)
        res = []
        for grp, inds in grpdict.items():
            # DS.Dat.iloc[inds]["strok"].tolist()
            # print(inds)

            if len(inds)>4:
                strokmean, _ = DS.cluster_compute_mean_stroke(inds, Npts=50, rescale_strokes_ver="stretch_to_1")

                # remove time
                # strokmean = strokmean[:, :2]
                res.append({
                    "strok":strokmean,
                    "grp":grp
                })

                for v, val in zip(vars_grp, grp):
                    res[-1][v]=val
        import pandas as pd
        df_strokmean = pd.DataFrame(res)
        df_strokmean
        # Plot each stroke, just for sanity check
        strokes = df_strokmean['strok'].tolist()
        fig, axes = plt.subplots(5,5)
        for ax, strok in zip(axes.flatten(), strokes):
            DS.plot_single_strok(strok, ax=ax)

        # Get pairwise distance between each mean stroke
        # -- can take a while   

        # Extract the basis set
        strokes = df_strokmean['strok'].tolist()
        labels = df_strokmean['grp'].tolist()

        strokes = strokes[:10]
        labels = labels[:10]
        # Which distance score
        # if False:
        # Before 1/15/24
        # list_distance_ver  =("euclidian_diffs", "euclidian", "hausdorff_alignedonset")
        # 1/15/24 - This is much better
        list_distance_ver  = ["dtw_vels_2d"]
        # list_distance_ver  = ["euclidian"]

        # Compute similarity
        Cl = DS._cluster_compute_sim_matrix_aggver(strokes, strokes, labels, labels, vars_grp, list_distance_ver, clustclass_rsa_mode=True)


        sort_order=(0,2,1)
        fig, ax = Cl.rsa_plot_heatmap(sort_order=sort_order)
        savefig(fig, f"{savedir}/distmat-sort={sort_order}.pdf")
        for var in Cl.rsa_labels_extract_label_vars():
            _, fig = Cl.rsa_distmat_construct_theoretical(var, True)
            savefig(fig, f"{savedir}/theor_distmat-var={var}.pdf")
        ### Get distnace scores -- violin plots
        dfdist = Cl.rsa_distmat_score_all_pairs_of_label_groups()
        # Cl.rsa_dataextract_with_labels_as_flattened_df() #

        # normalize distances 

        
        from pythonlib.tools.pandastools import append_col_with_grp_index

        dfdist = append_col_with_grp_index(dfdist, ["shape_same", "gridloc_same", "gridsize_same"], "same-shape_loc_size")

        dfdist = dfdist.drop("dist_yue_diff", axis=1)
        dfdist = dfdist.dropna()
        # normalize distnaces to 0,1
        minmax = np.percentile(dfdist["dist_mean"], [0, 100])


        dfdist["dist_mean_norm_v2"] = 1 - (dfdist["dist_mean"] - minmax[0])/(minmax[1]-minmax[0])
        import seaborn as sns
        sns.catplot(data=dfdist, x="same-shape_loc_size", y="dist_mean_norm_v2")

        import seaborn as sns
        sns.catplot(data=dfdist, x="same-shape_loc_size", y="dist_mean_norm_v2", kind="violin")


    from pythonlib.tools.pandastools import extract_trials_spanning_variable, append_col_with_grp_index

    inds, _ = extract_trials_spanning_variable(DS.Dat, "shape_loc_size", n_examples=N_TRIALS)
    strokes = DS.Dat.iloc[inds]["strok"].tolist()
    labels = [tuple(x) for x in DS.Dat.loc[inds, grpvars].values.tolist()]

    print("Scoring motor distance....")
    list_distance_ver  = ["dtw_vels_2d"]
    # Cl = DS._cluster_compute_sim_matrix_aggver(strokes, strokes, labels, labels, grpvars, list_distance_ver,
    #                                             clustclass_rsa_mode=True)
    Cl = DS.distgood_compute_beh_beh_strok_distances(strokes, strokes, None, labels, labels, grpvars, clustclass_rsa_mode=True)
    
    # Save, since this takes a long time.
    import pickle
    with open(f"{savedir}/Cl.pkl", "wb") as f:
        pickle.dump(Cl, f)

    ################### PLOTS

    # Plot heatmap
    for sort_order in [(0,1,2), (1,2, 0), (2,0,1)]:
        fig, ax = Cl.rsa_plot_heatmap(sort_order=sort_order)
        if fig is not None:
            savefig(fig, f"{savedir}/distmat-sort={sort_order}.pdf")
    for var in Cl.rsa_labels_extract_label_vars():  
        _, fig = Cl.rsa_distmat_construct_theoretical(var, True)
        if fig is not None:
            savefig(fig, f"{savedir}/theor_distmat-var={var}.pdf")
    # fig, ax = Cl.rsa_plot_heatmap()
    # savefig(fig, f"{savedir}/heatmap-trials.pdf")

    dfdist = Cl.rsa_distmat_score_all_pairs_of_label_groups_datapts()
    dfdist = dfdist.drop("dist_yue_diff", axis=1)
    dfdist = dfdist.dropna()
    from pythonlib.tools.pandastools import append_col_with_grp_index
    dfdist = append_col_with_grp_index(dfdist, ["shape_same", "gridloc_same", "gridsize_same"], "same-shape_loc_size")

    # Get normalized score (0 good, 1 bad)
    score_max = dfdist[dfdist["same-shape_loc_size"]=="1|1|1"]["dist_mean"].mean()
    score_min = dfdist[dfdist["same-shape_loc_size"]=="0|0|0"]["dist_mean"].mean()
    dfdist["dist_mean_norm_v2"] = 1 - (dfdist["dist_mean"] - score_min)/(score_max - score_min)

    # Agg, so that each (shape, loc, size) is single datapt
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.pandastools import grouping_print_n_samples, grouping_plot_n_samples_conjunction_heatmap
    dfdist_agg = aggregGeneral(dfdist, ["labels_1_datapt", "same-shape_loc_size"], ["dist_mean", "dist_mean_norm_v2"])
    
    fig = grouping_plot_n_samples_conjunction_heatmap(dfdist_agg, "labels_1_datapt", "same-shape_loc_size")
    savefig(fig, f"{savedir}/counts-dfdist_agg.pdf")

    import seaborn as sns
    for y in ["dist_mean", "dist_mean_norm_v2"]:
        fig = sns.catplot(data=dfdist, x="same-shape_loc_size", y=y)
        savefig(fig, f"{savedir}/catplot-trials-1.pdf")

        fig = sns.catplot(data=dfdist, x="same-shape_loc_size", y=y, kind="violin")
        savefig(fig, f"{savedir}/catplot-trials-2.pdf")

        fig = sns.catplot(data=dfdist, x="same-shape_loc_size", y=y, kind="bar", errorbar=("ci", 68))
        savefig(fig, f"{savedir}/catplot-trials-3.pdf")

        fig = sns.catplot(data=dfdist_agg, x="same-shape_loc_size", y=y)
        savefig(fig, f"{savedir}/catplot-grps-1.pdf")

        fig = sns.catplot(data=dfdist_agg, x="same-shape_loc_size", y=y, kind="violin")
        savefig(fig, f"{savedir}/catplot-grps-2.pdf")

        fig = sns.catplot(data=dfdist_agg, x="same-shape_loc_size", y=y, kind="bar", errorbar=("ci", 68))
        savefig(fig, f"{savedir}/catplot-grps-3.pdf")
        plt.close("all")

def fig3_charsyntax_wrapper(animal, DATE, SAVEDIR):
    """
    Everything for fig3 that is related to syntax for char.
    """
    # animal = "Diego"
    # DATE = 231205
    D = load_dataset_daily_helper(animal, DATE)
    D.preprocessGood(params=["no_supervision", "remove_online_abort"])
    D.Dat = D.Dat[D.Dat["task_kind"]=="character"].reset_index(drop=True)

    assert all(D.Dat["charclust_shape_seq_did_replace"]), "have not replaced all with clustered strokes"
    from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
    DS = preprocess_dataset_to_datstrokes(D, "all_no_abort_superv") # get all strokes.

    ##### Location heatmaps (vs. stroke index)
    savedir = f"{SAVEDIR}/location_heatmaps"
    os.makedirs(savedir, exist_ok=True)
    from pythonlib.dataset.scripts.analy_manuscript_figures import fig3_charsyntax_location_vs_si_heatmaps
    fig3_charsyntax_location_vs_si_heatmaps(D, DS, savedir)

    ##### Location vs. rank, generalizing across shapes [CODE GOOD]
    from pythonlib.dataset.scripts.analy_manuscript_figures import fig3_charsyntax_location_vs_si_generalization
    for var_shape in ["shape", "shape_semantic_grp"]:
        for do_consolidate_si in [False, True]:
            for pt_ver in ["all", "onset"]:
                savedir = f"{SAVEDIR}/location_heatmaps_generalization/varshape={var_shape}-consolidsi={do_consolidate_si}-ptver={pt_ver}"
                os.makedirs(savedir, exist_ok=True)
                fig3_charsyntax_location_vs_si_generalization(DS, savedir, var_shape, do_consolidate_si, pt_ver)

    ##### Shape vs. stroke index
    savedir = f"{SAVEDIR}/shape_vs_strokeindex"
    os.makedirs(savedir, exist_ok=True)
    # Only keep shapes that have at least N trials.
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good, grouping_plot_n_samples_conjunction_heatmap
    n_min = 20
    for var_shape in ["shape", "shape_semantic_grp"]:
        dfthis, inds_keep = extract_with_levels_of_var_good(DS.Dat, [var_shape], n_min)
        print(len(DS.Dat), len(dfthis))
        for norm_method in [None, "row_div", "col_div"]:
            fig, list_df = grouping_plot_n_samples_conjunction_heatmap(dfthis, var_shape, "stroke_index", 
                                                            annotate_heatmap=False, sort_rows_by_mean_col_value=True,
                                                            norm_method=norm_method, also_return_df=True);
            savefig(fig, f"{savedir}/counts-{var_shape}-vs-stroke_index-norm={norm_method}.pdf")
            
        # Also plot the shapes for axis.
        shapes = list_df[0].index.tolist()
        if var_shape=="shape_semantic_grp":
            shapes = [DS.shapesemantic_map_shapesemgrp_to_shape(ssg) for ssg in shapes]
        fig = DS.plotshape_row_figure_axis_shapes(shapes, "y")
        savefig(fig, f"{savedir}/axisdrawings-{var_shape}.pdf")
    plt.close("all")

    ##### Transition prob between shapes
    from pythonlib.dataset.scripts.analy_manuscript_figures import fig3_charsyntax_transition_matrix_all
    var_shape = "shape_semantic_grp"
    for transition_prob_kind in ["all", "div"]:
        # transition_prob_kind = "all"
        savedir = f"{SAVEDIR}/transition_probs/{var_shape}-txnkind={transition_prob_kind}"
        os.makedirs(savedir, exist_ok=True)
        fig3_charsyntax_transition_matrix_all(D, DS, savedir, var_shape)
    plt.close("all")
    print("Done!")

def fig3_charsyntax_location_vs_si_heatmaps(D, DS, savedir):
    """
    Plot heatmaps of location vs. stroke index --> sequential bias in stroke locations?

    """
    # For each stroke, collect its location in the image bounding box.
    # (convert locations to relative to image (bounding box of image))
    from pythonlib.tools.stroketools import strokes_bounding_box, strokes_bounding_box_dimensions
    import random
    from pythonlib.tools.plottools import plot_2d_binned_smoothed_heatmap

    ########### PREPROCESSING -- Extract strokes, rescaled to bounding box of character
    PLOT=False
    list_strokbeh_rescaled = []
    for ind in range(len(DS.Dat)):
        # Extract strokes (beh and task)
        strokes_task = DS.extract_strokes(inds=[ind], ver_behtask="task_entire") # list of array
        strok_beh = DS.Dat.iloc[ind]["strok"].copy() # array

        # rescale to bounding box 
        if PLOT:
            fig, ax = plt.subplots()
            DS.plot_multiple_strok(strokes_task, "task", ax=ax)
            DS.plot_multiple_strok([strok_beh], "beh", ax=ax)

        xmin, xmax, ymin, ymax = strokes_bounding_box(strokes_task)

        # Rescale to bounding box
        strok_beh[:,0] = (strok_beh[:,0]-xmin)/(xmax-xmin)
        strok_beh[:,1] = (strok_beh[:,1]-ymin)/(ymax-ymin)

        # Also collect task strokes
        
        if PLOT:
            print(xmin, xmax, ymin, ymax)
            fig, ax = plt.subplots()
            DS.plot_multiple_strok([strok_beh], "beh", ax=ax)
        
        list_strokbeh_rescaled.append(strok_beh)
    DS.Dat["strok_rescaled_to_task"] = list_strokbeh_rescaled

    # Do the same for the task -- to show uniform spread over image
    # - colect stroks across all tasks
    from pythonlib.tools.stroketools import strokes_bounding_box, strokes_bounding_box_dimensions
    PLOT=False
    list_stroktask_rescaled = []
    for ind in range(len(D.Dat)):
        # Extract strokes (beh and task)
        strokes_task = D.Dat.iloc[ind]["strokes_task"]
        strokes_task = [s.copy() for s in strokes_task] # copy
        
        xmin, xmax, ymin, ymax = strokes_bounding_box(strokes_task)

        def _rescale(strok):
            """ 
            Modifies in place, rescaling to (0,1), (0,1)
            """
            strok[:,0] = (strok[:,0]-xmin)/(xmax-xmin)
            strok[:,1] = (strok[:,1]-ymin)/(ymax-ymin)

        # Rescale to bounding box
        for strok in strokes_task:
            _rescale(strok)

        # Also collect task strokes    
        if PLOT:
            print(xmin, xmax, ymin, ymax)
            fig, ax = plt.subplots()
            D.plot_strokes(strokes_task, ax)
            assert False, "too many plots"
        
        list_stroktask_rescaled.extend(strokes_task)

    ################ PLOT HEATMAPS
    # Plot heatmap of all task strokes
    pad = 0.2
    x_min=0-pad
    x_max=1+pad
    y_min=0-pad
    y_max=1+pad

    num_bins = 20
    sigma = 3

    ### (1) Task image
    fig, axes = plt.subplots(2, 2)

    ### Strokes
    ax = axes.flatten()[0]
    k = 400
    inds = list(range(len(list_stroktask_rescaled)))
    if len(inds)>k:
        inds_rand = random.sample(inds, k)
    else:
        inds_rand = inds
    strokes = [list_stroktask_rescaled[i] for i in inds_rand]
    DS.plot_multiple_strok(strokes, ax=ax, alpha=0.1)

    ### Plot heatmap of all stroke points
    ax = axes.flatten()[1]
    ax.set_title("all pts")
    pts = np.concatenate(strokes, axis=0)[:, :2]
    plot_2d_binned_smoothed_heatmap(pts, x_min=0-pad, x_max=1+pad,
                                                                    y_min=0-pad, y_max=1+pad, ax=ax,
                                                                    num_bins=num_bins, sigma=sigma, plot_colorbar=False)

    ### Plot heatmap of stroke onsets
    ax = axes.flatten()[2]
    ax.set_title("onsets")
    pts = np.stack([s[0, :2] for s in strokes], axis=0)
    plot_2d_binned_smoothed_heatmap(pts, x_min=0-pad, x_max=1+pad,
                                                                    y_min=0-pad, y_max=1+pad, ax=ax,
                                                                    num_bins=num_bins, sigma=sigma)

    ### Plot heatmap of centers of mass
    from pythonlib.tools.stroketools import getCentersOfMass
    ax = axes.flatten()[3]
    ax.set_title("COM")
    pts = np.stack(getCentersOfMass(strokes), axis=0)
    plot_2d_binned_smoothed_heatmap(pts, x_min=0-pad, x_max=1+pad,
                                                                    y_min=0-pad, y_max=1+pad, ax=ax,
                                                                    num_bins=num_bins, sigma=sigma, plot_colorbar=False)

    savefig(fig, f"{savedir}/heatmaps-task_strokes.pdf")
    plt.close("all")

    ########### BEH STROKES
    from pythonlib.tools.pandastools import extract_trials_spanning_variable, grouping_append_and_return_inner_items_good
    min_trials = 10
    k = 15

    for grpvars in [
        ["shape_semantic_grp"], ["stroke_index"], ["stroke_index", "shape_semantic_grp"], ["shape_semantic_grp", "stroke_index"]
        # ["shape_semantic_grp"]
        ]:
        grpdict = grouping_append_and_return_inner_items_good(DS.Dat, grpvars)
        grpdict = {grp:inds for grp, inds in grpdict.items() if len(inds)>=min_trials}

        SIZE=4
        ncols = 8
        nrows = int(np.ceil(len(grpdict)/ncols))
        fig_0, axes_0 = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)
        fig_1, axes_1 = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)
        fig_2, axes_2 = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)
        fig_3, axes_3 = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)

        for i, (grp, inds) in enumerate(grpdict.items()):

            ### Strokes
            ax = axes_0.flatten()[i]
            ax.set_title(f"{grpvars}-{grp}", fontsize=8)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

            if len(inds)>k:
                inds_rand = random.sample(inds, k)
            else:
                inds_rand = inds
            strokes = DS.Dat.iloc[inds_rand]["strok_rescaled_to_task"].tolist()
            DS.plot_multiple_strok(strokes, ax=ax, alpha=0.5)

            ########################## HEATMAPS
            strokes = DS.Dat.iloc[inds]["strok_rescaled_to_task"].tolist()

            ### Plot heatmap of all stroke points
            ax = axes_1.flatten()[i]
            ax.set_title(f"{grpvars}-{grp}", fontsize=8)
            pts = np.concatenate(strokes, axis=0)[:, :2]
            plot_2d_binned_smoothed_heatmap(pts, x_min=0-pad, x_max=1+pad,
                                                                            y_min=0-pad, y_max=1+pad, ax=ax,
                                                                            num_bins=num_bins, sigma=sigma, plot_colorbar=False)
            
            ### Plot heatmap of stroke onsets
            ax = axes_2.flatten()[i]
            ax.set_title(f"{grpvars}-{grp}", fontsize=8)
            pts = np.stack([s[0, :2] for s in strokes], axis=0)
            num_bins = 20
            sigma = 3
            plot_2d_binned_smoothed_heatmap(pts, x_min=0-pad, x_max=1+pad,
                                                                            y_min=0-pad, y_max=1+pad, ax=ax,
                                                                            num_bins=num_bins, sigma=sigma)
            
            ### Plot heatmap of centers of mass
            from pythonlib.tools.stroketools import getCentersOfMass
            ax = axes_3.flatten()[i]
            ax.set_title(f"{grpvars}-{grp}", fontsize=8)
            pts = np.stack(getCentersOfMass(strokes), axis=0)
            plot_2d_binned_smoothed_heatmap(pts, x_min=0-pad, x_max=1+pad,
                                                                            y_min=0-pad, y_max=1+pad, ax=ax,
                                                                            num_bins=num_bins, sigma=sigma, plot_colorbar=False)
        
        savefig(fig_0, f"{savedir}/strokesbeh-heatmaps-grp={'|'.join(grpvars)}-strokes.pdf")
        savefig(fig_1, f"{savedir}/strokesbeh-heatmaps-grp={'|'.join(grpvars)}-allpts.pdf")
        savefig(fig_2, f"{savedir}/strokesbeh-heatmaps-grp={'|'.join(grpvars)}-onsets.pdf")
        savefig(fig_3, f"{savedir}/strokesbeh-heatmaps-grp={'|'.join(grpvars)}-COMs.pdf")

        plt.close("all")
        

def fig3_charsyntax_location_vs_si_generalization(DS, savedir, var_shape = "shape_semantic_grp", 
                                                  do_consolidate_si = False, pt_ver = "all",
                                                  min_trials_per_split=5):
    """
    Ask whether the relationship betwen si and location (i.e., spatial heatmaps showing
    change in stroke location) generlaizes across shapes 
    --> ie as evidence that this aspect of syntax is "invariant" to shape

    Does so by taking correaltion of heatmaps across all pairs of (si, shape)
    
    """
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper, grouping_append_and_return_inner_items_good
    import random
    from pythonlib.tools.plottools import plot_2d_binned_smoothed_heatmap
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap

    # do_consolidate_si = True
    do_consolidate_si_max = 2
    # # var_shape = "shape"
    # var_shape = "shape_semantic_grp"
    # min_trials_per_split = 6
    min_trials = min_trials_per_split*2
    # pt_ver = "all" # Note: in order of best to worst, in correlation for diff shape, same index: all, COM, onset.
    pad = 0.15
    num_bins = 20
    sigma = 3

    ### Consoldiate si (i.e. 0,1,2+)
    dfout = DS.Dat.copy()
    if do_consolidate_si:
        def _consolidate_stroke_index(si):
            if si>do_consolidate_si_max-1:
                return do_consolidate_si_max
            else:
                return si
        dfout["stroke_index"] = [_consolidate_stroke_index(si) for si in dfout["stroke_index"]]

    ### Clean up data, by conjucntions of (shape, si)
    # Keep only shapes that have multiple stroke indices
    dfout, _ = extract_with_levels_of_conjunction_vars_helper(dfout, "stroke_index", [var_shape], min_trials, f"{savedir}/counts_after_step1.pdf", 2)
    dfout, _ = extract_with_levels_of_conjunction_vars_helper(dfout, var_shape, ["stroke_index"], min_trials, f"{savedir}/counts_after_step2.pdf", 3)
    # dfout, dict_dfthis = extract_with_levels_of_conjunction_vars(DS.Dat, "stroke_index", [var_shape], None, min_trials, False, 2, prune_levels_with_low_n=True,
    #                                         balance_no_missed_conjunctions=True, plot_counts_heatmap_savepath="/tmp/test.pdf")

    ### Go thru each (si, shape)
    grpvars = ["stroke_index", var_shape]
    grpdict = grouping_append_and_return_inner_items_good(dfout, grpvars)
    grpdict = {grp:inds for grp, inds in grpdict.items() if len(inds)>=min_trials}

    # Get correlation coefficient between heatmaps
    # (1) Collect heatmaps
    list_heatmap = []
    list_grp = []
    for _, (grp, inds) in enumerate(grpdict.items()):

        # Get pts
        strokes = dfout.iloc[inds]["strok_rescaled_to_task"].tolist()

        # Split into two halves, to get self-similarity
        n_mid = int(np.ceil(len(strokes)/2))
        strokes = [s for s in strokes] # to copy
        random.shuffle(strokes) # shuffle strokes
        for i_half in range(2):
            if i_half==0:
                strokes_this = strokes[:n_mid]
            elif i_half==1:
                strokes_this = strokes[n_mid:]
            else:
                assert False

            if pt_ver == "COM":
                pts = np.stack(getCentersOfMass(strokes_this), axis=0)
            elif pt_ver == "onset":
                pts = np.stack([s[0, :2] for s in strokes_this], axis=0)
            elif pt_ver == "all":
                pts = np.concatenate(strokes_this, axis=0)[:, :2]
            else:
                assert False
            assert pts.shape[1]==2

            _, smoothed_heatmap, _ = plot_2d_binned_smoothed_heatmap(pts, x_min=0-pad, x_max=1+pad,
                                                                            y_min=0-pad, y_max=1+pad,
                                                                            num_bins=num_bins, sigma=sigma, 
                                                                            plot_colorbar=False, skip_plot=True)
            list_heatmap.append(smoothed_heatmap)
            list_grp.append(grp)

    ### Get pairwise corr between each (si, shape)
    from pythonlib.tools.distfunctools import distmat_construct_wrapper
    def _dist(hm1, hm2):
        """
        Pearson corr between two flattened heatmaps
        """
        x = hm1.flatten()
        y = hm2.flatten()
        correlation_coefficient = np.corrcoef(x, y)[0, 1]
        return correlation_coefficient
    labels = list_grp
    Cl = distmat_construct_wrapper(list_heatmap, list_heatmap, _dist, return_as_clustclass=True, clustclass_labels_1=labels,
                                        clustclass_labels_2=labels, clustclass_label_vars=grpvars)
    for sort_order in [(0,1), (1,0)]:
        fig, _ = Cl.rsa_plot_heatmap(sort_order=sort_order, diverge=True, zlims=[-1, 1])
        if fig is not None:
            savefig(fig, f"{savedir}/pairwise_corr-heatmap-sort_order={sort_order}.pdf")  

    ### Extract disitances
    dfdists = Cl.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=False)
    dfdists = dfdists[~dfdists['dist_mean'].isna()].reset_index(drop=True)

    # agg, so that one datapt per unique label
    var_var_same = f"same-stroke_index|{var_shape}"
    dfdists_agg = aggregGeneral(dfdists, ["labels_1", var_var_same], ["dist_mean"])
    if False:
        grouping_plot_n_samples_conjunction_heatmap(dfdists, "labels_1", "labels_2")
        grouping_plot_n_samples_conjunction_heatmap(dfdists, "labels_1", var_var_same)
        grouping_plot_n_samples_conjunction_heatmap(dfdists_agg, "labels_1", var_var_same)

    ### Plots
    dfthis = dfdists_agg
    fig = sns.catplot(data=dfthis, x=var_var_same, y="dist_mean", jitter=True, alpha=0.5)
    savefig(fig, f"{savedir}/distances-catplot-1.pdf")

    fig = sns.catplot(data=dfthis, x=var_var_same, y="dist_mean", kind="violin")
    savefig(fig, f"{savedir}/distances-catplot-2.pdf")

    fig = sns.catplot(data=dfthis, x=var_var_same, y="dist_mean", kind="bar", errorbar=("ci", 68))
    savefig(fig, f"{savedir}/distances-catplot-3.pdf")

    plt.close("all")

    return dfdists

def fig3_charsyntax_transition_matrix_all(D, DS, savedir,
                                          var_shape = "shape_semantic_grp", transition_prob_kind = "div"):
    """
    """
    import pandas as pd
    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    import numpy as np
    from scipy.stats import chi2_contingency

    ### Collect transition frequenceies between shapes
    res = []
    for ind in range(len(D.Dat)):
        Tkbeh = D.taskclass_tokens_extract_wrapper(ind, "beh_using_beh_data", False, True)
        # Tktask = D.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data", False, True)
        # assert len(Tktask.Tokens)==len(Tkbeh.Tokens)
        for i in range(len(Tkbeh.Tokens)-1):
            
            tok1 = Tkbeh.Tokens[i]
            tok2 = Tkbeh.Tokens[i+1]
            
            res.append({
                "inddat":ind,
                "indstroke_1":i,
                "indstroke_2":i+1,
                "shape1":tok1[var_shape],
                "shape2":tok2[var_shape]
            })    
    df = pd.DataFrame(res)
    shapes_ordered = sorted(set(df["shape1"].tolist() + df["shape2"].tolist()))
    data, fig, ax, rgba_values = convert_to_2d_dataframe(df, "shape1", "shape2", list_cat_1=shapes_ordered, list_cat_2=shapes_ordered)
    n_min_from = 20
    data = data.loc[data.sum(axis=1)>n_min_from, :]


    ### STATS - Transition probabilties different from uniform?

    # Perform the Chi-squared test
    try:
        chi2, p, dof, expected = chi2_contingency(data)
        # Output the results
        print("Chi-squared statistic:", chi2)
        print("p-value:", p)
        print("Degrees of freedom:", dof)
        from pythonlib.tools.expttools import writeDictToTxt
        writeDictToTxt({
            "Chi-squared statistic":chi2,
            "p-value":p,
            "Degrees of freedom":dof}, f"{savedir}/tmat-chi2-stats.txt")
    except Exception as err:
        SIZE = 4
        fig, axes = plt.subplots(1, 4, figsize=(5*SIZE, SIZE))    
        ax = axes.flatten()[0]
        ax.set_title("data")
        sns.heatmap(data, ax=ax)
        savefig(fig, f"{savedir}/DEBUG-heatmaps-all.pdf")
        raise err
        # Just skip things
        # from pythonlib.tools.expttools import writeDictToTxt
        # writeDictToTxt({}, f"{savedir}/tmat-chi2-stats-FAILED.txt")
    
    # Convert to normalized (outgoing fractions)
    if transition_prob_kind=="div":
        data_norm = data/np.sum(data.values, axis=1, keepdims=True)
        expected_norm = expected/np.sum(expected, axis=1, keepdims=True)
    elif transition_prob_kind=="all":
        data_norm = data/np.sum(data.values.flatten())
        expected_norm = expected/np.sum(expected.flatten())
    else:
        assert False

    ############## PLOTS

    ### Plot heatmaps
    SIZE = 4
    fig, axes = plt.subplots(1, 4, figsize=(5*SIZE, SIZE))
    
    ax = axes.flatten()[0]
    ax.set_title("data")
    sns.heatmap(data, ax=ax)

    ax = axes.flatten()[1]
    ax.set_title("expected")
    sns.heatmap(expected, ax=ax)


    ax = axes.flatten()[2]
    ax.set_title("data_norm")
    sns.heatmap(data_norm, ax=ax)

    ax = axes.flatten()[3]
    ax.set_title("expected_norm")
    sns.heatmap(expected_norm, ax=ax)

    savefig(fig, f"{savedir}/heatmaps-all.pdf")

    from pythonlib.tools.snstools import heatmap
    fig, ax = plt.subplots()
    data_min_expected = data_norm - expected_norm
    zlims = None
    heatmap(data_min_expected, ax, annotate_heatmap=False, zlims=zlims,
                                    diverge=True, norm_method=None)
    ax.set_xlabel("shape to")
    ax.set_ylabel("shape from")

    savefig(fig, f"{savedir}/heatmap-data_min_expected.pdf")
    
    
    ############## Graphs
    # Prune to a smaller matrix that maximizes deviations from expectstion
    n_shapes_keep = 4

    # Compute the sum of absolute weights for each row and column
    row_sums = data_min_expected.abs().sum(axis=1)
    col_sums = data_min_expected.abs().sum(axis=0)

    # Identify the top 5 rows and columns with the largest absolute weights
    top_rows = row_sums.nlargest(n_shapes_keep).index
    top_columns = col_sums.nlargest(n_shapes_keep).index

    # Subset the DataFrame to only include these rows and columns
    data_min_expected_top = data_min_expected.loc[top_rows, top_columns]

    ### Plot result
    zlims = None
    fig, axes = plt.subplots(1,2)

    ax = axes.flatten()[0]
    heatmap(data_min_expected, ax, annotate_heatmap=False, zlims=zlims,
                                    diverge=True, norm_method=None)
    ax.set_xlabel("shape to")
    ax.set_ylabel("shape from")

    ax = axes.flatten()[1]
    heatmap(data_min_expected_top, ax, annotate_heatmap=False, zlims=zlims,
                                    diverge=True, norm_method=None)
    ax.set_xlabel("shape to")
    ax.set_ylabel("shape from")

    plt.tight_layout()
    savefig(fig, f"{savedir}/heatmaps-pruned_to_high_effect.pdf")

    ### Graph plots
    # Get strokes data for each shape
    dfbasis, list_strok_basis, list_shape_basis = DS.stroke_shape_cluster_database_load_helper()
    from pythonlib.drawmodel.tokens import MAP_SHAPE_TO_SHAPESEMANTIC, MAP_SHAPESEM_TO_SHAPESEMGROUP
    def map_shape_to_shapesemgrp(shape):
        shapesem = MAP_SHAPE_TO_SHAPESEMANTIC[shape]
        shapesemgrp = MAP_SHAPESEM_TO_SHAPESEMGROUP[shapesem]
        return shapesemgrp
    if var_shape == "shape_semantic_grp":
        map_node_to_inset_plot_pts = {map_shape_to_shapesemgrp(sh):(strok[:,0], strok[:,1]) for sh, strok in zip(list_shape_basis, list_strok_basis)}
    else:
        map_node_to_inset_plot_pts = {sh:(strok[:,0], strok[:,1]) for sh, strok in zip(list_shape_basis, list_strok_basis)}

    for suffix, df in zip(
        ["fullmatrix", "subset"], [data_min_expected, data_min_expected_top]):
        threshold = 0.
        fig = fig3_charsyntax_transition_matrix_graph_plot(df, threshold = threshold, 
                                                           map_node_to_inset_plot_pts=map_node_to_inset_plot_pts)
        savefig(fig, f"{savedir}/graph-{suffix}.pdf")
    
    plt.close("all")


def fig3_charsyntax_transition_matrix_graph_plot(df, threshold, map_node_to_inset_plot_pts=None,
                                      with_labels=True, alpha=0.9):
    """
    Make directed graph plot, showing eges between df, assuming that each row is a single "from", with
    multipel "to".
    PARAMS:
    - df, 2d.
    - threshold, any edge below this are colored default gray and thin.
    - map_node_to_inset_plot_pts,  map from node (e.g., string label in df), (xs, ys) where xs is a 
    (n,) array of x coordiates. Will plot overlay on each node the curve (xs, ys), with ball at its onset.
    """
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt

    color_inset_stroke = "w"
    color_node = "k"
    # color_inset_stroke = "w"
    # color_node = "k"
    diverging_cmap = plt.cm.coolwarm

    # Update node labels using DataFrame rows and columns
    G = nx.DiGraph()

    # Add edges with weights and use DataFrame index and columns as labels
    for i, row in df.iterrows():
        for j, value in row.items():
            G.add_edge(i, j, weight=value)

    # Normalize edge weights for graph visualization
    weights = nx.get_edge_attributes(G, 'weight')
    max_weight = max(weights.values())
    min_weight = min(weights.values())


    edge_colors = []
    edge_widths = []
    for edge in G.edges():
        weight = weights[edge]
        if abs(weight) >= threshold:
            # Map significant weights to a diverging colormap
            normalized_weight = (weight - min_weight) / (max_weight - min_weight)
            edge_colors.append(diverging_cmap(normalized_weight))
            edge_widths.append(2 + 6 * (abs(weight) - threshold) / (max_weight - threshold))
        else:
            # Default style for insignificant weights
            edge_colors.append("gray")
            edge_widths.append(0.4)

    # Example x and y coordinate data for insets for each node
    # node_inset_data = {
    #     node: (np.random.rand(10), np.random.rand(10)) for node in G.nodes()
    # }

    # Function to rescale data to fit within a node
    def rescale_coordinates(x, y, scale=1):
        """
        Rescale coordinates to fit within the node.
        Parameters:
            x, y: Original coordinates.
            scale: Scaling factor to control the size of the inset relative to the node.
        Returns:
            Rescaled x and y coordinates.
        """
        x = (x - np.min(x)) / (np.max(x) - np.min(x))  # Normalize to [0, 1]
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        x = scale * (x - 0.5)  # Scale to [-scale/2, scale/2]
        y = scale * (y - 0.5)
        return x, y

    # Function to draw rescaled insets on nodes
    # def draw_rescaled_node_overlays(ax, pos, node_data):
    #     for node, (x, y) in pos.items():
    #         # Rescale the inset data for the current node
    #         inset_x, inset_y = rescale_coordinates(*map_node_to_inset_plot_pts[node])
            
    #         # Add the rescaled inset plot
    #         inset_ax = ax.inset_axes([x - 0.03, y - 0.03, 0.06, 0.06], transform=ax.transData)
    #         inset_ax.plot(inset_x, inset_y, 'o-', markersize=2, color='purple')
    #         inset_ax.set_xlim(-0.05, 0.05)
    #         inset_ax.set_ylim(-0.05, 0.05)
    #         inset_ax.axis('off')

            
    # Adjust scaling factor to make insets larger relative to the nodes
    def draw_rescaled_node_overlays_larger(ax, pos, node_data, scale=0.2):
        """
        Draw rescaled insets on nodes with increased size.
        Parameters:
            ax: Matplotlib axis to draw on.
            pos: Node positions.
            node_data: Data for the insets.
            scale: Size of the inset relative to the node.
        """
        for node, (x, y) in pos.items():
            # Rescale the inset data for the current node
            inset_x, inset_y = rescale_coordinates(*map_node_to_inset_plot_pts[node], scale=scale)
            
            # Add the rescaled inset plot
            inset_ax = ax.inset_axes([x - scale / 2, y - scale / 2, scale, scale], transform=ax.transData)
            inset_ax.plot(inset_x, inset_y, '-', markersize=3, color=color_inset_stroke)
            inset_ax.plot(inset_x[0], inset_y[0], 'o', markersize=4, color=color_inset_stroke)
            inset_ax.set_xlim(-1.1*scale / 2, 1.1*scale / 2)
            inset_ax.set_ylim(-1.1*scale / 2, 1.1*scale / 2)
            inset_ax.axis('off')

    # Draw the graph again with larger insets
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=with_labels,
        labels={node: node for node in G.nodes()},  # Use DataFrame labels
        node_color=color_node,
        # node_color="none",
        edge_color=edge_colors,
        width=edge_widths,
        alpha=alpha,
        edge_cmap=diverging_cmap,
        node_size=500,
        font_size=10,
        ax=ax
    )

    # Overlay the larger rescaled plots on the nodes
    draw_rescaled_node_overlays_larger(ax, pos, G.nodes(), scale=0.1)

    # Add color bar and title
    sm = plt.cm.ScalarMappable(cmap=diverging_cmap, norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
    sm.set_array([])  
    plt.colorbar(sm, label="Significant Edge Weight (Diverging)")
    ax.set_title("Directed Graph with Larger Rescaled Node Inset Plots")
    
    return fig

def fig2_categ_extract_dist_scores(DSmorphsets, SAVEDIR, cetegory_expt_version="switching"):
    """
    Extract pairiwse distances for all morphsets in DSmorphsets.Dat

    Ensures that those morphsets that are "switching" wil have accurate label for each index within
    (e.g., ambig, not-ambig). Does not guarantee that for "smototh" but that's ok since there
    I don't use this label info.

    """
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.dataset.scripts.analy_manuscript_figures import score_motor_distances
    from pythonlib.tools.stroketools import get_centers_strokes_list, strokes_centerize
    from pythonlib.tools.pandastools import stringify_values
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good

    ### Extract task strokes
    inds = list(range(len(DSmorphsets.Dat)))
    DSmorphsets.Dat["strok_task"]= DSmorphsets.extract_strokes(inds=inds, ver_behtask="task_entire")
    if cetegory_expt_version == "switching":
        DSmorphsets.Dat = append_col_with_grp_index(DSmorphsets.Dat, ["morph_idxcode_within_set", "assigned_base_simple"], 
                                                    "idxmorph_assigned", False) # To make this tuple, same as in Cl below.

    ### Collect distances
    DEBUG = False
    list_df = []
    list_df_index = []
    for morphset in DSmorphsets.Dat["morph_set_idx"].unique():
        dfdat = DSmorphsets.Dat[DSmorphsets.Dat["morph_set_idx"]==morphset].reset_index(drop=True)
        
        # Need at least 2 trials, or else distances will fail
        if cetegory_expt_version == "switching":
            dfdat, _ = extract_with_levels_of_var_good(dfdat, ["morph_idxcode_within_set", "assigned_base_simple"], 2)
        elif cetegory_expt_version == "smooth":
            dfdat, _ = extract_with_levels_of_var_good(dfdat, ["morph_idxcode_within_set"], 2)
        else:
            assert False

        for version in ["beh", "beh_imagedist", "task"]:
            savedir = f"{SAVEDIR}/extraction/morphset={morphset}-ver={version}"
            os.makedirs(savedir, exist_ok=True)

            # label
            if cetegory_expt_version == "switching":
                label_var = ["morph_idxcode_within_set", "assigned_base_simple"]
            elif cetegory_expt_version == "smooth":
                label_var = ["morph_idxcode_within_set"]
            else:
                assert False
            labels = [tuple(x) for x in dfdat.loc[:, label_var].values.tolist()]

            if version == "beh":
                strokes = dfdat["strok"].tolist()
                if DEBUG:
                    strokes = strokes[::2]
                    labels = labels[::2]
                Cl = DSmorphsets.distgood_compute_beh_beh_strok_distances(strokes, strokes, None, labels, labels, label_var, 
                                                                        clustclass_rsa_mode=True, PLOT=True, 
                                                                        savedir=savedir, savesuff=version)
                # Cl = score_motor_distances(DSmorphsets, strokes, strokes, labels, labels, "/tmp", "TEST", label_var=label_var)
            elif version == "task":
                strokes = dfdat["strok_task"].tolist()
                strokes = strokes_centerize(strokes, method="bounding_box")
                if DEBUG:
                    strokes = strokes[::2]
                    labels = labels[::2]
                Cl = DSmorphsets.distgood_compute_image_strok_distances(strokes, strokes, labels, labels, label_var,
                                                                do_centerize=True, clustclass_rsa_mode=True,
                                                                PLOT=True, savedir=savedir, savesuff=version)
            elif version == "beh_imagedist":
                strokes = dfdat["strok"].tolist()
                strokes = strokes_centerize(strokes, method="bounding_box")
                if DEBUG:
                    strokes = strokes[::2]
                    labels = labels[::2]
                Cl = DSmorphsets.distgood_compute_image_strok_distances(strokes, strokes, labels, labels, label_var,
                                                                do_centerize=True, clustclass_rsa_mode=True,
                                                                PLOT=True, savedir=savedir, savesuff=version)
            else:
                assert False

            # Plot example strokes
            inds = sorted(set(labels))
            strokes_this = [strokes[labels.index(i)] for i in inds]
            for overlay in [False, True]:
                fig, _ = DSmorphsets.plot_multiple_strok(strokes_this, overlay=overlay, titles=inds)
                savefig(fig, f"{savedir}/example_drawings-overlay={overlay}.pdf")
            plt.close("all")

            ### Compute distances
            dfdists = Cl.rsa_distmat_score_all_pairs_of_label_groups_datapts()
            dfproj_index = Cl.rsa_dfdist_to_dfproj_index_datapts(dfdists, var_effect="morph_idxcode_within_set", 
                                                                effect_lev_base1=0, effect_lev_base2=99)
            
            # Condition the data, i.e,, adding labels as newc olumns.
            if cetegory_expt_version == "switching":
                from pythonlib.dataset.scripts.analy_manuscript_figures import fig2_categ_switching_condition_dfdists
                fig2_categ_switching_condition_dfdists(dfdat, dfdists, dfproj_index)

            ### Plot
            if cetegory_expt_version == "switching":
                list_x_var = ["labels_1_datapt", "morph_idxcode_within_set", "assigned_base_simple"]
            elif cetegory_expt_version == "smooth":
                list_x_var = ["morph_idxcode_within_set"]
            else:
                assert False

            from pythonlib.tools.snstools import rotateLabel
            dfdists_str = stringify_values(dfdists)
            for x_var in list_x_var:
                x_order = sorted(dfdists_str[x_var].unique())
                
                fig = sns.catplot(data=dfdists_str, x=x_var, y="dist_mean", col="labels_2_grp", order=x_order)
                rotateLabel(fig)
                savefig(fig, f"{savedir}/catplot-dfdists-x={x_var}-1.pdf")
                
                fig = sns.catplot(data=dfdists_str, x=x_var, y="dist_mean", col="labels_2_grp", order=x_order, kind="point", errorbar=("ci", 68))
                rotateLabel(fig)
                savefig(fig, f"{savedir}/catplot-dfdists-x={x_var}-2.pdf")

            dfproj_index_str = stringify_values(dfproj_index)
            for x_var in ["labels_1_datapt", "morph_idxcode_within_set"]:
                x_order = sorted(dfproj_index_str[x_var].unique())
                
                fig = sns.catplot(data=dfproj_index_str, x=x_var, y="dist_index", order=x_order)
                rotateLabel(fig)
                savefig(fig, f"{savedir}/catplot-dfproj_index-x={x_var}-1.pdf")
                
                fig = sns.catplot(data=dfproj_index_str, x=x_var, y="dist_index", order=x_order, kind="point", errorbar=("ci", 68))
                rotateLabel(fig)
                savefig(fig, f"{savedir}/catplot-dfproj_index-x={x_var}-2.pdf")
            plt.close("all")

            ## Collect results
            dfdists["morphset"] = morphset
            dfdists["version"] = version
            dfproj_index["morphset"] = morphset
            dfproj_index["version"] = version

            list_df.append(dfdists)
            list_df_index.append(dfproj_index)

    DFDISTS = pd.concat(list_df).reset_index(drop=True)
    DFINDEX = pd.concat(list_df_index).reset_index(drop=True)
    pd.to_pickle(DFDISTS, f"{SAVEDIR}/DFDISTS.pkl")
    pd.to_pickle(DFINDEX, f"{SAVEDIR}/DFINDEX.pkl")

    return DFDISTS, DFINDEX

def fig2_categ_switching_condition_dfdists(ds_dat, dfdists, dfproj_index):
    """
    Run this separately for each morphset.
    PARAMS:
    - ds_dat, slice of DSmorphsets for this morphset.
    RETURNS:
    - modifies dfdists and dfproj_index, adding columns.
    """

    #  Map from idx|assign to label
    # idxmorph_assigned = (1, base1)...
    
    #################### ADD LABELS
    map_idxassign_to_label = {}
    map_idxassign_to_assignedbase = {}
    # map_idxassign_to_assignedbase_simple = {}
    # map_idxassign_to_idx_morph = {}
    for i, row in ds_dat.iterrows():
        if row["idxmorph_assigned"] not in map_idxassign_to_label:
            map_idxassign_to_label[row["idxmorph_assigned"]] = row["morph_assigned_label"]
            map_idxassign_to_assignedbase[row["idxmorph_assigned"]] = row["morph_assigned_to_which_base"]

            # map_idxassign_to_assignedbase_simple[row["idxmorph_assigned"]] = row["assigned_base_simple"]
            # map_idxassign_to_idx_morph[row["idxmorph_assigned"]] = row["idx_morph_temp"]
        else:
            assert map_idxassign_to_label[row["idxmorph_assigned"]] == row["morph_assigned_label"]
            assert map_idxassign_to_assignedbase[row["idxmorph_assigned"]] == row["morph_assigned_to_which_base"]

            # assert map_idxassign_to_assignedbase_simple[row["idxmorph_assigned"]] == row["assigned_base_simple"]
            # assert map_idxassign_to_idx_morph[row["idxmorph_assigned"]] == row["idx_morph_temp"]

    for df in [dfdists, dfproj_index]:
        # df["assigned_base_simple"] = [map_idxassign_to_assignedbase_simple[x] for x in df["idxmorph_assigned"]]
        df["morph_assigned_to_which_base"] = [map_idxassign_to_assignedbase[x] for x in df["labels_1_datapt"]] # (base, ambig, notambig)
        df["morph_assigned_label"] = [map_idxassign_to_label[x] for x in df["labels_1_datapt"]] # (base1, ambig1, ..., base2)
        # df["idx_morph_temp"] = [map_idxassign_to_idx_morph[x] for x in df["idxmorph_assigned"]]

if __name__=="__main__":
    import sys

    PLOTS_DO = [4.2]
    
    ###
    for plot_do in PLOTS_DO:
        if plot_do==1:
            # Show that subjects learn distinct prims, by comparison to other prims, and across subejcts, etc.

            # All dates for Pancho that have one location/size (and another day, 240509, just in case others dont work)
            pancho_date_idx = int(sys.argv[1])
            list_pancho_dates = [240509, 220718, 221217]

            ### Learned prims
            # The only date for Diego without loc/size variation
            animal1="Diego"
            date1=230616
            animal2="Pancho"
            date2 = list_pancho_dates[pancho_date_idx]

            # ENtire pipeline to compute pairwise distances, beh, image, etc.
            # Then, to make final plots, go to:
            # - notebook: /home/lucas/code/drawmonkey/notebooks_datasets/240912_MANUSCRIPT_FIGURES_1_shapes.ipynb
            # - "After running pipeline, reload and flexibly make final plots here"
            fig1_learned_prims_wrapper(animal1, date1, animal2, date2)
        elif plot_do==2:
            # SHow that shapes are invariant to (location, size).

            animal = "Diego"
            DATE = 240530

            D = load_dataset_daily_helper(animal, DATE)
            SAVEDIR = f"/lemur2/lucas/analyses/manuscripts/1_action_symbols/fig2_beh_abstraction"
            savedir = f"{SAVEDIR}/distance_matrix_2/{animal}-{DATE}"
            os.makedirs(savedir, exist_ok=True)

            ### Distance matrix between (shapes, locations, sizes)
            from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
            DS = preprocess_dataset_to_datstrokes(D, "singleprim") # get all strokes.
            N_TRIALS = 8
            fig1_motor_invariance(DS, N_TRIALS, SAVEDIR, False)
            
        elif plot_do==3:
            # Syntax, in characters tasks. Plots all.
            animal = sys.argv[1]
            date = sys.argv[2]
            SAVEDIR = f"/lemur2/lucas/analyses/manuscripts/1_action_symbols/fig3_char_syntax/{animal}-{date}"
            os.makedirs(SAVEDIR, exist_ok=True)

            fig3_charsyntax_wrapper(animal, date, SAVEDIR)
        
        elif plot_do==4.1:
            # Categories (switching), plotting motor distances of each index vs. endpoint (base prims), and doing
            # analyses/stats of that.
            # - To make final (MULT) plots, load the results from here and make plots, using notebook, section:
            # "### [Categorization, switching] Summary plots", notebook:
            # /home/lucas/code/drawmonkey/notebooks_datasets/240912_MANUSCRIPT_FIGURES_1_shapes.ipynb

            ### Load a daily dataset
            animal = sys.argv[1]
            DATE = sys.argv[2]
            SAVEDIR = f"/lemur2/lucas/analyses/manuscripts/1_action_symbols/fig2_categorization/{animal}-{DATE}"

            D = load_dataset_daily_helper(animal, DATE)

            ### Preprocess
            from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_preprocess_wrapper_GOOD
            savedir = f"{SAVEDIR}/preprocess"
            os.makedirs(savedir, exist_ok=True)
            NEURAL_PLOT_DRAWINGS = False
            DSmorphsets, map_tc_to_morph_info, map_morphset_to_basemorphinfo, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info, map_morphsetidx_to_assignedbase_or_ambig, map_tc_to_morph_status = psychogood_preprocess_wrapper_GOOD(D, 
                                                                                                                                                                                                                            NEURAL_VERSION=True, 
                                                                                                                                                                                                                            NEURAL_SAVEDIR=savedir,
                                                                                                                                                                                                                            NEURAL_PLOT_DRAWINGS=NEURAL_PLOT_DRAWINGS)
            
            from pythonlib.dataset.scripts.analy_manuscript_figures import fig2_categ_extract_dist_scores
            DFDISTS, DFINDEX = fig2_categ_extract_dist_scores(DSmorphsets, SAVEDIR)

        elif plot_do==4.2:
            # Categories (smooth), plotting motor distances of each index vs. endpoint (base prims), and doing
            # analyses/stats of that

            cetegory_expt_version = "smooth"

            ### Load a daily dataset
            animal = sys.argv[1]
            DATE = sys.argv[2]
            SAVEDIR = f"/lemur2/lucas/analyses/manuscripts/1_action_symbols/fig2_categorization/smooth/{animal}-{DATE}"

            D = load_dataset_daily_helper(animal, DATE)

            ### Preprocess
            from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_preprocess_wrapper_GOOD
            savedir = f"{SAVEDIR}/preprocess"
            os.makedirs(savedir, exist_ok=True)
            NEURAL_PLOT_DRAWINGS = False
            DSmorphsets, map_tc_to_morph_info, map_morphset_to_basemorphinfo, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info, map_morphsetidx_to_assignedbase_or_ambig, map_tc_to_morph_status = psychogood_preprocess_wrapper_GOOD(D, 
                                                                                                                                                                                                                            NEURAL_VERSION=True, 
                                                                                                                                                                                                                            NEURAL_SAVEDIR=savedir,
                                                                                                                                                                                                                            NEURAL_PLOT_DRAWINGS=NEURAL_PLOT_DRAWINGS)
            
            from pythonlib.dataset.scripts.analy_manuscript_figures import fig2_categ_extract_dist_scores
            DFDISTS, DFINDEX = fig2_categ_extract_dist_scores(DSmorphsets, SAVEDIR, cetegory_expt_version=cetegory_expt_version)

        else:
            assert False