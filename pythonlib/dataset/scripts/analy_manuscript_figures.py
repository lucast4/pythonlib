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


def load_pair_of_datasets(animal1, date1, animal2, date2, strokes_keep_only_main_prims=False):
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
    DS1.stroke_database_prune_to_basis_prims(keep_only_main_21=strokes_keep_only_main_prims)
    DS2.stroke_database_prune_to_basis_prims(keep_only_main_21=strokes_keep_only_main_prims)

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
    Given computed distances (in LIST_TRIAL_CL), make various plots
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
        if dist_ver in DFDIST["dist_ver"].unique().tolist():
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

    print("HERE", len(DFDIST))
    DFDIST["same_animal"] = DFDIST["animal_pair"].isin(["0|0", "1|1"])
    DFDIST["same_shape"] = DFDIST["shape_1"] == DFDIST["shape_2"]

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
                          label_var="shape", plot=True):
    """
    Score pairwise stroke trajectory distances between all pairs comparing strokes1 and strokes2
    """
    print("Scoring motor distance....")

    # Compute similarity
    Cl = DSthis.distgood_compute_beh_beh_strok_distances(strokes1, strokes2, labels_rows_dat=shapes1, 
                                                         labels_cols_feats=shapes2, label_var=label_var, 
                                                         clustclass_rsa_mode=True,
                                                         PLOT=plot, savedir=savedir, savesuff=savesuff,
                                                         invert_score=True)
    return Cl

def score_motor_distances_OLD(DSthis, strokes1, strokes2, shapes1, shapes2, savedir, savesuff,
                          label_var="shape"):
    """
    Score pairwise stroke trajectory distances between all pairs comparing strokes1 and strokes2
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
                                                         PLOT=True, savedir=savedir, savesuff=savesuff,
                                                         invert_score=True)
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


def fig1_score_pairwise_distances_motor(DS1, DS2, SAVEDIR, N_TRIALS = 10, plot=True):
    """
    Score all pairwise distances between strokes, stored in DS1, DS2, where these are datasets
    for two subjects (usually). Gets both within and across subejcts.
    """
    from pythonlib.tools.pandastools import extract_trials_spanning_variable

    savedir = f"{SAVEDIR}/raw_heatmaps"
    os.makedirs(savedir, exist_ok=True)

    # Helper functions
    ds = DS1 # just for the methods.

    def _score_motor_distances(strokes1, strokes2, shapes1, shapes2, savesuff):
        """
        """
        print("Scoring motor distance....")
        Cl = score_motor_distances(ds, strokes1, strokes2, shapes1, shapes2, savedir, savesuff, plot=plot)
        return Cl

    ### Compute all pairwise distances between strokes.
    LIST_TRIAL_CL = []

    ### (1) Score across animals    
    # First, get common shapes, and get n trials subsampling each shape
    shapes_common = sorted(set(DS1.Dat[DS1.Dat["shape"].isin(DS2.Dat["shape"].tolist())]["shape"]))

    inds, _ = extract_trials_spanning_variable(DS1.Dat, "shape", varlevels=shapes_common, n_examples=N_TRIALS)
    strokes1 = DS1.Dat.iloc[inds]["strok"].tolist()
    shapes1 = DS1.Dat.iloc[inds]["shape"].tolist()

    inds, _ = extract_trials_spanning_variable(DS2.Dat, "shape", varlevels=shapes_common, n_examples=N_TRIALS)
    strokes2 = DS2.Dat.iloc[inds]["strok"].tolist()
    shapes2 = DS2.Dat.iloc[inds]["shape"].tolist()

    assert shapes1 == shapes2, "then below will fail -- this is hacky, converting to rsa, which expects lables row and col to be identical. solve by updating code to also work with assymetric"
    
    # Second, do scoring
    Cl = _score_motor_distances(strokes1, strokes2, shapes1, shapes2, "01")
    LIST_TRIAL_CL.append({
        "animal_pair":(0,1),
        "dist_ver":"motor",
        "Cl":Cl
        })

    ### (2) Scores distances within each animal
    for i, DSthis in enumerate([DS1, DS2]):

        inds, _ = extract_trials_spanning_variable(DSthis.Dat, "shape", n_examples=N_TRIALS)
        strokes = DSthis.Dat.iloc[inds]["strok"].tolist()
        strokes_task = DSthis.extract_strokes(inds=inds, ver_behtask="task")
        shapes = DSthis.Dat.iloc[inds]["shape"].tolist()

        ### (2) Motor distance (beh-beh)
        if "motor" in which_distances:
            Cl = _score_motor_distances(strokes, strokes, shapes, shapes, i)
            LIST_TRIAL_CL.append({
                "animal_pair":(i,i),
                "dist_ver":"motor",
                "Cl":Cl
                })
        
        plt.close("all")

    ## Save the data 
    import pickle
    with open(f"{SAVEDIR}/LIST_TRIAL_CL.pkl", "wb") as f:
        pickle.dump(LIST_TRIAL_CL, f)

    return LIST_TRIAL_CL

def fig1_score_pairwise_distances(DS1, DS2, SAVEDIR, N_TRIALS = 10, which_distances=None):
    """
    Score all pairwise distances between strokes, stored in DS1, DS2, where these are datasets
    for two subjects (usually). Gets both within and across subejcts.
    """
    from pythonlib.tools.pandastools import extract_trials_spanning_variable

    if which_distances is None:
        which_distances = ["image", "motor", "image_beh_task"]

    savedir = f"{SAVEDIR}/raw_heatmaps"
    os.makedirs(savedir, exist_ok=True)

    # Helper functions
    ds = DS1 # just for the methods.
    def _score_image_distances(strokes1, strokes2, shapes1, shapes2, savesuff):
        """
        """
        print("Scoring image distance....")

        Cl = ds.distgood_compute_image_strok_distances(strokes1, strokes2, shapes1, shapes2, do_centerize=True,
                                                            clustclass_rsa_mode=True, PLOT=True, savedir=savedir, savesuff=savesuff)
        return Cl

    def _score_motor_distances(strokes1, strokes2, shapes1, shapes2, savesuff):
        """
        """
        print("Scoring motor distance....")
        Cl = score_motor_distances(ds, strokes1, strokes2, shapes1, shapes2, savedir, savesuff)
        return Cl

    ### Compute all pairwise distances between strokes.
    LIST_TRIAL_CL = []


    ### (1) Score across animals    
    # First, get common shapes, and get n trials subsampling each shape
    shapes_common = sorted(set(DS1.Dat[DS1.Dat["shape"].isin(DS2.Dat["shape"].tolist())]["shape"]))

    inds, _ = extract_trials_spanning_variable(DS1.Dat, "shape", varlevels=shapes_common, n_examples=N_TRIALS)
    strokes1 = DS1.Dat.iloc[inds]["strok"].tolist()
    shapes1 = DS1.Dat.iloc[inds]["shape"].tolist()

    inds, _ = extract_trials_spanning_variable(DS2.Dat, "shape", varlevels=shapes_common, n_examples=N_TRIALS)
    strokes2 = DS2.Dat.iloc[inds]["strok"].tolist()
    shapes2 = DS2.Dat.iloc[inds]["shape"].tolist()

    assert shapes1 == shapes2, "then below will fail -- this is hacky, converting to rsa, which expects lables row and col to be identical. solve by updating code to also work with assymetric"
    
    # Second, do scoring
    if "image" in which_distances:
        ### (1) Image distance
        Cl = _score_image_distances(strokes1, strokes2, shapes1, shapes2, "01")
        LIST_TRIAL_CL.append({
            "animal_pair":(0,1),
            "dist_ver":"image",
            "Cl":Cl
            })    
    
    if "motor" in which_distances:
        ### (2) Motor distance
        Cl = _score_motor_distances(strokes1, strokes2, shapes1, shapes2, "01")
        LIST_TRIAL_CL.append({
            "animal_pair":(0,1),
            "dist_ver":"motor",
            "Cl":Cl
            })

    ### (2) Scores distances within each animal
    for i, DSthis in enumerate([DS1, DS2]):

        inds, _ = extract_trials_spanning_variable(DSthis.Dat, "shape", n_examples=N_TRIALS)
        strokes = DSthis.Dat.iloc[inds]["strok"].tolist()
        strokes_task = DSthis.extract_strokes(inds=inds, ver_behtask="task")
        shapes = DSthis.Dat.iloc[inds]["shape"].tolist()

        ### (1) Image distance (beh-beh)
        if "image" in which_distances:
            Cl = _score_image_distances(strokes, strokes, shapes, shapes, i)
            LIST_TRIAL_CL.append({
                "animal_pair":(i,i),
                "dist_ver":"image",
                "Cl":Cl
                })    

        ### (2) Motor distance (beh-beh)
        if "motor" in which_distances:
            Cl = _score_motor_distances(strokes, strokes, shapes, shapes, i)
            LIST_TRIAL_CL.append({
                "animal_pair":(i,i),
                "dist_ver":"motor",
                "Cl":Cl
                })
        
        ### (3) Image distance (beh-task)
        if "image_beh_task" in which_distances:
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

    return LIST_TRIAL_CL

def fig1_learned_prims_wrapper(animal1="Diego", date1=230616, animal2="Pancho", date2=240509, SAVEDIR=None,
                               resave_datasets=False, strokes_keep_only_main_prims=False, manuscript_version=False):
    """
    Entire pipeline
    """
    ### Load dataset
    # animal1="Diego"
    # date1=230616
    # animal2="Pancho"
    # date2=240509
    _, _, DS1, DS2, _SAVEDIR = load_pair_of_datasets(animal1, date1, animal2, date2, strokes_keep_only_main_prims=strokes_keep_only_main_prims)
    if SAVEDIR is None:
        SAVEDIR = _SAVEDIR

    if resave_datasets:
        DS1.save(SAVEDIR, filename="DS1", manuscript_version=manuscript_version)
        DS2.save(SAVEDIR, filename="DS2", manuscript_version=manuscript_version)


    LIST_TRIAL_CL = fig1_score_pairwise_distances(DS1, DS2, SAVEDIR)

    # DSthis = DS1
    # savedir = f"{SAVEDIR}/raw_heatmaps"
    # os.makedirs(savedir, exist_ok=True)

    # def _score_image_distances(strokes1, strokes2, shapes1, shapes2, savesuff):
    #     """
    #     """
    #     print("Scoring image distance....")

    #     Cl = DSthis.distgood_compute_image_strok_distances(strokes1, strokes2, shapes1, shapes2, do_centerize=True,
    #                                                         clustclass_rsa_mode=True, PLOT=True, savedir=savedir, savesuff=savesuff)
    #     # Cl = Cl.convert_copy_to_rsa_dist_version("shape", "image")


    #     # if N_TRIALS<12:
    #     #     fig, _ = Cl.rsa_plot_heatmap()
    #     #     if fig is not None:
    #     #         savefig(fig, f"{savedir}/TRIAL_DISTS-image-{savesuff}.pdf")  

    #     # ### Aggregate to get distance between groups (shapes)
    #     # _, Clagg = Cl.rsa_distmat_score_all_pairs_of_label_groups(return_as_clustclass=True, return_as_clustclass_which_var_score="dist_mean")

    #     # # Cl.rsa_distmat_score_all_pairs_of_label_groups_datapts
    #     # fig, _ = Clagg.rsa_plot_heatmap()
    #     # savefig(fig, f"{savedir}/TRIAL_DISTS_mean_over_shapes-image-{savesuff}.pdf")  

    #     return Cl

    # def _score_motor_distances(strokes1, strokes2, shapes1, shapes2, savesuff):
    #     """
    #     """
    #     print("Scoring motor distance....")
    #     Cl = score_motor_distances(DSthis, strokes1, strokes2, shapes1, shapes2, savedir, savesuff)

    #     return Cl


    # ### Compute all pairwise distances between strokes.
    # from pythonlib.tools.pandastools import extract_trials_spanning_variable

    # N_TRIALS = 10
    # LIST_TRIAL_CL = []

    # ### Also get distnace across animals
    # # First, get common shapes
    # shapes_common = sorted(set(DS1.Dat[DS1.Dat["shape"].isin(DS2.Dat["shape"].tolist())]["shape"]))

    # inds, _ = extract_trials_spanning_variable(DS1.Dat, "shape", varlevels=shapes_common, n_examples=N_TRIALS)
    # strokes1 = DS1.Dat.iloc[inds]["strok"].tolist()
    # shapes1 = DS1.Dat.iloc[inds]["shape"].tolist()

    # inds, _ = extract_trials_spanning_variable(DS2.Dat, "shape", varlevels=shapes_common, n_examples=N_TRIALS)
    # strokes2 = DS2.Dat.iloc[inds]["strok"].tolist()
    # shapes2 = DS2.Dat.iloc[inds]["shape"].tolist()

    # assert shapes1 == shapes2, "then below will fail -- this is hacky, converting to rsa, which expects lables row and col to be identical. solve by updating code to also work with assymetric"

    # ### (1) Image distance
    # Cl = _score_image_distances(strokes1, strokes2, shapes1, shapes2, "01")
    # LIST_TRIAL_CL.append({
    #     "animal_pair":(0,1),
    #     "dist_ver":"image",
    #     "Cl":Cl
    #     })    
        
    # ### (2) Motor distance
    # Cl = _score_motor_distances(strokes1, strokes2, shapes1, shapes2, "01")
    # LIST_TRIAL_CL.append({
    #     "animal_pair":(0,1),
    #     "dist_ver":"motor",
    #     "Cl":Cl
    #     })

    # # Get distances within each animal
    # for i, DSthis in enumerate([DS1, DS2]):

    #     inds, _ = extract_trials_spanning_variable(DSthis.Dat, "shape", n_examples=N_TRIALS)
    #     strokes = DSthis.Dat.iloc[inds]["strok"].tolist()
    #     strokes_task = DSthis.extract_strokes(inds=inds, ver_behtask="task")
    #     shapes = DSthis.Dat.iloc[inds]["shape"].tolist()

    #     ### (1) Image distance (beh-beh)
    #     Cl = _score_image_distances(strokes, strokes, shapes, shapes, i)
    #     LIST_TRIAL_CL.append({
    #         "animal_pair":(i,i),
    #         "dist_ver":"image",
    #         "Cl":Cl
    #         })    

    #     ### (2) Motor distance (beh-beh)
    #     Cl = _score_motor_distances(strokes, strokes, shapes, shapes, i)
    #     LIST_TRIAL_CL.append({
    #         "animal_pair":(i,i),
    #         "dist_ver":"motor",
    #         "Cl":Cl
    #         })
        
    #     ### (3) Image distance (beh-task)
    #     Cl = _score_image_distances(strokes, strokes_task, shapes, shapes, f"beh_task-{i}")
    #     LIST_TRIAL_CL.append({
    #         "animal_pair":(i,i),
    #         "dist_ver":"image_beh_task",
    #         "Cl":Cl
    #         })    

    #     plt.close("all")

    # ## Save the data 
    # import pickle
    # with open(f"{SAVEDIR}/LIST_TRIAL_CL.pkl", "wb") as f:
    #     pickle.dump(LIST_TRIAL_CL, f)


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


def fig1_motor_invariance(DS, N_TRIALS, savedir, DEBUG=False,
                          PLOT_DRAWINGS=True):
    """
    Entire pipeline, to analyze invariance of motor parameters due to loation and size.
    Expt is assumed to have both loc and size vairation.

    Does: Make drawing plots, Compute pairwise dsitances, plot scores.

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
    dfthis, inds_keep = extract_with_levels_of_var_good(DS.Dat, grpvars, min([3, N_TRIALS]))
    DS.Dat = dfthis

    DS.Dat = append_col_with_grp_index(DS.Dat, ["gridloc", "gridsize"], "locsize")
    DS.Dat = append_col_with_grp_index(DS.Dat, grpvars, "shape_loc_size")

    # Drawings
    if PLOT_DRAWINGS:
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

    ### Compute pairwise trajectory distance between strokes (trials).
    from pythonlib.tools.pandastools import extract_trials_spanning_variable, append_col_with_grp_index

    inds, _ = extract_trials_spanning_variable(DS.Dat, "shape_loc_size", n_examples=N_TRIALS)
    strokes = DS.Dat.iloc[inds]["strok"].tolist()
    labels = [tuple(x) for x in DS.Dat.loc[inds, grpvars].values.tolist()]

    print("Scoring motor distance....")
    # list_distance_ver  = ["dtw_vels_2d"]
    Cl = DS.distgood_compute_beh_beh_strok_distances(strokes, strokes, None, labels, labels, grpvars, clustclass_rsa_mode=True, invert_score=True)
    
    # Save, since this takes a long time.
    import pickle
    with open(f"{savedir}/Cl.pkl", "wb") as f:
        pickle.dump(Cl, f)

    ### Plots
    dfdist, dfdist_agg = fig1_motor_invariance_plots(Cl, savedir, plot_heatmaps=True)

    return dfdist, dfdist_agg

def fig1_motor_invariance_OLD(DS, N_TRIALS, savedir, DEBUG=False,
                          PLOT_DRAWINGS=True):
    """
    Entire pipeline, to analyze invariance of motor parameters due to loation and size.
    Expt is assumed to have both loc and size vairation.

    Does: Make drawing plots, Compute pairwise dsitances, plot scores.

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
    dfthis, inds_keep = extract_with_levels_of_var_good(DS.Dat, grpvars, min([3, N_TRIALS]))
    DS.Dat = dfthis

    DS.Dat = append_col_with_grp_index(DS.Dat, ["gridloc", "gridsize"], "locsize")
    DS.Dat = append_col_with_grp_index(DS.Dat, grpvars, "shape_loc_size")

    # Drawings
    if PLOT_DRAWINGS:
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
    Cl = DS.distgood_compute_beh_beh_strok_distances(strokes, strokes, None, labels, labels, grpvars, clustclass_rsa_mode=True, invert_score=True)
    
    # Save, since this takes a long time.
    import pickle
    with open(f"{savedir}/Cl.pkl", "wb") as f:
        pickle.dump(Cl, f)

    ### Plots
    dfdist, dfdist_agg = fig1_motor_invariance_plots(Cl, savedir, plot_heatmaps=True)

    return dfdist, dfdist_agg


def fig1_motor_invariance_plots(Cl, savedir, plot_heatmaps=True):
    """
    After getting Cl, make the invariance plots.
    This is split from code to construct Cl, as the latter is slow.
    Can load pre-saved Cl and then run this from notebook
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.pandastools import grouping_print_n_samples, grouping_plot_n_samples_conjunction_heatmap
    import seaborn as sns

    ### PLOTS
    
    # Plot heatmap
    # - First agg (over trials), or else plots too large
    _, Clagg = Cl.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=False, return_as_clustclass=True,
                                                              return_as_clustclass_which_var_score="dist_mean")

    ### Heatmaps
    if plot_heatmaps:
        for sort_order in [(0,1,2), (1,2, 0), (2,0,1)]:
            fig, ax = Clagg.rsa_plot_heatmap(sort_order=sort_order)
            if fig is not None:
                savefig(fig, f"{savedir}/distmat-sort={sort_order}.pdf")
        for var in Clagg.rsa_labels_extract_label_vars():  
            _, fig = Clagg.rsa_distmat_construct_theoretical(var, True)
            if fig is not None:
                savefig(fig, f"{savedir}/theor_distmat-var={var}.pdf")

    dfdist = Cl.rsa_distmat_score_all_pairs_of_label_groups_datapts()
    dfdist = dfdist.drop("dist_yue_diff", axis=1)
    dfdist = dfdist.dropna()
    dfdist = append_col_with_grp_index(dfdist, ["shape_same", "gridloc_same", "gridsize_same"], "same-shape_loc_size")

    # Get normalized score (0 good, 1 bad)
    score_max = dfdist[dfdist["same-shape_loc_size"]=="1|1|1"]["dist_mean"].mean()
    score_min = dfdist[dfdist["same-shape_loc_size"]=="0|0|0"]["dist_mean"].mean()
    dfdist["dist_mean_norm_v2"] = 1 - (dfdist["dist_mean"] - score_min)/(score_max - score_min)

    # Agg, so that each (shape, loc, size) is single datapt
    dfdist_agg = aggregGeneral(dfdist, ["labels_1_datapt", "same-shape_loc_size"], ["dist_mean", "dist_mean_norm_v2"])
    
    fig = grouping_plot_n_samples_conjunction_heatmap(dfdist_agg, "labels_1_datapt", "same-shape_loc_size")
    savefig(fig, f"{savedir}/counts-dfdist_agg.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(dfdist, "labels_1_datapt", "same-shape_loc_size")
    savefig(fig, f"{savedir}/counts-dfdist.pdf")

    path = f"{savedir}/counts-dfdist.txt"
    grouping_print_n_samples(dfdist, ["labels_1_datapt", "same-shape_loc_size"], savepath=path)        

    # Print sample size   
    path = f"{savedir}/counts_trial.txt"
    grouping_print_n_samples(dfdist, ["same-shape_loc_size"], savepath=path)        
    
    path = f"{savedir}/counts_grp.txt"
    grouping_print_n_samples(dfdist_agg, ["same-shape_loc_size"], savepath=path)        

    ### Plots
    for y in ["dist_mean", "dist_mean_norm_v2"]:
        fig = sns.catplot(data=dfdist, x="same-shape_loc_size", y=y)
        savefig(fig, f"{savedir}/catplot-trials-{y}-1.pdf")

        fig = sns.catplot(data=dfdist, x="same-shape_loc_size", y=y, kind="violin")
        savefig(fig, f"{savedir}/catplot-trials-{y}-2.pdf")

        fig = sns.catplot(data=dfdist, x="same-shape_loc_size", y=y, kind="bar", errorbar="se")
        savefig(fig, f"{savedir}/catplot-trials-{y}-3.pdf")

        fig = sns.catplot(data=dfdist_agg, x="same-shape_loc_size", y=y)
        savefig(fig, f"{savedir}/catplot-grps-{y}-1.pdf")

        fig = sns.catplot(data=dfdist_agg, x="same-shape_loc_size", y=y, kind="violin")
        savefig(fig, f"{savedir}/catplot-grps-{y}-2.pdf")

        fig = sns.catplot(data=dfdist_agg, x="same-shape_loc_size", y=y, kind="bar", errorbar="se")
        savefig(fig, f"{savedir}/catplot-grps-{y}-3.pdf")
        plt.close("all")

    ### Stats
    from pythonlib.tools.statstools import signrank_wilcoxon_from_df
    datapt_vars = ["labels_1_datapt"] # data
    contrast_var = "same-shape_loc_size"
    lev_base = "1|1|1"
    value_var = "dist_mean"

    res = []
    for lev1 in dfdist_agg["same-shape_loc_size"].unique():
        for lev2 in dfdist_agg["same-shape_loc_size"].unique():
            if not lev1==lev2:
                contrast_levels = [lev1, lev2]
                path = f"{savedir}/stats-agg-{lev1}-vs-{lev2}.txt"
                out, fig = signrank_wilcoxon_from_df(dfdist_agg, datapt_vars, contrast_var, contrast_levels, value_var, PLOT=True,
                                                save_text_path=path)


                savefig(fig, f"{savedir}/stats-agg-{lev1}-vs-{lev2}.pdf")
                plt.close("all")

                res.append({
                    "p":out["p"],
                    "lev1":lev1,
                    "lev2":lev2
                })
    pd.DataFrame(res).to_csv(f"{savedir}/stats_summary.csv")

    # MORE_MORE_figs from notebook
    from pythonlib.tools.statstools import compute_all_pairwise_signrank_wrapper
    savedir_this = f"{savedir}/signrank_pairwise"
    os.makedirs(savedir_this, exist_ok=True)
    compute_all_pairwise_signrank_wrapper(dfdist_agg, ["labels_1_datapt"], "same-shape_loc_size", "dist_mean", True, savedir_this)

    return dfdist, dfdist_agg

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

def fig2_categ_extract_dist_scores_manuscript(DSmorphsets, SAVEDIR, list_version=None, heatmap_plot_trials=True):
    """
    Extract pairiwse distances (between all trials) separately for each
    morphsets in DSmorphsets.Dat.

    Ensures that wil have accurate label for each index within (e.g., ambig, not-ambig)
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.stroketools import strokes_centerize
    from pythonlib.tools.pandastools import stringify_values
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good


    if list_version is None:
        list_version = ["beh", "task"]
    else:
        assert isinstance(list_version, list)

    ### Extract task strokes
    inds = list(range(len(DSmorphsets.Dat)))
    DSmorphsets.Dat["strok_task"]= DSmorphsets.extract_strokes(inds=inds, ver_behtask="task_entire")
    DSmorphsets.Dat = append_col_with_grp_index(DSmorphsets.Dat, ["morph_idxcode_within_set", "assigned_base_simple"], 
                                                "idxmorph_assigned", False) # To make this tuple, same as in Cl below.

    ### Collect distances
    DEBUG = False
    list_df = []
    list_df_index = []
    for morphset in DSmorphsets.Dat["morph_set_idx"].unique():
        dfdat = DSmorphsets.Dat[DSmorphsets.Dat["morph_set_idx"]==morphset].reset_index(drop=True)
        
        # Need at least 2 trials, or else distances will fail
        dfdat, _ = extract_with_levels_of_var_good(dfdat, ["morph_idxcode_within_set", "assigned_base_simple"], 2)

        for version in list_version:
            savedir = f"{SAVEDIR}/extraction/morphset={morphset}-ver={version}"
            os.makedirs(savedir, exist_ok=True)

            # label
            label_var = ["morph_idxcode_within_set", "assigned_base_simple"]
            labels = [tuple(x) for x in dfdat.loc[:, label_var].values.tolist()]

            if version == "beh":
                strokes = dfdat["strok"].tolist()
                if DEBUG:
                    strokes = strokes[::2]
                    labels = labels[::2]
                Cl = DSmorphsets.distgood_compute_beh_beh_strok_distances(strokes, strokes, None, labels, labels, label_var, 
                                                                        clustclass_rsa_mode=True, PLOT=True, 
                                                                        savedir=savedir, savesuff=version, invert_score=True,
                                                                        plot_trials=heatmap_plot_trials)
            elif version == "task":
                strokes = dfdat["strok_task"].tolist()
                strokes = strokes_centerize(strokes, method="bounding_box")
                if DEBUG:
                    strokes = strokes[::2]
                    labels = labels[::2]
                Cl = DSmorphsets.distgood_compute_image_strok_distances(strokes, strokes, labels, labels, label_var,
                                                                do_centerize=True, clustclass_rsa_mode=True,
                                                                PLOT=True, savedir=savedir, savesuff=version,
                                                                plot_trials=heatmap_plot_trials)
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
            fig2_categ_switching_condition_dfdists(dfdat, dfdists, dfproj_index)

            ### Plot
            from pythonlib.tools.snstools import rotateLabel
            list_x_var = ["labels_1_datapt", "morph_idxcode_within_set", "assigned_base_simple"]

            # Dfdists
            dfdists_str = stringify_values(dfdists)
            for x_var in list_x_var:
                x_order = sorted(dfdists_str[x_var].unique())
                
                fig = sns.catplot(data=dfdists_str, x=x_var, y="dist_mean", col="labels_2_grp", order=x_order)
                rotateLabel(fig)
                savefig(fig, f"{savedir}/catplot-dfdists-x={x_var}-1.pdf")
                
                fig = sns.catplot(data=dfdists_str, x=x_var, y="dist_mean", col="labels_2_grp", order=x_order, kind="point", errorbar="se")
                rotateLabel(fig)
                savefig(fig, f"{savedir}/catplot-dfdists-x={x_var}-2.pdf")
            
            # Dfproj_index
            dfproj_index_str = stringify_values(dfproj_index)
            for x_var in ["labels_1_datapt", "morph_idxcode_within_set"]:
                x_order = sorted(dfproj_index_str[x_var].unique())
                
                fig = sns.catplot(data=dfproj_index_str, x=x_var, y="dist_index", order=x_order)
                rotateLabel(fig)
                savefig(fig, f"{savedir}/catplot-dfproj_index-x={x_var}-1.pdf")
                
                fig = sns.catplot(data=dfproj_index_str, x=x_var, y="dist_index", order=x_order, kind="point", errorbar="se")
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

def fig2_categ_extract_dist_scores(DSmorphsets, SAVEDIR, cetegory_expt_version="switching", 
                                   list_version=None, heatmap_plot_trials=True):
    """
    Extract pairiwse distances (between all trials) separately for each
    morphsets in DSmorphsets.Dat.

    Ensures that those morphsets that are "switching" wil have accurate label for each index within
    (e.g., ambig, not-ambig). (Does not guarantee that for "smototh" but that's ok since there
    I don't use this label info.)
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.dataset.scripts.analy_manuscript_figures import score_motor_distances
    from pythonlib.tools.stroketools import get_centers_strokes_list, strokes_centerize
    from pythonlib.tools.pandastools import stringify_values
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good

    if list_version is None:
        list_version = ["beh_imagedist_reversed", "beh_imagedist", "beh_imagedist_bb", "beh", "task"]
    else:
        assert isinstance(list_version, list)

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

        for version in list_version:
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
                                                                        savedir=savedir, savesuff=version, invert_score=True,
                                                                        plot_trials=heatmap_plot_trials)
                # Cl = score_motor_distances(DSmorphsets, strokes, strokes, labels, labels, "/tmp", "TEST", label_var=label_var)
            elif version == "task":
                strokes = dfdat["strok_task"].tolist()
                strokes = strokes_centerize(strokes, method="bounding_box")
                if DEBUG:
                    strokes = strokes[::2]
                    labels = labels[::2]
                Cl = DSmorphsets.distgood_compute_image_strok_distances(strokes, strokes, labels, labels, label_var,
                                                                do_centerize=True, clustclass_rsa_mode=True,
                                                                PLOT=True, savedir=savedir, savesuff=version,
                                                                plot_trials=heatmap_plot_trials)
            elif version == "beh_imagedist":
                strokes = dfdat["strok"].tolist()
                strokes = strokes_centerize(strokes, method="bounding_box")
                if DEBUG:
                    strokes = strokes[::2]
                    labels = labels[::2]
                Cl = DSmorphsets.distgood_compute_image_strok_distances(strokes, strokes, labels, labels, label_var,
                                                                do_centerize=True, clustclass_rsa_mode=True,
                                                                PLOT=True, savedir=savedir, savesuff=version,
                                                                plot_trials=heatmap_plot_trials)
            elif version == "beh_imagedist_bb":
                # Same, but using bounding box.
                strokes = dfdat["strok"].tolist()
                strokes = strokes_centerize(strokes, method="bounding_box")

                # interpolate spatially.
                strokes = strokesInterpolate2(strokes, N=["npts", 70], base="space")

                Cl = DSmorphsets.distgood_compute_image_strok_distances(strokes, strokes, labels, labels, label_var,
                                                                do_centerize=False, clustclass_rsa_mode=True,
                                                                PLOT=True, savedir=savedir, savesuff=version,
                                                                plot_trials=heatmap_plot_trials)
            elif version == "beh_imagedist_reversed":
                # Reverse the direction of one of the beh strokes. Do this as sanity check that the motor has no effect. ie. this 
                # really is image-level
                strokes = dfdat["strok"].tolist()
                strokes = strokes_centerize(strokes, method="bounding_box")
                # Do the reverse
                strokes1 = [strok.copy() for strok in strokes]
                strokes2 = [strok.copy()[::-1] for strok in strokes]
                Cl = DSmorphsets.distgood_compute_image_strok_distances(strokes1, strokes2, labels, labels, label_var,
                                                                do_centerize=True, clustclass_rsa_mode=True,
                                                                PLOT=True, savedir=savedir, savesuff=version,
                                                                plot_trials=heatmap_plot_trials)
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
    Postprocessing of dataset, helper. Assigns new columns reflecting task variables.
    Run this separately for each morphset.
    PARAMS:
    - ds_dat, slice of DSmorphsets for this morphset.
    RETURNS:
    - modifies dfdists and dfproj_index, adding columns.
    """

    #################### ADD LABELS
    # (1) Collect mappers
    map_idxassign_to_label = {}
    map_idxassign_to_assignedbase = {}
    # map_idxassign_to_assignedbase_simple = {}
    # map_idxassign_to_idx_morph = {}
    for _, row in ds_dat.iterrows():
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

    # (2) Assign new columns based on mappers
    for df in [dfdists, dfproj_index]:
        # df["assigned_base_simple"] = [map_idxassign_to_assignedbase_simple[x] for x in df["idxmorph_assigned"]]
        df["morph_assigned_to_which_base"] = [map_idxassign_to_assignedbase[x] for x in df["labels_1_datapt"]] # (base, ambig, notambig)
        df["morph_assigned_label"] = [map_idxassign_to_label[x] for x in df["labels_1_datapt"]] # (base1, ambig1, ..., base2)
        # df["idx_morph_temp"] = [map_idxassign_to_idx_morph[x] for x in df["idxmorph_assigned"]]

def fig2_categ_switching_mult_load_manuscript():
    """
    Load and process pre-extracted data for psychometric experimets.

    RETURNS:
    - DFINDEX, datapt = trial
    - DFINDEX_AGG_1, datapt = labels_1_datapt (e.g., 4|base1)
    - DFINDEX_AGG_2, datapt = morph_assigned_to_which_base (e.g., ambig_base2)
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good

    # Load pre-computed DFDISTS
    list_animal_date = [("Diego", 240517), ("Diego", 240521), ("Diego", 240523), ("Diego", 240730), 
                        ("Pancho", 240516), ("Pancho", 240521), ("Pancho", 240524)]

    SAVEDIR_LOAD = f"/lemur2/lucas/analyses/manuscripts/1_action_symbols/REPRODUCED_FIGURES/fig2h/switching"
    SAVEDIR_MULT = f"{SAVEDIR_LOAD}/MULT/switching"
    os.makedirs(SAVEDIR_MULT, exist_ok=True)

    # Collect data across animals/dates
    list_dfindex = []
    for animal, date in list_animal_date:
        savedir = f"{SAVEDIR_LOAD}/{animal}-{date}"
        dfindex = pd.read_pickle(f"{savedir}/DFINDEX.pkl")
        dfindex["animal"] = animal
        dfindex["date"] = date
        list_dfindex.append(dfindex)   

    DFINDEX = pd.concat(list_dfindex).reset_index(drop=True)
    DFINDEX = append_col_with_grp_index(DFINDEX, ["animal", "date", "morphset"], "ani_date_ms")
    DFINDEX["assigned_base_simple"] = [x[1] for x in DFINDEX["labels_1_datapt"]]
    # remove indices without at least 2 trials...
    DFINDEX = DFINDEX[~DFINDEX["morph_assigned_label"].isin(["not_enough_trials"])].reset_index(drop=True)

    ### Clean up, making sure that "ambig" groups have trials in both base1 and base2
    list_df =[]
    for morph_assigned_label in ["not_ambig", "ambig", "base"]:
        dfthis = DFINDEX[DFINDEX["morph_assigned_label"] == morph_assigned_label].reset_index(drop=True)
        if morph_assigned_label=="ambig":
            # any cases that are ambiguous -- they must have trials for both (base1 and base2)
            # ie. every (ani, date, ms, idxcode) must have at least 1 trial each in base1, base2
            _dfthis, _ = extract_with_levels_of_conjunction_vars_helper(dfthis, "assigned_base_simple", 
                                                        ["ani_date_ms", "morph_idxcode_within_set"],
                                                        plot_counts_heatmap_savepath=None,
                                                        lenient_allow_data_if_has_n_levels=2)
            print(len(dfthis), len(_dfthis), len(_dfthis)/len(dfthis))
            assert len(_dfthis)>0.75*len(dfthis), "pruned so many. unexpcted"
            dfthis = _dfthis
        list_df.append(dfthis)
    DFINDEX = pd.concat(list_df).reset_index(drop=True)
    DFINDEX = DFINDEX.drop(["_index", "vars_others"], axis=1)

    # Add a "label_good" column
    def _new_x_label(morph_assigned_to_which_base):
        """Helper to rename labels for a certain purpose"""
        if morph_assigned_to_which_base in ["base1", "base2", "not_ambig_base1", "not_ambig_base2"]:
            return morph_assigned_to_which_base
        elif morph_assigned_to_which_base in ["ambig_base1", "ambig_base2"]:
            return "ambig"
        else:
            print(morph_assigned_to_which_base)
            assert False
    DFINDEX["label_good"] = [_new_x_label(x) for x in DFINDEX["morph_assigned_to_which_base"]]

    # Normalized dist_index, so that ranges from (0,1) for index (0, 99)
    # NOTE: this is not acrtually used in plots.
    grpvars = ["animal", "date", "morphset", "version"] # norm within this group
    grpdict = grouping_append_and_return_inner_items_good(DFINDEX, grpvars)
    list_df = []
    for grp, inds in grpdict.items():
        dfindex = DFINDEX.iloc[inds].reset_index(drop=True)

        # Normalize to range of min,max (of means)
        valmin = dfindex.groupby(["labels_1_datapt"])["dist_index"].mean().min()
        valmax = dfindex.groupby(["labels_1_datapt"])["dist_index"].mean().max()
        dfindex["dist_index_norm"] = (dfindex["dist_index"]-valmin)/(valmax-valmin)

        list_df.append(dfindex)
    DFINDEX = pd.concat(list_df).reset_index(drop=True)

    # Make aggregrated across trials.
    # datapt = labels_1_datapt (e.g., 4|base1)
    from pythonlib.tools.pandastools import aggregGeneral
    DFINDEX_AGG_1 = aggregGeneral(DFINDEX, ["label_good", "animal", "date", "morphset", "labels_1_datapt", "version"], 
                                  ["dist_index", "dist_index_norm"], nonnumercols="all")

    from pythonlib.tools.pandastools import stringify_values
    DFINDEX = stringify_values(DFINDEX)
    DFINDEX_AGG_1 = stringify_values(DFINDEX_AGG_1)

    # Also agg again to label categories.
    # datapt = morph_assigned_to_which_base (e.g., ambig_base2)
    DFINDEX_AGG_2 = aggregGeneral(DFINDEX_AGG_1, ["label_good", "animal", "date", "morphset", "morph_assigned_to_which_base", "version"], ["dist_index", "dist_index_norm"], nonnumercols="all")
    DFINDEX_AGG_2 = stringify_values(DFINDEX_AGG_2)

    return DFINDEX, DFINDEX_AGG_1, DFINDEX_AGG_2, SAVEDIR_MULT

def fig2_categ_switching_mult_load(cetegory_expt_version, manuscript_version=False):
    """
    Load pre-extracted data, for categ switching.
    """
    from pythonlib.dataset.dataset_analy.psychometric_singleprims import params_good_morphsets_switching, params_good_morphsets_no_switching, params_has_intermediate_shape
    from pythonlib.dataset.dataset_analy.psychometric_singleprims import params_image_distance_progression_not_linear

    # Load pre-computed DFDISTS
    if cetegory_expt_version=="switching":
        list_animal_date = [("Diego", 240517), ("Diego", 240521), ("Diego", 240523), ("Diego", 240730), ("Pancho", 240516), ("Pancho", 240521), ("Pancho", 240524)]
    elif cetegory_expt_version=="smooth":
        list_animal_date = [("Diego", 240515), ("Diego", 240517), ("Diego", 240523), ("Diego", 240731), ("Diego", 240801), 
                    ("Diego", 240802), ("Pancho", 240516), ("Pancho", 240521), ("Pancho", 240524), ("Pancho", 240801), ("Pancho", 240802)]
        # list_animal_date = [("Diego", 240515), ("Diego", 240517), ("Diego", 240523), 
        #                     ("Pancho", 240516), ("Pancho", 240521), ("Pancho", 240524)]
    else:
        assert False

    if manuscript_version:
        SAVEDIR_LOAD = f"/lemur2/lucas/analyses/manuscripts/1_action_symbols/REPRODUCED_FIGURES/fig2h/{cetegory_expt_version}"
    else:
        SAVEDIR_LOAD = f"/lemur2/lucas/analyses/manuscripts/1_action_symbols/fig2_categorization/{cetegory_expt_version}"
    SAVEDIR_MULT = f"{SAVEDIR_LOAD}/MULT/{cetegory_expt_version}"
    os.makedirs(SAVEDIR_MULT, exist_ok=True)

    list_dfindex = []
    for animal, date in list_animal_date:
        savedir = f"{SAVEDIR_LOAD}/{animal}-{date}"
        dfindex = pd.read_pickle(f"{savedir}/DFINDEX.pkl")
        dfindex["animal"] = animal
        dfindex["date"] = date

        if False:
            dfdists = pd.read_pickle(f"{savedir}/DFDISTS.pkl")
            dfdists["animal"] = animal
            dfdists["date"] = date
            morphsets_list_smooth = params_good_morphsets_no_switching(animal, date)

        # Keep just the good switching dates
        if cetegory_expt_version=="switching":
            # Pick out just the good morphsets
            morphsets_map_set_to_indices_switching = params_good_morphsets_switching(animal, date)
            dfindex = dfindex[dfindex["morphset"].isin(list(morphsets_map_set_to_indices_switching.keys()))]
        elif cetegory_expt_version=="smooth":
            # Pick out just the good morphsets
            morphsets_list_smooth = params_good_morphsets_no_switching(animal, date)
            dfindex = dfindex[dfindex["morphset"].isin(list(morphsets_list_smooth))]
            # No intermediate shape
            morphsets_bad = params_has_intermediate_shape(animal, date)
            dfindex = dfindex[~dfindex["morphset"].isin(list(morphsets_bad))]
            # Image should be linear.
            bad_expts = params_image_distance_progression_not_linear()
            dfindex = dfindex[[(animal, date, ms) not in bad_expts for ms in dfindex["morphset"]]].reset_index(drop=True)        
        else:
            assert False
        list_dfindex.append(dfindex)   

    DFINDEX = pd.concat(list_dfindex).reset_index(drop=True)

    ### Plot, separately for each (animal, date, morphset)
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper

    DFINDEX = append_col_with_grp_index(DFINDEX, ["animal", "date", "morphset"], "ani_date_ms")

    if cetegory_expt_version=="switching":
        DFINDEX = DFINDEX[~DFINDEX["morph_assigned_label"].isin(["not_enough_trials"])].reset_index(drop=True)
        DFINDEX["assigned_base_simple"] = [x[1] for x in DFINDEX["labels_1_datapt"]]

        # any cases that are ambiguous -- they must have trials for both 
        # ie. every (ani, date, ms, idxcode) must have at least 1 trial each in base1, base2
        list_df =[]
        for morph_assigned_label in ["not_ambig", "ambig", "base"]:
            dfthis = DFINDEX[DFINDEX["morph_assigned_label"] == morph_assigned_label].reset_index(drop=True)
            if morph_assigned_label=="ambig":
                _dfthis, dict_dfthis = extract_with_levels_of_conjunction_vars_helper(dfthis, "assigned_base_simple", 
                                                            ["ani_date_ms", "morph_idxcode_within_set"],
                                                            plot_counts_heatmap_savepath=None,
                                                            lenient_allow_data_if_has_n_levels=2)
                print(len(dfthis), len(_dfthis), len(_dfthis)/len(dfthis))
                assert len(_dfthis)>0.75*len(dfthis), "pruned so many. unexpcted"
                dfthis = _dfthis
            list_df.append(dfthis)
        DFINDEX = pd.concat(list_df).reset_index(drop=True)
        DFINDEX = DFINDEX.drop(["_index", "vars_others"], axis=1)

        def _new_x_label(morph_assigned_to_which_base):
            if morph_assigned_to_which_base in ["base1", "base2", "not_ambig_base1", "not_ambig_base2"]:
                return morph_assigned_to_which_base
            elif morph_assigned_to_which_base in ["ambig_base1", "ambig_base2"]:
                return "ambig"
            else:
                print(morph_assigned_to_which_base)
                assert False
        DFINDEX["label_good"] = [_new_x_label(x) for x in DFINDEX["morph_assigned_to_which_base"]]
    elif cetegory_expt_version=="smooth":
        DFINDEX["label_good"] = DFINDEX["morph_idxcode_within_set"]
    else:
        assert False

    # normalize, so that ranges from (0,1) for index (0, 99)
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    grpvars = ["animal", "date", "morphset", "version"]
    grpdict = grouping_append_and_return_inner_items_good(DFINDEX, grpvars)
    list_df = []
    for grp, inds in grpdict.items():
        dfindex = DFINDEX.iloc[inds].reset_index(drop=True)

        # Normalize to range of min,max (of means)
        valmin = dfindex.groupby(["labels_1_datapt"])["dist_index"].mean().min()
        valmax = dfindex.groupby(["labels_1_datapt"])["dist_index"].mean().max()
        dfindex["dist_index_norm"] = (dfindex["dist_index"]-valmin)/(valmax-valmin)

        # Flip direction if this is beh
        if False: # Not any more...
            if dfindex["version"].values[0] == "beh":
                dfindex["dist_index_norm"] = 1 - dfindex["dist_index_norm"]

        list_df.append(dfindex)
    DFINDEX = pd.concat(list_df).reset_index(drop=True)

    from pythonlib.tools.pandastools import aggregGeneral
    DFINDEX_AGG_1 = aggregGeneral(DFINDEX, ["label_good", "animal", "date", "morphset", "labels_1_datapt", "version"], ["dist_index", "dist_index_norm"], nonnumercols="all")

    from pythonlib.tools.pandastools import stringify_values
    DFINDEX = stringify_values(DFINDEX)
    DFINDEX_AGG_1 = stringify_values(DFINDEX_AGG_1)

    if cetegory_expt_version=="switching":
        DFINDEX_AGG_2 = aggregGeneral(DFINDEX_AGG_1, ["label_good", "animal", "date", "morphset", "morph_assigned_to_which_base", "version"], ["dist_index", "dist_index_norm"], nonnumercols="all")
        DFINDEX_AGG_2 = stringify_values(DFINDEX_AGG_2)

    return DFINDEX, DFINDEX_AGG_1, DFINDEX_AGG_2, SAVEDIR_MULT

def fig2_categ_switching_mult_plot_stats(DFINDEX_AGG_2, SAVEDIR_MULT, cetegory_expt_version, manuscript_version=False):
    """
    PARAMS:
    - DFINDEX_AGG_2, datapt = morph_assigned_to_which_base (e./g., ambig_base1), rows also split by "version", 
    which is beh or task.

    Plot stats, results for psychometric categories, switching beh
    """
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap, grouping_append_and_return_inner_items_good
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.statstools import signrank_wilcoxon_from_df

    assert cetegory_expt_version == "switching"
    
    # Plot counts
    fig = grouping_plot_n_samples_conjunction_heatmap(DFINDEX_AGG_2, "ani_date_ms", "morph_assigned_to_which_base", ["version"])
    savefig(fig, f"{SAVEDIR_MULT}/stats-counts.pdf")

    ### (1) Show the the change from base --> not_ambig is small for beh compared to for image. 
    # (i.e., test that the flat part of sigmoid is flat)
    # The test compares flatness of beh to flatness of task(image) score.
    grpdict = grouping_append_and_return_inner_items_good(DFINDEX_AGG_2, ["ani_date_ms", "version"])
    res = []
    for grp, inds in grpdict.items():
        dfthis = DFINDEX_AGG_2.iloc[inds]

        # (1) Not-ambig minus base
        a = dfthis[dfthis["morph_assigned_to_which_base"]=="base1"]["dist_index_norm"].mean()
        b = dfthis[dfthis["morph_assigned_to_which_base"]=="not_ambig_base1"]["dist_index_norm"].mean()
        c = dfthis[dfthis["morph_assigned_to_which_base"]=="not_ambig_base2"]["dist_index_norm"].mean()
        d = dfthis[dfthis["morph_assigned_to_which_base"]=="base2"]["dist_index_norm"].mean()
        
        res.append({
            "ani_date_ms":grp[0],
            "version":grp[1],
            "score_name":"notambig-base",
            "score":np.nanmean([(b-a), (d-c)]) # the smaller this value, the more sigmoidal it is.
        })
    dfstats = pd.DataFrame(res)
    assert not any(dfstats["score"].isna())
    # Plot
    _, fig = signrank_wilcoxon_from_df(dfstats, ["ani_date_ms"], "version", ["beh", "task"], 
                            "score", True, f"{SAVEDIR_MULT}/stats-flat_from_base_to_notambig.txt", assert_no_na_rows=True)
    savefig(fig, f"{SAVEDIR_MULT}/stats-flat_from_base_to_notambig.pdf")
    dfstats.to_csv(f"{SAVEDIR_MULT}/stats-flat_from_base_to_notambig.csv")

    ### (2) Test for trial by trial switching (ambig: base 2 minus base 1)
    dfthis = DFINDEX_AGG_2[DFINDEX_AGG_2["version"]=="beh"].reset_index(drop=True)
    _, fig = signrank_wilcoxon_from_df(dfthis, ["ani_date_ms"], "morph_assigned_to_which_base", ["ambig_base1", "ambig_base2"], 
                            "dist_index_norm", True, f"{SAVEDIR_MULT}/stats-ambig_base2_vs_base1.txt", assert_no_na_rows=True)
    savefig(fig, f"{SAVEDIR_MULT}/stats-ambig_base2_vs_base1.pdf")
    plt.close("all")

    ### (3) At each condition (base1, base2,...) compare image to beh, to show that beh is more sigmoidal than image. 
    if not manuscript_version:
        grpdict = grouping_append_and_return_inner_items_good(DFINDEX_AGG_2, ["ani_date_ms", "morph_assigned_to_which_base"])
        y = "dist_index_norm"
        res = []
        for grp, inds in grpdict.items():
            dfthis = DFINDEX_AGG_2.iloc[inds]

            val_beh = dfthis[dfthis["version"]=="beh"][y].mean()
            val_task = dfthis[dfthis["version"]=="task"][y].mean()    
            morph_assigned_to_which_base = grp[1]
            assert len(dfthis["morph_assigned_label"].unique())==1
            morph_assigned_label = dfthis["morph_assigned_label"].values[0]

            # The diff is defined such that it is always in the direction where if it is positive, then it is consistent
            # with a sigmoidal curve for beh (i.e, beh more sigmoidal than task)
            if morph_assigned_to_which_base in ["base1", "not_ambig_base1", "ambig_base1"]:
                val_diff = val_task - val_beh
            elif morph_assigned_to_which_base in ["base2", "not_ambig_base2", "ambig_base2"]:
                val_diff = val_beh - val_task
            else:
                assert False

            res.append({
                "ani_date_ms":grp[0],
                "morph_assigned_to_which_base":morph_assigned_to_which_base,
                "score_name":"diff_image_vs_beh",
                "score":val_diff,
                "morph_assigned_label":morph_assigned_label
            })
        dfstats = pd.DataFrame(res)

        # Further, average across two bases, to get 3 values per (ani, date, ms): base, notambig, ambig.
        dfstats = aggregGeneral(dfstats, ["ani_date_ms", "morph_assigned_label"], ["score"], ["score_name"])
        
        for morph_assigned_label in ["base", "not_ambig", "ambig"]:
            _, fig = signrank_wilcoxon_from_df(dfstats, ["ani_date_ms"], "morph_assigned_label", [morph_assigned_label], 
                                    "score", True, f"{SAVEDIR_MULT}/stats-sigmoidalness_beh_vs_image-{morph_assigned_label}.txt", assert_no_na_rows=True)
            savefig(fig, f"{SAVEDIR_MULT}/stats-sigmoidalness_beh_vs_image-{morph_assigned_label}.pdf")

        dfstats.to_csv(f"{SAVEDIR_MULT}/stats-sigmoidalness_beh_vs_image.csv")

    plt.close("all")


def fig2_categ_switching_mult_plot_manuscript(DFINDEX, DFINDEX_AGG_1, DFINDEX_AGG_2, SAVEDIR_MULT):
    """
    Plot prim indices scores in many ways, across all data, combining
    single-trial scatterplots and average summary plots.

    Combines "each_expt" and "all_expt" plots

    Combines two sets of plots:
    - fig2_categ_switching_mult_plot_eachexpt()
    - fig2_categ_switching_mult_plot_allexpt()
    """

    map_xlabel_to_orders = {
        "label_good":["base1", "not_ambig_base1", "ambig",  "not_ambig_base2", "base2"],
        "morph_assigned_label":["base", "not_ambig", "ambig"],
        "morph_assigned_to_which_base":["base1", "not_ambig_base1", "ambig_base1", "ambig_base2", "not_ambig_base2", "base2"],
    }

    y_vars = ["dist_index"]
    for y in y_vars:
        
        # Single-trial plots
        x_var = "morph_idxcode_within_set"
        x_order = sorted(DFINDEX[x_var].unique())
        fig = sns.catplot(data=DFINDEX, x=x_var, y=y, hue="assigned_base_simple", 
                        col="ani_date_ms", row="version", order=x_order, alpha=0.5, jitter=True)
        savefig(fig, f"{SAVEDIR_MULT}/eachexpt-x_var={x_var}-y={y}-2.pdf")
        plt.close("all")

        # Summary plots
        x_var = "label_good"
        x_order = map_xlabel_to_orders[x_var]
        for row in [None, "animal"]:
            fig = sns.catplot(data=DFINDEX_AGG_2, x=x_var, y=y, hue="assigned_base_simple", 
                            col="version", order=x_order, jitter=True, row=row)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR_MULT}/summary-x_var={x_var}-y={y}-row={row}-1.pdf")

            fig = sns.catplot(data=DFINDEX_AGG_2, x=x_var, y=y, hue="assigned_base_simple", 
                            col="version", order=x_order, kind="violin", row=row)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR_MULT}/summary-x_var={x_var}-y={y}-row={row}-2.pdf")

            fig = sns.catplot(data=DFINDEX_AGG_2, x=x_var, y=y, hue="assigned_base_simple", 
                            col="version", order=x_order, kind="point", row=row)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR_MULT}/summary-x_var={x_var}-y={y}-row={row}-3.pdf")

            order =["base1", "not_ambig_base1", "ambig_base1", "ambig_base2", "not_ambig_base2", "base2"]
            fig = sns.catplot(data=DFINDEX_AGG_2, x="morph_assigned_to_which_base", y=y, hue="version", 
                            kind="point", order=order, row=row)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR_MULT}/summary-x_var={x_var}-y={y}-row={row}-4.pdf")

            plt.close("all")

def fig2_categ_switching_mult_plot_eachexpt(DFINDEX, DFINDEX_AGG_1, SAVEDIR_MULT, cetegory_expt_version):
    """
    Plot prim indices scores in many ways, across all data
    """

    if cetegory_expt_version=="switching":
        map_xlabel_to_orders = {
            "label_good":["base1", "not_ambig_base1", "ambig",  "not_ambig_base2", "base2"],
            "morph_assigned_label":["base", "not_ambig", "ambig"],
            "morph_assigned_to_which_base":["base1", "not_ambig_base1", "ambig_base1", "ambig_base2", "not_ambig_base2", "base2"],
        }
    else:
        assert False, "Add labels here."


    ### Plot, separately for each (animal, date, morphset)
    if cetegory_expt_version=="switching":
        x_var = "morph_idxcode_within_set"
        x_order = sorted(DFINDEX[x_var].unique())
        for y in ["dist_index", "dist_index_norm"]:
            fig = sns.catplot(data=DFINDEX, x=x_var, y=y, hue="assigned_base_simple", 
                            col="ani_date_ms", row="version", order=x_order, kind="point")
            savefig(fig, f"{SAVEDIR_MULT}/eachexpt-x_var={x_var}-y={y}-1.pdf")

            fig = sns.catplot(data=DFINDEX, x=x_var, y=y, hue="assigned_base_simple", 
                            col="ani_date_ms", row="version", order=x_order, alpha=0.5, jitter=True)
            savefig(fig, f"{SAVEDIR_MULT}/eachexpt-x_var={x_var}-y={y}-2.pdf")

            plt.close("all")

        x_var = "label_good"
        x_order = map_xlabel_to_orders[x_var]
        for y in ["dist_index", "dist_index_norm"]:
            fig = sns.catplot(data=DFINDEX_AGG_1, x=x_var, y=y, hue="assigned_base_simple", 
                            col="ani_date_ms", row="version", order=x_order, kind="point")
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR_MULT}/eachexpt-x_var={x_var}-y={y}.pdf")
            
            plt.close("all")
            
        x_var = "morph_assigned_label"
        x_order = map_xlabel_to_orders[x_var]
        for y in ["dist_index", "dist_index_norm"]:
            fig = sns.catplot(data=DFINDEX_AGG_1, x=x_var, y=y, hue="assigned_base_simple", 
                            col="ani_date_ms", row="version", order=x_order, kind="point")
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR_MULT}/eachexpt-x_var={x_var}-y={y}.pdf")
            plt.close("all")
            
    elif cetegory_expt_version=="smooth":
        x_var = "morph_idxcode_within_set"
        x_order = sorted(DFINDEX[x_var].unique())
        for y in ["dist_index", "dist_index_norm"]:
            fig = sns.catplot(data=DFINDEX, x=x_var, y=y, col="ani_date_ms", col_wrap=6,
                            hue="version", order=x_order, alpha=0.5, jitter=True)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR_MULT}/eachexpt-x_var={x_var}-y={y}-1.pdf")

            fig = sns.catplot(data=DFINDEX, x=x_var, y=y, col="ani_date_ms", col_wrap=6,
                            hue="version", order=x_order, kind="point")
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR_MULT}/eachexpt-x_var={x_var}-y={y}-2.pdf")
            plt.close("all")

    else:
        assert False
    plt.close("all")

def fig2_categ_switching_mult_plot_allexpt(DFINDEX_AGG_1, DFINDEX_AGG_2, SAVEDIR_MULT, cetegory_expt_version):
    """
    Plot summary plots across experiments
    """
    
    ### Summarize across all expts.
    # One plot per "version"
    if cetegory_expt_version=="switching":
        map_xlabel_to_orders = {
            "label_good":["base1", "not_ambig_base1", "ambig",  "not_ambig_base2", "base2"],
            "morph_assigned_label":["base", "not_ambig", "ambig"],
            "morph_assigned_to_which_base":["base1", "not_ambig_base1", "ambig_base1", "ambig_base2", "not_ambig_base2", "base2"],
        }
    else:
        assert False

    if cetegory_expt_version=="switching":
        x_var = "label_good"
        x_order = map_xlabel_to_orders[x_var]
        for y in ["dist_index", "dist_index_norm"]:
            for row in [None, "animal"]:
                fig = sns.catplot(data=DFINDEX_AGG_2, x=x_var, y=y, hue="assigned_base_simple", 
                                col="version", order=x_order, jitter=True, row=row)
                rotateLabel(fig)
                savefig(fig, f"{SAVEDIR_MULT}/summary-x_var={x_var}-y={y}-row={row}-1.pdf")

                fig = sns.catplot(data=DFINDEX_AGG_2, x=x_var, y=y, hue="assigned_base_simple", 
                                col="version", order=x_order, kind="violin", row=row)
                rotateLabel(fig)
                savefig(fig, f"{SAVEDIR_MULT}/summary-x_var={x_var}-y={y}-row={row}-2.pdf")

                fig = sns.catplot(data=DFINDEX_AGG_2, x=x_var, y=y, hue="assigned_base_simple", 
                                col="version", order=x_order, kind="point", row=row)
                rotateLabel(fig)
                savefig(fig, f"{SAVEDIR_MULT}/summary-x_var={x_var}-y={y}-row={row}-3.pdf")

                order =["base1", "not_ambig_base1", "ambig_base1", "ambig_base2", "not_ambig_base2", "base2"]
                fig = sns.catplot(data=DFINDEX_AGG_2, x="morph_assigned_to_which_base", y=y, hue="version", 
                                kind="point", order=order, row=row)
                rotateLabel(fig)
                savefig(fig, f"{SAVEDIR_MULT}/summary-x_var={x_var}-y={y}-row={row}-4.pdf")

                plt.close("all")

    elif cetegory_expt_version=="smooth":
        ### One datapt per each morph_idxcode (good esp for smooth expts)
        assert False, "replace dist_index_norm with both dist_index_norm and dist_index (see above)"
        for row in [None, "animal"]:
            fig = sns.catplot(data=DFINDEX_AGG_1, x="morph_idxcode_within_set", y="dist_index_norm", 
                            col="version", jitter=True, row=row)
            savefig(fig, f"{SAVEDIR_MULT}/eachmorphidxcode-catplot-row={row}-1.pdf")

            fig = sns.catplot(data=DFINDEX_AGG_1, x="morph_idxcode_within_set", y="dist_index_norm", 
                            col="version", kind="point", errorbar=("ci", 68), row=row)
            savefig(fig, f"{SAVEDIR_MULT}/eachmorphidxcode-catplot-row={row}-2.pdf")

        # For smoothing expts
        from pythonlib.tools.pandastools import pivot_table
        DFINDEX_AGG_1_WIDE = pivot_table(DFINDEX_AGG_1, ["ani_date_ms", "animal", "date", "morphset", "labels_1_datapt"], 
                                        ["version"], ["dist_index_norm"], flatten_col_names=True)

        nbins = 8
        bin_edges = np.linspace(0, 1.0, nbins+1)
        from pythonlib.tools.nptools import bin_values
        DFINDEX_AGG_1_WIDE["task_image_binned"] = [int(x) for x in bin_values(DFINDEX_AGG_1_WIDE["dist_index_norm-task"], nbins)]

        for x_var in ["dist_index_norm-task", "task_image_binned"]:
            for yvar in ["dist_index_norm-beh", "dist_index_norm-beh_imagedist"]:
                fig = sns.relplot(data=DFINDEX_AGG_1_WIDE, x=x_var, 
                            y=yvar, col="ani_date_ms", col_wrap=6, kind="line")
                # for ax in fig.axes.flatten():
                #     ax.plot([0,0], [1,1], "--k", alpha=0.5)
                savefig(fig, f"{SAVEDIR_MULT}/eachmorphidxcode-replot-x_var={x_var}-yvar={yvar}-1.pdf")

                fig = sns.relplot(data=DFINDEX_AGG_1_WIDE, x=x_var, 
                            y=yvar, hue="ani_date_ms", kind="line", legend=False)
                # for ax in fig.axes.flatten():
                #     ax.plot([0,0], [1,1], "--k", alpha=0.5)
                savefig(fig, f"{SAVEDIR_MULT}/eachmorphidxcode-replot-x_var={x_var}-yvar={yvar}-2.pdf")

                fig = sns.relplot(data=DFINDEX_AGG_1_WIDE, x=x_var, 
                            y=yvar, hue="ani_date_ms", legend=False)
                # for ax in fig.axes.flatten():
                #     ax.plot([0,0], [1,1], "--k", alpha=0.5)
                savefig(fig, f"{SAVEDIR_MULT}/eachmorphidxcode-replot-x_var={x_var}-yvar={yvar}-3.pdf")

                if x_var == "task_image_binned":
                    fig = sns.catplot(data=DFINDEX_AGG_1_WIDE, x=x_var, 
                                y=yvar, alpha=0.5, jitter=True, legend=False)
                    savefig(fig, f"{SAVEDIR_MULT}/eachmorphidxcode-catplot-x_var={x_var}-yvar={yvar}-1.pdf")

                    fig = sns.catplot(data=DFINDEX_AGG_1_WIDE, x=x_var, 
                                y=yvar, kind="point", legend=False)
                    savefig(fig, f"{SAVEDIR_MULT}/eachmorphidxcode-catplot-x_var={x_var}-yvar={yvar}-2.pdf")

            plt.close("all")
    else:
        assert False


def recording_units_counts_plot(DFall, savedir):
    """
    All plots of num units, across all recording dates.

    """
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.pandastools import grouping_count_n_samples_return_df
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.expttools import writeStringsToFile

    ### Get data of counts
    dfcount = grouping_count_n_samples_return_df(DFall, ["ani_date", "animal", "date", "spikes_version", "label_final", "region"])
    dfcount_comb = grouping_count_n_samples_return_df(DFall, ["ani_date", "animal", "date", "spikes_version", "region"])

    ### PLOTS
    if False: # Skip, because x orders are not shared across plots
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
        from pythonlib.tools.plottools import rotate_x_labels

        grpdict = grouping_append_and_return_inner_items_good(DFall, ["animal", "date"])

        ncols = 4
        nrows = int(np.ceil(len(grpdict)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), sharex=True, sharey=True)
        for i, ((animal, date), inds) in enumerate(grpdict.items()):

            ax = axes.flatten()[i]
            
            df = DFall.iloc[inds].reset_index(drop=True)
            plot_bar_stacked_histogram_counts(df, "region", "label_final", ax=ax)
            ax.set_title(f"{animal}-{date}-{df['spikes_version'].values[0]}")

            rotate_x_labels(ax, 90)

    fig = grouping_plot_n_samples_conjunction_heatmap(DFall, "region", "label_final", ["animal", "date"])
    savefig(fig, f"{savedir}/counts_heatmap.pdf")

    fig = sns.catplot(data=dfcount, x="region", y="count", hue="label_final", col="ani_date", col_wrap=4, kind="bar")
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot_counts_all.pdf")

    plt.close("all")

    # Plot summary
    fig = sns.catplot(data=dfcount, x="region", y="count", hue="label_final", col="spikes_version", col_wrap=4, kind="bar", aspect=1.5)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot_catplot_counts-1.pdf")

    fig = sns.catplot(data=dfcount, x="region", y="count", hue="label_final", col="spikes_version", col_wrap=4, jitter=True, alpha=0.5, aspect=1.5)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot_catplot_counts-2.pdf")

    fig = sns.catplot(data=dfcount, x="region", y="count", hue="label_final", row="animal", col="spikes_version", kind="bar", aspect=1.5)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot_catplot_counts-1a.pdf")

    fig = sns.catplot(data=dfcount, x="region", y="count", hue="label_final", row="animal", col="spikes_version", jitter=True, alpha=0.5, aspect=1.5)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot_catplot_counts-2a.pdf")

    fig = sns.catplot(data=dfcount_comb, x="region", y="count", col="spikes_version", col_wrap=4, jitter=True, alpha=0.5)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot_catplot_counts_comb-1.pdf")

    fig = sns.catplot(data=dfcount_comb, x="region", y="count", col="spikes_version", col_wrap=4, kind="bar")
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot_catplot_counts_comb-2.pdf")

    plt.close("all")
    
    ### Print results
    dfcount.to_csv(f"{savedir}/count_summary_all.csv")
    df = aggregGeneral(dfcount, ["animal", "spikes_version", "label_final", "region"], ["count"], aggmethod=["mean", "std"])
    df.to_csv(f"{savedir}/count_summary_split.csv")

    # Print text for manuscript
    animals = df["animal"].unique().tolist()
    bregions = df["region"].unique().tolist()

    strings = []
    for a in animals:
        for b in bregions:
            for label in ["mua", "su"]:
                tmp = df[(df["animal"] == a) & (df["region"]==b) & (df["spikes_version"]=="kilosort") & (df["label_final"]==label)]
                if len(tmp)==1:
                    if label=="mua":
                        _label = "MU"
                    elif label == "su":
                        _label = "SU"
                    else:
                        assert False
                    s = f"{a} - {b} ({_label}: {tmp['count_mean'].values[0]:.1f} +/- {tmp['count_std'].values[0]:.1f})"
                    print(s)
                    strings.append(s)
                else:
                    assert len(tmp)==0
    writeStringsToFile(f"{savedir}/count_summary_split_strings-1.txt", strings)


    df = aggregGeneral(dfcount_comb, ["animal", "spikes_version", "region"], ["count"], aggmethod=["mean", "std"])
    df.to_csv(f"{savedir}/count_summary_comb.csv")

    # Print text for manuscript
    animals = df["animal"].unique().tolist()
    bregions = df["region"].unique().tolist()

    strings = []
    for a in animals:
        for b in bregions:
            tmp = df[(df["animal"] == a) & (df["region"]==b) & (df["spikes_version"]=="kilosort")]
            if len(tmp)==1:
                s = f"{a} - {b} - {tmp['count_mean'].values[0]:.1f} +/- {tmp['count_std'].values[0]:.1f}"
                s = f"{a} - {b} ({tmp['count_mean'].values[0]:.1f} +/- {tmp['count_std'].values[0]:.1f})"
                print(s)
                strings.append(s)
            else:
                assert len(tmp)==0

    writeStringsToFile(f"{savedir}/count_summary_split_strings-2.txt", strings)

def stroke_shape_cluster_database_save_shuffled(animal):
    """
    Generate and save new shuffled prims, splitting into two halves and reconnecting them.
    Takes the original prims, and makes a new prim from each pair, and then samples a new set, doing this 
    like 5 times. Each time apply filters to make sure the prims are different from eahc other, and different
    from the base prims (to some sufficient extent).

    RETURNS:
    - Saves the shuffled prims, each of the sets, as pickled strokes, along with figures of the prims
    in a grid.
    """
    from pythonlib.dataset.dataset_strokes import DatStrokes
    from pythonlib.tools.stroketools import merge_interpolate_concat_strokes_halves

    ### (1) Load original basis set
    DS = DatStrokes()
    which_basis_set = animal
    which_shapes = "main_21"
    dfbasis, _, _ = DS.stroke_shape_cluster_database_load_helper(
        which_basis_set=which_basis_set,
        which_shapes=which_shapes, plot_examples=True)
    # list_strok_basis = dfbasis["strok"].tolist()
    # list_shape_basis = dfbasis["shape"].tolist()

    ### (2) Get new basis that concats halves, doing this across all possible pairs
    dfbasis_merged = merge_interpolate_concat_strokes_halves(dfbasis)
    
    if False: # using different method, it checks distance to orig stroke below.
        strokes_base = dfbasis["strok"].tolist()
        Clorig = DS.distgood_compute_beh_beh_strok_distances(strokes_base, strokes_base, PLOT=False, invert_score=True)
        # Filter this list to those that are not too close to any of the original prims.
        ma = Clorig._rsa_matindex_generate_upper_triangular()
        fig, ax = plt.subplots()
        x = Clorig.Xinput[ma].flatten()
        ax.hist(x, bins=20)

        min_score = np.percentile(x, [5])[0]

        ma_bad_close_to_orig = Clmerged.Xinput<min_score

        Clmerged.rsa_matindex_plot_bool_mask(ma_bad_close_to_orig)
    
    # Find the cases that are too twisted (intersect with self)
    from pythonlib.tools.stroketools import has_self_intersection
    dfbasis_merged["self_intersects"] = [has_self_intersection(strok) for strok in dfbasis_merged["strok"]]
    inds_bad_self_intersect = dfbasis_merged[dfbasis_merged["self_intersects"]==True].index.tolist()

    ### (3) Determine how many new base prims to sample from each bin (binning pairwise distances)
    # Extract a random subset of prims
    # Do this in separate subsets of data based on their similarity, trying to match
    # the original data's distribtuion of pairwise distances
    from pythonlib.tools.nptools import bin_values
    strokes_base = dfbasis["strok"].tolist()
    Clorig = DS.distgood_compute_beh_beh_strok_distances(strokes_base, strokes_base, PLOT=False, invert_score=True)
    ma_ut = Clorig._rsa_matindex_generate_upper_triangular()
    dists_orig = Clorig.Xinput[ma_ut].flatten()

    nbins = 6
    dists_orig_binned, bins = bin_values(dists_orig, nbins, return_bins=True) # 1, 2, 3. ,, (strings). 
    dict_bins_nsamp = {}
    for i, (b1, b2) in enumerate(zip(bins[:-1], bins[1:])):
        key = f"{i+1}"
        n = sum([x==key for x in dists_orig_binned]) # num cases (pairs of strokes)
        dict_bins_nsamp[key] = [(b1, b2), n]

    # Normalize the n, i.e., determine how many strokes to take per bin
    ntot = sum([v[1] for v in dict_bins_nsamp.values()])
    ntake_per_bin = [int(np.ceil(len(dfbasis)*v[1]/ntot)) for v in dict_bins_nsamp.values()]
    print(ntake_per_bin)
    ntake_per_bin = [np.max([2, n]) for n in ntake_per_bin] # make sure at least 2, so that you are taking a distance too.
    while sum(ntake_per_bin)>len(dfbasis):
        indmax = np.argmax(ntake_per_bin)
        ntake_per_bin[indmax] -= 1

    # Update the final dict.
    for i, (k, v) in enumerate(dict_bins_nsamp.items()):
        v[1] = ntake_per_bin[i]
    print("Final n indices to take per bin:", ntake_per_bin)
    print(dict_bins_nsamp)

    ### (4) Prep, get pairwise between all new strokes vs. base strokes, for use below.
    if False:
        strokes_base = dfbasis["strok"].tolist()
        strokes_merged = dfbasis_merged["strok"].tolist()
        Clmerged = DS.distgood_compute_beh_beh_strok_distances(strokes_merged, strokes_base, PLOT=False, invert_score=True)

    ### (5) Sample random sets
    # MIN_SCORE_VS_OTHERS, MIN_SCORE_VS_BASIS = np.percentile(dists_orig, [0, 2.5])
    MIN_SCORE_VS_OTHERS, MIN_SCORE_VS_BASIS = np.percentile(dists_orig, [0, 0])
    if animal=="Diego":
        MIN_SCORE_VS_OTHERS = MIN_SCORE_VS_OTHERS - 0.02
    elif animal=="Pancho":
        # He has fewer prims, pool of prims, so this is more lenient.
        MIN_SCORE_VS_OTHERS = MIN_SCORE_VS_OTHERS - 0.05
        MIN_SCORE_VS_BASIS = MIN_SCORE_VS_BASIS - 0.05
    else:
        print(animal)
        assert False

    # Params for testing curvature
    from pythonlib.tools.stroketools import strokesCurvature, sample_rate_from_strokes
    MAX_CURV = 0.8 # curvature
    strok = dfbasis_merged["strok"].values[0]
    npts = strok.shape[0]
    curv_fs = sample_rate_from_strokes([strok])
    curv_on = int(np.ceil(0.1*npts))
    curv_off = int(np.floor(0.9*npts))

    from pythonlib.tools.expttools import makeTimeStamp
    timestamp = makeTimeStamp()
    SAVEDIR, _ = DS._stroke_shape_cluster_database_path(animal=which_basis_set, 
                                                        expt="shuffled_first_second_half", 
                                                        date=timestamp)
    os.makedirs(SAVEDIR, exist_ok=True)
    
    ### Save dataframes Save additional things
    dfbasis.to_pickle(f"{SAVEDIR}/dfbasis_orig.pkl")
    dfbasis_merged.to_pickle(f"{SAVEDIR}/dfbasis_merged.pkl")

    n_iters = 10
    for iter_num in range(n_iters):
        
        try:
            # For each bin, sample n cases
            import random

            # For each bin, sample n cases
            inds_take_all = []
            inds_taken_orig_onset = []
            inds_taken_orig_offset = []

            for key, (vals, nget) in dict_bins_nsamp.items():
                print(key, (vals, nget))

                if False:
                    # This was incorrect -- it was using novel_prim vs. old_prims, where in fact
                    # I wanted to use novel_prim vs. novel_prim. But the problem is that ocmputing those
                    # takes a long time. In the end it doesnt seem to make much of a difference.
                    Clmerged.Xinput>vals[0]
                    Clmerged.Xinput<=vals[1]
                    inds = np.argwhere((Clmerged.Xinput>vals[0]) & (Clmerged.Xinput<=vals[1]))
                    inds_pool = sorted(set(inds.flatten()))
                else:
                    # Hacky, just take all inds and place in pool.
                    inds_pool = list(range(len(dfbasis_merged)))

                # Exclude those that self-intersect
                inds_pool = [i for i in inds_pool if i not in inds_bad_self_intersect]
                
                # sample randomly from pool
                random.shuffle(inds_pool)

                # inds_take_this = []
                ngotten=0
                while ngotten<nget:
                    ind_candidate = inds_pool.pop()
                    
                    if ind_candidate in inds_take_all:
                        # This already taken. 
                        continue

                    # Skip if the onset of this merged has already been taken
                    ind_basis_onset = dfbasis_merged.iloc[ind_candidate]["i1"] # index into original bases
                    ind_basis_offset = dfbasis_merged.iloc[ind_candidate]["i2"] # index into original bases
                    if False: # Skip, since it restricts it a lot, and ends up using up all inds and faling.
                        if ind_basis_onset in inds_taken_orig_onset:
                            print("skipping, since onset already gotten: ", ind_basis_onset)
                            continue
                    else:
                        # More lenient, allow max 2 cases
                        if sum([i==ind_basis_onset for i in inds_taken_orig_onset])>=2:
                            print("skipping, since onset already gotten: ", ind_basis_onset)
                            continue

                    if False: # Skip, since it restricts it a lot, and ends up using up all inds and faling.
                        if ind_basis_offset in inds_taken_orig_offset:
                            print("skipping, since offset already gotten: ", ind_basis_offset)
                            continue

                    # Checks, based on stroke similarty
                    strok_this = dfbasis_merged["strok"].values[ind_candidate]
                    
                    # (1) Check that its not too similar to previous strokes
                    if len(inds_take_all)>0:
                        strokes_taken_so_far = dfbasis_merged.iloc[inds_take_all]["strok"].tolist()
                        cl = DS.distgood_compute_beh_beh_strok_distances([strok_this], strokes_taken_so_far, invert_score=True)
                        if np.any(cl.Xinput<MIN_SCORE_VS_OTHERS):
                            # Then this strok is too close to one already in the pool
                            print("skipping, since too similar to one already gotten")
                            continue    
                    
                    # (2) Check that this is not too similar to any of the original basis strokes
                    strokes_basis = dfbasis["strok"].tolist()
                    cl = DS.distgood_compute_beh_beh_strok_distances([strok_this], strokes_basis, invert_score=True)
                    if np.any(cl.Xinput<MIN_SCORE_VS_BASIS):
                        # Then this strok is too close to one already in the pool
                        print("skipping, since too similar to basis stroke")
                        # print("skipping, since too similar to basis stroke", cl.Xinput)
                        continue   

                    # (3) Check that is not too acute of an angle
                    plot_final_simple = False # make this True to see how this works
                    strokcurv = strokesCurvature([strok_this], curv_fs, plot_final_simple=plot_final_simple)[0]
                    if np.any(strokcurv[curv_on:curv_off, 0] > MAX_CURV):
                        continue                      
                    
                    ### Got here, good, means keep it
                    inds_take_all.append(ind_candidate)
                    inds_taken_orig_onset.append(ind_basis_onset)
                    inds_taken_orig_offset.append(ind_basis_offset)

                    ngotten+=1
                    print("taking :", ind_candidate, ", now got: ", ngotten)
            assert len(set(inds_take_all))==len(dfbasis)

            ### Take the set of new prims
            strokes_basis_new = dfbasis_merged.iloc[inds_take_all]["strok"].tolist()
            dfbasis_merged_take = dfbasis.copy()
            dfbasis_merged_take["strok"] = strokes_basis_new
            dfbasis_merged_take["inds_from_dfbasis_merged"] = inds_take_all
            for col_bad in ["strok_task", "shape_cat_abstract", "char_first_instance"]:
                del dfbasis_merged_take[col_bad]

            ### Save
            savedir = f"{SAVEDIR}/iter={iter_num}"
            os.makedirs(savedir, exist_ok=True)
            dfbasis_merged_take.to_pickle(f"{savedir}/dfbasis_merged_take.pkl")

            ### Plots
            from pythonlib.tools.plottools import savefig

            # Get pariwise dist between each other
            cl = DS.distgood_compute_beh_beh_strok_distances(dfbasis_merged_take["strok"].tolist(), 
                                                                    dfbasis_merged_take["strok"].tolist(), invert_score=True)
            fig, X, labels_col, labels_row, ax = cl.plot_heatmap_data();
            ax.set_ylabel("new prim")
            ax.set_xlabel("new prim")
            savefig(fig, f"{savedir}/heatmap_self_pairs.pdf")
            ma_ut = cl._rsa_matindex_generate_upper_triangular()
            dists_final_within = cl.Xinput[ma_ut].flatten()

            # Get pariwise dist to original basis set
            cl = DS.distgood_compute_beh_beh_strok_distances(dfbasis_merged_take["strok"].tolist(), 
                                                                dfbasis["strok"].tolist(), invert_score=True)
            fig, X, labels_col, labels_row, ax = cl.plot_heatmap_data();
            ax.set_ylabel("new prim")
            ax.set_xlabel("original prim")
            savefig(fig, f"{savedir}/heatmap_vs_basis_orig.pdf")
            dists_final_vs_orig = cl.Xinput.flatten()

            assert np.all(dists_final_within >= MIN_SCORE_VS_OTHERS)
            assert np.all(dists_final_vs_orig >= MIN_SCORE_VS_OTHERS)

            ### Plot the distances
            fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10, 10))

            # ax = axes.flatten()[0]
            # ax.hist(dists_merged, bins=20);
            # for v in bins:
            #     ax.axvline(v, color="k")
            # ax.set_title("pairs of ALL merged strokes")

            ax = axes.flatten()[1]
            ax.hist(dists_orig, bins=20);
            for v in bins:
                ax.axvline(v, color="k")
            ax.set_title("original bases")

            ax = axes.flatten()[2]
            ax.hist(dists_final_within, bins=20);
            for v in bins:
                ax.axvline(v, color="k")
            ax.set_title("pairs of TAKEN merged strokes")

            ax = axes.flatten()[3]
            ax.hist(dists_final_vs_orig, bins=20);
            for v in bins:
                ax.axvline(v, color="k")
            ax.set_title("taken merged vs. orig bases")

            savefig(fig, f"{savedir}/distances_hist.pdf")

            # Plot each stroke
            # sort by name of prim
            # dfbasis_merged_take = dfbasis_merged_take.sort_values("shape").reset_index(drop=True)
            dfbasis_merged_take = dfbasis_merged_take.sort_values("inds_from_dfbasis_merged").reset_index(drop=True)
            list_strok = dfbasis_merged_take["strok"].tolist()
            list_shape = dfbasis_merged_take["shape"].tolist()
            list_inds_orig = dfbasis_merged_take["inds_from_dfbasis_merged"].tolist()
            # # centerize task strokes, otherwise they are in space
            # list_strok_task = dfdat["strok_task"].tolist()
            # list_strok_task = [x-np.mean(x, axis=0, keepdims=True) for x in list_strok_task]

            for i, titles in enumerate([list_inds_orig]):
                fig, axes = DS.plot_multiple_strok(list_strok, overlay=False, ncols=9, titles=titles)
                savefig(fig, f"{savedir}/mean_stroke_each_shape-{i}-BEH.pdf")

                # fig, axes = self.plot_multiple_strok(list_strok_task, ver="task", overlay=False, ncols=9, titles=titles);
                # savefig(fig, f"{sdir}/mean_stroke_each_shape-{i}-TASK.pdf")

            plt.close("all")
        except Exception as e:
            print("Error in iter_num: ", iter_num)
            print(e)
            continue

def revision_eye_fixation_load_data(animal, date):
    """
    Load and concats PIG, combining fixation-aligned and regular data.
    """
    from neuralmonkey.classes.population_mult import load_handsaved_wrapper

    combine_areas = True

    # Method 2 - Combine two dfallpa
    question = "PIG_BASE_saccade_fix_on"
    DFallpa1 = load_handsaved_wrapper(animal=animal, date=date, version="saccade_fix_on", combine_areas=combine_areas, question=question)

    which_level = "trial"
    question = "PIG_BASE_trial"
    DFallpa2 = load_handsaved_wrapper(animal=animal, date=date, version=which_level, combine_areas=combine_areas, question=question)

    DFallpa = pd.concat([DFallpa1, DFallpa2]).reset_index(drop=True)

    return DFallpa

def revision_eye_fixation_shape_decode(DFallpa, bregion, SAVEDIR_ALL):
    """
    To show that, in PIG, decoding is strong for what going to draw, regardless of where you are looking.

    Trains decoder (decoder_moment using SP data for a given day, then uses this to test alinged to fixations
    for PIG, and plots decoding score).

    To make point that PMv encodes the planned shape to draw, even when fixating on a different shape.

    Code development, see:
    # See plot_all in analyeyefixdecodemoment gen
    # See _analy_chars_score_postsamp_plot_timecourse
    # Also this for more splitting and stuff: _timeseries_plot_by_shape_drawn_order
    """

    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good, aggregGeneral, grouping_plot_n_samples_conjunction_heatmap_helper, append_col_with_grp_index, plot_45scatter_means_flexible_grouping

    ########## TRAIN DECODER
    from neuralmonkey.analyses.decode_moment import pipeline_train_test_scalar_score

    PLOT_DECODER = True
    n_min_per_var = 5

    # Train a single decoder on SP data
    event_train = "03_samp"
    # twind_train = (0.05, 1.2)
    twind_train = (0.05, 1.0)
    filterdict_train = {
        "FEAT_num_strokes_task":[1],
        "task_kind":["prims_single"],
    }
    which_level_train = "trial"
    var_train = "seqc_0_shape"

    # var_test = "shape-fixation"
    var_test = "seqc_0_shape"
    list_twind_test = [(0.05, 0.3)]
    assert len(list_twind_test)==1, "assumes this."
    which_level_test = "flex"
    event_test = "fixon_preparation"
    filterdict_test = {
        "FEAT_num_strokes_task":list(range(2,10)),
        "task_kind":["prims_on_grid"],
    }

    for twind_test in list_twind_test:
        SAVEDIR = f"{SAVEDIR_ALL}/twind={twind_test}"
        os.makedirs(SAVEDIR, exist_ok=True)
        print("SAVING AT: ", SAVEDIR)

        savedir = f"{SAVEDIR}/decoder_training"
        os.makedirs(savedir, exist_ok=True)
        DFSCORES, Dc, _, PAtest = pipeline_train_test_scalar_score(DFallpa, bregion, 
                                            var_train, event_train, twind_train, filterdict_train,
                                            var_test, event_test, [twind_test], filterdict_test,
                                            savedir, prune_labels_exist_in_train_and_test=True, PLOT=PLOT_DECODER,
                                            which_level_train=which_level_train, which_level_test=which_level_test, 
                                            n_min_per_var=n_min_per_var,
                                            allow_multiple_twind_test=True)

    ### Postprocess of dfscores (e.g., add columns)
    dflab = PAtest.Xlabels["trials"]
    dflab = append_col_with_grp_index(dflab, ["shape-fixation", "seqc_0_shape"], "shape-fix_draw")
    PAtest.Xlabels["trials"] = dflab

    inds_pa = DFSCORES["pa_idx"].tolist()
    for col in ["shape-fixation", "seqc_0_shape", "early-or-late-planning-period", "shape-macrosaccade-index"]:
        if col in DFSCORES:
            assert DFSCORES[col].tolist()==dflab.iloc[inds_pa][col].tolist()
        DFSCORES[col] = dflab.iloc[inds_pa][col].values
    assert all(DFSCORES[var_test] == DFSCORES["pa_class"])
    DFSCORES["dcd_eq_fix"] = DFSCORES["decoder_class"] == DFSCORES["shape-fixation"]
    DFSCORES["dcd_eq_draw"] = DFSCORES["decoder_class"] == DFSCORES["seqc_0_shape"]
    DFSCORES = append_col_with_grp_index(DFSCORES, ["dcd_eq_fix", "dcd_eq_draw"], "dcd_eq_fix|draw")

    # Sanity check
    dfscores_good = DFSCORES[DFSCORES["shape-fixation"] != DFSCORES["seqc_0_shape"]]
    n_fix_good = len(dfscores_good["pa_idx"].unique()) # This where fix and draw are dissociated
    assert sum(DFSCORES["dcd_eq_fix|draw"]=="0|1") == sum(DFSCORES["dcd_eq_fix|draw"]=="1|0") == n_fix_good, "sanity check, each fixation has a single datapt"

    ### AGG so that the final score is not biased by frequencies of draw/fixation.
    DFSCORES_AGG = aggregGeneral(DFSCORES, ["early-or-late-planning-period", "shape-fixation", "seqc_0_shape", "decoder_class"], ["score"], 
                  ["dcd_eq_fix", "dcd_eq_draw", "dcd_eq_fix|draw"])

    ############### PLOTS
    ### Get timecourses 
    # Split by (what draw, what looking at)
    dflab = PAtest.Xlabels["trials"]
    twind_test = (-0.35, 0.35)
    shapes_unique = sorted(set(dflab["shape-fixation"].unique().tolist() + dflab["seqc_0_shape"].unique().tolist()))
    for early_late_this in ["early", "late"]:
        grpdict = grouping_append_and_return_inner_items_good(dflab, ["shape-fixation", "seqc_0_shape", "early-or-late-planning-period"])
        grpdict = {k:v for k, v in grpdict.items() if k[2]==early_late_this} # Keep just early or late.
        assert len(grpdict)>0
        
        n = len(shapes_unique)
        SIZE = 3
        fig, axes = plt.subplots(n, n, figsize=(SIZE*n, SIZE*n), sharex=True, sharey=True)

        for i, (grp, indtrials) in enumerate(grpdict.items()):
            
            _, probs_mat_all, times, labels = Dc.timeseries_score_wrapper(PAtest, 
                                                                                    twind_test, indtrials, 
                                                                                    labels_in_order_keep=shapes_unique)

            ax = axes.flatten()[i]
            plot_legend = i==0
            Dc._timeseries_plot_flex(probs_mat_all, times, labels, MAP_INDEX_TO_COL=None, ax=ax, plot_legend=plot_legend)
            ax.set_title(f"shape-fixation={grp[0]}|seqc_0_shape={grp[1]}", fontsize=6)
            ax.set_ylim([0, 1])

        savefig(fig, f"{SAVEDIR}/timecourses-final-{early_late_this}.pdf")
        plt.close("all")

    ### Plot heatmap summaries

    # Split by (what draw, what looking at)
    dflab = PAtest.Xlabels["trials"]
    grpdict = grouping_append_and_return_inner_items_good(dflab, ["shape-fixation", "early-or-late-planning-period"])
    n = len(shapes_unique)
    SIZE = 3

    # for norm_method, annotate_heatmap in [(None, True), ("row_div", False), ("col_div", False), ("all_div", False)]:
    for norm_method, annotate_heatmap in [(None, True), ("row_div", False), ("col_div", False)]:
        fig, axes = plt.subplots(2, n, figsize=(SIZE*n, SIZE*2), sharex=True, sharey=True)
        agg_method = "mean"
        val_name = "score"
        diverge = False
        zlims = [0, 1]
        # annotate_heatmap = False
        # norm_method = "row_div"
        for i, (grp, indtrials) in enumerate(grpdict.items()):
            dfscores = DFSCORES[DFSCORES["pa_idx"].isin(indtrials)].reset_index(drop=True)
            ax = axes.flatten()[i]
            if len(dfscores)>0:
                df2d, _, _, rgba_values = convert_to_2d_dataframe(dfscores, "pa_class", "decoder_class", True,
                                        agg_method,
                                        val_name,
                                        ax=ax, annotate_heatmap=annotate_heatmap,
                                        diverge=diverge, zlims=zlims, norm_method=norm_method,
                                        list_cat_1 = shapes_unique, list_cat_2=shapes_unique)
            ax.set_title(f"shape-fixation={grp[0]}", fontsize=6)
        savefig(fig, f"{SAVEDIR}/heatmaps-final-norm={norm_method}-annot={annotate_heatmap}.pdf")    
    # assert False
    ### Plot scatter
    _, fig = plot_45scatter_means_flexible_grouping(DFSCORES, "dcd_eq_fix|draw", "0|1", "1|0", "early-or-late-planning-period", 
                                        "score", "trialcode", False, alpha=0.1, SIZE=4, plot_error_bars=False);
    savefig(fig, f"{SAVEDIR}/scatter-datapt=trialcode.pdf")

    _, fig = plot_45scatter_means_flexible_grouping(DFSCORES, "dcd_eq_fix|draw", "0|1", "1|0", "early-or-late-planning-period", 
                                        "score", "decoder_class", True);
    savefig(fig, f"{SAVEDIR}/scatter-datapt=decoder_class.pdf")
    plt.close("all")

    ### Final score, split by saccade idx
    fig = sns.catplot(data=DFSCORES, x="dcd_eq_fix|draw", y="score", hue="shape-macrosaccade-index", kind="bar")
    savefig(fig, f"{SAVEDIR}/catplot-shape-macrosaccade-index.pdf")

    ### Final summary, score compare looking at vs. draw
    for suff, dfscores in [
        ("datapt=trial", DFSCORES), 
        ("datapt=aggcondition", DFSCORES_AGG), 
    ]:

        # - 
        fig = sns.catplot(data=dfscores, x="dcd_eq_fix|draw", y="score", col="early-or-late-planning-period", alpha=0.1, jitter=True)
        savefig(fig, f"{SAVEDIR}/catplot-1-{suff}.pdf")

        fig = sns.catplot(data=dfscores, x="dcd_eq_fix|draw", y="score", hue="early-or-late-planning-period", kind="boxen")
        savefig(fig, f"{SAVEDIR}/catplot-2-{suff}.pdf")

        fig = sns.catplot(data=dfscores, x="dcd_eq_fix|draw", y="score", hue="early-or-late-planning-period", kind="bar")
        savefig(fig, f"{SAVEDIR}/catplot-4-{suff}.pdf")

    ### Relationship between fixation index and early/late
    if False:
        grouping_print_n_samples(dflab, ["is-first-macrosaccade", "event_idx_within_trial", "early-or-late-planning-period"]) #"event_idx_within_trial"]);

    fig = grouping_plot_n_samples_conjunction_heatmap_helper(dflab, ["shape-macrosaccade-index", "shape-fix_draw", "early-or-late-planning-period"]);    
    savefig(fig, f"{SAVEDIR}/counts.pdf")   

    plt.close("all")

    ### STATS, pairwise
    from pythonlib.tools.statstools import signrank_wilcoxon, signrank_wilcoxon_from_df, ttest_unpaired, compute_all_pairwise_stats_wrapper
    vars_grp = "dcd_eq_fix|draw"
    var_score = "score"

    for test_ver in ["ttest", "rank_sum"]:
        for early_late in ["early", "late"]:
            savedir = f"{SAVEDIR}/stats_datpt=agg-{early_late}-test_ver={test_ver}"
            os.makedirs(savedir, exist_ok=True)
            dfscores_this = DFSCORES_AGG[DFSCORES_AGG["early-or-late-planning-period"] == early_late].reset_index(drop=True)
            compute_all_pairwise_stats_wrapper(dfscores_this, vars_grp, var_score, doplots=True, savedir=savedir, test_ver=test_ver)

            savedir = f"{SAVEDIR}/stats_datpt=fix-{early_late}-test_ver={test_ver}"
            os.makedirs(savedir, exist_ok=True)
            dfscores_this = DFSCORES[DFSCORES["early-or-late-planning-period"] == early_late].reset_index(drop=True)
            compute_all_pairwise_stats_wrapper(dfscores_this, vars_grp, var_score, doplots=True, savedir=savedir, test_ver=test_ver)
            plt.close("all")
            
    return DFSCORES, DFSCORES_AGG

if __name__=="__main__":
    import sys

    PLOTS_DO = [2]
    # PLOTS_DO = [5.1]
    
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

            cetegory_expt_version = "switching"

            ### Load a daily dataset
            animal = sys.argv[1]
            DATE = sys.argv[2]
            SAVEDIR = f"/lemur2/lucas/analyses/manuscripts/1_action_symbols/fig2_categorization/{cetegory_expt_version}/{animal}-{DATE}"

            D = load_dataset_daily_helper(animal, DATE)

            ### Preprocess
            from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_preprocess_wrapper_GOOD
            savedir = f"{SAVEDIR}/preprocess"
            os.makedirs(savedir, exist_ok=True)
            NEURAL_PLOT_DRAWINGS = False
            DSmorphsets, map_tc_to_morph_info, map_morphset_to_basemorphinfo, \
                map_tcmorphset_to_idxmorph, map_tcmorphset_to_info, map_morphsetidx_to_assignedbase_or_ambig, \
                map_tc_to_morph_status = psychogood_preprocess_wrapper_GOOD(D, NEURAL_VERSION=True, NEURAL_SAVEDIR=savedir,
                                                NEURAL_PLOT_DRAWINGS=NEURAL_PLOT_DRAWINGS, cetegory_expt_version=cetegory_expt_version)
            
            from pythonlib.dataset.scripts.analy_manuscript_figures import fig2_categ_extract_dist_scores
            DFDISTS, DFINDEX = fig2_categ_extract_dist_scores(DSmorphsets, SAVEDIR, cetegory_expt_version=cetegory_expt_version)

        elif plot_do==5.1:
            ### Categories, switching, plot drawings in grid, for final manuscript.
            # goal is to show examples of drawing and task.
            
            animal = sys.argv[1]
            DATE = sys.argv[2]
            D = load_dataset_daily_helper(animal, DATE)
            
            ### Choose one of these

            # # Original plots of stats, keeping all data
            # NEURAL_VERSION = False # For stats (older plots)
            # remove_flankers=False
            # cetegory_expt_version=None

            # For making draings, grids, restrict to good trials
            NEURAL_VERSION = True # For making good draing figures (grid of drawings)
            remove_flankers=True
            cetegory_expt_version="switching"
            from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_preprocess_wrapper_GOOD

            savedir = "/tmp/PSYCHO"
            os.makedirs(savedir, exist_ok=True)
            NEURAL_PLOT_DRAWINGS = False 

            DSmorphsets, map_tc_to_morph_info, map_morphset_to_basemorphinfo, \
                map_tcmorphset_to_idxmorph, map_tcmorphset_to_info, \
                    map_morphsetidx_to_assignedbase_or_ambig, map_tc_to_morph_status = \
                        psychogood_preprocess_wrapper_GOOD(D, NEURAL_VERSION=NEURAL_VERSION, 
                                                        NEURAL_SAVEDIR=savedir, 
                                                        NEURAL_PLOT_DRAWINGS=NEURAL_PLOT_DRAWINGS,
                                                        remove_flankers=remove_flankers,
                                                        cetegory_expt_version=cetegory_expt_version)

            #  Plotting drawings, combining all image sets.
            SAVEDIR = f"/lemur2/lucas/analyses/manuscripts/1_action_symbols/fig2_categorization/DRAWING_GRID/{animal}-{DATE}"
            os.makedirs(SAVEDIR, exist_ok=True)


            from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_plot_drawings_morphsets_manuscript
            psychogood_plot_drawings_morphsets_manuscript(D, DSmorphsets, SAVEDIR)

        elif plot_do==4.2:
            # Categories (smooth), plotting motor distances of each index vs. endpoint (base prims), and doing
            # analyses/stats of that

            cetegory_expt_version = "smooth"

            ### Load a daily dataset
            animal = sys.argv[1]
            DATE = sys.argv[2]
            SAVEDIR = f"/lemur2/lucas/analyses/manuscripts/1_action_symbols/fig2_categorization/{cetegory_expt_version}/{animal}-{DATE}"

            D = load_dataset_daily_helper(animal, DATE)

            ### Preprocess
            from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_preprocess_wrapper_GOOD
            savedir = f"{SAVEDIR}/preprocess"
            os.makedirs(savedir, exist_ok=True)
            NEURAL_PLOT_DRAWINGS = False
            DSmorphsets, map_tc_to_morph_info, map_morphset_to_basemorphinfo, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info, map_morphsetidx_to_assignedbase_or_ambig, map_tc_to_morph_status = psychogood_preprocess_wrapper_GOOD(D, 
                                                                                                                                                                                                                            NEURAL_VERSION=True, 
                                                                                                                                                                                                                            NEURAL_SAVEDIR=savedir,
                                                                                                                                                                                                                            NEURAL_PLOT_DRAWINGS=NEURAL_PLOT_DRAWINGS,
                                                                                                                                                                                                                            cetegory_expt_version=cetegory_expt_version)
            
            from pythonlib.dataset.scripts.analy_manuscript_figures import fig2_categ_extract_dist_scores
            DFDISTS, DFINDEX = fig2_categ_extract_dist_scores(DSmorphsets, SAVEDIR, cetegory_expt_version=cetegory_expt_version)

        elif plot_do==6:
            """
            Generate and save new shuffled prims, splitting into two halves and reconnecting them.
            Takes the original prims, and makes a new prim from each pair, and then samples a new set, doing this 
            like 5 times. Each time apply filters to make sure the prims are different from eahc other, and different
            from the base prims (to some sufficient extent).
            """

            animal = "Pancho"
            stroke_shape_cluster_database_save_shuffled(animal)
            
        elif plot_do==7:
            """
            revision_eye_fixation_shape_decode

            Decoding (trained on single prims) testing on fixation data, asking whether PMv cares more about what shape fixated (vision) or 
            what prim plan to draw.

            Note: There is not code to agg mult days, since in the paper I just included I day per animal.

            To get single-trial example plot, use:

            sn.beh_eye_fixation_extract_and_assign_task_shape, in notebook:
            /home/lucas/code/neuralmonkey/neuralmonkey/notebooks/230430_eyetracking_overview.ipynb
            """
            from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_wrapper
            from neuralmonkey.analyses.decode_moment import pipeline_get_dataset_params_from_codeword, train_decoder_helper, _test_decoder_helper

            for animal, date in [
                ("Diego", 230615),
                ("Diego", 230628),
                ("Diego", 230630),
                ("Diego", 240625),
                ("Pancho", 230620),
                ("Pancho", 230622),
                ("Pancho", 230623), 
                ("Pancho", 230626),
                ("Pancho", 240612),
                ("Pancho", 240612),
                ]:

                try:
                    # (1) Load data
                    DFallpa = revision_eye_fixation_load_data(animal, date)
                    dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date)

                    # (2) Run thru each brain region
                    SAVEDIR = f"/lemur2/lucas/analyses/manuscripts/1_action_symbols/REVISION_PIG_eyetracking_decode/{animal}-{date}"
                    for bregion in DFallpa["bregion"].unique():
                        savedir = f"{SAVEDIR}/{bregion}"
                        os.makedirs(savedir, exist_ok=True)
                        revision_eye_fixation_shape_decode(DFallpa, bregion, savedir)
                except FileNotFoundError as err:
                    print(err)
                    print("Dfallpa not yet extracted, most likely")
                    continue
                except Exception as err:
                    raise err
                    
        else:
            assert False