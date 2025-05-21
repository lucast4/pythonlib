""" analysis of characters, including strokiness
Extract prims, gets their dsitance to basis sets of strokes, 
plots.
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pythonlib.tools.plottools import savefig
from pythonlib.tools.listtools import sort_mixed_type

def analy_preprocess_dataset(D, Dc, SAVEDIR):
    """
    Various preprocessing, including 
    label characters, novel, perfblocky, etc.
    """

    # (1) Novel chars
    savedir = f"{SAVEDIR}/preprocess"
    os.makedirs(savedir, exist_ok=True)

    # for Dthis in [D, Dc]:
    for Dthis in [D]:
        list_los = params_novel_chars_tasksets(Dthis, savedir)

        ### perfblocky chars.
        # For each los_set, check which are the task prims
        map_los_set_to_perfblocky = params_perfblocky_chars(Dthis, savedir)

    # Preprocess both same way
    for Dthis in [D, Dc]:
        if Dthis is not None:
            Dthis.preprocessGood(params=["no_supervision", "remove_online_abort"])

    # Final image accuracy (append, i.e, each char, convert to pts)
    D.score_visual_distance()
    if False: # Plot
        list_char = D.Dat[D.Dat["task_kind"]=="character"]["character"].unique().tolist()
        D.taskcharacter_find_plot_sorted_by_score("hdoffline", True, "/tmp", 1, 20, list_char=list_char)

    # Group chars by complexity
    D.Dat["los_set"] = [x[:2] for x in D.Dat["los_info"]] # (setnum, setind), without taskind.

    D.extract_beh_features()


def params_perfblocky_chars(D, savedir):
    """
    Are there perfblocky chars this date? 
    Will find them automatically

    RETURNS:
    - map_los_set_to_perfblocky
    (ALso modifies D.Dat, adding column <is_perfblocky>)
    """

    animal = D.animals(True)[0]
    date = int(D.dates(True)[0])
    if (animal, date) == ("Diego", 231204):
        perfblocky_exists = True
    elif (animal, date) == ("Diego", 231205):
        # NOTE: confirmed (compared to traingetterv2 Notes) that this works!
        perfblocky_exists = True
    elif (animal, date) == ("Pancho", 230122):
        # Didnt plan this...
        perfblocky_exists = True
    elif (animal, date) == ("Pancho", 230125):
        # NOTE: confirmed (compared to traingetterv2 Notes) that this works!
        perfblocky_exists = True
    elif (animal, date) == ("Pancho", 230126):
        # Did plan this
        perfblocky_exists = True
    elif (animal, date) == ("Pancho", 230127):
        # Did plan this
        perfblocky_exists = True
    else:
        # Just assume true, and then check that final results (text file below), since it is strict criteria for calling
        # it perfblocky
        perfblocky_exists = True
        # print(animal, date)
        # assert False

    if perfblocky_exists:
        map_los_set_to_perfblocky = {}
        for los_set in D.Dat["los_set"].unique().tolist():

            # collect all shapes for these los_set
            inds = D.Dat[D.Dat["los_set"] == los_set].index.tolist()
            shapes = []
            for i in inds:
                shapes.extend(D.taskclass_shapes_extract(i))

            # count how many are lines
            n_lines = sum([sh[:5]=="line-" for sh in shapes])
            n_tot = len(shapes)

            # many lines --> this is perfblocky
            is_perfblocky = n_lines>0.98*n_tot

            # Save
            map_los_set_to_perfblocky[los_set] = is_perfblocky
        
        D.Dat["is_perfblocky"] = [map_los_set_to_perfblocky[los_set] for los_set in D.Dat["los_set"]]
    else:
        D.Dat["is_perfblocky"] = False

    # Save list of novel chars
    D.grouping_print_n_samples(["is_perfblocky", "probe", "los_info", "block"], savepath=f"{savedir}/perfblocky_assignments-1.txt")
    D.grouping_print_n_samples(["block", "probe", "is_perfblocky", "los_info"], savepath=f"{savedir}/perfblocky_assignments-2.txt")
    D.grouping_print_n_samples(["block", "probe", "is_perfblocky", "los_set"], savepath=f"{savedir}/perfblocky_assignments-3.txt")
    D.grouping_print_n_samples(["block", "probe", "is_perfblocky", "task_kind"], savepath=f"{savedir}/perfblocky_assignments-4.txt")
    
    return map_los_set_to_perfblocky

def params_novel_chars_tasksets(D, savedir):
    """
    For each character, label whether it is a novel char, based on hand written notes in gslides for this experiment.
    This is the best way, since otherwise hard to auto detect across previuos days.
    RETURNS;
    - list_los
    ALSO modifies D.Dat, adding column: "novel_char"

    INSTRUCTION: find the list_los in probeGetterv2.m in dragmonkey. Then
    convert to strings to paste here using stringify_list_code_for_python(list_code)
    """

    # This allows for not manually inputing what is novel. Instaed, take probes, then take first trial of
    # that across all datasets and call that novel.
    HACK_CALL_PROBES_NOVEL = True 

    animal = D.animals(True)[0]
    date = int(D.dates(True)[0])

    if (animal, date) == ("Diego", 231130):
        list_los = []
    elif (animal, date) == ("Diego", 231122):
        # Somewhat confident these are novel.
        list_los = [
            (14, [ 176, 174, 161, 21, 182, 42, 124, 10, 137, 27, 104], "character"),
            (15, [ 85, 161, 157, 145, 91, 191, 55, 99, 146, 115, 185, 1, 31, 13, 42, 15, 43, 37], "character"),
        ]
    elif (animal, date) == ("Diego", 231128):
        list_los = []
    elif (animal, date) == ("Diego", 231129):
        list_los = []
    elif (animal, date) == ("Diego", 231201):
        # Lenient -- included all probes
        list_los = [
            (18, [ 11, 53, 18, 59, 25, 2, 36, 51, 14, 27, 56, 47, 22, 6, 50, 60, 13, 9, 49, 3, 20], "charstrokeseq"),
            (18, [ 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 80], "charstrokeseq"),
            (18, [ 8, 21, 24, 33, 34, 48, 52, 55, 57, 78, 1, 19, 35, 38, 58, 79], "charstrokeseq"),
            (19, [ 60, 74, 16, 51, 45, 33, 69, 48, 84, 73, 88, 44], "charstrokeseq"),
            (20, [ 19, 173, 191, 26, 11, 194, 199, 189, 35, 179], "charstrokeseq"),
            (19, [ 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 117, 118, 119, 120], "charstrokeseq"),
            (20, [ 173, 176, 179, 183, 185, 187, 189, 191, 192, 194, 196, 197, 199, 200], "charstrokeseq"),
            (19, [ 1, 2, 3, 6, 7, 9, 10, 11, 12, 18, 19, 20, 21, 22], "charstrokeseq"),
            (20, [ 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95], "charstrokeseq"),
        ]
    elif (animal, date) == ("Diego", 231204):
        list_los = [
            (34, [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59], "charstrokeseq"),
            (35, [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], "charstrokeseq"),
            (37, [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33], "charstrokeseq"),
        ]
    elif (animal, date) == ("Diego", 231211):
        list_los = [
            (63, [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64], "charstrokeseq"),
            (64, [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "charstrokeseq"),
            (65, [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64], "charstrokeseq"),
            (66, [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "charstrokeseq"),
        ]

    elif (animal, date) == ("Diego", 231205):
        list_los = [
            (41, None, 'charstrokeseq'),
            (42, None, 'charstrokeseq'),
            (43, None, 'charstrokeseq'),
            (44, None, 'charstrokeseq'),
            (45, None, 'charstrokeseq'),
            (46, None, 'charstrokeseq'),
            (47, None, 'charstrokeseq')
        ]
    elif (animal, date) == ("Pancho", 230122):
        # None listed
        list_los = []
    elif (animal, date) == ("Pancho", 230125):
        # These are thes ones I called:
        # 2. Blocky tasks done only by Diego. [Generated by lines (char star and reg), done across all epochs.]
        # 3. (Any) tasks done by Diego only (all epoch, all prims tasks).
        # (in the gslides). I assume that the other ones were previously done by Pancho a long time ago (80% confident),
        # Otherwise should consider including them too.
        list_los = [
            (9, [ 2, 14], "charstrokeseq"),
            (10, [ 14, 24], "charstrokeseq"),
            (12, [ 19, 45], "charstrokeseq"),
            (13, [ 9, 8, 61, 28, 84, 31], "charstrokeseq"),
            (14, [ 6, 89, 59, 100, 70, 12, 34, 49, 88, 50, 47, 52, 20, 48], "charstrokeseq"),
            (15, [ 13, 99, 43, 80], "charstrokeseq"),
            (16, [ 7, 11], "singleprims"),
            (17, [ 45, 11, 35, 16, 55, 58], "singleprims"),
            (18, [ 1, 42, 3], "singleprims"),
            (20, [ 43, 6, 8], "singleprims"),
            (21, [ 45, 96, 24, 21, 42, 74, 22, 54, 86, 2], "singleprims"),
            (22, [ 20, 68, 93, 51, 32, 47, 90, 61, 17, 84, 38, 8], "singleprims"),
        ]
    elif (animal, date) == ("Pancho", 230126):
        # None listed
        list_los = []
    elif (animal, date) == ("Pancho", 230127):
        # None listed -- note there are some but those he did long time ago
        list_los = []
    else:
        if HACK_CALL_PROBES_NOVEL:
            list_los = "probes_equals_novel"
        else:
            print(animal, date)
            assert False

    if HACK_CALL_PROBES_NOVEL:
        D.Dat["novel_char"] = D.Dat["probe"]==1
    else:
        def f(x):
            """ Any inds with None, this signifies get all indices, so
            so replace with very large list
            """
            if isinstance(x, (list, tuple)) and isinstance(x[0], int):
                return x
            elif x is None:
                return list(range(1, 10000))
            else:
                print(x)
                assert False
        # Convert each los to format (setname, setind, taskind)
        list_los = [[x[2], x[0], f(x[1])] for x in list_los]

        ### modify D.Dat.
        # (1) Novel chars
        def _check_if_los_in_list(los, list_los):
            for los_check in list_los:
                if los[0]==los_check[0]: # Setname
                    if los[1]==los_check[1]: # set ind
                        if los[2] in los_check[2]: # this ind, los[2], in list of inds, los_check[2]?
                            return True
            return False

        D.Dat["novel_char"] = [_check_if_los_in_list(los, list_los) for los in D.Dat["los_info"]]

    # Save list of novel chars
    D.grouping_print_n_samples(["novel_char", "probe", "los_info", "block"], savepath=f"{savedir}/novel_char_assignments-1.txt")
    D.grouping_print_n_samples(["block", "probe", "novel_char", "los_info"], savepath=f"{savedir}/novel_char_assignments-2.txt")
    D.grouping_print_n_samples(["block", "probe", "novel_char", "los_set"], savepath=f"{savedir}/novel_char_assignments-3.txt")
    D.grouping_print_n_samples(["block", "probe", "novel_char", "task_kind"], savepath=f"{savedir}/novel_char_assignments-4.txt")

    return list_los

                
def debug_eyeball_distance_metric_goodness(D):
    """ Script to comapre distnaces by eye to what loloks good. Iterates over distance metrics,
    and for eahc plots example strokes along with printing their scores. See that the scores
    match expected by eye (strokiness).
    NMOTE: in progress, but skeleton is there.
    """
        
    # collect data, once for each distance metric
    # list_distance_ver=["euclidian_diffs", "euclidian", "hausdorff_alignedonset"]
    list_distance_ver = ["dtw_vels_2d"]
    LIST_RES = []
    for dist in list_distance_ver:
        RES = generate_data(D, list_distance_ver=[dist])
        LIST_RES.append(RES)

    # plot for each metric
    list_inds = get_random_inds()
    for RES in LIST_RES:
        plot_example_trials(RES, list_inds=list_inds)


def pipeline_generate_and_plot_all(D, do_plots=True, filter_taskkind = "character",
    suffix=None):
    """
    Full pipeline to generate and plot and save all, for character analysis.
    """
    
    # if list_distance_ver is None:
    #     # list_distance_ver=("euclidian_diffs", "euclidian", "hausdorff_alignedonset")
    #     list_distance_ver = ["dtw_vels_2d"]

    savedir = D.make_savedir_for_analysis_figures("character_strokiness")
    if suffix is not None:
        savedir = f"{savedir}/{suffix}"

    assert len(D.Dat)>0

    RES = generate_data(D, filter_taskkind=filter_taskkind,
        # list_distance_ver=list_distance_ver,
        which_features="beh_motor_sim")

    if do_plots and RES is not None:

        DS = RES["DS"]
        Dthis = DS.Dataset # Because this is a copy of D that has been modded
        list_strok_basis = RES["list_strok_basis"]
        list_shape_basis = RES["list_shape_basis"]

        if "microstim_epoch_code" in DS.Dataset.Dat.columns:
            plot_epoch_effects_paired_chars_microstim_wrapper(DS, savedir)

        plot_clustering(DS, list_strok_basis, list_shape_basis, savedir)
        plot_learning_and_characters(Dthis, savedir)

        for MIN_SCORE in [0., 0.7]:
            for sorted_unique_shapes in [True, False]:
                plot_prim_sequences(RES, Dthis, savedir, MIN_SCORE,
                    sorted_unique_shapes)

    return RES, savedir

def generate_data(D, version_trial_or_shapemean="trial", 
        trial_summary_score_ver="clust_sim_max",
        plot_score_hist=False, filter_taskkind = None,
        ds_clean_methods = None, ds_clean_params = None,
        ds_filterdict = None,
        which_features = "beh_motor_sim"):
    """
    Initial data generation: extracts strokes, extracts basis
    set of strokes to compare them to, does similarity matrix
    PARAMS;
    - which_basis_set, which_shapes, params for getting bassis set of 
    strokes (see within)
    - trial_summary_score_ver, str, which statistic to use as the
    summary score (i.e, a strokiness score)
    - which_features, str, HACKY -- extract just a single feature space, 
    event hough multiple are computed, e.g, "beh_motor_sim"
    RETURNS:
    - RES, dict of results. -OR-
    - (or) None, if no tasks left after filter (e.g.,, filter to just chars)
    """
    from pythonlib.dataset.dataset_strokes import DatStrokes

    # assert list_distance_ver is None, "does nothing.."

    ### Generate Strokes data
    Dthis = D.copy()
    Dthis.preprocessGood(params=["no_supervision", "remove_online_abort"])

    DS = DatStrokes(Dthis)
    if ds_clean_methods is not None:
        DS.clean_preprocess_data(methods=ds_clean_methods, params=ds_clean_params)
    # Filter to just "character" tasks
    if filter_taskkind:
        DS.filter_dataframe({"task_kind":[filter_taskkind]}, True)
    if ds_filterdict is not None:
        DS.filter_dataframe(ds_filterdict, True)

    if len(DS.Dat)==0:
        # usually beuase not "character" taskkinds
        return None

    ## Score each trial across all features.
    ClustDict, ParamsDict, ParamsGeneral, dfdat = DS.features_wrapper_generate_all_features(version_trial_or_shapemean)

    ## Assign trials to clusters, based on a single feature space.
    DS.clustergood_assign_data_to_cluster(ClustDict, ParamsDict, 
            ParamsGeneral, dfdat,
            which_features = which_features,
            trial_summary_score_ver=trial_summary_score_ver)

    ###  order the prims by complexity
    def map_shape_to_complexity(sh):
        """ Had coded, the complexity of shapes based on how many segments..."""
        lev0 = ["line"]
        lev1 = ["Lcentered", "V", "arcdeep"]
        lev2 = ["squiggle3", "usquare", "zigzagSq", "circle"]

        if any([x in sh for x in lev0]):
            return 0
        elif any([x in sh for x in lev1]):
            return 1
        if any([x in sh for x in lev2]):
            return 2
        else:
            print(sh)
            assert False

    complexities = []
    for i in range(len(DS.Dat)):
        complexities.append(map_shape_to_complexity(DS.Dat.iloc[i]["clust_sim_max_colname"]))
    DS.Dat["clust_sim_max_colname_complexity"]=complexities

    # Prune dataset
    DS.dataset_prune_to_match_self()

    ## OUT
    RES = {
        "version_trial_or_shapemean":version_trial_or_shapemean,
        "ClustDict":ClustDict,
        "ParamsDict":ParamsDict,
        "ParamsGeneral":ParamsGeneral,
        "dfdat":dfdat,
        "trial_summary_score_ver":trial_summary_score_ver,
        "DS":DS,
        "which_features":which_features,
        "list_strok_basis":ParamsDict[which_features]["list_strok_basis"],
        "list_shape_basis":ParamsDict[which_features]["list_shape_basis"],
        # "list_distance_ver":list_distance_ver
        # "which_shapes":which_shapes,
        # "Cl":Cl,
    }
    
    if plot_score_hist:        
        plt.figure()
        plt.hist(Cl.Xinput)
    
    return RES


def plot_example_trials(RES, nrand=5, list_inds=None):
    """ Useful plots and prints for evaluating quality of strok-strok distance/simialrity
    matrix calcualtion. 
    Plot n example trials (i.e, extract strokes) overlaid on their dataset trials 
    (e.g., entire character). Also print similarituy and clustering results.
    """

    DS = RES["DS"]
    Cl = RES["Cl"]
    
    # pick random strokes and plot their concentration
    if list_inds is None:
        import random
        list_inds = random.sample(range(len(DS.Dat)), nrand)

    # DS.plot_multiple([1,2,100], titles_by_dfcolumn="clust_sim_max")
    # fig, axes, list_inds = DS.plot_multiple(None, titles_by_dfcolumn="clust_sim_max_colname", nrand=5)
    fig, axes, inds_trials_dataset = DS.plot_multiple_overlay_entire_trial(list_inds)
    fig, axes, inds_trials_dataset = DS.plot_multiple_overlay_entire_trial(list_inds, overlay_beh_or_task="task")

    # # plot task images
    # D.plotMultTrials(inds_trials_dataset, which_strokes="strokes_task")
    # DS.plot_beh_and_aligned_task_strokes(list_inds)

    # 
    # Print info about each stroke
    print("vec labels:", Cl.LabelsCols)
    print("")
    for ind in list_inds:
        simvec = DS.Dat.iloc[ind]["clust_sim_vec"]
        simmax = DS.Dat.iloc[ind]["clust_sim_max"]
        simmaxind = DS.Dat.iloc[ind]["clust_sim_max_ind"]
        simmaxname = DS.Dat.iloc[ind]["clust_sim_max_colname"] 
        simconc = DS.Dat.iloc[ind]["clust_sim_concentration"]
        # alignshape = DS.Dat.iloc[ind]["shape"]

        print(ind, int(100*simmax), int(100*simconc), simmaxname)
        print(simmaxind, simvec)
        print(" ")

def _plot_microstim_effect(DS, savedir):
    """ Plot effect of microstim on characters. Each plot is a scatter, with each char being a single datapt
    This only works if "microstim_epoch_code" is a column.
    Saves in (e.g) /gorilla1/analyses/main/character_strokiness/Diego_charstimdiego1_231209/epoch_effects
    """
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    import os

    Dthis = DS.Dataset

    if "microstim_epoch_code" not in Dthis.Dat.columns:
        return

    # Include Other metrics of strokes
    Dthis.extract_beh_features()

    # First, keep only chars that have both epochs
    dfthis, dict_dfthis = extract_with_levels_of_conjunction_vars(Dthis.Dat, "epoch", ["character"],
                                                                  n_min_across_all_levs_var=1)

    # Get names of stim epochs.
    stim_epochs = sort_mixed_type(dfthis["microstim_epoch_code"].unique().tolist())
    stim_epochs_on = [e for e in stim_epochs if not e=="off"]

    sdir = f"{savedir}/epoch_effects_charlevel"
    os.makedirs(sdir, exist_ok=True)
    print("Saving plots at: ", sdir)
    for var_outer in ["task_stagecategory", "epoch_orig"]:
        for feat in ["FEAT_angle_overall", "FEAT_num_strokes_beh", "FEAT_circ", "FEAT_dist", "strokes_clust_score"]:
            for stim_epoch in stim_epochs_on:
                for plot_label in [False, True]:
                    dfres, fig = plot_45scatter_means_flexible_grouping(dfthis, "microstim_epoch_code",
                                                                        "off", y_lev_manip=stim_epoch,
                                                                        var_value=feat, var_subplot=var_outer,
                                                                        var_datapt="character",
                                                           plot_text=plot_label, alpha=0.2, SIZE=4)
                    savefig(fig, f"{sdir}/by_{var_outer}-score_{feat}-scatter-vs-{stim_epoch}-label_{plot_label}.pdf")


            fig = sns.catplot(data=dfthis, x=var_outer, y=feat, hue="microstim_epoch_code", kind="point", errorbar=("ci", 68))
            path =f"{sdir}/by_{var_outer}-score_{feat}-pointplot.pdf"
            print(path)
            savefig(fig, path)
            plt.close("all")

def plot_epoch_effects_paired_chars_microstim_wrapper(DS, savedir, paired_by_char=True,
                                    SAMPLE_RANDOM_SINGLE_TRIAL=False):
    """
    Plot effects of microstim, where chars have both microstim and no-stim trials.
    :param DS:
    :return:
    """
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    from pythonlib.tools.pandastools import convert_to_1d_dataframe_hist
    from scipy.stats import entropy
    from pythonlib.tools.nptools import bin_values_categorical_factorize


    # Extract and preprocess
    DS = DS.copy()
    DS.dataset_append_column("microstim_epoch_code")
    Dthis = DS.Dataset

    # Keep only if char has both epochs
    if paired_by_char:
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
        Dthis.Dat, _ = extract_with_levels_of_conjunction_vars(Dthis.Dat, "epoch", ["character"],
                                                               n_min_across_all_levs_var=1)
        # - Prune DS so that chars are matched across epochs.
        DS.dataset_replace_dataset(Dthis)
        DS.dataset_prune_self_to_match_dataset()

    ## Pair up chars even more by getting a single trial per char
    if SAMPLE_RANDOM_SINGLE_TRIAL:
        from pythonlib.tools.pandastools import extract_trials_spanning_variable, append_col_with_grp_index
        DS.Dataset.Dat = append_col_with_grp_index(DS.Dataset.Dat, ["character", "epoch"],
                                                   "char_epoch", use_strings=True,
                                                   strings_compact=True)
        inds, chars = DS.Dataset.taskcharacter_extract_examples(var="char_epoch")
        trialcodes = Dthis.Dat.iloc[inds]["trialcode"].tolist()
        assert len(set(trialcodes))==len(trialcodes)

        # slice DS by trialcodes
        from pythonlib.tools.pandastools import slice_by_row_label
        dfthis = slice_by_row_label(DS.Dat, "trialcode", trialcodes)
    else:
        # interim, just use all data
        dfthis = DS.Dat
    DS.Dat = dfthis

    #### Datapt = char (Paired chars (scatter))
    _plot_microstim_effect(DS, savedir)

    #### LOCATION/SHAPE/INDEX distribitions
    _plot_location_shape_index_distributions(DS, savedir)

    #### MOTOR TIMING
    _plot_timing(DS, savedir)

    #### OTHER PLOTS
    sdir = f"{savedir}/epoch_effects_strokelevel"
    os.makedirs(sdir, exist_ok=True)

    ### PLOTS
    fig = sns.catplot(data=dfthis, x="clust_sim_max_colname", y="clust_sim_max", hue="epoch", aspect=1.5)
    rotateLabel(fig)
    savefig(fig, f"{sdir}/clust_sim_max-vs-clust_sim_max_colname-1.pdf")

    fig = sns.catplot(data=dfthis, x="clust_sim_max_colname", y="clust_sim_max", hue="epoch", aspect=1.5,
                      kind="point", errorbar=("ci", 68))
    rotateLabel(fig)
    savefig(fig, f"{sdir}/clust_sim_max-vs-clust_sim_max_colname-2.pdf")

    fig, ax = plt.subplots(1,1)
    sns.histplot(data=dfthis, x="clust_sim_max", hue="microstim_epoch_code", ax=ax)
    savefig(fig, f"{sdir}/hist_1d-clust_sim_max.pdf")

    fig = sns.catplot(data=dfthis, x="microstim_epoch_code", y="clust_sim_max", kind="point")
    rotateLabel(fig)
    savefig(fig, f"{sdir}/pointplot-clust_sim_max.pdf")

    # Histrograms
    fig, ax = plt.subplots()
    epochs = dfthis["epoch"].unique().tolist()
    for ep in epochs:
        dftmp = dfthis[dfthis["epoch"]==ep]
        # shapes, vals, fig, ax = convert_to_1d_dataframe_hist(dftmp, "clust_sim_max_colname", True, ax=ax)
        convert_to_1d_dataframe_hist(dftmp, "clust_sim_max_colname", True, ax=ax)
        # # fig.savefig(f"{sdir}/hist_n_matches.pdf")
    savefig(fig, f"{sdir}/hist_n_matches.pdf")

    for var in ["clust_sim_max_colname", "clust_sim_max_colname_complexity"]:
        _, fig, _, _ = convert_to_2d_dataframe(dfthis, var, "epoch", True, annotate_heatmap=True)
        savefig(fig, f"{sdir}/hist_2d-{var}.pdf")

        _, fig, _, _ = convert_to_2d_dataframe(dfthis, var, "epoch", True, norm_method="col_div",
                                               annotate_heatmap=True)
        savefig(fig, f"{sdir}/hist_2d-{var}-col_div.pdf")

        fig, ax = plt.subplots(1,1)
        sns.histplot(data=dfthis, x=var, hue="epoch", stat="probability",
                     multiple="dodge", ax=ax)
        from pythonlib.tools.plottools import rotate_x_labels
        rotate_x_labels(ax, 45)
        savefig(fig, f"{sdir}/hist_1d-{var}.pdf")


    ##### get entropy of shape names as a function of thresholding
    fig, ax = plt.subplots()
    threshes = np.linspace(0, 1, 20)
    for epoch in epochs:
        entropies = []
        list_n = []
        for thresh in threshes:

            # shapes = dfthis[dfthis["clust_sim_max"]>thresh]["clust_sim_max_colname"].factorize()
            shapes = dfthis[(dfthis["clust_sim_max"]>thresh) & (dfthis["epoch"]==epoch)]["clust_sim_max_colname"].tolist()
            shapes_int = bin_values_categorical_factorize(shapes)
            entropies.append(entropy(shapes_int))
            list_n.append(len(shapes_int))

        ax.plot(threshes, entropies, "-o", label=epoch)
        print("list n: ", epoch, list_n)
    ax.legend()
    ax.set_title("entropy of shape lables")
    ax.set_ylabel("entropy")
    ax.set_xlabel("thresh (keep data above this)")
    savefig(fig, f"{sdir}/entropy-vs-threshold_clust_stim.pdf")

    plt.close("all")


def _plot_timing(DS, savedir, microstim_version=True):
    """
    Plot effect of micrositm on timing...
    :param DS:
    :param savedir:
    :param microstim_version:
    :return:
    """
    # go cue to each stroke
    from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_plot_pipeline, plot_triallevel_results
    from pythonlib.dataset.dataset_analy.motortiming import _gapstrokes_preprocess_extract_strokes_gaps

    sdir = f"{savedir}/timing"
    os.makedirs(sdir, exist_ok=True)

    D = DS.Dataset

    # Dcopy = D.copy()
    # DS, SAVEDIR, dfres, grouping = preprocess_plot_pipeline(Dcopy, microstim_version=True, PLOT=False)

    # microstim_version = True
    # Merge the DS from strokes clust with DS from primtiveness.
    prune_strokes = True
    VARS_CONTEXT = None
    DS = _gapstrokes_preprocess_extract_strokes_gaps(DS, microstim_version, prune_strokes, VARS_CONTEXT)

    # Metric of stroke complexity

    # plot timing where context is simply stroke index (here shape is less relevant)
    contrast = "epoch"
    # savedir = "/tmp"
    if False:
        context = "context"
        # context var: stroke index x assigned shape
        DS.grouping_append_and_return_inner_items(["stroke_index", "clust_sim_max_colname"], new_col_name="context")
    else:
        context = "stroke_index"
    yvars = ["circularity", "time_duration", "velocity", "distcum", "gap_from_prev_dur",
            "gap_from_prev_dist", "gap_from_prev_vel"]
    plot_triallevel_results(DS, contrast, sdir, context, yvars)

def _plot_location_shape_index_distributions(DS, savedir):
    """
    Plot effect of microstim on distribiotn of shapes,. locations.
    :param RES:adfasdfasdf
    :param RES:adfasdfasdf
    :return:
    """
    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    from pythonlib.tools.plottools import color_make_pallete_categories

    sdir = f"{savedir}/loc_shape_index_distros"
    os.makedirs(sdir, exist_ok=True)

    # Thresholds, for keeping only decent shapes for this plot.
    thresh = 0.62
    max_nstrokes = 4

    # First, threshold it to just those that are good matches.
    # DS = RES["DS"]

    # get the center of the actual beh stroke
    def F(x):
        return x["Stroke"].extract_center()
    DS.Dat = applyFunctionToAllRows(DS.Dat, F, "beh_center")

    # Colors for each stroke index
    cmap = color_make_pallete_categories(DS.Dat, "stroke_index", "turbo")

    ### Plot distrubtions for each epoch
    list_epoch = DS.Dat["epoch"].unique()
    for epoch in list_epoch:
        dfthis = DS.Dat[(DS.Dat["clust_sim_max"]>thresh) & (DS.Dat["stroke_index"]<=max_nstrokes) & (DS.Dat["epoch"]==epoch)].reset_index(drop=True)
        # print(len(DS.Dat))
        # print(len(dfthis))

        # Combine all shapes, plot location
        SIZE = 4
        var = "beh_center"
        fig, ax = plt.subplots(1,1, figsize=(SIZE, SIZE))
        # locations
        locations = np.asarray(dfthis[var].tolist())
        indexes = np.asarray(dfthis["stroke_index"].tolist())
        colors = [cmap[i] for i in indexes]
        ax.scatter(locations[:,0], locations[:,1], c=colors, marker=".", alpha=0.3)
        savefig(fig, f"{sdir}/locations-ALLDATA-{epoch}.pdf")

        # Separate supblots for each level of (shape) or (stroke_index)
        for var_subplot in ["clust_sim_max_colname", "stroke_index"]:
            list_lev = sort_mixed_type(dfthis[var_subplot].unique())
            SIZE = 3
            var = "beh_center"
            ncols = 4
            nrows = int(np.ceil(len(list_lev)/ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE))

            for sh, ax in zip(list_lev, axes.flatten()):
                dfthisthis = dfthis[dfthis[var_subplot]==sh]
                ax.set_title(sh)

                # locations
                locations =  np.asarray(dfthisthis[var].tolist())
                indexes = np.asarray(dfthisthis["stroke_index"].tolist())
                colors = [cmap[i] for i in indexes]
                ax.scatter(locations[:,0], locations[:,1], c=colors, marker=".", alpha=0.25)
                # ax.hist2d(locations[:,0], locations[:,1])
                # ax.plot(locations[:,0], locations[:,1], '.k', alpha=0.2)

            savefig(fig, f"{sdir}/locations-{var_subplot}-{epoch}.pdf")

        # Bias for shapes
        _, fig, _, _ = convert_to_2d_dataframe(dfthis, "clust_sim_max_colname",
                                               "stroke_index", True, annotate_heatmap=False)
        savefig(fig, f"{sdir}/2dheat-shape-stroke_index-NONORM-{epoch}.pdf")

        for norm_method in ["row_div", "col_div"]:
            _, fig, _, _ = convert_to_2d_dataframe(dfthis, "clust_sim_max_colname",
                                                   "stroke_index", True, annotate_heatmap=False,
                                                   norm_method=norm_method)
            savefig(fig, f"{sdir}/2dheat-shape-stroke_index-{norm_method}-{epoch}.pdf")


def plot_learning_and_characters(D, savedir, scorename = "strokes_clust_score"):
    """ Plot and save all things related to:
    - strokiness of each trial, grouped by block, char, and day
    - changes in strokiness (i.e., learning)
    (Meant for across day analyssi)
    """ 
    from scipy.stats import linregress as lr
    from pythonlib.tools.nptools import rankItems

    sdir = f"{savedir}/figs_character_strokiness"
    import os
    os.makedirs(sdir, exist_ok=True)
    print("Saving figures at:", sdir)
    
    ## COrrelation between online scores (e.g, strokiness) and offline computed strokiness
    # sdir = f"{savedir}/corr_scores"
    # import os
    # os.makedirs(sdir, exist_ok=True)
    list_blocks = sorted(D.Dat["block"].unique().tolist())
    vars_to_compare = ["rew_total", "strokinessv2", "pacman", "hausdorff", "ft_decim", "numstrokesorig", "ft_minobj", "frac_strokes", "shortness", "score_final"]
    vars_to_compare = [var for var in vars_to_compare if var in D.Dat.columns]
    for bk in list_blocks:
        df = D.Dat[D.Dat["block"]==bk]
        if len(df)>0:
            try:
                if sum(~df[scorename].isna())>0:
                    fig = sns.pairplot(data=df, x_vars=vars_to_compare, y_vars=[scorename])
                    savefig(fig, f"{sdir}/corr-rew_vs_strokiness-bk{bk}.pdf")
                    # fig.savefig(f"{sdir}/corr-rew_vs_strokiness-bk{bk}.pdf")
                plt.close("all")
            except Exception as err:
                print("----", bk)
                print(len(df))
                print(df.keys())
                raise err

    ## Score for each trial by block and date
    fig = sns.catplot(data=D.Dat, x="block", y=scorename, aspect=2, row="date")
    fig.savefig(f"{sdir}/score_by_block-1.pdf")

    fig = sns.catplot(data=D.Dat, x="block", y=scorename, kind="violin", aspect=2, row="date")
    fig.savefig(f"{sdir}/score_by_block-2.pdf")

    fig = sns.catplot(data=D.Dat, x="block", y=scorename, kind="point", aspect=2, hue="date")
    fig.savefig(f"{sdir}/score_by_block-3.pdf")

    fig = sns.catplot(data=D.Dat, x="date", y=scorename, kind="point", aspect=2, hue="block")
    fig.savefig(f"{sdir}/score_by_date-1.pdf")

    fig = sns.catplot(data=D.Dat, col="block", col_wrap=4, x="date", y=scorename, aspect=1, kind="point")
    fig.savefig(f"{sdir}/score_by_date-3.pdf")

    fig = sns.relplot(data=D.Dat, col="block", col_wrap=3, x="tvalfake", y=scorename, aspect=2, kind="scatter")
    fig.savefig(f"{sdir}/score_by_tval_withinblock.pdf")

    if "strokinessv2" in D.Dat.columns:
        fig = sns.relplot(data=D.Dat, col="block", col_wrap=3, x="tvalfake", y="strokinessv2", aspect=2, kind="scatter")
        fig.savefig(f"{sdir}/strokinessv2_by_tval_withinblock.pdf")

    plt.close("all")

    ## For each character
    # Plot chars, sorted by score, for ALL and for each category.
    D.taskcharacter_print_score_per_char(scorename, sdir)

    # First, sort characters
    # Sort all characters by score, and plot them in a grid
    list_char, list_score, list_n = D.taskcharacter_find_plot_sorted_by_score(scorename,
                                                                      plot=True,
                                                                      sdir=sdir)

    # # Plot
    # # -- get one trial for each char
    #
    # from pythonlib.tools.pandastools import extract_trials_spanning_variable
    # n_iter = 3
    # for i in range(n_iter):
    #     inds, chars = extract_trials_spanning_variable(D.Dat, "character", list_char)
    #     assert chars == list_char
    #
    #     if len(inds)<60:
    #         indsthis = inds
    #         charsthis = chars
    #         list_score_this = list_score
    #     else:
    #         indsthis = inds[:30] + inds[-30:]
    #         charsthis = chars[:30] + chars[-30:]
    #         list_score_this = list_score[:30] + list_score[-30:]
    #
    #     # -- plot
    #     fig, axes, idxs = D.plotMultTrials2(indsthis, titles=charsthis, SIZE=3);
    #     fig.savefig(f"{sdir}/drawings_sorted_byscore-iter{i}-beh.pdf")
    #     fig, axes, idxs = D.plotMultTrials2(indsthis, "strokes_task", titles=list_score_this);
    #     fig.savefig(f"{sdir}/drawings_sorted_byscore-iter{i}-task.pdf")
    #
    #     plt.close("all")

    # which carhacters best
    fig = sns.catplot(data=D.Dat, y="character", x=scorename, height=10, hue="block",
               order = list_char)
    fig.savefig(f"{sdir}/char_sorted_score-1.pdf")

    fig = sns.catplot(data=D.Dat, y="character", x=scorename, height=8, col="block", col_wrap=2,
               order = list_char)
    fig.savefig(f"{sdir}/char_sorted_score-2.pdf")

    fig = sns.relplot(data=D.Dat, x="block", y=scorename, col="character", col_wrap=4, 
               col_order=list_char)
    fig.savefig(f"{sdir}/char_learning_block.pdf")
    # fig = sns.catplot(data=D.Dat, x="block", y=scorename, col="character", col_wrap=4, 
    #            col_order=list_char)
    # fig.savefig(f"{sdir}/char_score_by_block-1.pdf")

    fig = sns.relplot(data=D.Dat, x="tvalfake", y=scorename, col="character", col_wrap=4, 
               col_order=list_char, hue="block")
    fig.savefig(f"{sdir}/char_learning_tval.pdf")

    plt.close("all")
    
    ### Change in score over trials (each char one slope)
    list_char_alpha = sorted(D.Dat["character"].unique().tolist())
        
    # First, prune dataset to avoid (i) trials with supervision and (ii) trials
    # without enough strokes, both of which have artifically high strokiness.
    # Each trial, got correct n strokes?
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    def F(x):
        # Returns True if n beh strokes = or > than n task strokes.
        ntask = len(x["strokes_task"])
        nbeh = len(x["strokes_beh"])
        return nbeh>=ntask
    D.Dat = applyFunctionToAllRows(D.Dat, F, "nbeh_match_ntask")

    # For each character, get its change in score
    ONLY_TRIALS_WITH_ENOUGH_STROKES = True
    ONLY_TRIALS_IN_NOSUP_BLOCKS = True

    Dcopy = D.copy()
    if ONLY_TRIALS_IN_NOSUP_BLOCKS:
        Dcopy.preprocessGood(params=["no_supervision"])
    DF = Dcopy.Dat
    print("orignial", len(DF))
    # if ONLY_TRIALS_IN_NOSUP_BLOCKS:
    #     DF = DF[DF["supervision_stage_concise"]=="off|0||0"]
    # print("ONLY_TRIALS_IN_NOSUP_BLOCKS", len(DF))
    if ONLY_TRIALS_WITH_ENOUGH_STROKES:
        DF = DF[DF["nbeh_match_ntask"]==True]
    print("ONLY_TRIALS_IN_NOSUP_BLOCKS", len(DF))

    if "strokinessv2" in df.columns:
        list_slope = []
        list_slope_sk2 = []
        list_slope_rew = []
        for char in list_char_alpha:
            df = DF[DF["character"]==char]
            t = df["tvalfake"]
            v = df[scorename]
            strokinessv2 = df["strokinessv2"]
            rew_total = df["rew_total"]


            try:
                if len(t)>=5:
                    # convert to to rank
                    t = rankItems(t)

                    if len(np.unique(v))>1:
                        slope = lr(t, v)[0]
                    else:
                        slope = np.nan
                    list_slope.append(slope)

                    # make sure score and strokiness correlate with offline score
                    if len(np.unique(strokinessv2))>1:
                        slope = lr(strokinessv2, v)[0]
                    else:
                        slope = np.nan
                    list_slope_sk2.append(slope)

                    if len(np.unique(rew_total))>1:
                        slope = lr(rew_total, v)[0]
                    else:
                        slope = np.nan
                    list_slope_rew.append(slope)      

                else:
                    list_slope.append(np.nan)
                    list_slope_rew.append(np.nan)
                    list_slope_sk2.append(np.nan)
            except Exception as err:
                print(t)
                print(strokinessv2)
                print(v)
                print(rew_total)
                raise err


        # only keep chars with enough data
        inds = np.where(~np.isnan(list_slope))[0].tolist()
        list_char_alpha = [list_char_alpha[i] for i in inds]
        list_slope = [list_slope[i] for i in inds]
        list_slope_rew = [list_slope_rew[i] for i in inds]
        list_slope_sk2 = [list_slope_sk2[i] for i in inds]

        # Plot (each char)
        fig, axes = plt.subplots(1,3, figsize=(15, len(list_char_alpha)*0.16))

        if len(list_slope)>0:
            ax=axes.flatten()[0]
            ax.plot(list_slope, list_char_alpha, "ok")
            ax.axvline(0)
            ax.set_xlabel(f"slope ({scorename}/trials)")
            ax.grid(True)

            ax=axes.flatten()[1]
            ax.plot(list_slope_sk2, list_char_alpha, "ok")
            ax.axvline(0)
            ax.set_xlabel(f"slope ({scorename}/strokinessv2)")
            ax.grid(True)

            ax=axes.flatten()[2]
            ax.plot(list_slope_rew, list_char_alpha, "ok")
            ax.axvline(0)
            ax.set_xlabel(f"slope ({scorename}/rew_total)")
            ax.grid(True)

        fig.savefig(f"{sdir}/slope_score_vs_trial-each_char.pdf")

        # Plot, historgram across cahar
        fig, axes = plt.subplots(1,3, figsize=(9,2))

        if len(list_slope)>0:
            ax=axes.flatten()[0]
            ax.hist(list_slope, bins=20)
            ax.axvline(0, color="k")
            ax.set_xlabel(f"slope ({scorename}/trials)")

            ax=axes.flatten()[1]
            ax.hist(list_slope_sk2, bins=20)
            ax.axvline(0, color="k")
            ax.set_xlabel(f"slope ({scorename}/strokinessv2)")

            ax=axes.flatten()[2]
            ax.hist(list_slope_rew, bins=20)
            ax.axvline(0, color="k")
            ax.set_xlabel(f"slope ({scorename}/rew_total)")

        fig.savefig(f"{sdir}/slope_score_vs_trial-hist.pdf")

        # Does having higher slope for (rew vs. score) predict learning?
        fig, axes = plt.subplots(2,2, figsize=(8,8))

        if len(list_slope)>0:

            ax = axes.flatten()[0]
            ax.plot(list_slope_rew, list_slope, 'ok')
            ax.set_xlabel(f"slope ({scorename}/rew_total)")
            ax.set_ylabel(f"slope ({scorename}/trials)")

            ax = axes.flatten()[1]
            ax.plot(list_slope_sk2, list_slope, 'ok')
            ax.set_xlabel(f"slope ({scorename}/strokinessv2)")
            ax.set_ylabel(f"slope ({scorename}/trials)")

        fig.savefig(f"{sdir}/scatter-slopes_vs_slopes.pdf")

    # Close
    plt.close("all")


def plot_clustering(DS, list_strok_basis, list_shape_basis, savedir):
    """ Clustering beh stroke based on max similiaryt to prims. Plots example
    strokes (beh) to get a sense of how clustered are the strokes.
    """

    sdir = f"{savedir}/clustering_by_basis_prims"
    import os
    os.makedirs(sdir, exist_ok=True)

    from pythonlib.tools.pandastools import convert_to_1d_dataframe_hist
    shapes, vals, fig, ax = convert_to_1d_dataframe_hist(DS.Dat, "clust_sim_max_colname", True)
    # fig.savefig(f"{sdir}/hist_n_matches.pdf")
    savefig(fig, f"{sdir}/hist_n_matches.pdf")

    # fig = DS.plot_examples_grid("clust_sim_max_colname", col_levels=shapes, nrows=5)
    fig = DS.plotshape_multshapes_trials_grid("clust_sim_max_colname", 
        col_levels=shapes, nrows=5)
    fig.savefig(f"{sdir}/drawings_examplegrid_sorted_by_nmatches.pdf")

    # Plot the basis set stroke
    # sort by order
    list_strok_basis_sorted = [list_strok_basis[list_shape_basis.index(sh)] for sh in shapes]
    fig, axes = DS.plot_multiple_strok(list_strok_basis_sorted, overlay=False, ncols=len(list_strok_basis_sorted), titles=shapes);
    fig.savefig(f"{sdir}/drawings_examplegrid_sorted_by_nmatches-basis.pdf")

    figholder = DS.plotshape_multshapes_egstrokes(key_subplots = "clust_sim_max_colname",
                n_examples_total_per_shape = 5, color_by=None, list_shape=None);

    for i, x in enumerate(figholder):
        fig = x[0]
        fig.savefig(f"{sdir}/drawings_examples_overlaid-iter{i}.pdf")

    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    dfthis, fig, ax, rgba_values = convert_to_2d_dataframe(DS.Dat, "character", "clust_sim_max_colname", True, annotate_heatmap=False);
    # fig.savefig(f"{sdir}/hist_n_matches_2d_heat_characters.pdf")
    savefig(fig, f"{sdir}/hist_n_matches_2d_heat_characters.pdf")


    # for each basis strok, get the distribution of scores for beh strokes that match it
    fig = sns.catplot(data=DS.Dat, x="clust_sim_max_colname", y="clust_sim_max", aspect=2, order=shapes)
    from pythonlib.tools.snstools import rotateLabel
    rotateLabel(fig)
    fig.savefig(f"{sdir}/scoredist_shapes-1.pdf")

    fig = sns.catplot(data=DS.Dat, x="clust_sim_max_colname", y="clust_sim_max", aspect=2, order=shapes, kind="bar")
    from pythonlib.tools.snstools import rotateLabel
    rotateLabel(fig)
    fig.savefig(f"{sdir}/scoredist_shapes-2.pdf")

    plt.close("all")

    # For each basis stroke, plot examples, sorted by score
    from pythonlib.tools.pandastools import extract_trials_spanning_variable
    nrand = 40
    trials_dict = extract_trials_spanning_variable(DS.Dat, "clust_sim_max_colname", shapes, 
                                                   n_examples=nrand, return_as_dict=True, 
                                                   method_if_not_enough_examples="prune_subset")[0]

    for sh in shapes:
        inds = trials_dict[sh]
        scores = DS.Dat.iloc[inds]["clust_sim_max"].values.tolist()
        
        # sort
        x = [(i, s) for i,s in zip(inds, scores)]
        x = sorted(x, key=lambda x:-x[1])
        inds = [xx[0] for xx in x]
        scores = [xx[1] for xx in x]
        fig, axes, list_inds = DS.plot_multiple(inds, titles=scores, ncols=8)
        fig.savefig(f"{sdir}/behstrokes_matching_basis-{sh}-sorted_by_score.pdf")    

    # Also look in entire pool, regardless of shape
    from pythonlib.tools.listtools import random_inds_uniformly_distributed
    inds = random_inds_uniformly_distributed(DS.Dat["clust_sim_max"].tolist(), nrand)
    scores = DS.Dat.iloc[inds]["clust_sim_max"].values.tolist()
        
    # sort
    nrand = 64
    from pythonlib.tools.listtools import random_inds_uniformly_distributed
    inds = list(random_inds_uniformly_distributed(DS.Dat["clust_sim_max"].tolist(), nrand))
    scores = DS.Dat.iloc[inds]["clust_sim_max"].values.tolist()
    shapes = DS.Dat.iloc[inds]["clust_sim_max_colname"]

    titles = [f"{sh}|{sc:.2f}" for sh, sc in zip(shapes, scores)]

    fig, axes, list_inds = DS.plot_multiple(inds, titles=titles, ncols=8)
    fig.savefig(f"{sdir}/behstrokes_ALL-sorted_by_score.pdf")    

def plot_prim_sequences(RES, D, savedir, MIN_SCORE = 0., sorted_unique_shapes=False):
    """ Extract for each trial a sequence of strokes (shape names), and plot
    things about these (e.g. distribution across trialsS).
    PARAMS:
    - MIN_SCORE, scalar, only keep trials with mean strokes_clust_score >= this
    - sorted_unique_shapes, bool, then first get sorted(unqiue()) shapes for each trial.
    """
    from pythonlib.tools.plottools import makeColors, rotate_x_labels

    # MIN_SCORE = 0.75
    sdir = f"{savedir}/prim_sequences-MIN_SCORE_{MIN_SCORE}-sorteduniq_{sorted_unique_shapes}"
    import os
    os.makedirs(sdir, exist_ok=True)
    print("Saving at:", sdir)
    DS = RES["DS"]

    def _extract_shapes_beh_order(tc):
        """ get list of shapes for this trial in order of beh strokes
        """        
        # 1) shapes, based on simmat alignment
        shapes_ordered_by_beh_strokes_datseg = DS.dataset_extract_strokeslength_list(tc, "shape")

        # 2) shapes, based on clustering each stroke
        shapes_ordered_by_beh_strokes_cluster = DS.dataset_extract_strokeslength_list(tc, "clust_sim_max_colname")

        return shapes_ordered_by_beh_strokes_datseg, shapes_ordered_by_beh_strokes_cluster

    # For each char, get its "best parse"
    # list_scores = []
    list_shapes_datseg = []
    list_shapes_cluster = []
    for i in range(len(D.Dat)):
        tc = D.Dat.iloc[i]["trialcode"]
        if D.Dat.iloc[i]["strokes_clust_score"]>=MIN_SCORE:
            shapes_datseg, shapes_cluster = _extract_shapes_beh_order(tc)
            list_shapes_cluster.append(shapes_cluster)
            list_shapes_datseg.append(shapes_datseg)
        else:
            list_shapes_cluster.append(None)
            list_shapes_datseg.append(None)
    
    # Remove Nones
    A = []
    B = []
    for sh_c, sh_d in zip(list_shapes_cluster, list_shapes_datseg):
        if sh_c is None:
            continue
        A.append(sh_c)
        # B.append(sh_d)
    list_shapes_cluster = A
    # list_shapes_datseg = A

    # Colors, for plotting ranks.
    n = max([len(x) for x in list_shapes_cluster])
    pcols, fig, ax = makeColors(n, ploton=True)
    fig.savefig(f"{sdir}/legend_rank_colors.pdf")

    # "Waterfall"-like plot, showing sequences ordered by their freqeuncy.
    if sorted_unique_shapes:
        def F(x):
            x = tuple(sorted(set(x)))
            return x
    else:
        def F(x):
            x = tuple(x)
            return x
    for Nprims in [None, 1, 2, 3,4,5]:

        list_shapes_cluster_unique = [F(x) for x in list_shapes_cluster if x is not None]

        # only keep if has Nprims
        if Nprims is not None:
            list_shapes_cluster_unique = [x for x in list_shapes_cluster_unique if len(x)==Nprims]

        from pythonlib.tools.listtools import tabulate_list
        list_shapes_sorted = sorted(tabulate_list(list_shapes_cluster_unique, True), key=lambda x:-x[1])

        # Plot it
        fig, axes = plt.subplots(1,2, figsize=(10, 6))

        # 1) Waterfall
        ax = axes.flatten()[0]
        for i, shapes in enumerate(list_shapes_sorted):
            for j, (sh, col) in enumerate(zip(shapes[0], pcols)):
                ax.plot(sh, i, 'x', color=col, alpha=0.7)
        ax.set_ylabel('rank in ordered by freq (0, most frequent sequence)')
        ax.set_xlabel('shapes in sequence')
        ax.set_title(f"[Nprims={Nprims}], Most common sequences")
        # plt.xticks(rotation=45);
        # ax.draw(ax.figure.canvas.renderer)
        rotate_x_labels(ax, 45)

        #2) Hist, n strokes in sew
        ax = axes.flatten()[1]
        ax.set_xlabel("n shapes in sequence")
        list_n = [len(x[0]) for x in list_shapes_sorted]
        ax.hist(list_n)
        plt.xticks(rotation=45);

        fig.savefig(f"{sdir}/waterfall_hist-Nprims_{Nprims}.pdf")

        plt.close("all")


def analy_score_base_prim_reuse_score_match(DS, savedir, 
                                            shape_var_for_getting_threshold = "shape_orig", 
                                            shape_var_for_applying_threshold="clust_sim_max_colname", dist_var="clust_sim_max", 
                                            match_better_if_dist_high=True, map_shape_to_mindist_input=None,
                                            remove_outliers=False):
    """
    [NOTE] this mixes the calculation of distance thresholds (if map_shape_to_mindist_input is None) and
    the application of thresholds (no matter what map_shape_to_mindist_input is)

    Score how likely a stroke is to be assigned to a base prim. 
    Modifies by adding column: DS.Dat[matches_base_prim]

    Determines threshold to use to call "match" by taking a percentile of SP/PIG strokes, with assumption that
    those are ground-truth how close you would expect if he were indeed matching the base prims.

    # NOTE: this also internally does preprocessing to add onto DS and DSc the "match" labels for each stroke.

    RETURNS:
    - modifies DS.Dat. see note above.
    """
    from pythonlib.tools.plottools import savefig

    # # (1) Load basis shapes
    # if False:
    #     dfbasis, list_strok_basis, list_shape_basis = DS.stroke_shape_cluster_database_load_helper(plot_examples=True)
    # else:
    #     # Better. doesnt require DS and Dataset
    #     list_shape_basis = sorted(DS.Dat[shape_var_for_getting_threshold].unique().tolist())

    ### Give binary score to each stroke, whether it matches base prim, based on score distribgtuion during SP and PIG trials.
    # For each shape, what fraction are within the bounds based on SP and PIG
    def _label_whether_stroke_matches_baseprim(MIN_PRCTILE = 2.5, HACK=False):
        """
        Modifies DS.Dat
        """
        task_kinds_control = ["prims_single", "prims_on_grid"]

        if map_shape_to_mindist_input is None:
            # Get it auto, from SP/PIG tasks.
            map_shape_to_mindist = {}

            # Also add unique shapes from dataset
            list_shape_basis = sorted(DS.Dat[shape_var_for_getting_threshold].unique().tolist()) + DS.Dat[shape_var_for_getting_threshold].unique().tolist()
            list_shape_basis = set(list_shape_basis)
            for sh in list_shape_basis:
                df = DS.Dat[(DS.Dat[shape_var_for_getting_threshold] == sh) & (DS.Dat["task_kind"].isin(task_kinds_control))]
                if len(df)<8:
                    print("WARNING, too few trials for controls: ", sh)
                    map_shape_to_mindist[sh] = np.nan
                else:
                    # First remove outliers
                    if remove_outliers:
                        from pythonlib.tools.pandastools import remove_outlier_values_tukey
                        n1 = len(df)
                        df = remove_outlier_values_tukey(df, col=dist_var, niqr=3.5, method="remove_rows", PLOT=False, remove_directions="lower")
                        if len(df)<n1:
                            print("Removed, due to outlers:", n1, "-->", len(df), sh)
                        # assert False
                    map_shape_to_mindist[sh] = np.percentile(df[dist_var], [MIN_PRCTILE])[0]

            # # Also add unique shapes from dataset
            # for sh in DS.Dat[shape_var_for_getting_threshold].unique():
            #     df = DS.Dat[(DS.Dat[shape_var_for_getting_threshold] == sh) & (DS.Dat["task_kind"].isin(task_kinds_control))]
            #     if len(df)<8:
            #         print("WARNING, too few trials for controls: ", sh)
            #         map_shape_to_mindist[sh] = np.nan
            #     else:
            #         map_shape_to_mindist[sh] = np.percentile(df[dist_var], [MIN_PRCTILE])[0]
        else:
            # Use inputed mapping
            map_shape_to_mindist = map_shape_to_mindist_input

            # Any shapes in dataset that are not presnet, give them nan
            for sh in DS.Dat[shape_var_for_getting_threshold].unique():
                if sh not in map_shape_to_mindist:
                    map_shape_to_mindist[sh] = np.nan

        if HACK:
            # Cases without neough SP or PIG, replace with avbearger over others.
            val_replacement = np.mean([x for x in map_shape_to_mindist.values() if not np.isnan(x)])        
            map_shape_to_mindist = {sh:val if ~np.isnan(val) else val_replacement for sh, val in map_shape_to_mindist.items()}

        DS.Dat["match_thresh"] = [map_shape_to_mindist[sh] for sh in DS.Dat[shape_var_for_applying_threshold]]
        if match_better_if_dist_high:
            DS.Dat["matches_base_prim"] = DS.Dat[dist_var]>=DS.Dat["match_thresh"]
        else:
            DS.Dat["matches_base_prim"] = DS.Dat[dist_var]<=DS.Dat["match_thresh"]
        
        return map_shape_to_mindist

    # Collect frac match
    LIST_MIN_PRCTILE = np.linspace(0, 100, 30)
    list_frac_match = []
    for THRESH_PRCTILE in LIST_MIN_PRCTILE:
        _label_whether_stroke_matches_baseprim(THRESH_PRCTILE, HACK=True)
        frac_match = 100*np.mean(DS.Dat[DS.Dat["task_kind"] == "character"]["matches_base_prim"])
        list_frac_match.append(frac_match)
    # Pick a single min threshold to define "good match" and make plots.
    if match_better_if_dist_high:
        THRESH_PRCTILE_FINAL = 2.5
    else:
        THRESH_PRCTILE_FINAL = 97.5

    if savedir is not None:
        # Plot
        fig, ax = plt.subplots(1,1, figsize=(6, 6))
        ax.set_title("task_kind=character")
        ax.set_xlabel("Threshold score (percentile of SP/PIG)")
        ax.set_ylabel("Frac char strokes match base prim")
        ax.plot(LIST_MIN_PRCTILE, list_frac_match, "-ok")
        ax.axvline(THRESH_PRCTILE_FINAL, color="r")

        # - overlay expected, if is SP/PIG
        if match_better_if_dist_high:
            ax.plot(LIST_MIN_PRCTILE, 100-LIST_MIN_PRCTILE)
        else:
            ax.plot(LIST_MIN_PRCTILE, LIST_MIN_PRCTILE, "-ok")

        savefig(fig, f"{savedir}/frac_match-vs-chosen_threshold.pdf")

    ### Pick a single min threshold to define "good match" and make plots.
    # NOTE: this modifies DS in place.
    map_shape_to_mindist = _label_whether_stroke_matches_baseprim(THRESH_PRCTILE_FINAL, HACK=True)

    plt.close("all")

    return map_shape_to_mindist

def analy_score_base_prim_reuse_score_match_apply_(DS, map_shape_to_mindist_input,
                                                   shape_var="clust_sim_max_colname", dist_var="clust_sim_max", 
                                                   match_better_if_dist_high=True):
    """
    Score whether each stroke is a match or not, based on user inputed dict of match thresholds (map_shape_to_mindist_input)
    Modifies by adding column: DS.Dat[matches_base_prim]

    NOTE: is component of analy_score_base_prim_reuse_score_match, but here just applies a score to determine match. Above
    does too much

    RETURNS:
    - modifies DS.Dat. see note above.
    """
    # Any shapes in dataset that are not presnet, give them nan
    for sh in DS.Dat[shape_var].unique():
        if sh not in map_shape_to_mindist_input:
            print("Shape missing from map_shape_to_mindist_input (possibly one of many): ", sh)
            print(map_shape_to_mindist_input)
            print(DS.Dat[shape_var].unique())
            assert False, "the inputed dict is missing at least one shape"
            # map_shape_to_mindist_input[sh] = np.nan

    # if HACK:
    #     # Cases without neough SP or PIG, replace with avbearger over others.
    #     val_replacement = np.mean([x for x in map_shape_to_mindist_input.values() if not np.isnan(x)])        
    #     map_shape_to_mindist_input = {sh:val if ~np.isnan(val) else val_replacement for sh, val in map_shape_to_mindist_input.items()}

    DS.Dat["match_thresh"] = [map_shape_to_mindist_input[sh] for sh in DS.Dat[shape_var]]
    if match_better_if_dist_high:
        DS.Dat["matches_base_prim"] = DS.Dat[dist_var]>=DS.Dat["match_thresh"]
    else:
        DS.Dat["matches_base_prim"] = DS.Dat[dist_var]<=DS.Dat["match_thresh"]

def analy_score_base_prim_reuse_score_recompute_thresh(DictDS, MapsShapesToMindist, PLOT=False, savedir=None):
    """
    To reassign, to each shape in MapsShapesToMindist, what its threahold is, 
    in order to "align" the thresholds to the mean score for the shape (across data). 
    Does this by sorting the thresholds, and sorting the mean scores, and then taking 
    the lowest threshold for the shape wtih lowest mean score, etc...

    NOTE: this is usedul for applying actual thrahsolds to data from made-up shuffled prims.
    """

    MapsShapesToMindistReassigned = {}
    for basis in ["Pancho", "Diego"]:
        # # (1) For each shape, get its actual distribution of scores

        ds = DictDS[(basis, basis)]

        # (2) Get shapes in order of their mean dist in the actual data
        shapes_in_incr_order = ds.Dat.groupby("clust_sim_max_colname")["clust_sim_max"].mean().sort_values().index.tolist()

        # (3) Get saved thresholds
        if False:
            values_in_incr_order = sorted([MapsShapesToMindist[basis][sh] for sh in shapes_in_incr_order])
        else:
            # - ignore what the shapes actualyl are, jsut use the values as a pool of values
            thresh_pool = list(MapsShapesToMindist[basis].values())
            if len(thresh_pool)<len(shapes_in_incr_order):
                # if there are not enough in the pool, sample more randomly.
                from pythonlib.tools.listtools import random_inds_uniformly_distributed
                nget = len(shapes_in_incr_order) - len(thresh_pool)
                thresh_extras = random_inds_uniformly_distributed(thresh_pool, nget, return_original_values=True)
                thresh_pool = thresh_pool + thresh_extras
            values_in_incr_order = sorted(thresh_pool)
        assert len(values_in_incr_order) == len(shapes_in_incr_order)

        # re-allocate values
        MapsShapesToMindistReassigned[basis] = {sh:val for sh, val in zip(shapes_in_incr_order, values_in_incr_order)}

        if PLOT:
            from pythonlib.tools.plottools import rotate_x_labels
            fig, ax = plt.subplots()
            map_shape_to_mean_dist = ds.Dat.groupby("clust_sim_max_colname")["clust_sim_max"].mean().to_dict() # {shape:meandist}

            x = shapes_in_incr_order
            y = [map_shape_to_mean_dist[xx] for xx in x]
            ax.plot(x, y, "ok", label="mean dist in dataset")

            x = shapes_in_incr_order
            y = [MapsShapesToMindistReassigned[basis][xx] for xx in x]
            ax.plot(x, y, "or", label="new threshold")

            rotate_x_labels(ax)
            ax.legend()
            ax.set_title(basis)

            if savedir is not None:
                savefig(fig, f"{savedir}/mean_dist-vs-threshold-basis={basis}.pdf")

    return MapsShapesToMindistReassigned

def analy_score_base_prim_reuse_score_match_apply_wrapper(DictDS, MapsShapesToMindist, SAVEDIR,
                                                          PLOT=False, 
                                                          USE_REALLOCATED_THRESH=False,
                                                          SKIP_IMAGE=False):
    """
    Helper to apply thresholds to score whether each datapt is match or not, modifying DS.

    PARAMS:
    - MapsShapesToMindist, dict, holding thresholds.
    - USE_REALLOCATED_THRESH, bool, if True, then recomputes MapsShapesToMindist, but sorting the thresholds
    and mapping to shapes sorted by their distances in data.
    RETURNS:
    - (Modifies ds in DictDS)
    """
    from pythonlib.dataset.dataset_analy.characters import analy_score_base_prim_reuse_score_match_apply_

    # import copy
    # TMP = copy.deepcopy(MapsShapesToMindist)

    if USE_REALLOCATED_THRESH:
        MapsShapesToMindist = analy_score_base_prim_reuse_score_recompute_thresh(DictDS, MapsShapesToMindist, 
                                                                                 PLOT=PLOT, savedir=SAVEDIR)

    ### (1) Apply threshold to score whether strokes are beh matches to prims
    for animal_data in ["Diego", "Pancho"]:
        for basis in ["Diego", "Pancho"]:
            ds = DictDS[(animal_data, basis)]
            
            # Then use the thresholds for the basis (vs its own animal).
            map_shape_to_mindist = MapsShapesToMindist[basis]
            analy_score_base_prim_reuse_score_match_apply_(ds, map_shape_to_mindist, 
                                                        "clust_sim_max_colname", "clust_sim_max", 
                                                        match_better_if_dist_high=True)
            ds.Dat["FINAL_match"] = ds.Dat["matches_base_prim"]

    ### (2) Apply threshold to decide whether stroke matches image.
    if not SKIP_IMAGE:
        for animal_data in ["Diego", "Pancho"]:
            ds = DictDS[(animal_data, "image")]
            savedir = f"{SAVEDIR}/scores_summary-data={animal_data}-image_distance"
            os.makedirs(savedir, exist_ok=True)
            analy_score_base_prim_reuse_score_match(ds, savedir, "shape", "shape", "dist_beh_task_strok", 
                                                    match_better_if_dist_high=False)
            ds.Dat["IMAGE_matches"] = ds.Dat["matches_base_prim"]

        if False:
            ### OLD VERSION -- replaced with separate functions for stroke match (which can use the saved distances dict) and image dist
            # Now do preprocessing for each separately (getting shape matches).
            tmp2 = {}
            for animal_data in ["Diego", "Pancho"]:
                
                # (1) Against each basis
                for basis in ["Diego", "Pancho"]:
                    ds = DictDS[(animal_data, basis)]
                    
                    if animal_data!=basis:
                        # Then use the thresholds for the basis (vs its own animal).
                        map_shape_to_mindist = MapsShapesToMindist[basis]
                    else:
                        # Get automatically
                        map_shape_to_mindist = None

                    savedir = f"{SAVEDIR}/scores_summary-data={animal_data}-basis={basis}"
                    os.makedirs(savedir, exist_ok=True)

                    analy_score_base_prim_reuse_score_match(ds, savedir, "clust_sim_max_colname", "clust_sim_max", 
                                                            match_better_if_dist_high=True, 
                                                            map_shape_to_mindist_input=map_shape_to_mindist)

                    tmp2[(animal_data, basis)] = ds.Dat["matches_base_prim"].tolist()

                # (1) Against image
                ds = DictDS[(animal_data, "image")]
                savedir = f"{SAVEDIR}/scores_summary-data={animal_data}-image_distance"
                os.makedirs(savedir, exist_ok=True)
                analy_score_base_prim_reuse_score_match(ds, savedir, "shape", "dist_beh_task_strok", 
                                                        match_better_if_dist_high=False)
            
    if PLOT:
        print("Making plots...")
        savedir = f"{SAVEDIR}/final_labeling_matches"
        os.makedirs(savedir, exist_ok=True)

        # Plot the final thresholds
        for animal in ["Pancho", "Diego"]:
            for basis in ["Pancho", "Diego"]:
                ds = DictDS[(animal, basis)]
                fig = sns.catplot(data=ds.Dat, x="task_kind", y="clust_sim_max", hue="matches_base_prim", jitter=True, alpha=0.2, 
                            col="clust_sim_max_colname", col_wrap=6, order=["prims_single", "prims_on_grid", "character"])
                savefig(fig, f"{savedir}/all_scores_and_thresholds-animal={animal}-basis={basis}.pdf")
                        
                plt.close("all")

                # [GOOD] useful plots, show correspondance between label and ground-truth prim name 
                for tk in ["prims_single", "prims_on_grid"]:

                    for only_match in [False, True]:
                        
                        if only_match:
                            df = ds.Dat[(ds.Dat["task_kind"].isin([tk])) & (ds.Dat["matches_base_prim"]==True)]
                        else:
                            df = ds.Dat[ds.Dat["task_kind"].isin([tk])]

                        fig = sns.displot(df, x="clust_sim_max_colname", y="shape_semantic", height=6, aspect=1.2)
                        from pythonlib.tools.snstools import rotateLabel
                        rotateLabel(fig)

                        savefig(fig, f"{savedir}/pairwise_labels-animal={animal}-basis={basis}-taskkind={tk}-onlymatch={only_match}.pdf")
                        plt.close("all")

def analy_score_base_prim_reuse_plots(DS, savedir, shape_var="clust_sim_max_colname", 
                                      dist_var="clust_sim_max", matches_var="matches_base_prim", PLOT_DRAWINGS=True,
                                      n_rows_plots=8):
    """
    All plots scoring, across all strokes, similarity to base prims, or to image prims, depending
    on the values of inputs.
    PARAMS:
    - 
    """
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.snstools import rotateLabel

    list_shape_basis = sorted(DS.Dat[shape_var].unique().tolist())

    ### SAVe DS for each task kind
    DS_dict_task_kind = {}
    for task_kind in DS.Dat["task_kind"].unique():
        ds = DS.copy()
        ds.Dat = ds.Dat[ds.Dat["task_kind"]==task_kind].reset_index(drop=True)
        DS_dict_task_kind[task_kind] = ds

    if PLOT_DRAWINGS:
        ### Plot example strokes, cols = shapes, rows = sorted by feature score.
        col_levels = list_shape_basis + ["IGN"]
        if True: # Order columns to match the stacked bar
            shapes_ordered_by_counts = DS.Dat["clust_sim_max_colname"].value_counts().reset_index()["clust_sim_max_colname"].tolist() # High to low
            col_levels = shapes_ordered_by_counts
        recenter_strokes = True
        for task_kind, ds in DS_dict_task_kind.items():
            print("Plotting task_kind:", task_kind)
            if task_kind=="character":
                niter = 1 # moreis not random, as this is interpolating. 
            else:
                niter = 1
            for i_iter in range(niter):
                fig = ds.plotshape_multshapes_trials_grid_sort_by_feature(shape_var, 
                    col_levels=col_levels, nrows=n_rows_plots, sort_rows_by_this_feature=dist_var, SIZE=2, 
                    recenter_strokes=recenter_strokes)
                if fig is not None:
                    savefig(fig, f"{savedir}/drawings-sort_by_{dist_var}-taskkind={task_kind}-recenter={recenter_strokes}-iter={i_iter}.pdf")
        plt.close("all")

    # Frac strokes which are assigned as "match" to base prims
    fig = sns.catplot(data=DS.Dat, x="task_kind", y=matches_var, kind="bar")
    for ax in fig.axes.flatten():
        ax.set_ylim([0, 1])
    savefig(fig, f"{savedir}/frac_match-task_kind.pdf")
    
    # What fraction of trials are "good match"
    fig = sns.catplot(data=DS.Dat, x=shape_var, y=matches_var, kind="bar", row="task_kind")
    rotateLabel(fig)
    savefig(fig, f"{savedir}/frac_match-task_kind-shape.pdf")

    # ### Distribution of scores across all strokes    
    # - for each shape, compare its distribution across PIG, SP, and char
    fig = sns.catplot(data=DS.Dat, x="task_kind", y=dist_var, col=shape_var, hue=matches_var, 
                col_wrap=6, jitter=True, alpha=0.5)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot-dist_vs_shape-1.pdf")

    fig = sns.catplot(data=DS.Dat, x=shape_var, y=dist_var, row="task_kind", hue=matches_var, 
                jitter=True, alpha=0.5)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot-dist_vs_shape-2.pdf")

    fig = sns.catplot(data=DS.Dat, x="task_kind", y=dist_var, hue=matches_var, jitter=True, alpha=0.35)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot-dist_vs_shape-3.pdf")

    fig = sns.catplot(data=DS.Dat, hue="task_kind", y=dist_var, x=shape_var, kind="bar", aspect=2.5, height=4)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot-dist_vs_shape-4.pdf")

    # [GOOD] stacked frequency plot
    from pythonlib.tools.pandastools import plot_bar_stacked_histogram_counts
    from pythonlib.tools.plottools import rotate_x_labels
    fig, ax = plt.subplots()
    plot_bar_stacked_histogram_counts(DS.Dat, shape_var, matches_var, ax)
    rotate_x_labels(ax, 90)
    ax.set_ylabel("Counts")
    savefig(fig, f"{savedir}/stackedbar-shapes_matches.pdf")

    # also bar, just the matches
    fig, ax = plt.subplots()
    df = DS.Dat[DS.Dat[matches_var]==True].reset_index(drop=True)
    plot_bar_stacked_histogram_counts(df, shape_var, matches_var, ax)
    rotate_x_labels(ax, 90)
    ax.set_ylabel("Counts")
    savefig(fig, f"{savedir}/stackedbar-shapes-only_matches.pdf")

    ### Plot frac match vs. num cases (one dot for each shape)
    # vs. number of trials that exist
    shape_N_var = f"{shape_var}_N"
    for task_kind, ds in DS_dict_task_kind.items():
        dfchar = ds.Dat
        _dfcounts = dfchar.groupby(shape_var).size().reset_index()
        _dfcounts= _dfcounts.rename(columns={0:shape_N_var})
        dfchar = pd.merge(dfchar, _dfcounts, on=shape_var)
        dfchar_agg = aggregGeneral(dfchar, [shape_var], [matches_var, dist_var, shape_N_var])

        fig, ax = plt.subplots()
        sns.scatterplot(dfchar_agg, x=shape_N_var, y=matches_var, hue=dist_var, ax=ax)    
        savefig(fig, f"{savedir}/scatter-matches-vs-n_counts-taskkind={task_kind}.pdf")

    plt.close("all")

def analy_score_match_baseprim_vs_matchimage_strokes(DS, savedir):
    """
    For each stroke, compare whether is matched to base prim and/or ot image. 

    """
    from pythonlib.tools.pandastools import plot_subplots_heatmap

    # (1) [dat=strokes] Match to base prims is higher than match to images
    fig, axes = plt.subplots(2,2, figsize=(10,10), sharex=True, sharey=True)
    for ax, matchkind in zip(axes.flatten(), ["exclmatch_prim_image", "exclmatch_prim", "exclmatch_image", "exclmatch_none"]):
        sns.barplot(data=DS.Dat, x="task_kind", y=matchkind, ax=ax)
        ax.set_title(matchkind)
    savefig(fig, f"{savedir}/catplot-matches.pdf")

    fig, _ = plot_subplots_heatmap(DS.Dat, "matches_base_prim", "IMAGE_matches", "val", "task_kind", agg_method="counts", 
                        row_values=[True, False], col_values=[False, True], annotate_heatmap=True)
    savefig(fig, f"{savedir}/heatplot-matches-1.pdf")
    
    fig, _ = plot_subplots_heatmap(DS.Dat, "task_kind", "match_prim_imag_LABEL", "val", None, agg_method="counts", annotate_heatmap=True)
    savefig(fig, f"{savedir}/heatplot-matches-2.pdf")

    plt.close("all")

def analy_score_match_baseprim_vs_matchimage_trials(DS, savedir,  just_get_dfcounts=False):
    """
    For each trial(i.e., baseiclaly character), compare how mnay of its beh strokes are matched
    to base prim vs. to the task image.
    
    Also include stuff related to beahvior-image distnace (taking into account entire drawing, not strokes)
    """
    from pythonlib.tools.pandastools import aggregGeneral, append_col_with_grp_index
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping_from_wideform
    from pythonlib.tools.pandastools import plot_subplots_heatmap
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good
    from pythonlib.tools.pandastools import plot_subplots_heatmap

    if False:
        # Visaulize characters with diff visual distancse
        list_char = D.Dat[D.Dat["task_kind"]=="character"]["character"].unique().tolist()
        D.taskcharacter_find_plot_sorted_by_score("hdoffline", True, "/tmp", 1, 20, list_char=list_char)

    # Get dataframe where each trialcode is a single row -- with n matches as columns
    dfcounts = aggregGeneral(DS.Dat, ["trialcode"], ["FEAT_num_strokes_beh", "IMAGE_matches", "matches_base_prim", "hdoffline"], aggmethod=["sum", "mean"], nonnumercols=["character", "task_kind", "los_set"])
    dfcounts = append_col_with_grp_index(dfcounts, ["task_kind_Nbeh_first", "los_set_first"], "tkset")
    dfcounts = append_col_with_grp_index(dfcounts, ["matches_base_prim_sum", "IMAGE_matches_sum"], "n_matches_mot_imag")
    dfcounts["IMAGE_matches_sum_str"] = [str(x) for x in dfcounts["IMAGE_matches_sum"]]
    dfcounts["matches_base_prim_sum_str"] = [str(x) for x in dfcounts["matches_base_prim_sum"]]

    # use this for visaul distance stuff -- clean.
    dfcounts_clean, _ = extract_with_levels_of_var_good(dfcounts, 
                                                        ["matches_base_prim_sum", "IMAGE_matches_sum","task_kind_Nbeh_first"], 5)

    if not just_get_dfcounts:
        
        ### Plot - 
        _, fig = plot_45scatter_means_flexible_grouping_from_wideform(dfcounts, "IMAGE_matches_sum", "matches_base_prim_sum", 
                                                            "task_kind_Nbeh_first", "trialcode", shareaxes=True, plot_error_bars=False, 
                                                            plot_text=False, SIZE=4, alpha=0.1, jitter_value=0.2)
        savefig(fig, f"{savedir}/scatter-matches.pdf")

        ### Plot heatmap
        nmax = max([dfcounts["matches_base_prim_sum"].max(), dfcounts["IMAGE_matches_sum"].max()])
        row_values = list(range(nmax, -1, -1))
        col_values = list(range(0, nmax+1))
        for annotate_heatmap in [False, True]:
            fig, axes = plot_subplots_heatmap(dfcounts, "matches_base_prim_sum", "IMAGE_matches_sum", "val", "task_kind_Nbeh_first", 
                                            agg_method="counts", row_values=row_values, col_values=col_values, annotate_heatmap=annotate_heatmap)
            savefig(fig, f"{savedir}/heatmap-matches-splitby_tk_n-annotate={annotate_heatmap}.pdf")
        plt.close("all")

        ### Scatterplot vs. image distance
        # for col in ["task_kind_Nbeh_first", "tkset"]:
        for col in ["task_kind_Nbeh_first"]:

            # for xvar in ["matches_base_prim_sum", "IMAGE_matches_sum"]:
            #     fig = sns.catplot(data=dfcounts, x=xvar, y="hdoffline_mean", col="tkset", col_wrap = 6, alpha=0.2, jitter=True)
            #     savefig(fig, f"{savedir}/visual_score-catplot-x={xvar}-colvar={col}-1.pdf")

            #     fig = sns.catplot(data=dfcounts, x=xvar, y="hdoffline_mean", col="tkset", col_wrap = 6, kind="point")
            #     savefig(fig, f"{savedir}/visual_score-catplot-x={xvar}-colvar={col}-2.pdf")

            # Plot, overlaying two stroke-prim distances
            xlim = [-2, dfcounts_clean["hdoffline_mean"].max()+2]
            ylim = [-1, dfcounts_clean["matches_base_prim_sum"].max()+1]

            fig = sns.FacetGrid(dfcounts_clean, col=col, col_wrap = 6, xlim=xlim, ylim=ylim)
            fig.map(sns.kdeplot, "hdoffline_mean", "IMAGE_matches_sum", alpha=0.2)
            fig.map(sns.kdeplot, "hdoffline_mean", "matches_base_prim_sum", alpha=0.2, color="r")
            for ax in fig.axes.flatten():
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            savefig(fig, f"{savedir}/visual_score-overlaid-colvar={col}-1.pdf")

            fig = sns.FacetGrid(dfcounts_clean, col=col, col_wrap = 6, xlim=xlim, ylim=ylim)
            fig.map(sns.stripplot, "hdoffline_mean", "IMAGE_matches_sum", alpha=0.15, jitter=True, orient="h")
            fig.map(sns.stripplot, "hdoffline_mean", "matches_base_prim_sum", alpha=0.15, jitter=True, color="r", orient="h")
            for ax in fig.axes.flatten():
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            savefig(fig, f"{savedir}/visual_score-overlaid-colvar={col}-2.pdf")

            # Regression
            fig = sns.FacetGrid(dfcounts_clean, col=col, col_wrap = 6, xlim=xlim, ylim=ylim)
            fig.map(sns.regplot, "hdoffline_mean", "IMAGE_matches_sum")
            savefig(fig, f"{savedir}/visual_score-regr-IMAGE_matches_sum.pdf")

            fig = sns.FacetGrid(dfcounts_clean, col=col, col_wrap = 6, xlim=xlim, ylim=ylim)
            fig.map(sns.regplot, "hdoffline_mean", "matches_base_prim_sum", color="r")
            savefig(fig, f"{savedir}/visual_score-regr-matches_base_prim_sum.pdf")

            plt.close("all")

            # Key points:
            # (1) Even without matching task components, is still getting decent image similarity
            fig = sns.catplot(data=dfcounts_clean, x="matches_base_prim_sum", y="hdoffline_mean", hue="IMAGE_matches_sum_str", col=col, 
                        col_wrap=6, jitter=True, alpha=0.5)
            savefig(fig, f"{savedir}/visual_score-catplot-matches_base_prim_sum=colvar={col}-1.pdf")

            fig = sns.catplot(data=dfcounts_clean, x="IMAGE_matches_sum", y="hdoffline_mean", hue="matches_base_prim_sum_str", col=col, 
                        col_wrap=6, jitter=True, alpha=0.5)
            savefig(fig, f"{savedir}/visual_score-catplot-IMAGE_matches_sum=colvar={col}-1.pdf")

            fig = sns.catplot(data=dfcounts_clean, x="matches_base_prim_sum", y="hdoffline_mean", hue="IMAGE_matches_sum_str", col=col, 
                        col_wrap=6, kind="point")
            savefig(fig, f"{savedir}/visual_score-catplot-matches_base_prim_sum=colvar={col}-2.pdf")

            fig = sns.catplot(data=dfcounts_clean, x="IMAGE_matches_sum", y="hdoffline_mean", hue="matches_base_prim_sum_str", col=col, 
                        col_wrap=6, kind="point")
            savefig(fig, f"{savedir}/visual_score-catplot-IMAGE_matches_sum=colvar={col}-2.pdf")

            plt.close("all")

        # Good.
        nmax = max([dfcounts_clean["matches_base_prim_sum"].max(), dfcounts_clean["IMAGE_matches_sum"].max()])
        row_values = list(range(nmax, -1, -1))
        col_values = list(range(0, nmax+1))
        fig, _ = plot_subplots_heatmap(dfcounts_clean, "matches_base_prim_sum", "IMAGE_matches_sum", "hdoffline_mean", "task_kind_Nbeh_first", 
                            agg_method="mean", row_values=row_values, col_values=col_values, annotate_heatmap=False, share_zlim=True);
        savefig(fig, f"{savedir}/heatmap-matches_vs_visualdist.pdf")

        plt.close("all")

    return dfcounts, dfcounts_clean


def analy_score_combined_animals_bases(DictDS, savedir, QUICK=False):
    """
    Combining animals and basis prims into a single dataset, and plots to directly compare them.
    
    Also finds common characters across aniamls, and plots just those.

    RETURNS:
    - DScomb, DS holding all data crossing (animal, basis) [does NOT prune to common chars]
    - dfchar_shared, df, just the chars that are common across animals.
    """
    from pythonlib.dataset.dataset_strokes import concat_dataset_strokes_minimal, concat_dataset_strokes
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from pythonlib.tools.pandastools import aggregGeneral, append_col_with_grp_index
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from pythonlib.tools.snstools import rotateLabel

    def _plot(DFTHIS, savesuff):
        from pythonlib.tools.pandastools import grouping_print_n_samples, grouping_plot_n_samples_conjunction_heatmap
        from pythonlib.tools.pandastools import stringify_values

        sdir = f"{savedir}/{savesuff}"
        os.makedirs(sdir, exist_ok=True)
        
        DFTHIS = stringify_values(DFTHIS)

        fig = sns.catplot(data=DFTHIS, x="FINAL_basis", y="FINAL_dist", col="task_kind", hue="novel|blocky", 
                    row="animal", jitter=True, alpha=0.2)
        savefig(fig, f"{sdir}/catplot-FINAL_dist-all-1.pdf")

        fig = sns.catplot(data=DFTHIS, x="FINAL_basis", y="FINAL_dist", col="task_kind", 
                        row="animal", jitter=True, alpha=0.2)
        savefig(fig, f"{sdir}/catplot-FINAL_dist-all-2.pdf")

        fig = sns.catplot(data=DFTHIS, x="FINAL_basis", y="FINAL_dist", col="task_kind", hue="novel|blocky", 
                    row="animal", kind="violin")
        savefig(fig, f"{sdir}/catplot-FINAL_dist-all-3.pdf")

        fig = sns.catplot(data=DFTHIS, x="FINAL_basis", y="FINAL_dist", col="task_kind", hue="novel_char", 
                    row="animal", kind="violin")
        savefig(fig, f"{sdir}/catplot-FINAL_dist-all-4.pdf")

        fig = sns.catplot(data=DFTHIS, x="FINAL_basis", y="FINAL_match", col="task_kind", hue="novel|blocky", 
                    row="animal", kind="bar")
        savefig(fig, f"{sdir}/catplot-matches_base_prim-all-3.pdf")

        fig = sns.catplot(data=DFTHIS, x="FINAL_basis", y="FINAL_match", col="task_kind", hue="novel_char", 
                    row="animal", kind="bar")
        savefig(fig, f"{sdir}/catplot-matches_base_prim-all-4.pdf")

        ### Print useful things       
        path = f"{sdir}/counts-1.txt"
        grouping_print_n_samples(DFTHIS, ["task_kind", "animal", "novel_char", "FINAL_basis"], savepath=path)  

        # path = f"{sdir}/counts-2.txt"
        # grouping_print_n_samples(DFTHIS, ["task_kind", "animal", "date", "novel_char", "FINAL_basis"], savepath=path)   

        ### Print useful things       
        if "los_info" in DFTHIS.columns:
            path = f"{sdir}/counts-3.txt"
            grouping_print_n_samples(DFTHIS, ["task_kind", "animal", "novel_char", "los_info", "FINAL_basis"], savepath=path)   

            path = f"{sdir}/counts-4_testing_why_los_has_mult_trials.txt"
            grouping_print_n_samples(DFTHIS, ["task_kind", "FINAL_basis", "los_info", "animal", "novel_char"], savepath=path)   

        ### Stats
        from pythonlib.tools.statstools import signrank_wilcoxon_from_df
        for is_novel in [False, True]:
            for animal in ["Diego", "Pancho"]:
                dfthis = DFTHIS[(DFTHIS["novel_char"] == is_novel) & (DFTHIS["animal"] == animal)].reset_index(drop=True)

                if "los_info" in DFTHIS.columns:
                    datapt_vars = ["los_info"] # data
                else:
                    datapt_vars = ["los_set"] # data

                value_var = "FINAL_match"
                contrast_var = "FINAL_basis"
                contrast_levels = ["Diego", "Pancho"]

                path = f"{sdir}/stats-animal={animal}-is_novel={is_novel}-contrast_var={contrast_var}-datapt={datapt_vars}.txt"
                out, fig = signrank_wilcoxon_from_df(dfthis, datapt_vars, contrast_var, contrast_levels, value_var, PLOT=True,
                                                save_text_path=path)
                savefig(fig, f"{sdir}/stats-animal={animal}-is_novel={is_novel}-contrast_var={contrast_var}-datapt={datapt_vars}.pdf")
                plt.close("all")

        ###################
        fig = sns.catplot(data=DFTHIS, x="animal", y="FEAT_num_strokes_beh", hue="novel|blocky", kind="violin", aspect=2)
        rotateLabel(fig)
        savefig(fig, f"{sdir}/catplot-FEAT_num_strokes_beh-1.pdf")

        fig = sns.catplot(data=DFTHIS, x="FEAT_num_strokes_beh", y="FINAL_match", col="animal|basis", hue="novel|blocky", kind="point")
        rotateLabel(fig)
        savefig(fig, f"{sdir}/catplot-FEAT_num_strokes_beh-2.pdf")
        
        fig = sns.catplot(data=DFTHIS, x="FEAT_num_strokes_beh", y="FINAL_match", col="animal|basis", hue="novel_char", kind="point")
        rotateLabel(fig)
        savefig(fig, f"{sdir}/catplot-FEAT_num_strokes_beh-3.pdf")

        # Scatterplot, for each stroke, compare its score against the two base prims
        if False: # TOo slow.
            for animal_data in ["Diego", "Pancho"]:
                task_kind = "character"
                dfthis = DFTHIS[(DFTHIS["animal"] == animal_data) & (DFTHIS["task_kind"] == task_kind)].reset_index(drop=True)
                _, fig = plot_45scatter_means_flexible_grouping(dfthis, "FINAL_basis", "Diego", "Pancho", "novel|blocky", 
                                                                "FINAL_dist", "index_datapt", plot_text=False, 
                                                                plot_error_bars=False, alpha=0.05, SIZE=3.5)
                savefig(fig, f"{sdir}/scatter-vs-basis-animal_data={animal_data}.pdf")

        plt.close("all")

        # (1) Agg, so los_info is single datapt
        # dfcounts_shared = aggregGeneral(dfchar_shared, ["animal", "FINAL_basis", "los_info", "novel|blocky"], ["FEAT_num_strokes_beh", "FINAL_dist", "FINAL_match"], aggmethod=["mean"], nonnumercols=["task_kind", "los_set"])
        # dfchar_shared_str = stringify_values(dfchar_shared)
        # assert pd.crosstab(dfchar_shared_str["los_info"], dfchar_shared_str["animal"]).min(axis=1).min()>0, "failure means a los_info was not done by all the animals."

        if "los_info" in DFTHIS.columns:
            var_data = "los_info"
        else:
            var_data = "los_set"

        for var_value, plot_error_bars in [
            ("FINAL_match", False),
            # ("los_set", "FINAL_match", True),
            ("FINAL_dist", False),
            # ("los_set", "FINAL_dist", True),
            ]:
            # for alpha in [0.1, 0.3]:
            for alpha in [0.2]:
                _, fig = plot_45scatter_means_flexible_grouping(DFTHIS, "FINAL_basis", "Diego", "Pancho", "animal", 
                                                                var_value, var_data, False, shareaxes=True, plot_error_bars=plot_error_bars, 
                                                                alpha=alpha, SIZE=3.5)
                savefig(fig, f"{savedir}/scatter-data={var_data}-value={var_value}-alpha={alpha}-errorbars={plot_error_bars}.pdf")

        _, fig = plot_45scatter_means_flexible_grouping(DFTHIS, "novel_char", True, False, "animal|basis", "FINAL_match", "los_set", 
                                            False, shareaxes=True, alpha=0.5)
        savefig(fig, f"{savedir}/scatterNOVEL.pdf")

        plt.close("all")


    ### Genreate single combined dataset
    # list_ds = []
    # for animal_data in ["Diego", "Pancho"]:
    #     ### (1) Collect df, across bases, and concat as new orjggt
    #     # for basis in ["Diego", "Pancho", "image"]:
    #     for basis in ["Diego", "Pancho"]:
    #         ds = DictDS[(animal_data, basis)]
    #         ds.Dat["FINAL_basis"] = basis
            
    #         if basis in ["Diego", "Pancho"]:
    #             ds.Dat["FINAL_shape"] = ds.Dat["clust_sim_max_colname"]
    #             ds.Dat["FINAL_dist"] = ds.Dat["clust_sim_max"]
    #             ds.Dat["FINAL_match"] = ds.Dat["matches_base_prim"]
    #         elif basis == "image":
    #             ds.Dat["FINAL_shape"] = ds.Dat["shape"]
    #             ds.Dat["FINAL_dist"] = ds.Dat["dist_beh_task_strok"]
    #             ds.Dat["FINAL_match"] = ds.Dat["IMAGE_matches"]
    #             # sanity
    #             assert ds.Dat["dist_beh_task_strok"] == ds.Dat["IMAGE_dist"]
    #         else:
    #             assert False
    #         list_ds.append(ds)
    # DScomb = concat_dataset_strokes_minimal(list_ds)
    # DScomb.Dat = append_col_with_grp_index(DScomb.Dat, ["novel_char", "is_perfblocky"], "novel|blocky")
    # DScomb.Dat = append_col_with_grp_index(DScomb.Dat, ["animal","FINAL_basis"], "animal|basis")

    # Check that columns exist and are correct. Previously I assigned these columns here, but now done in previous
    # function.
    list_ds = []
    for animal_data in ["Diego", "Pancho"]:
        ### (1) Collect df, across bases, and concat as new orjggt
        # for basis in ["Diego", "Pancho", "image"]:
        for basis in ["Diego", "Pancho"]:
            ds = DictDS[(animal_data, basis)]
            assert all(ds.Dat["FINAL_basis"] == basis)
            
            if basis in ["Diego", "Pancho"]:
                assert all(ds.Dat["FINAL_shape"] == ds.Dat["clust_sim_max_colname"])
                assert all(ds.Dat["FINAL_dist"] == ds.Dat["clust_sim_max"])
                assert all(ds.Dat["FINAL_match"] == ds.Dat["matches_base_prim"])
            elif basis == "image":
                assert all(ds.Dat["FINAL_shape"] == ds.Dat["shape"])
                assert all(ds.Dat["FINAL_dist"] == ds.Dat["dist_beh_task_strok"])
                assert all(ds.Dat["FINAL_match"] == ds.Dat["IMAGE_matches"])
            else:
                assert False
            list_ds.append(ds)
    DScomb = concat_dataset_strokes_minimal(list_ds)
    # DScomb.Dat = append_col_with_grp_index(DScomb.Dat, ["novel_char", "is_perfblocky"], "novel|blocky")
    # DScomb.Dat = append_col_with_grp_index(DScomb.Dat, ["animal","FINAL_basis"], "animal|basis")

    # Just chars
    task_kind = "character"
    DScomb.Dat = DScomb.Dat[(DScomb.Dat["task_kind"] == task_kind)].reset_index(drop=True)

    ### Filtering steps
    if True:
        # First, keep only those (animal, date, los_set[??]) which has both novel and trained.
        dfchar_shared, _ = extract_with_levels_of_conjunction_vars_helper(DScomb.Dat,
                                                    "novel_char", ["animal", "date", "los_set"], 
                                                    lenient_allow_data_if_has_n_levels=2)
        print("after pruning to jus t(animal, date, los_set) with both novel and not-novel: ", len(DScomb.Dat), " --> ",  len(dfchar_shared))
        assert len(dfchar_shared)>0, "no common chars between them"
    else:
        dfchar_shared = DScomb.Dat

    # Find characters that are tested for both animals
    levels_var = ["Diego", "Pancho"]
    n1 = len(dfchar_shared)
    dfchar_shared, _ = extract_with_levels_of_conjunction_vars_helper(dfchar_shared,
                                                "animal", ["character"], 
                                                lenient_allow_data_if_has_n_levels=2, 
                                                levels_var=levels_var)
    print("after pruning to shared cahracters between subects: ", n1, " --> ",  len(dfchar_shared))
    assert len(dfchar_shared)>0, "no common chars between them"

    # Finally, agg over trials, so that each los_info (char) has a single datapt.
    dfchar_shared = aggregGeneral(dfchar_shared, ["animal", "FINAL_basis", "los_info", "novel|blocky"], 
                ["FEAT_num_strokes_beh", "FINAL_dist", "FINAL_match"], 
                aggmethod=["mean"], 
                nonnumercols=["task_kind", "los_set", "novel_char"])
    
    #### PLOTS
    if QUICK:
        # Just the ones in paper.
        list_conditions = [(dfchar_shared, "shared_chars")]
        dfchar_shared_agg = None
    else:
        # Each los_set has one datapt (this is most conservative)
        dfchar_shared_agg = aggregGeneral(dfchar_shared, ["animal", "FINAL_basis", "los_set", "novel|blocky"], 
                    ["FEAT_num_strokes_beh", "FINAL_dist", "FINAL_match"], 
                    aggmethod=["mean"], 
                    nonnumercols=["task_kind", "novel_char"])
        list_conditions = [(DScomb.Dat, "all_chars"), (dfchar_shared, "shared_chars"), (dfchar_shared_agg, "shared_chars_datapt=los_set")]

    if savedir is not None:
        for dfthis, savesuff in list_conditions:
            # NOTE:
            # - DScomb.Dat is all data
            # - dfchar_shared is stringent, shared chars, only (dates, los_set) with both novel and not-novel.
            print("Plotting for savesuff: ", savesuff)
            _plot(dfthis, savesuff)

    #### Pairwise plots, specifically just for shared chars
    # # (1) Agg, so los_info is single datapt
    # dfcounts_shared = aggregGeneral(dfchar_shared, ["animal", "FINAL_basis", "los_info", "novel|blocky"], ["FEAT_num_strokes_beh", "FINAL_dist", "FINAL_match"], aggmethod=["mean"], nonnumercols=["task_kind", "los_set"])
    # from pythonlib.tools.pandastools import stringify_values
    # dfchar_shared_str = stringify_values(dfchar_shared)
    # assert pd.crosstab(dfchar_shared_str["los_info"], dfchar_shared_str["animal"]).min(axis=1).min()>0, "failure means a los_info was not done by all the animals."

    # for var_data, var_value, plot_error_bars in [
    #     ("los_info", "FINAL_match", False),
    #     ("los_set", "FINAL_match", True),
    #     ("los_info", "FINAL_dist", False),
    #     ("los_set", "FINAL_dist", True),
    #     ]:
    #     for alpha in [0.1, 0.3]:
    #         _, fig = plot_45scatter_means_flexible_grouping(dfcounts_shared, "FINAL_basis", "Diego", "Pancho", "animal", 
    #                                                         var_value, var_data, False, shareaxes=True, plot_error_bars=plot_error_bars, 
    #                                                         alpha=alpha, SIZE=3.5)
    #         savefig(fig, f"{savedir}/SHAREDCHAR-scatter-data={var_data}-value={var_value}-alpha={alpha}.pdf")
    # plt.close("all")

    if False:
        # based on FINAL_Basis
        
        # each subplot is a basis set.
        # plot histograms for each (animal, task_kind)
        fig, ax = plt.subplots()
        dfthis = DScomb.Dat
        col = "k"
        nbins = 25

        grpvars = ["animal", "task_kind", "FINAL_basis"]
        grpdict = grouping_append_and_return_inner_items_good(DScomb.Dat, grpvars)

        from pythonlib.tools.plottools import makeColors
        pcols = makeColors(len(grpdict))

        for (grp, inds), pcol in zip(grpdict.items(), pcols):
            dfthis = DScomb.Dat.iloc[inds]
            # sns.histplot(data=dfthis, x="clust_sim_max", stat="probability", color=col,
            #                 ax=ax, bins=np.linspace(0, 1, nbins+1), element="step", alpha=0.1, label="test")
            sns.histplot(data=dfthis, x="clust_sim_max", stat="probability", color=pcol,
                            ax=ax, bins=np.linspace(0, 1, nbins+1), element="step", alpha=0.1, label=grp)
            ax.legend()

        # Plot means. Do it here so that it is at the ymax
        YMAX = ax.get_ylim()[1]
        i=0
        for (grp, inds), pcol in zip(grpdict.items(), pcols):
            dfthis = DScomb.Dat.iloc[inds]
            # Place marker for mean
            valmean = np.mean(dfthis["clust_sim_max"])
            ax.plot(valmean, YMAX, "v", color=pcol)

    # FInal dumb sanity check that chars are indeed done by both aniamls.
    ds1 = DictDS[("Diego", "Diego")]
    ds2 = DictDS[("Pancho", "Pancho")]

    for char in dfchar_shared["character"].unique().tolist():
        assert char in ds1.Dat["character"].unique().tolist()
        assert char in ds2.Dat["character"].unique().tolist()

    for char in dfchar_shared["los_info"].unique().tolist():
        assert char in ds1.Dat["los_info"].unique().tolist()
        assert char in ds2.Dat["los_info"].unique().tolist()

    return DScomb, dfchar_shared, dfchar_shared_agg

def plot_drawing_task_colorby_matches_matched_trials(DictDS, LIST_ANIMAL_DATE, LIST_D, SAVEDIR,
                                                     novel=None, perfblocky=None,
                                                     plot_ax_beh_task=False):
    """
    PARAMS:
    - novel, perfblocky, either None (ignores) or bool (filters to just keep those)
    """
    from pythonlib.tools.pandastools import filterPandas
    from pythonlib.tools.plottools import color_make_map_discrete_labels

    savedir = f"{SAVEDIR}/drawings_matched_chars-novel={novel}-perfblock={perfblocky}"
    os.makedirs(savedir, exist_ok=True)
    
    ### Overall filter
    filtdict = {}
    if novel is not None:
        filtdict["novel_char"] = [novel]
    if perfblocky is not None:
        filtdict["is_perfblocky"] = [perfblocky]

    ### Get common chars
    ds1 = DictDS[("Diego", "Diego")]
    ds2 = DictDS[("Pancho", "Pancho")]
    chars1 = filterPandas(ds1.Dat, filtdict, False)["character"].unique().tolist()
    chars2 = filterPandas(ds2.Dat, filtdict, False)["character"].unique().tolist()
    # chars1 = ds1.Dat[(ds1.Dat["task_kind"] == "character") & (ds1.Dat["novel_char"] == novel) & (ds1.Dat["is_perfblocky"] == perfblocky)]["character"].unique().tolist()
    # chars2 = ds2.Dat[(ds2.Dat["task_kind"] == "character") & (ds2.Dat["novel_char"] == novel) & (ds2.Dat["is_perfblocky"] == perfblocky)]["character"].unique().tolist()
    chars_shared = [ch for ch in chars1 if ch in chars2]

    print("This many shared chars: ", len(chars_shared))

    ### Sort chars by quality
    list_dist_1 = []
    list_dist_2 = []
    for char in chars_shared:
        list_dist_1.append(np.mean(ds1.Dat[ds1.Dat["character"] == char]["FINAL_dist"]))
        list_dist_2.append(np.mean(ds2.Dat[ds2.Dat["character"] == char]["FINAL_dist"]))
    dfchars = pd.DataFrame({"character":chars_shared, "dist_1":list_dist_1, "dist_2":list_dist_2})
    dfchars["dist_min"] = [np.min([row["dist_1"], row["dist_2"]]) for i, row in dfchars.iterrows()]
    dfchars_sorted = dfchars.sort_values("dist_min", ascending=False).reset_index(drop=True)
    if False: # pairwise distribtuion of distances
        plt.figure()
        plt.plot(dfchars["dist_1"], dfchars["dist_2"], "ok")
        
    # Sanity check
    for char in dfchars_sorted["character"].unique().tolist():
        assert char in ds1.Dat["character"].tolist()
        assert char in ds2.Dat["character"].tolist()

    map_animal_to_basis_data={}
    for animal in ["Diego", "Pancho"]:
        # (1) First, Plot all the base prims in a grid
        _, list_strok_basis, list_shape_basis = ds1.stroke_shape_cluster_database_load_helper(which_basis_set=animal, plot_examples=False)
        map_baseprim_to_color, _, _ = color_make_map_discrete_labels(list_shape_basis)

        # center them
        from pythonlib.tools.stroketools import strokes_centerize
        list_strok_basis = strokes_centerize(list_strok_basis, "bounding_box")
        ncols = 5
        nrows = int(np.ceil(len(list_shape_basis)/ncols))
        SIZE = 1.7
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)
        for i, (ax, shape, strok) in enumerate(zip(axes.flatten(), list_shape_basis, list_strok_basis)):
            color = map_baseprim_to_color[shape]
            ds1.plot_single_strok(strok, "beh", ax=ax, color=color, alpha_beh=1)
            from pythonlib.drawmodel.strokePlots import formatDatStrokesPlot
            formatDatStrokesPlot(ax, naked_axes=True)
            ax.set_title(f"#{i}")
        fig.tight_layout()
        savefig(fig, f"{savedir}/base_prims-{animal}.pdf")

        map_animal_to_basis_data[animal] = (list_shape_basis, map_baseprim_to_color)

    ### Plots
    # from pythonlib.dataset.dataset_analy.characters import plot_drawing_task_colorby_matches_wrapper
    # plot_drawing_task_colorby_matches_wrapper(DictDS, LIST_ANIMAL_DATE, LIST_D, SAVEDIR)
    from pythonlib.tools.plottools import color_make_map_discrete_labels
    from pythonlib.dataset.dataset_analy.characters import plot_drawing_task_colorby_matches_trial
    from pythonlib.tools.pandastools import aggregGeneral

    # Go thru blocks of 10 trials (from best to worst)
    for ind_char_1 in range(0, len(dfchars_sorted), 10):
        ind_char_2 = ind_char_1 + 10
        print(ind_char_1, " to ", ind_char_2)
        # ### User params
        # ind_char_1 = 10
        # ind_char_2 = 20

        # Get characters, sorted so plot the best (across animals)
        if False: # Doesnt work, for some reason. Returns chars tat dont exist across both
            df = DScomb.Dat[DScomb.Dat["task_kind"] == "character"].reset_index(drop=True)
            df = aggregGeneral(df, ["animal", "character", "novel|blocky", "task_kind"], ["clust_sim_max"])
            dfmin = df.groupby(["character"])["clust_sim_max"].min().reset_index()
            dfmin["tmp"] = -df["clust_sim_max"]
            dfmin = dfmin.sort_values("tmp").reset_index(drop=True)
            # Get the top <n_char> characters
            # chars_get = dfmin["character"][:n_char_plot].tolist()
        else:
            chars_get = dfchars_sorted[ind_char_1:ind_char_2]["character"].tolist()

        print("Plotting these chars: ", chars_get)

        for only_color_if_good_match in [False, True]:
            for i_iter in range(1):
                n_char_plot = ind_char_2 - ind_char_1
                n_animal = 2
                nrows = n_animal*3
                SIZE = 2
                fig, axes = plt.subplots(nrows, n_char_plot, figsize=(n_char_plot*SIZE, nrows*SIZE), sharex=True, sharey=True)

                # Plot
                for i_an, animal in enumerate(["Diego", "Pancho"]):
                    
                    list_shape_basis, map_baseprim_to_color = map_animal_to_basis_data[animal]

                    # Get data for this animal
                    ds = DictDS[(animal, animal)]
                    if False:
                        trialcodes = ds.Dat["trialcode"][:5]
                    else:
                        # Get trialcodes givne input list of chars
                        from pythonlib.tools.pandastools import extract_trials_spanning_variable
                        list_inds, _ = extract_trials_spanning_variable(ds.Dat, "character", 
                                                                                chars_get, F=filtdict,
                                                                                method_if_not_enough_examples="fail")
                        trialcodes = ds.Dat.iloc[list_inds]["trialcode"].tolist()

                    for i_tc, tc in enumerate(trialcodes):
                        print(tc)
                        ax_task = axes[3*i_an][i_tc]
                        ax_beh_prim = axes[3*i_an+1][i_tc]
                        if plot_ax_beh_task:
                            ax_beh_task = axes[3*i_an+2][i_tc]
                        else:
                            ax_beh_task = None

                        ax_task.set_title(f"{animal}-{tc}")
                        plot_drawing_task_colorby_matches_trial(ds, LIST_ANIMAL_DATE, LIST_D, tc, 
                                                                        list_shape_basis, map_baseprim_to_color,
                                                                        ax_task=ax_task, ax_beh_prim=ax_beh_prim, ax_beh_task=ax_beh_task,
                                                                        only_color_if_good_match=only_color_if_good_match)
                savefig(fig, f"{savedir}/inds={ind_char_1}-{ind_char_2}-onlycolormatch={only_color_if_good_match}-iter={i_iter}.pdf")
                plt.close("all")
                    
def plot_drawing_task_colorby_matches_wrapper(DictDS, LIST_ANIMAL_DATE, LIST_D, SAVEDIR):
    """
    To Make all plots of single trial characters, here iterates over each animal x basis prims,
    and does relevant preprocessing and data extraction.
    """
    from pythonlib.tools.plottools import color_make_map_discrete_labels
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    from pythonlib.dataset.dataset_analy.characters import plot_drawing_task_colorby_matches
    from pythonlib.tools.plottools import share_axes
        

    # This requires original dataset -- maybe best to run separately for each day.
    n_examples = 10

    for animal, basis in [
        ("Pancho", "Pancho"),
        ("Diego", "Diego")]:
    # for animal, basis in [
    #     ("Pancho", "Pancho")]:

        if animal!=basis:
            # Just to speed things up.
            continue
        
        # Get the dataset
        DSorig = DictDS[(animal, basis)]
        DS = DSorig.copy()

        # Keep just chars
        DS.Dat = DS.Dat[DS.Dat["task_kind"] == "character"].reset_index(drop=True)

        # Go thru each date
        for date in DSorig.Dat["date"].unique().tolist():
            
            DS.Dat = DSorig.Dat[DSorig.Dat["date"] == date].reset_index(drop=True)

            savedir = f"{SAVEDIR}/match_prim-vs-match_image-DRAWINGS-data={animal}-basis={basis}-date={date}"
            os.makedirs(savedir, exist_ok=True)
            print(savedir)

            dfcounts, dfcounts_clean = analy_score_match_baseprim_vs_matchimage_trials(DS, None, just_get_dfcounts=True)
            
            if False:
                fig, ax = plt.subplots()
                plot_icon_heatmap_base_prims_used(prim_matches_shape, ax)

            # (1) First, Plot all the base prims in a grid
            _, list_strok_basis, list_shape_basis = DS.stroke_shape_cluster_database_load_helper(which_basis_set=basis, plot_examples=False)
            map_baseprim_to_color, _, _ = color_make_map_discrete_labels(list_shape_basis)

            ncols = 5
            nrows = int(np.ceil(len(list_shape_basis)/ncols))
            SIZE = 1.7
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)
            for i, (ax, shape, strok) in enumerate(zip(axes.flatten(), list_shape_basis, list_strok_basis)):
                color = map_baseprim_to_color[shape]
                DS.plot_single_strok(strok, "beh", ax=ax, color=color, alpha_beh=1)
                from pythonlib.drawmodel.strokePlots import formatDatStrokesPlot
                formatDatStrokesPlot(ax, naked_axes=True)
                ax.set_title(f"#{i}")
            fig.tight_layout()
            # from matplotlib.pyplot import subplots_adjust
            # plt.subplots_adjust(wspace=-0.2, hspace=0.2)
            savefig(fig, f"{savedir}/base_prims.pdf")

            # (2) Plot example trials
            grpdict = grouping_append_and_return_inner_items_good(dfcounts_clean, ["task_kind_Nbeh_first", "matches_base_prim_sum", "IMAGE_matches_sum"])

            # plot_beh_and_aligned_task_strokes
            SIZE = 3

            for grp, inds_dfcounts in grpdict.items():
                # print(inds_dfcounts)
                
                trialcodes = dfcounts_clean.iloc[inds_dfcounts]["trialcode"].tolist()
                np.random.shuffle(trialcodes)
                trialcodes = trialcodes[:n_examples]

                # fig, axes = plt.subplots(4, n_examples, figsize=(SIZE*n_examples, SIZE*3), sharex=True, sharey=True)
                fig, axes = plt.subplots(4, n_examples, figsize=(SIZE*n_examples, SIZE*3))

                for i_col, tc in enumerate(trialcodes):
                    ax_task = axes[0][i_col]
                    ax_beh_prim = axes[1][i_col]
                    ax_beh_task = axes[2][i_col]
                    ax_base_prims_icon = axes[3][i_col]

                    plot_drawing_task_colorby_matches_trial(DS, LIST_ANIMAL_DATE, LIST_D, tc, 
                                                                list_shape_basis, map_baseprim_to_color,
                                                                ax_task, ax_beh_prim, ax_beh_task, 
                                                                ax_base_prims_icon, nrows, ncols)
                    
                    # Give title.
                    tmp = dfcounts[dfcounts["trialcode"] == tc]
                    assert len(tmp)==1
                    n_match_prim = tmp["matches_base_prim_sum"].values[0]
                    n_match_image = tmp["IMAGE_matches_sum"].values[0]
                    n_strokes_beh = tmp["FEAT_num_strokes_beh_mean"].values[0]

                    ax_task.set_title(tc)
                    ax_beh_prim.set_title(f"n match prim: {n_match_prim}/{n_strokes_beh}")
                    ax_beh_task.set_title(f"n match task: {n_match_image}/{n_strokes_beh}")


                #     # inddat = 456
                #     # D.plotSingleTrial(inddat)
                #     # tc = D.Dat.iloc[inddat]["trialcode"]
                #     inds = DS.Dat[DS.Dat["trialcode"]==tc].index.tolist()
                #     if False:
                #         DS.plot_multiple_overlay_entire_trial(inds)
                    
                #     # Append Dataset back to DS
                #     animal = DS.Dat.iloc[inds]["animal"].unique()[0]
                #     date = str(DS.Dat.iloc[inds]["date"].unique()[0])
                #     idx = LIST_ANIMAL_DATE.index((animal, date))
                #     Dthis = LIST_D[idx]
                #     DS.dataset_replace_dataset(Dthis)

                #     # Determine if strokes match base prims
                #     prim_matches = DS.Dat.iloc[inds]["matches_base_prim"].tolist()
                #     prim_matches_shape = DS.Dat.iloc[inds]["clust_sim_max_colname"].tolist()
                #     if False:
                #         prim_matches_index = [list_shape_basis.index(sh) for sh in prim_matches_shape]
                #     else:
                #         # Dont plot text on strokes.
                #         prim_matches_index= None
                #     colors_prim = [map_baseprim_to_color[sh] for sh in prim_matches_shape]

                #     # colors_match_prim = [map_baseprim_to_color[sh] if match else color_nonmatch for match, sh in zip(prim_matches, prim_matches_shape)]
                #     image_matches = DS.Dat.iloc[inds]["IMAGE_matches"].tolist()
                #     image_matches_shape = DS.Dat.iloc[inds]["shape"].tolist()

                #     ax_task = axes[0][i_col]
                #     ax_beh_prim = axes[1][i_col]
                #     ax_beh_task = axes[2][i_col]
                #     ax_base_prims_icon = axes[3][i_col]

                #     plot_drawing_task_colorby_matches(DS, inds, ax_task, ax_beh_prim, ax_beh_task, 
                #                                             prim_matches, image_matches, colors_prim, prim_matches_index,
                #                                             color_nonmatch=color_nonmatch,
                #                                             color_task_by_matches=False)

                #     tmp = dfcounts[dfcounts["trialcode"] == tc]
                #     assert len(tmp)==1
                #     n_match_prim = tmp["matches_base_prim_sum"].values[0]
                #     n_match_image = tmp["IMAGE_matches_sum"].values[0]
                #     n_strokes_beh = tmp["FEAT_num_strokes_beh_mean"].values[0]

                #     ax_task.set_title(tc)
                #     ax_beh_prim.set_title(f"n match prim: {n_match_prim}/{n_strokes_beh}")
                #     ax_beh_task.set_title(f"n match task: {n_match_image}/{n_strokes_beh}")

                #     # Add icon of which base prims are gotten
                #     prim_matches_shape_good = [sh for sh, ismatch in zip(prim_matches_shape, prim_matches) if ismatch]
                #     plot_icon_heatmap_base_prims_used(prim_matches_shape_good, ax_base_prims_icon)

                #     # Share the drawing axes
                #     share_axes(np.array([ax_task, ax_beh_prim, ax_beh_task]), which="both")

                # fig.tight_layout()
                savefig(fig, f"{savedir}/example_drawings-grp={grp}.pdf")
                # assert False
                plt.close("all")


def plot_drawing_task_colorby_matches_trial(DS, LIST_ANIMAL_DATE, LIST_D, trialcode, 
                                            list_shape_basis, map_baseprim_to_color,
                                            ax_task, ax_beh_prim, ax_beh_task, 
                                            ax_base_prims_icon=None, nrows_icon=None, ncols_icon=None,
                                            only_color_if_good_match=True):
    """
    Does all things for plotting drwaing, including showing matches to base prims, for this trial.
    Plots a single trial.    

    """
    from pythonlib.tools.plottools import share_axes

    color_nonmatch = [0.3, 0.3, 0.3]
    color_task_by_matches = False

    def plot_icon_heatmap_base_prims_used(base_prim_shapes, ax):
        from pythonlib.tools.snstools import heatmap_mat
        list_index = [list_shape_basis.index(sh) for sh in base_prim_shapes]
        ma = np.zeros((nrows_icon, ncols_icon))

        for idx in list_index:
            y = int(np.floor(idx/ncols_icon)) # from top
            x = idx%ncols_icon
            ma[y, x] = 1

        # sns.heatmap(ma, ax=ax)
        heatmap_mat(ma, ax=ax, annotate_heatmap=False, cbar=False)

        # plot the index text
        if False: # Too large
            for idx in list_index:
                y = int(np.floor(idx/ncols)) # from top
                x = idx%ncols
                print(x,y, nrows-y-1)
                ax.text(x+0.25, y+0.625, f"#{idx}", color="k", fontsize=15)


        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(axis='both', which='both',length=0)

    # inddat = 456
    # D.plotSingleTrial(inddat)
    # tc = D.Dat.iloc[inddat]["trialcode"]
    inds = DS.Dat[DS.Dat["trialcode"]==trialcode].index.tolist()
    if False:
        DS.plot_multiple_overlay_entire_trial(inds)
    
    # Append Dataset back to DS
    # animal = DS.animal()
    # date = str(DS.date())
    tmp = DS.Dat.iloc[inds]["animal"].unique()
    assert len(tmp)==1
    animal = tmp[0]
    
    tmp = DS.Dat.iloc[inds]["date"].unique()
    assert len(tmp)==1
    date = str(tmp[0])

    idx = LIST_ANIMAL_DATE.index((animal, date))
    Dthis = LIST_D[idx]
    DS.dataset_replace_dataset(Dthis)

    # Determine if strokes match base prims
    prim_matches_shape = DS.Dat.iloc[inds]["clust_sim_max_colname"].tolist()
    if only_color_if_good_match:
        prim_matches = DS.Dat.iloc[inds]["matches_base_prim"].tolist()
    else:
        prim_matches = [True for _ in range(len(prim_matches_shape))]
    if False:
        prim_matches_index = [list_shape_basis.index(sh) for sh in prim_matches_shape]
    else:
        # Dont plot text on strokes.
        prim_matches_index= None
    colors_prim = [map_baseprim_to_color[sh] for sh in prim_matches_shape]

    # colors_match_prim = [map_baseprim_to_color[sh] if match else color_nonmatch for match, sh in zip(prim_matches, prim_matches_shape)]
    image_matches = DS.Dat.iloc[inds]["IMAGE_matches"].tolist()
    # image_matches_shape = DS.Dat.iloc[inds]["shape"].tolist()

    plot_drawing_task_colorby_matches(DS, inds, ax_task, ax_beh_prim, ax_beh_task, 
                                            prim_matches, image_matches, colors_prim, prim_matches_index,
                                            color_nonmatch=color_nonmatch,
                                            color_task_by_matches=color_task_by_matches)
    
    # Add icon of which base prims are gotten
    if ax_base_prims_icon is not None:
        if ax_base_prims_icon is not None:
            assert nrows_icon is not None, "needed to know how to plot grid icon"
            assert ncols_icon is not None, "needed to know how to plot grid icon"
        prim_matches_shape_good = [sh for sh, ismatch in zip(prim_matches_shape, prim_matches) if ismatch]
        plot_icon_heatmap_base_prims_used(prim_matches_shape_good, ax_base_prims_icon)

    # Share the drawing axes
    _axes = [_ax for _ax in [ax_task, ax_beh_prim, ax_beh_task] if _ax is not None]

    share_axes(np.array(_axes), which="both")


def plot_drawing_task_colorby_matches(DS, inds, ax_task, ax_beh_prim, ax_beh_task, 
                                        prim_matches, image_matches, colors_prim, prim_matches_index=None,
                                        color_nonmatch="k",
                                        color_task_by_matches=False):
    """
    Create plot of beh and task image, showing matches., comparing match to base prims vs. match to image.
    PARAMS:
    - inds, indices into DS.Dat -- the strokes that will plot
    - ax_task, ax_beh_prim, ax_beh_task, axes where will plot (i) task, (ii) beh, colored by match to base prims and (iii)
    beh, colored by whether match to image stroke.
    - prim_matches, list len(inds) of bools, whether this stroke matches a base prim
    - image_matcheslist len(inds) of bools, whether this stroke matches an image stroke.
    - colors_prim, list of colors for each beh stroke. These are usually the canonical colors for the base prims that are matched.
    - prim_matches_index, list of int indices that are simply labels on the plot, indicating which base prim it is.
    - color_task_by_matches, bool, if True, then the image will have color on strokes that are matched by beh.
    """
    # Plot beh strokes, coloring only if match base prims
    # DS.plot_multiple_overlay_entire_trial_single_plot(ax_beh, inds, separate_axis_image=ax_task, list_colors=colors_match_prim)

    if prim_matches_index is None:
        prim_matches_index = [None for _ in range(len(prim_matches))]

    ### Plot beh strokes, coloring only if match task image
    # number the beh strokes
    for i, _ind in enumerate(inds):
        if prim_matches[i]:
            _color = colors_prim[i]
            idx_prim = prim_matches_index[i]
        else:
            _color = color_nonmatch
            idx_prim = None
        DS.plot_single(_ind, ax_beh_prim, color=_color, label_onset=idx_prim, label_color=[0.2, 0.2, 0.2])
    # Number the stroke by their prim?


    # Plot image, coloring image strokes same as beh strokes, if they match
    # inddat = DS._dataset_index(ind)
    # strokes_task = DS.Dataset.Dat.iloc[inddat]["strokes_task"]
    # strokes_task_ordered_by_beh = [strokes_task[i] for i in DS.Dat.iloc[inds]["ind_taskstroke_orig"]]

    strokes_task = DS.dataset_extract("strokes_task", inds[0])
    map_idxtaskorig_to_color = {}
    for idx_task_orig, ismatch, color in zip(DS.Dat.iloc[inds]["ind_taskstroke_orig"].tolist(), image_matches, colors_prim):
        if ismatch:
            map_idxtaskorig_to_color[idx_task_orig] = color
        else:
            map_idxtaskorig_to_color[idx_task_orig] = color_nonmatch

    ### Plot task image
    if ax_task is not None:
        for i, st in enumerate(strokes_task):
            if color_task_by_matches and (i in map_idxtaskorig_to_color):
                color = map_idxtaskorig_to_color[i]
            else:
                color = color_nonmatch
            DS.plot_single_strok(st, "task_colored", ax=ax_task, color=color, alpha_beh=1)


    # strokes_task_ordered_by_beh = DS.extract_strokes(inds =inds, ver_behtask="task_aligned_single_strok")
    # for i, st in enumerate(strokes_task_ordered_by_beh):
    #     if color_task_by_matches and image_matches[i]:
    #         color = colors_match_task[i]
    #     else:
    #         color="k" 
    #     DS.plot_single_strok(st, "task_colored", ax=ax_task, color=color, alpha_beh=1)

    ### Plot beh strokes, coloring only if match task image
    if ax_beh_task is not None:
        for i, _ind in enumerate(inds):
            if image_matches[i]:
                _color = colors_prim[i]
            else:
                _color = color_nonmatch
            DS.plot_single(_ind, ax_beh_task, color=_color)




def analy_collect_all_los_datetimes_using_presaved(animal):
    """
    Across historical data, get mappings from all char los to the list of datetimes that iwas done.
    Useful for etermining whther trials were "novel".

    Uses data pre-extracted and saved.
    """
    import re

    # D.Dat.loc[:, ["character", "los_info", "task_kind", "trialcode", "datetime", "probe"]].to_csv(f"{SAVEDIR}/los_info-{animal}-{date}.csv")
    
    # List of dates that have had their los extracted previously.
    list_animal_date = [
        ("Pancho", "220531"),
        ("Pancho", "220601"),
        ("Pancho", "220602"),
        ("Pancho", "220603"),
        ("Pancho", "220613"),
        ("Pancho", "220614"),
        ("Pancho", "220615"),
        ("Pancho", "220616"),
        ("Pancho", "220617"),
        ("Pancho", "220618"),
        ("Pancho", "220621"),
        ("Pancho", "220622"),
        ("Pancho", "220624"),
        ("Pancho", "230112"),
        ("Pancho", "230117"),
        ("Pancho", "230118"),
        ("Pancho", "230119"),
        ("Pancho", "230120"),
        ("Pancho", "230122"),
        ("Pancho", "230125"),
        ("Pancho", "230126"),
        ("Pancho", "230127"),
        ("Diego", "231122"),
        ("Diego", "231128"),
        ("Diego", "231129"),
        ("Diego", "231130"),
        ("Diego", "231201"),
        ("Diego", "231204"),
        ("Diego", "231205"),
        ("Diego", "231206"),
        ("Diego", "231207"),
        ("Diego", "231211"),
        ("Diego", "231213"),
        ("Diego", "231218"),
        ("Diego", "231219"),
        ("Diego", "231220"),
    ]

    animals = sorted(set([x[0] for x in list_animal_date]))
    dates = sorted(set([x[1] for x in list_animal_date]))
    savesuff = "_".join(animals) + "|" + f"{min(dates)}_{max(dates)}|N={len(list_animal_date)}"
    savedir = f"/lemur2/lucas/analyses/main/strokes_clustering_similarity/analysis_match_base_prims/MULT/{savesuff}"
    print(savedir)

    # Go thru all dates and collect each los.
    map_los_to_datetimes = {}
    for a, date in list_animal_date:
        if a == animal:
            path = f"{savedir}/los_info-{animal}-{date}.csv"
            print(path)
            df = pd.read_csv(path)

            if False:
                display(df)
                for col in df.columns:
                    print(col, " -- ", type(df[col].values[0]))
                assert False

            # get all los and their list of datetimes
            # (1) Collect ALL dates for each los across all expts (characters)
            # los_novels = D.Dat[D.Dat["novel_char"]]["los_info"].unique().tolist()
            los_all = df[df["task_kind"]=="character"]["los_info"].unique().tolist()
            for los in los_all:
                if los == "(None, None, None)":
                    # Then this is a random task...
                    continue
                
                # Convert from string to tuple (los)
                _inds = [m.start() for m in re.finditer(",", los)]
                
                try:
                    los_tuple = (los[2:_inds[0]-1], int(los[_inds[0]+2:_inds[1]]), int(los[_inds[1]+2:-1]))
                except Exception as err:
                    print(los)
                    print(type(los))
                    print(_inds)
                    raise err
                dts = df[(df["los_info"]==los) & (df["task_kind"]=="character")]["datetime"].unique().tolist()
                
                if los_tuple == (None, None, None):
                    # Should have been caught above
                    assert False

                if los_tuple in map_los_to_datetimes:
                    map_los_to_datetimes[los_tuple].extend(dts)
                else:
                    map_los_to_datetimes[los_tuple] = dts                

    # sort datetimes
    map_los_to_datetimes = {los:sorted(set(dts)) for los, dts in map_los_to_datetimes.items()}

    return map_los_to_datetimes

def analy_collect_all_los_datetimes(animal):
    # (1) Collect ALL dates for each los across all expts (characters)
    map_los_to_datetimes = {}
    for D, (animal, date) in zip(LIST_D, LIST_ANIMAL_DATE):
        if animal == animal_get:
            # los_novels = D.Dat[D.Dat["novel_char"]]["los_info"].unique().tolist()
            los_all = D.Dat[D.Dat["task_kind"]=="character"]["los_info"].unique().tolist()
            for los in los_all:
                dts = D.Dat[(D.Dat["los_info"]==los) & (D.Dat["task_kind"]=="character")]["datetime"].unique().tolist()
                if los in map_los_to_datetimes:
                    map_los_to_datetimes[los].extend(dts)
                else:
                    map_los_to_datetimes[los] = dts                
    # sort datetimes
    map_los_to_datetimes = {los:sorted(set(dts)) for los, dts in map_los_to_datetimes.items()}

    return map_los_to_datetimes


def analy_update_novel_label(LIST_D, LIST_ANIMAL_DATE, SAVEDIR, animals=("Diego", "Pancho")):
    """
    For each trial, determine if it is a novel trial, the first trial of that
    character, and I hand labeled it as novel. 
    Then replace the column "novel_char" with this output.

    NOTE: is OK to run this repeatedly, as it does not overwrite old labels.

    """

    savedir = f"{SAVEDIR}/novel_chars_finalize"
    os.makedirs(savedir, exist_ok=True)

    # To be called novel, must be (i) "novel" on that day. (ii) 
    # NOTE: is fine to run this repeatedyl

    # Get mapping, los --> list of datetimes across all data
    # For a given los, this gets only the trials that are novel_char==True

    for animal_get in animals:

        if True:
            # Better -- uses all char dates
            map_los_to_datetimes = analy_collect_all_los_datetimes_using_presaved(animal_get)
        else:
            # Use just what is in the dataset
            map_los_to_datetimes = analy_collect_all_los_datetimes(animal_get)

        # # (1) Collect ALL dates for each los across all expts (characters)
        # map_los_to_datetimes = {}
        # for D, (animal, date) in zip(LIST_D, LIST_ANIMAL_DATE):
        #     if animal == animal_get:
        #         # los_novels = D.Dat[D.Dat["novel_char"]]["los_info"].unique().tolist()
        #         los_all = D.Dat[D.Dat["task_kind"]=="character"]["los_info"].unique().tolist()
        #         for los in los_all:
        #             dts = D.Dat[(D.Dat["los_info"]==los) & (D.Dat["task_kind"]=="character")]["datetime"].unique().tolist()
        #             if los in map_los_to_datetimes:
        #                 map_los_to_datetimes[los].extend(dts)
        #             else:
        #                 map_los_to_datetimes[los] = dts                
        # # sort datetimes
        # map_los_to_datetimes = {los:sorted(set(dts)) for los, dts in map_los_to_datetimes.items()}

        # (2) For each candidate los (already hand-labeled as novel_char), determine if it is the first trial ever
        # (for any los, it is only called novel if its time matches the first attempt)
        def _is_novel(row):
            """ Return True if this is the first trial
            this los ever found, in this dataset
            """
            los = row["los_info"]
            dt = row["datetime"]
            # if (los in map_los_to_datetimes)

            if (row["task_kind"]=="character") and (row["random_task"]==False):
                assert los in map_los_to_datetimes, "PRoblem with extraction/saving of old los"
                return (row["novel_char_hand_old"]==True) and (map_los_to_datetimes[los][0] == dt)
            else:
                # This is not novel (becuase it is either not char or is random char)
                # print(row["los_info"])
                # print(row["datetime"])
                # assert False, "weird, all chars should be in dict"
                return False

        # To be called novel, must be (i) probe/novel. and (ii) first time done
        # Go thru all datasetes again, this time replacing novel label with correct one.
        list_df = []
        for D, (animal, _) in zip(LIST_D, LIST_ANIMAL_DATE):
            if animal == animal_get:

                # novels = []
                # for _, row in D.Dat.iterrows():
                #     novels.append(_is_novel(row))
                novels = ([_is_novel(row) for i, row in D.Dat.iterrows()])
                
                # Update "novel char"
                D.Dat["novel_char"] = novels

                # Finally, print and save all the finalized novel chars.
                a = D.animals(True)[0]
                d = D.dates(True)[0]

                D.grouping_print_n_samples(["novel_char", "probe", "los_info", "block"], savepath=f"{savedir}/{a}-{d}-novel_char_assignments-1.txt")
                D.grouping_print_n_samples(["block", "probe", "novel_char", "los_info"], savepath=f"{savedir}/{a}-{d}-novel_char_assignments-2.txt")
                D.grouping_print_n_samples(["block", "probe", "novel_char", "los_set"], savepath=f"{savedir}/{a}-{d}-novel_char_assignments-3.txt")
                D.grouping_print_n_samples(["block", "probe", "novel_char", "task_kind"], savepath=f"{savedir}/{a}-{d}-novel_char_assignments-4.txt")

                # Sanity check -- confirm that each novel_char is unique los
                if any(D.Dat["novel_char"]==True):
                    try:
                        assert D.Dat[D.Dat["novel_char"]==True]["los_info"].value_counts().max()<2
                        assert D.Dat[D.Dat["novel_char"]==True]["character"].value_counts().max()<2
                    except Exception as err:
                        print(D.Dat[D.Dat["novel_char"]==True]["los_info"].value_counts().max())
                        print(D.Dat[D.Dat["novel_char"]==True]["character"].value_counts().max())
                        raise err


                list_df.append(D.Dat)

        df = pd.concat(list_df).reset_index(drop=True)
        from pythonlib.tools.pandastools import grouping_print_n_samples
        grp_vars = ["animal", "date", "probe", "novel_char", "block"]
        grouping_print_n_samples(df, grp_vars, savepath=f"{savedir}/{animal_get}-ALL-novel_char_assignments-3.txt")

        grp_vars = ["animal", "date", "probe", "novel_char", "los_set"]
        grouping_print_n_samples(df, grp_vars, savepath=f"{savedir}/{animal_get}-ALL-novel_char_assignments-2.txt")

        grp_vars = ["animal", "date", "probe", "novel_char", "los_info"]
        grouping_print_n_samples(df, grp_vars, savepath=f"{savedir}/{animal_get}-ALL-novel_char_assignments-1.txt")

def analy_update_novel_label_ALL_wrapper(LIST_D, LIST_ANIMAL_DATE, savesuff):
    """
    For all trialcodes currently designated as "novel_char"==True, determine whether this is atually
    novel by comparing it to every previous trial, using image distances, and calling noivel only if is not similar
    to EVERY previous trial. 

    Only run this AFTER already pre-saved all data across days.

    Very stringent.
    """
    
    
    for animal in list(set([x[0] for x in LIST_ANIMAL_DATE])):

        ### Load dataset of all trials.
        DFALL, SAVEDIR_ALL = analy_update_novel_label_ALL_load(animal)

        savedir = f"{SAVEDIR_ALL}/final_list_novel/version={savesuff}/{animal}"
        os.makedirs(savedir, exist_ok=True)
        analy_update_novel_label_ALL_run(DFALL, LIST_D, LIST_ANIMAL_DATE, animal, savedir)


def analy_update_novel_label_ALL_load(animal_get):
    """
    Load database of data across ALL dates.
    """
    import pickle

    ##### Load data across all day (just the "raw" df)
    LIST_ANIMAL_DATE_ALL = [
        ("Pancho", "220531"),
        ("Pancho", "220601"),
        ("Pancho", "220602"),
        ("Pancho", "220603"),
        ("Pancho", "220613"),
        ("Pancho", "220614"),
        ("Pancho", "220615"),
        ("Pancho", "220616"),
        ("Pancho", "220617"),
        ("Pancho", "220618"),
        ("Pancho", "220621"),
        ("Pancho", "220622"),
        ("Pancho", "220624"),
        ("Pancho", "220625"), # added 2/8/25
        ("Pancho", "220626"), # added 2/8/25
        ("Pancho", "220627"), # added 2/8/25
        ("Pancho", "220628"), # added 2/8/25
        ("Pancho", "220629"), # added 2/8/25
        ("Pancho", "220630"), # added 2/8/25
        ("Pancho", "230112"),
        ("Pancho", "230117"),
        ("Pancho", "230118"),
        ("Pancho", "230119"),
        ("Pancho", "230120"),
        ("Pancho", "230122"),
        ("Pancho", "230124"),
        ("Pancho", "230125"),
        ("Pancho", "230126"),
        ("Pancho", "230127"),
        ("Diego", "230421"), # added 2/8/25
        ("Diego", "230422"), # added 2/8/25
        ("Diego", "230423"), # added 2/8/25
        ("Diego", "230424"), # added 2/8/25
        ("Diego", "231120"), # added 2/8/25
        ("Diego", "231121"), # added 2/8/25
        ("Diego", "231122"),
        ("Diego", "231128"),
        ("Diego", "231129"),
        ("Diego", "231130"),
        ("Diego", "231201"),
        ("Diego", "231204"),
        ("Diego", "231205"),
        ("Diego", "231206"),
        ("Diego", "231207"),
        ("Diego", "231209"), # added 2/8/25
        ("Diego", "231211"),
        ("Diego", "231213"),
        ("Diego", "231214"), # added 2/8/25
        ("Diego", "231215"), # added 2/8/25
        ("Diego", "231218"),
        ("Diego", "231219"),
        ("Diego", "231220"),
    ]

    animals = sorted(set([x[0] for x in LIST_ANIMAL_DATE_ALL]))
    dates = sorted(set([x[1] for x in LIST_ANIMAL_DATE_ALL]))
    savesuff = "_".join(animals) + "|" + f"{min(dates)}_{max(dates)}|N={len(LIST_ANIMAL_DATE_ALL)}"
    SAVEDIR_ALL = f"/lemur2/lucas/analyses/main/strokes_clustering_similarity/analysis_match_base_prims/MULT/{savesuff}"
    print(SAVEDIR_ALL)

    list_df = []
    # LIST_D = []
    for animal, date in LIST_ANIMAL_DATE_ALL:
        if animal==animal_get:

            print(animal, date)
            import pickle
            path = f"{SAVEDIR_ALL}/dat_unique_los-{animal}-{date}.pkl.pkl"

            with open(path, "rb") as f:
                Dthis = pickle.load(f)

                # Condition, for relate to global things.
                if False: # I stopped using tvalfake...
                    # - get all time relative same date
                    date_start = min([x[1] for x in LIST_ANIMAL_DATE_ALL])
                    Dthis._get_standard_time(date_start)

                # Collect data
                Dthis.Dat = Dthis.Dat[Dthis.Dat["task_kind"]=="character"].reset_index(drop=True)
                list_df.append(Dthis.Dat)

    # Concatenate into a single
    DFALL = pd.concat(list_df).reset_index(drop=True)

    return DFALL, SAVEDIR_ALL


def analy_update_novel_label_ALL_run(DFALL, LIST_D, LIST_ANIMAL_DATE, animal, savedir):
    """
    GOOD - uses ALL trials across dates in rig.

    Saves data, one row for each index that is novel_char, in each D in LIST_D.


    """
    from pythonlib.tools.stringtools import trialcode_to_scalar

    ### Filter to just this animal
    DFall = DFALL[DFALL["animal"]==animal].reset_index(drop=True)
    list_D = [D for D, anidate in zip(LIST_D, LIST_ANIMAL_DATE) if anidate[0]==animal]
    # Make copy, as below will prune.

    ### Preprocessing, to make below faster
    # # - sort by tval_fake
    # DFall = DFall.sort_values("tvalfake").reset_index(drop=True)

    # (1) all data
    DFall["trialcode_num"]= [trialcode_to_scalar(tc) for tc in DFall["trialcode"]]

    # (2) specific data
    for D in list_D:
        D.Dat["trialcode_num"]= [trialcode_to_scalar(tc) for tc in D.Dat["trialcode"]]

    # keep only first trial for each los_info
    from pythonlib.tools.pandastools import extract_first_trial_each_level
    DFall = extract_first_trial_each_level(DFall, "los_info", "trialcode_num")

    # # Plot to check that tvals are monotonic, correct.
    # sns.catplot(data=DFall, x="tvalfake", y="date", col="animal", sharex=False)
    # # For both datasets (current, and loaded_all_days), match them in the following ways:
    DEBUG = False
    for D in list_D:
        # # (1) tval_fake to same date.
        # date_start = min([x[1] for x in LIST_ANIMAL_DATE_ALL])
        # D._get_standard_time(date_start)

        # (2) Keep just first trial
        if False: # No need, becuase below only pulls out novel chars, which are by definition just one trial.
            df = D.analy_subsample_first_trial_each_level("los_info")
            if DEBUG:
                print(len(D.Dat), len(df))
                # Confirm that tvalfake and trialcode_num are similar.
                sns.scatterplot(data=D.Dat, x="tvalfake", y="trialcode_num")        
            assert sum(D.Dat["novel_char"]) == sum(df["novel_char"]), "keeping only first trial should NEVER throw out novel chars..."
            D.Dat = df

    ### Use trialcode global.

    # Sanity check that all are monotonic increasing
    if False:
        DFall = DFall.sort_values("trialcode_num")
        assert np.all(np.diff(DFall["trialcode_num"])>0)
        assert np.all(np.diff(DFall["tvalfake"])>0)
        sns.scatterplot(data=DFall, x="trialcode_num", y="tvalfake")

    ### Go thru each novel char, and compare it to all pevious trials.
    from pythonlib.dataset.dataset_strokes import DatStrokes
    ds = DatStrokes()

    res = []
    for D in list_D:
        
        # THIS IS MAIN CODE.
        # for each novel char, compare it to all the previous chars
        inds_novel = D.Dat[D.Dat["novel_char"]==True].index.tolist()

        for ind in inds_novel:

            ### Information for this novel 
            tc_this = D.Dat.iloc[ind]["trialcode"]
            tcnum_this = D.Dat.iloc[ind]["trialcode_num"]
            los_this = D.Dat.iloc[ind]["los_info"]
            novel_this = D.Dat.iloc[ind]["novel_char"]
            animal_this = D.Dat.iloc[ind]["animal"]
            date_this = D.Dat.iloc[ind]["date"]
            pts_this = np.concatenate(D.Dat.iloc[ind]["strokes_task"])

            print(tc_this)

            ### Information for trials before this date
            df_others = DFall[DFall["trialcode_num"] < tcnum_this]      

            if len(df_others)==0:
                # Then this is earliser... there are no preceding trials. Fake it as not having match.
                res.append({
                    "this_trialcode":tc_this,
                    "this_trialcode_num":tcnum_this,
                    "this_los":los_this,
                    "this_novel":novel_this,
                    "this_animal":animal_this,
                    "this_date":date_this,
                    "other_trialcode":"IGNORE",
                    "other_los":"IGNORE",
                    "dist":10
                })
            else: 
                # list_pts_others = [np.concatenate(strokes) for strokes in df_others["strokes_task"]]
                # # list_novel_others = df_others["novel_char"].tolist()
                # list_los_others = df_others["los_info"].tolist()
                # # list_trialcode_others = df_others["trialcode"].tolist()

                # For each of the others, compare to this.
                # Get all their distances to this one
                for _, row in df_others.iterrows():
                    pts_other = np.concatenate(row["strokes_task"])
                    dist = ds._dist_strok_pair(pts_this, pts_other, do_centerize=True, centerize_method="bounding_box")
                    res.append({
                        "this_trialcode":tc_this,
                        "this_trialcode_num":tcnum_this,
                        "this_los":los_this,
                        "this_novel":novel_this,
                        "this_animal":animal_this,
                        "this_date":date_this,
                        "other_trialcode":row["trialcode"],
                        "other_los":row["los_info"],
                        # "other_novel":row["novel_char"],
                        "dist":dist
                    })
    dfres = pd.DataFrame(res)
    
    ### Final saves and plots
    # Can evaluate, for each novel char, whether it has any mathces (failures)
    dfres["failure_same_los"] = (dfres["this_los"] == dfres["other_los"])

    thresh = 1.0
    dfres["failure_images_same"] = dfres["dist"]<thresh

    dfres["failure_any"] = (dfres["failure_same_los"]) | (dfres["failure_images_same"])

    dfres.groupby(["failure_same_los", "failure_images_same"]).size().to_csv(f"{savedir}/failure_kinds.csv")

    # save list of all the bad cases
    # Save this somewhere global, then can reference this later no matter what dataset use.
    dfres_bad = dfres[dfres["failure_any"]].reset_index(drop=True)
    dfres_bad.to_csv(f"{savedir}/failures_list.csv")

    # Save, finally, all the novel trials.
    # list_trialcode = dfres["this_trialcode"].unique().tolist()
    # list_trialcode
    # trialcodes_good = []
    # trialcodes_bad = []
    # for tc in list_trialcode:
    #     if any(dfres[(dfres["this_trialcode"]==tc)]["failure_any"]):
    #         trialcodes_bad.append(tc)
    #     else:
    #         trialcodes_good.append(tc)
    # print(len(trialcodes_bad), len(trialcodes_good))

    from pythonlib.tools.pandastools import aggregGeneral
    # Cnofirmed -- this gets failure_any true if any cases are faile.d
    dfres_agg = aggregGeneral(dfres, ["this_trialcode", "this_los"], ["failure_any"], aggmethod=["any"], nonnumercols="all")

    # save this
    dfres_agg.to_csv(f"{savedir}/final_results_each_los.csv")
    dfres_agg.to_pickle(f"{savedir}/final_results_each_los.pkl")
    
    dfres.to_pickle(f"{savedir}/final_results_each_pair.pkl")

def analy_update_novel_label_ALL_apply(LIST_D, LIST_ANIMAL_DATE, savesuff):
    """
    
    PARAMS:
    - savesuff, 
    --- "original_novel_labels", DATA = subset of dates (all trials), but using ALL data to determine if data are data in DATA are novel.
    --- "shared_all_not_just_novel", DATA = chars shared across subjects, first trial only, across all days.
    """
    # Finally, update labels using previuosly saved data    
    SAVEDIR_LABELS = "/lemur2/lucas/analyses/main/strokes_clustering_similarity/analysis_match_base_prims/MULT/Diego_Pancho|220531_231220|N=52"
    # TODO HERE: Load it, and keep just those trialcodes.

    # dict_failures = {}
    dict_res = {}
    for animal in ["Pancho", "Diego"]:

        # path = f"{SAVEDIR_LABELS}/final_list_novel/{animal}/final_results_each_los.pkl"
        # pd.read_pickle(path)

        # path = f"{SAVEDIR_LABELS}/final_list_novel/{animal}/failures_list.csv"
        # df_failures = pd.read_csv(path)

        path = f"{SAVEDIR_LABELS}/final_list_novel/version={savesuff}/{animal}/final_results_each_los.pkl"
        df_res = pd.read_pickle(path)
        dict_res[animal] = df_res
        
        # Keep only dist less than 1.0
        if False:
            thresh_better = 1.0
            df_failures = df_failures[df_failures["dist"]<thresh_better].reset_index(drop=True)

            dict_failures[animal] = df_failures

    ### Apply all updates to novel chars
    for D, ani_date in zip(LIST_D, LIST_ANIMAL_DATE):
        animal = ani_date[0]
        date = ani_date[1]
        # df_failures = dict_failures[animal]
        df_res = dict_res[animal]

        def _is_novel(tc):
            # Return if this new label is novel (F/T). Error if tc doesnt exist in database.
            tmp = df_res[df_res["this_trialcode"]==tc]
            assert len(tmp)==1
            return not tmp["failure_any"].item()

        novels = []
        for _, row in D.Dat.iterrows():
            if row["novel_char"] and not (row["trialcode"] in df_res["this_trialcode"].tolist()):
                # Previuosly novel, but not included in database... PRoblem, this cannot happen... 
                print(animal, date, row["trialcode"])
                assert False, "bug,  not sure why"
            elif not row["novel_char"]:
                # Not novel. check that it is not in database
                if False: # this sometimes fails, for good reason
                    assert not (row["trialcode"] in df_res["this_trialcode"].tolist())
                novels.append(False)
            elif row["novel_char"] and _is_novel(row["trialcode"]):
                # previously novel, and is still novel
                novels.append(True)
            elif row["novel_char"] and not _is_novel(row["trialcode"]):
                print("Changed from novel to not novel:  ", row["trialcode"])
                # Peviously novel, but changed now
                novels.append(False)
            else:
                assert False, "cannot happen"
            
        # Update novels
        D.Dat["novel_char"] = novels

def analy_plot_variety_wrapper(DictDS, SAVEDIR, DO_PLOTS=True, SKIP_IMAGE=False,
                               PLOT_DRAWINGS = False):
    """
    One of the main plots, variety, to get just those in paper, use SKIP_IMAGE=True
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import grouping_print_n_samples, grouping_append_and_return_inner_items_good

    if SKIP_IMAGE:
        # Just the useful qucik plots, not using image distance
        for animal in ["Pancho", "Diego"]:
            for basis in ["Diego", "Pancho"]:
                SAVEDIR_THIS = f"{SAVEDIR}/SEPARATE_DATA-animal={animal}-vs-basis={basis}"
                print("Saving to: ", SAVEDIR_THIS)
                
                DS = DictDS[(animal, basis)] # strokes vs. his own base prims
                
                if DO_PLOTS:
                    ### (1) Combine across all kinds of char
                    ### STROKES, summarizing match stats
                    # --- vs. Prims
                    savedir = f"{SAVEDIR_THIS}/scores_summary-clust_max_dist-allchars"
                    os.makedirs(savedir, exist_ok=True)
                    analy_score_base_prim_reuse_plots(DS, savedir, "clust_sim_max_colname", "clust_sim_max", PLOT_DRAWINGS=PLOT_DRAWINGS)
    else:
        # The original plots, a lot, might take time. Incluses the step from above too.
        for animal in ["Pancho", "Diego"]:
            for basis in ["Diego", "Pancho"]:

                if animal!=basis:
                    # To reduce how long this takes.
                    continue

                SAVEDIR_THIS = f"{SAVEDIR}/SEPARATE_DATA-animal={animal}-vs-basis={basis}"
                print("Saving to: ", SAVEDIR_THIS)
                
                ########################################
                ### First, merge image to basis distances
                
                DS = DictDS[(animal, basis)] # strokes vs. his own base prims
                DSc = DictDS[(animal, "image")] 
                
                #########################################
                ### Merge DS and DSc into one.
                # --> ONLY USE DS from now on
                trialcodes = DS.Dat["trialcode"].tolist()
                stroke_indexes = DS.Dat["stroke_index"].tolist()

                # slice from the others to append
                DSc.Dat = DSc.dataset_slice_by_trialcode_strokeindex(trialcodes, stroke_indexes)
                assert np.all(DSc.Dat["character"] == DS.Dat["character"])

                # put all columns into same dataframe
                DS.Dat["IMAGE_shape"] = DSc.Dat["shape"]
                DS.Dat["IMAGE_dist"] = DSc.Dat["dist_beh_task_strok"]
                DS.Dat["IMAGE_matches"] = DSc.Dat["matches_base_prim"]

                # DS.Dat = append_col_with_grp_index(DS.Dat, ["matches_base_prim", "IMAGE_matches"], "matches_prim_image")
                DS.Dat["exclmatch_prim_image"] = DS.Dat["IMAGE_matches"] & DS.Dat["matches_base_prim"]
                DS.Dat["exclmatch_prim"] = ~DS.Dat["IMAGE_matches"] & DS.Dat["matches_base_prim"]
                DS.Dat["exclmatch_image"] = DS.Dat["IMAGE_matches"] & ~DS.Dat["matches_base_prim"]
                DS.Dat["exclmatch_none"] = ~DS.Dat["IMAGE_matches"] & ~DS.Dat["matches_base_prim"]
                DS.Dat = append_col_with_grp_index(DS.Dat, ["task_kind", "FEAT_num_strokes_beh"], "task_kind_Nbeh")

                # give a single label
                DS.Dat = append_col_with_grp_index(DS.Dat, ["matches_base_prim", "IMAGE_matches"], "match_prim_imag_LABEL")
                
                #########################################
                if DO_PLOTS:
                    ### (1) Combine across all kinds of char
                    ### STROKES, summarizing match stats
                    # --- vs. Prims
                    savedir = f"{SAVEDIR_THIS}/scores_summary-clust_max_dist-allchars"
                    os.makedirs(savedir, exist_ok=True)
                    analy_score_base_prim_reuse_plots(DS, savedir, "clust_sim_max_colname", "clust_sim_max", PLOT_DRAWINGS=PLOT_DRAWINGS)

                    # --- vs. Image
                    savedir = f"{SAVEDIR_THIS}/scores_summary-image_distance-allchars"
                    os.makedirs(savedir, exist_ok=True)
                    analy_score_base_prim_reuse_plots(DS, savedir, "IMAGE_shape", "IMAGE_dist", "IMAGE_matches", PLOT_DRAWINGS=PLOT_DRAWINGS)

                    ### STROKES, comparing match to prim vs. match to image shape.
                    savedir = f"{SAVEDIR_THIS}/match_prim-vs-match_image-STROKES-allchars"
                    os.makedirs(savedir, exist_ok=True)
                    analy_score_match_baseprim_vs_matchimage_strokes(DS, savedir)

                    ### TRIALS, comapring n matches to prim vs. image
                    savedir = f"{SAVEDIR_THIS}/match_prim-vs-match_image-TRIALS-allchars"
                    os.makedirs(savedir, exist_ok=True)
                    dfcounts, dfcounts_clean = analy_score_match_baseprim_vs_matchimage_trials(DS, savedir)

                    if False: # Skip, since the summary -- combined -- will make these plots.
                        ### (2) For each kind of char, make separate set of plots.
                        grpvars = ["animal", "novel_char", "is_perfblocky"]
                        grpdict = grouping_append_and_return_inner_items_good(DS.Dat, grpvars)
                        for grp, inds in grpdict.items():
                            print(grp)

                            # Get copy of DS
                            ds = DS.copy()
                            ds.Dat = DS.Dat.iloc[inds].reset_index(drop=True)

                            ### STROKES, summarizing match stats
                            # --- vs. Prims
                            savedir = f"{SAVEDIR_THIS}/scores_summary-clust_max_dist-ani_novel_blocky={grp}"
                            os.makedirs(savedir, exist_ok=True)
                            analy_score_base_prim_reuse_plots(ds, savedir, "clust_sim_max_colname", "clust_sim_max", PLOT_DRAWINGS=PLOT_DRAWINGS)

                            # --- vs. Image
                            savedir = f"{SAVEDIR_THIS}/scores_summary-image_distance-ani_novel_blocky={grp}"
                            os.makedirs(savedir, exist_ok=True)
                            analy_score_base_prim_reuse_plots(ds, savedir, "IMAGE_shape", "IMAGE_dist", "IMAGE_matches", PLOT_DRAWINGS=PLOT_DRAWINGS)

                            ### STROKES, comparing match to prim vs. match to image shape.
                            savedir = f"{SAVEDIR_THIS}/match_prim-vs-match_image-STROKES-ani_novel_blocky={grp}"
                            os.makedirs(savedir, exist_ok=True)
                            analy_score_match_baseprim_vs_matchimage_strokes(ds, savedir)

                            ### TRIALS, comapring n matches to prim vs. image
                            savedir = f"{SAVEDIR_THIS}/match_prim-vs-match_image-TRIALS-ani_novel_blocky={grp}"
                            os.makedirs(savedir, exist_ok=True)
                            dfcounts, dfcounts_clean = analy_score_match_baseprim_vs_matchimage_trials(ds, savedir)


def analy_update_novel_label_ALL_plot_counts(LIST_D, LIST_ANIMAL_DATE, SAVEDIR):
    """
    Print how many unqiue los that were practiced vs novel.
    """
    
    from pythonlib.tools.pandastools import extract_first_trial_each_level
    from pythonlib.tools.pandastools import aggregGeneral

    ### Plot and 
    savedir = f"{SAVEDIR}/novel_counts_final"
    os.makedirs(savedir, exist_ok=True)
    print("Saving to: ", savedir)

    # Count how many novels
    res_counts = []
    for D, ani_date in zip(LIST_D, LIST_ANIMAL_DATE):
        animal = ani_date[0]
        date = ani_date[1]

        # Make sure only first trial for each char
        df = D.analy_subsample_first_trial_each_level("character")
        
        # keep only character
        df = df[df["task_kind"] == "character"].reset_index(drop=True)
        
        # Save stats on novel chars -- n novel per day.
        dfagg = aggregGeneral(df, ["character"], ["novel_char"], aggmethod=["any"])
        n_practiced = sum(dfagg["novel_char"] == False)
        n_novel = sum(dfagg["novel_char"] == True)

        res_counts.append({
            "animal":animal,
            "date":date,
            "n_practiced_los":n_practiced,
            "n_novel_los":n_novel,
        })
    # Finalisze and save, where is clean -- one los (unique task) 
    # MINOR PROBLEM: some tasks may be repeats. But confirmed that this is NEVER the case for novel chars (i..e each novel
    # char is first time it is encountred.)
    dfres_counts = pd.DataFrame(res_counts)
    dfres_counts.to_csv(f"{savedir}/all_counts.csv")
    dfres_counts.loc[:, ["n_practiced_los", "n_novel_los"]].mean().to_csv(f"{savedir}/mean_counts.csv")
    dfres_counts.loc[:, ["n_practiced_los", "n_novel_los"]].std().to_csv(f"{savedir}/stdl_counts.csv")
    for animal in ["Diego", "Pancho"]:
        dfthis = dfres_counts[dfres_counts["animal"] == animal]
        dfthis.loc[:, ["n_practiced_los", "n_novel_los"]].mean().to_csv(f"{savedir}/mean_counts-{animal}.csv")
        dfthis.loc[:, ["n_practiced_los", "n_novel_los"]].std().to_csv(f"{savedir}/stdl_counts-{animal}.csv")

    # ##### Older code, using distnace matrix, which is not good for large data -- blows up.
    # D.Dat["taskconfig_shploc"]

    # fig, axes, idxs = D.plotMultTrials(30, return_idxs=True)
    # D.plotMultTrials(idxs, "strokes_task", return_idxs=True)

    # for i in idxs:
    #     tk = D.Dat.iloc[i]["taskconfig_shploc_SHSEM"]
    #     print(i, " --- ", tk)

    # for i in idxs:
    #     print(i, " --- ", D.Dat.iloc[i]["los_info"])


    # # Get pairwise distances
    # from pythonlib.tools.distfunctools import distmat_construct_wrapper

    # list_pts = [np.concatenate(D.Dat.iloc[i]["strokes_task"]) for i in idxs]
    # labels = [tuple(x) for x in D.Dat.loc[idxs, ["animal", "date", "trialcode", "novel_char", "los_info"]].values.tolist()]

    # # With this dist_func, <1 means this is same image (actually, <5 even is very rare)
    # def dist_func(pts1, pts2):
    #     return ds._dist_strok_pair(pts1, pts2, do_centerize=True, centerize_method="bounding_box")
    # Cl = distmat_construct_wrapper(list_pts, list_pts, dist_func, PLOT=True, return_as_clustclass=True, 
    #                                clustclass_labels_1=labels, clustclass_labels_2=labels, clustclass_label_vars="index")

    # # def dist_func(pts1, pts2):
    # #     return ds._dist_strok_pair(pts1, pts2, do_centerize=True, centerize_method="bounding_box", distancever="hausdorff_max")
    # # Cl = distmat_construct_wrapper(list_pts, list_pts, dist_func, PLOT=True, return_as_clustclass=True, 
    # #                                clustclass_labels_1=labels, clustclass_labels_2=labels, clustclass_label_vars="index")


    # # Plot distribution of pairwise distances
    # fig, ax = plt.subplots()
    # dat = Cl.dataextract_masked_upper_triangular_flattened()
    # ax.hist(dat, bins=50)

    # # Find cases that are similar. Make sure do sanity checks.
    # # - cannot be (novel, practiced) --> remove the "novel" label.
    # # - cannot be (novel, novel) --> keep only the first novel.

    # dflabs = Cl.rsa_labels_return_as_df()

    # ma_ut = Cl._rsa_matindex_generate_upper_triangular()
    # ma = Cl.Xinput<1 & ma_ut
    # ind_pairs = Cl._rsa_matindex_convert_from_mask_to_rowcol(ma)

    # for inds in ind_pairs:
    #     _i = 3
    #     novel1 = dflabs.iloc[inds[0]]["index"][_i]
    #     novel2 = dflabs.iloc[inds[1]]["index"][_i]
    #     if novel1==True or novel2==True:
    #         print(inds)
    #         assert False

def analy_extract_strokes_wrapper(LIST_D, LIST_Dc, LIST_ANIMAL_DATE, SAVEDIR,
                                  WHICH_SHAPES_DIEGO, WHICH_SHAPES_PANCHO, 
                                  USE_SHUFFLED_PRIMS, SHUFFLED_IDX,
                                  SKIP_IMAGE = False):
    """
    Lots of stuff -- helper to extract FINAL data, which is each stroke, for each combinatoin of 
    (animal_data, animal_basis), in which image is included as a basis if SKIP_IMAGE is True.

    The actual cluster scores are extracted by loading, ie they are not held within D or Dc; those are just 
    there to be data from which strokes are extracted.

    PARAMS:
    - WHICH_SHAPES_DIEGO, str, telling which shape set to use. None for default (also main_21)
    - WHICH_SHAPES_PANCHO, <same>, usually use None
    - USE_SHUFFLED_PRIMS, bool, if True, then uses the half1-half2 shuffled prims, which is a control dataset.

    RETURNS:
    - DictDS, e.g., {(Pancho, Diego):ds}, which is data for Pancho, using Diego prims basis.
    """
    from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes

    if SKIP_IMAGE==False:
        # Have to fix code below, for some rason DS for image ends up lacks "novel_char" field for some reason,
        # Which is fine here, but it ends up failing in later code.
        #TODO: extract novel_char to ds for image. You might have to do this upstream where Dc is extracted. 
        assert False

    # For plotting novel drawings, I did not pre-extract clusters for cross-animal, so I just skip it, since its not
    # used anyway.
    HACK_IGNORE_GETTING_CROSS_ANIMAL_SCORE = False

    ### RUN (Hard coded)
    WHICH_BASIS_DIEGO = "Diego"
    WHICH_BASIS_PANCHO = "Pancho"

    if USE_SHUFFLED_PRIMS:
        if SHUFFLED_IDX==0:
            WHICH_BASIS_DIEGO += "_shuffled"
            WHICH_BASIS_PANCHO += "_shuffled"
        else:
            WHICH_BASIS_DIEGO += f"_shuffled_{SHUFFLED_IDX}"
            WHICH_BASIS_PANCHO += f"_shuffled_{SHUFFLED_IDX}"

    LIST_DS_Diegobase = [] # distance to Diego's prims
    LIST_DS_Panchobase = [] # distance to Pancho's prims
    LIST_DSc = [] # distance to image
    for D, Dc, (animal, date) in zip(LIST_D, LIST_Dc, LIST_ANIMAL_DATE):
        # Convert to DS
        DS = preprocess_dataset_to_datstrokes(D, "all_no_clean")

        # Get two prim distance sets (to each animals' basis set)
        # Must have previously run analyses that cluster the character strokes.
        if True:
            DSdiego, _ = DS.clustergood_load_saved_cluster_shape_classes_input_basis_set(WHICH_BASIS = WHICH_BASIS_DIEGO, 
                                                                                        WHICH_SHAPES=WHICH_SHAPES_DIEGO)
            
            DSpancho, _ = DS.clustergood_load_saved_cluster_shape_classes_input_basis_set(WHICH_BASIS = WHICH_BASIS_PANCHO, 
                                                                                        WHICH_SHAPES=WHICH_SHAPES_PANCHO)

            ### SANITY CHECK (remove this later)
            # - because I did not replace these with char_lust labels. MAke sure code downstream does not fail. If
            # fails, make sure these columns are not really useful.
            # - Also note: shape is the original, has nothing to do with cluster labels. Useful to keep it for determining thresholds
            DSdiego.Dat["shape_orig"] = DSdiego.Dat["shape"]
            DSpancho.Dat["shape_orig"] = DSpancho.Dat["shape"]

            del DSdiego.Dat["shape"]
            del DSdiego.Dat["shape_oriented"]
            del DSpancho.Dat["shape"]
            del DSpancho.Dat["shape_oriented"]

            if HACK_IGNORE_GETTING_CROSS_ANIMAL_SCORE:
                if animal=="Diego":
                    DSpancho = DSdiego.copy()
                elif animal=="Pancho":
                    DSdiego = DSpancho.copy()
                else:
                    assert False

            assert DSdiego is not None
            assert DSpancho is not None
        else:
            DS, params_clust = DS.clustergood_load_saved_cluster_shape_classes(
                skip_if_labels_not_found=False)

        ### Other preprocessing
        # (1) For DS against base prim
        for ds in [DSdiego, DSpancho]:
            ds.dataset_append_column("hdoffline")
            ds.dataset_append_column("los_set")
            ds.dataset_append_column("FEAT_num_strokes_beh")
            for feat in ["novel_char", "is_perfblocky"]:
                ds.dataset_append_column(feat)

        if not SKIP_IMAGE:
            # Dumb to run this twice, but it works. should optimized to be quicker.
            DSc = preprocess_dataset_to_datstrokes(Dc, "all_no_clean") # This is only here to get image distance

            # (2) For DS against image
            DSc.distgood_compute_beh_task_strok_distances()
            
            list_ds_tmp = [DSdiego, DSpancho, DSc]
        else:
            DSc = None
            list_ds_tmp = [DSdiego, DSpancho]

        # (3) For all
        for ds in list_ds_tmp:
            ds.Dat["animal"] = ds.Dataset.animals(True)[0]
            ds.Dat["date"] = int(ds.Dataset.dates(True)[0])
            ds.dataset_append_column("los_info")
            ds.dataset_append_column("los_info")

        LIST_DS_Diegobase.append(DSdiego)
        LIST_DS_Panchobase.append(DSpancho)
        LIST_DSc.append(DSc)
    
    ### Concat DS across dates
    # Instead of concatting, which is failing and slow, just concat the dataframe, and use DS just for the methods (i.e., minimal use)
    from pythonlib.dataset.dataset_strokes import concat_dataset_strokes_minimal, concat_dataset_strokes
    DSDiegobase = concat_dataset_strokes_minimal(LIST_DS_Diegobase)
    DSPanchobase = concat_dataset_strokes_minimal(LIST_DS_Panchobase)
    if not SKIP_IMAGE:
        DSc = concat_dataset_strokes_minimal(LIST_DSc)

    # from pythonlib.dataset.analy_dlist import concatDatasets
    # Dall = concatDatasets(LIST_D)

    # for DS in LIST_DS:
    #     DS.Dataset._TaskclassCheckPrimsExtraParams = False
    # from pythonlib.dataset.dataset_strokes import concat_dataset_strokes
    # concat_dataset_strokes(LIST_DS)

    ## No, this takes way too long, and stil fails a lot.
    # for DS in LIST_DS_Diegobase:
    #     DS.Dataset._TaskclassCheckPrimsExtraParams = False
    # DSDiegobase = concat_dataset_strokes(LIST_DS_Diegobase)

    ### SPLIT -- Get each (animal, base) as a separate DS
    DictDS = {}
    for animal_data in ["Diego", "Pancho"]:
        
        # Against Diego base
        df = DSDiegobase.Dat[DSDiegobase.Dat["animal"] == animal_data].reset_index(drop=True)
        ds = DSDiegobase.copy()
        ds.Dat = df
        assert len(df)>0
        # # - finalize columns
        # ds.Dat["FINAL_shape"] = ds.Dat["clust_sim_max_colname"]
        # ds.Dat["FINAL_dist"] = ds.Dat["clust_sim_max"]
        # - save
        DictDS[(animal_data, "Diego")] = ds

        # Against Pancho base
        df = DSPanchobase.Dat[DSPanchobase.Dat["animal"] == animal_data].reset_index(drop=True)
        ds = DSPanchobase.copy()
        ds.Dat = df
        assert len(df)>0
        # # - finalize columns
        # ds.Dat["FINAL_shape"] = ds.Dat["clust_sim_max_colname"]
        # ds.Dat["FINAL_dist"] = ds.Dat["clust_sim_max"]
        # - save
        DictDS[(animal_data, "Pancho")] = ds

        if not SKIP_IMAGE:
            # Against image
            df = DSc.Dat[DSc.Dat["animal"] == animal_data].reset_index(drop=True)
            ds = DSc.copy()
            ds.Dat = df
            assert len(df)>0
            # # - finalize columns
            # ds.Dat["FINAL_shape"] = ds.Dat["shape"]
            # ds.Dat["FINAL_dist"] = ds.Dat["dist_beh_task_strok"]
            # - save
            DictDS[(animal_data, "image")] = ds


    ### Final conditining
    from pythonlib.tools.pandastools import append_col_with_grp_index

    if SKIP_IMAGE:
        list_basis = ["Diego", "Pancho"]
    else:
        list_basis = ["Diego", "Pancho", "image"]

    tmp = {}
    for animal_data in ["Diego", "Pancho"]:
        for basis in list_basis:

            ds = DictDS[(animal_data, basis)]
            ds.Dat["FINAL_basis"] = basis
            
            if basis in ["Diego", "Pancho"]:
                ds.Dat["FINAL_shape"] = ds.Dat["clust_sim_max_colname"]
                ds.Dat["FINAL_dist"] = ds.Dat["clust_sim_max"]
            elif basis == "image":
                ds.Dat["FINAL_shape"] = ds.Dat["shape"]
                ds.Dat["FINAL_dist"] = ds.Dat["dist_beh_task_strok"]
            else:
                assert False

            ds.Dat = append_col_with_grp_index(ds.Dat, ["novel_char", "is_perfblocky"], "novel|blocky")
            ds.Dat = append_col_with_grp_index(ds.Dat, ["animal","FINAL_basis"], "animal|basis")

            tmp[(animal_data, basis)] = ds
    DictDS = tmp

    return DictDS

