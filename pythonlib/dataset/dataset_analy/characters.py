""" analysis of characters, including strokiness
Extract prims, gets their dsitance to basis sets of strokes, 
plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def debug_eyeball_distance_metric_goodness(D):
    """ Script to comapre distnaces by eye to what loloks good. Iterates over distance metrics,
    and for eahc plots example strokes along with printing their scores. See that the scores
    match expected by eye (strokiness).
    NMOTE: in progress, but skeleton is there.
    """
        
    # collect data, once for each distance metric
    list_distance_ver=["euclidian_diffs", "euclidian", "hausdorff_alignedonset"]
    for dist in list_distance_ver:
        RES = generate_data(D, list_distance_ver=[dist])
        LIST_RES.append(RES)

    # plot for each metric
    list_inds = get_random_inds()
    for RES in LIST_RES:
        plot_example_trials(RES, list_inds=list_inds)


def pipeline_generate_and_plot_all(D):
    """
    Full pipeline to generate and plot and save all.
    """

    savedir = D.make_savedir_for_analysis_figures("character_strokiness")

    RES = generate_data(D)

    DS = RES["DS"]
    list_strok_basis = RES["list_strok_basis"]
    list_shape_basis = RES["list_shape_basis"]
    plot_clustering(DS, list_strok_basis, list_shape_basis, savedir)

    plot_learning_and_characters(D, savedir)

    return RES, savedir


def generate_data(D, which_basis_set="standard_17", 
        which_shapes="all_basis",
        trial_summary_score_ver="clust_sim_max",
        list_distance_ver=("euclidian_diffs", "euclidian", "hausdorff_alignedonset"),
        plot_score_hist=False):
    """
    Initial data generation: extracts strokes, extracts basis
    set of strokes to compare them to, does similarity matrix
    PARAMS;
    - which_basis_set, which_shapes, params for getting bassis set of 
    strokes (see within)
    - trial_summary_score_ver, str, which statistic to use as the
    summary score (i.e, a strokiness score)
    RETURNS:
    - RES, dict of results.
    """
    from pythonlib.dataset.dataset_strokes import DatStrokes

    ### Generate Strokes data
    DS = DatStrokes(D)
    # Filter to just "character" tasks
    DS.filter_dataframe({"task_kind":["character"]}, True)

    ### Generate basis set of strokes
    dfstrokes, list_strok_basis, list_shape_basis = DS.stroke_shape_cluster_database_load_helper(
        which_basis_set=which_basis_set, which_shapes=which_shapes)

    ### Generate dataset of beh strokes
    list_strok = DS.Dat["strok"].tolist()
    list_shape = DS.Dat["shape"].tolist()

    ### Compute similarity
    Cl = DS._cluster_compute_sim_matrix_aggver(list_strok, list_strok_basis, list_distance_ver,
                                                        labels_for_Clusters = list_shape, 
                                                        labels_for_basis = list_shape_basis)

    ### Extract scalar values summarizing the simialrity scores (e.g,, clustering)
    # For each beh stroke, get (i) match and (ii) uniqueness.
    sims_max = Cl.Xinput.max(axis=1)
    sims_min = Cl.Xinput.min(axis=1)
    # sims_mean = Cl.Xinput.mean(axis=1)
    sims_concentration = (sims_max - sims_min)/(sims_max + sims_min)
    # which shape does it match the best
    inds_maxsim = np.argmax(Cl.Xinput, axis=1)
    cols_maxsim = [Cl.LabelsCols[i] for i in inds_maxsim]

    ### Slide back into DS
    DS.Dat["clust_sim_max"] = sims_max
    DS.Dat["clust_sim_concentration"] = sims_concentration
    DS.Dat["clust_sim_max_ind"] = inds_maxsim
    DS.Dat["clust_sim_max_colname"] = cols_maxsim
    DS.Dat["clust_sim_vec"] = [vec for vec in Cl.Xinput]

    ### Slide back in to D: Collect scores (avg over strokes for a trial) and put back into D
    list_scores = []
    for i in range(len(D.Dat)):
        # get all rows in DS
        inds = DS._dataset_index_here_given_dataset(i)
        if len(inds)>0:
            score = np.mean(DS.Dat.iloc[inds][trial_summary_score_ver])
        else:
            # assert False, "not sure why this D trial has no strokes..."
            score = np.nan
        list_scores.append(score)
    D.Dat["strokes_clust_score"] = list_scores

    # OUT
    RES = {
        "which_basis_set":which_basis_set,
        "trial_summary_score_ver":trial_summary_score_ver,
        "which_shapes":which_shapes,
        "DS":DS,
        "Cl":Cl,
        "list_strok_basis":list_strok_basis,
        "list_shape_basis":list_shape_basis,
        "list_distance_ver":list_distance_ver
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
        alignshape = DS.Dat.iloc[ind]["shape"]

        print(ind, int(100*simmax), int(100*simconc), simmaxname)
        print(simmaxind, simvec)
        print(" ")


def plot_learning_and_characters(D, savedir, scorename = "strokes_clust_score"):
    """ Plot and save all things related to:
    - strokiness of each trial, grouped by block, char, and day
    - changes in strokiness (i.e., learning)
    (Meant for across day analyssi)
    """ 

    sdir = f"{savedir}/figs_character_strokiness"
    import os
    os.makedirs(sdir, exist_ok=True)
    print("Saving figures at:", sdir)
    

    ## Score for each trial by block and date
    fig = sns.catplot(data=D.Dat, x="block", y=scorename, aspect=2, row="date")
    fig.savefig(f"{sdir}/score_by_block-1.pdf")

    fig = sns.catplot(data=D.Dat, x="block", y=scorename, kind="violin", aspect=2, row="date")
    fig.savefig(f"{sdir}/score_by_block-2.pdf")

    fig = sns.catplot(data=D.Dat, x="block", y=scorename, kind="point", aspect=2, hue="date")
    fig.savefig(f"{sdir}/score_by_block-3.pdf")

    fig = sns.catplot(data=D.Dat, x="date", y=scorename, kind="point", aspect=2, hue="block")
    fig.savefig(f"{sdir}/score_by_date-1.pdf")

    fig = sns.relplot(data=D.Dat, col="block", col_wrap=3, x="tvalfake", y=scorename, aspect=2, kind="scatter")
    fig.savefig(f"{sdir}/score_by_date-2.pdf")

    fig = sns.catplot(data=D.Dat, col="block", col_wrap=4, x="date", y=scorename, aspect=1, kind="point")
    fig.savefig(f"{sdir}/score_by_date-3.pdf")


    ## For each character
    # First, sort characters
    # Sort all characters by score, and plot them in a grid
    list_char, list_score = D.taskcharacter_find_sorted_by_score("strokes_clust_score")
    # Plot
    # -- get one trial for each char
    from pythonlib.tools.pandastools import extract_trials_spanning_variable
    n_iter = 3
    for i in range(n_iter):
        inds, chars = extract_trials_spanning_variable(D.Dat, "character", list_char)
        assert chars == list_char

        # -- plot
        fig, axes, idxs = D.plotMultTrials2(inds, titles=chars, SIZE=3);
        fig.savefig(f"{sdir}/drawings_sorted_byscore-iter{i}-beh.pdf")
        fig, axes, idxs = D.plotMultTrials2(inds, "strokes_task", titles=list_score);
        fig.savefig(f"{sdir}/drawings_sorted_byscore-iter{i}-task.pdf")


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


def plot_clustering(DS, list_strok_basis, list_shape_basis, savedir):
    """ Clustering beh stroke based on max similiaryt to prims. Plots example
    strokes (beh) to get a sense of how clustered are the strokes.
    """

    sdir = f"{savedir}/clustering_by_basis_prims"
    import os
    os.makedirs(sdir, exist_ok=True)

    from pythonlib.tools.pandastools import convert_to_1d_dataframe_hist
    shapes, vals, fig, ax = convert_to_1d_dataframe_hist(DS.Dat, "clust_sim_max_colname", True)
    fig.savefig(f"{sdir}/hist_n_matches.pdf")

    fig = DS.plot_examples_grid("clust_sim_max_colname", col_levels=shapes, nrows=5)
    fig.savefig(f"{sdir}/drawings_examplegrid_sorted_by_nmatches.pdf")

    # Plot the basis set stroke
    # sort by order
    list_strok_basis_sorted = [list_strok_basis[list_shape_basis.index(sh)] for sh in shapes]
    fig, axes = DS.plot_multiple_strok(list_strok_basis_sorted, overlay=False, ncols=len(list_strok_basis_sorted), titles=shapes);
    fig.savefig(f"{sdir}/drawings_examplegrid_sorted_by_nmatches-basis.pdf")

    figholder = DS.plot_egstrokes_grouped_by_shape(key_subplots = "clust_sim_max_colname",
                n_examples = 5, color_by=None, list_shape=None);

    for i, x in enumerate(figholder):
        fig = x[0]
        fig.savefig(f"{sdir}/drawings_examples_overlaid-iter{i}.pdf")

    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    dfthis, fig, ax, rgba_values = convert_to_2d_dataframe(DS.Dat, "character", "clust_sim_max_colname", True, annotate_heatmap=False);
    fig.savefig(f"{sdir}/hist_n_matches_2d_heat_characters.pdf")


    # for each basis strok, get the distribution of scores for beh strokes that match it
    fig = sns.catplot(data=DS.Dat, x="clust_sim_max_colname", y="clust_sim_max", aspect=2, order=shapes)
    from pythonlib.tools.snstools import rotateLabel
    rotateLabel(fig)
    fig.savefig(f"{sdir}/scoredist_shapes-1.pdf")

    fig = sns.catplot(data=DS.Dat, x="clust_sim_max_colname", y="clust_sim_max", aspect=2, order=shapes, kind="bar")
    from pythonlib.tools.snstools import rotateLabel
    rotateLabel(fig)
    fig.savefig(f"{sdir}/scoredist_shapes-2.pdf")


    # For each basis stroke, plot examples, sorted by score
    from pythonlib.tools.pandastools import extract_trials_spanning_variable
    nrand = 40
    trials_dict = extract_trials_spanning_variable(DS.Dat, "clust_sim_max_colname", shapes, 
                                                   n_examples=nrand, return_as_dict=True, method_if_not_enough_examples="prune_subset")[0]

    for sh in shapes:
        inds = trials_dict[sh]
        scores = DS.Dat["clust_sim_max"].values.tolist()
        
        # sort
        x = [(i, s) for i,s in zip(inds, scores)]
        x = sorted(x, key=lambda x:-x[1])
        inds = [xx[0] for xx in x]
        scores = [xx[1] for xx in x]
        fig, axes, list_inds = DS.plot_multiple(inds, titles=scores, ncols=8)
        fig.savefig(f"{sdir}/behstrokes_matching_basis-{sh}-sorted_by_score.pdf")    

