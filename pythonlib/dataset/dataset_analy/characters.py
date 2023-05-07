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
    LIST_RES = []
    for dist in list_distance_ver:
        RES = generate_data(D, list_distance_ver=[dist])
        LIST_RES.append(RES)

    # plot for each metric
    list_inds = get_random_inds()
    for RES in LIST_RES:
        plot_example_trials(RES, list_inds=list_inds)


def pipeline_generate_and_plot_all(D, do_plots=True, filter_taskkind = "character",
    list_distance_ver=None):
    """
    Full pipeline to generate and plot and save all.
    """
    
    if list_distance_ver is None:
        list_distance_ver=("euclidian_diffs", "euclidian", "hausdorff_alignedonset")

    savedir = D.make_savedir_for_analysis_figures("character_strokiness")
    
    RES = generate_data(D, filter_taskkind=filter_taskkind, list_distance_ver=list_distance_ver)

    if do_plots: 
        DS = RES["DS"]
        list_strok_basis = RES["list_strok_basis"]
        list_shape_basis = RES["list_shape_basis"]
        plot_clustering(DS, list_strok_basis, list_shape_basis, savedir)

        plot_learning_and_characters(D, savedir)

        for MIN_SCORE in [0., 0.7]:
            for sorted_unique_shapes in [True, False]:
                plot_prim_sequences(RES, D, savedir, MIN_SCORE, 
                    sorted_unique_shapes)

    return RES, savedir


def generate_data(D, which_basis_set="standard_17", 
        which_shapes="all_basis",
        trial_summary_score_ver="clust_sim_max",
        list_distance_ver=None, 
        plot_score_hist=False, filter_taskkind = None,
        ds_clean_methods = None, ds_clean_params = None,
        ds_filterdict = None):
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

    if list_distance_ver is None:
        list_distance_ver  =("euclidian_diffs", "euclidian", "hausdorff_alignedonset")
    ### Generate Strokes data
    DS = DatStrokes(D)

    if ds_clean_methods is not None:
        DS.clean_preprocess_data(methods=ds_clean_methods, params=ds_clean_params)
    
    # Filter to just "character" tasks
    if filter_taskkind:
        DS.filter_dataframe({"task_kind":[filter_taskkind]}, True)

    if ds_filterdict is not None:
        DS.filter_dataframe(ds_filterdict, True)

    ### Generate basis set of strokes
    if D.animals()==["Pancho"]:
        which_basis_set = "standard_17"
    elif D.animals()==["Diego"]:
        which_basis_set = "diego_all_minus_open_to_left"
    else:
        print(D.animals())
        assert False
        
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
        if sum(~df[scorename].isna())>0:
            fig = sns.pairplot(data=df, x_vars=vars_to_compare, y_vars=[scorename])
            fig.savefig(f"{sdir}/corr-rew_vs_strokiness-bk{bk}.pdf")

        plt.close("all")


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
    # First, sort characters
    # Sort all characters by score, and plot them in a grid
    list_char, list_score = D.taskcharacter_find_plot_sorted_by_score(scorename)

    # Plot
    # -- get one trial for each char
    from pythonlib.tools.pandastools import extract_trials_spanning_variable
    n_iter = 3
    for i in range(n_iter):
        inds, chars = extract_trials_spanning_variable(D.Dat, "character", list_char)
        assert chars == list_char

        if len(inds)<60:
            indsthis = inds
            charsthis = chars
            list_score_this = list_score
        else:
            indsthis = inds[:30] + inds[-30:]
            charsthis = chars[:30] + chars[-30:]
            list_score_this = list_score[:30] + list_score[-30:]

        # -- plot
        fig, axes, idxs = D.plotMultTrials2(indsthis, titles=charsthis, SIZE=3);
        fig.savefig(f"{sdir}/drawings_sorted_byscore-iter{i}-beh.pdf")
        fig, axes, idxs = D.plotMultTrials2(indsthis, "strokes_task", titles=list_score_this);
        fig.savefig(f"{sdir}/drawings_sorted_byscore-iter{i}-task.pdf")

        plt.close("all")

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

    DF = D.Dat
    print("orignial", len(DF))
    if ONLY_TRIALS_IN_NOSUP_BLOCKS:
        DF = DF[DF["supervision_stage_concise"]=="off|0||0"]
    print("ONLY_TRIALS_IN_NOSUP_BLOCKS", len(DF))
    if ONLY_TRIALS_WITH_ENOUGH_STROKES:
        DF = DF[DF["nbeh_match_ntask"]==True]
    print("ONLY_TRIALS_IN_NOSUP_BLOCKS", len(DF))

    list_slope = []
    list_slope_sk2 = []
    list_slope_rew = []
    for char in list_char_alpha:
        df = DF[DF["character"]==char]
        t = df["tvalfake"]
        v = df[scorename]
        strokinessv2 = df["strokinessv2"]
        rew_total = df["rew_total"]

        if len(t)>=4:
            # convert to to rank
            t = rankItems(t)
            slope = lr(t, v)[0]
            list_slope.append(slope)

            # make sure score and strokiness correlate with offline score
            slope = lr(strokinessv2, v)[0]
            list_slope_sk2.append(slope)

            slope = lr(rew_total, v)[0]
            list_slope_rew.append(slope)      
        else:
            list_slope.append(np.nan)
            list_slope_rew.append(np.nan)
            list_slope_sk2.append(np.nan)

    # only keep chars with enough data
    inds = np.where(~np.isnan(list_slope))[0].tolist()
    list_char_alpha = [list_char_alpha[i] for i in inds]
    list_slope = [list_slope[i] for i in inds]
    list_slope_rew = [list_slope_rew[i] for i in inds]
    list_slope_sk2 = [list_slope_sk2[i] for i in inds]

    # Plot (each char)
    fig, axes = plt.subplots(1,3, figsize=(15, len(list_char_alpha)*0.16))

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
    fig.savefig(f"{sdir}/hist_n_matches.pdf")

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

    plt.close("all")

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

    def _extract_shapes_beh_order(ind_dataset):
        """ get list of shapes for this trial in order of beh strokes
        """        
        # 1) shapes, based on simmat alignment
        shapes_ordered_by_beh_strokes_datseg = DS.dataset_extract_strokeslength_list(ind_dataset, "shape")

        # 2) shapes, based on clustering each stroke
        shapes_ordered_by_beh_strokes_cluster = DS.dataset_extract_strokeslength_list(ind_dataset, "clust_sim_max_colname")

        return shapes_ordered_by_beh_strokes_datseg, shapes_ordered_by_beh_strokes_cluster

    # For each char, get its "best parse"
    # list_scores = []
    list_shapes_datseg = []
    list_shapes_cluster = []
    for i in range(len(D.Dat)):
        if D.Dat.iloc[i]["strokes_clust_score"]>=MIN_SCORE:
            shapes_datseg, shapes_cluster = _extract_shapes_beh_order(i)
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
        B.append(sh_d)
    list_shapes_cluster = A
    list_shapes_datseg = A

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
