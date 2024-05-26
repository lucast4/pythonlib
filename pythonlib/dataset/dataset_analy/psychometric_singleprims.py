"""
5/20/24 - To deal with (usually single prim) expts with single sahpes, varying in graded fashion (therefore called
"psychometric"), doing the extraction of the psycho params and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pythonlib.tools.plottools import savefig
import os
import seaborn as sns

def preprocess_and_plot(D, var_psycho, PLOT=True):
    """
    Wrapper to do preprocess and plots, two kinds, either with strong restriction on strokes (only
    clean like singleprims) or lenient, including all even abort (useful to see vacillation).
    """
    SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("psycho_singleprims")
    SAVEDIR = f"{SAVEDIR}/{var_psycho}"

    DS, DSlenient, map_shape_to_psychoparams = preprocess(D, var_psycho)

    if PLOT:
        # Plot both strict and lenient
        plot_overview(DS, D, f"{SAVEDIR}/clean_strokes_singleprim", var_psycho=var_psycho)
        plot_overview(DSlenient, D, f"{SAVEDIR}/lenient_strokes", var_psycho=var_psycho)
    
    #####################
    # Also make plot of mean_sim_score (trial by trial var)
    savedir = f"{SAVEDIR}/using_primitivenessv2"
    os.makedirs(savedir, exist_ok=True)

    from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_plot_pipeline
    # First, extract all the derived metrics
    PLOT=True
    plot_methods = ("tls", "drw")
    DSnew, _, dfres, grouping = preprocess_plot_pipeline(D, PLOT=PLOT, plot_methods=plot_methods)
    _apply_psychoparams_to_ds(DSnew.Dat, map_shape_to_psychoparams, var_psycho)
    _apply_psychoparams_to_ds(dfres, map_shape_to_psychoparams, var_psycho)

    # Make plots
    fig = sns.catplot(data=dfres, x="angle_idx_within_shapeorig", y="mean_sim_score", col="shapeorig", 
        kind="point", sharey=True)
    savefig(fig, f"{savedir}/mean_sim_score-1.pdf")
    fig = sns.catplot(data=dfres, x="angle_idx_within_shapeorig", y="mean_sim_score", col="shapeorig", 
        alpha=0.5, sharey=True)
    savefig(fig, f"{savedir}/mean_sim_score-2.pdf")

    plt.close("all")

    return DS, DSlenient, map_shape_to_psychoparams

def _apply_psychoparams_to_ds(df, map_shape_to_psychoparams, var_psycho):
    """
    """

    var_psycho_unique = f"{var_psycho}_unique"
    var_psycho_str = f"{var_psycho}_str"
    var_psycho_idx = f"{var_psycho}_idx_within_shapeorig"

    names = ["shapeorig_psycho", "shapeorig", var_psycho_unique, var_psycho_str, var_psycho_idx]
    for i, na in enumerate(names):
        # D.Dat[f"seqc_0_{na}"] = [map_shape_to_psychoparams[sh][i] for sh in D.Dat["seqc_0_shape"]]
        df[na] = [map_shape_to_psychoparams[sh][i] for sh in df["shape"]]
        # DSlenient.Dat[na] = [map_shape_to_psychoparams[sh][i] for sh in DSlenient.Dat["shape"]]

def preprocess(D, var_psycho="angle", SANITY=True):
    """
    For psychometric variables, such as angle, determines for each stroke what its original shape is, and 
    its angle relative to that shape. 
    """
    from pythonlib.tools.pandastools import find_unique_values_with_indices
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import append_col_with_index_number_in_group, append_col_with_index_of_level, append_col_with_index_of_level_after_grouping
    from pythonlib.tools.pandastools import grouping_print_n_samples
    from pythonlib.tools.checktools import check_objects_identical
    from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes

    assert var_psycho == "angle", "coded only for this currently."

    # First, for each token, extract variables that reflect the psycho variable/param.
    token_ver = "task"
    for i, row in D.Dat.iterrows():
        # for each shape get its concrete params
        Tk = D.taskclass_tokens_extract_wrapper(i, token_ver, return_as_tokensclass=True)
        Tk.features_extract_wrapper(["loc_on", "angle"], angle_twind=[0, 2])
    df = D.tokens_extract_variables_as_dataframe(["shape", "loc_on", "angle", "Prim", "gridloc"], token_ver)
    
    # Extract the original shape, which should have been overwritten in rprepovessing, but is useufl as a category
    # to anchor the variations.
    list_shape_orig = []
    for P in df["Prim"]:
        list_shape_orig.append(P.shape_oriented())
    df["shape_orig"] = list_shape_orig

    # For each shape_orig, get an ordered indices for angle. m
    col = var_psycho
    var_psycho_unique = f"{var_psycho}_unique"
    var_psycho_str = f"{var_psycho}_str"
    var_psycho_idx = f"{var_psycho}_idx_within_shapeorig"

    unique_values, indices, map_index_to_value = find_unique_values_with_indices(df, col, 
        append_column_with_unique_values_colname=var_psycho_unique)
    df[var_psycho_str] = [f"{a:.2f}" for a in df[f"{var_psycho}_unique"]]

    if SANITY:
        df["shape_hash"] = [P.label_classify_prim_using_stroke(return_as_string=True, version="hash") for P in df["Prim"]]
        assert np.all(df["shape_hash"] == df["shape"]), "for psycho, you need to turn on reclassify_shape_using_stroke_version=hash in preprocess/general"

    # Redefine each token to have a unique identifier (shape-psycho)
    df = append_col_with_grp_index(df, ["shape_orig", var_psycho_str], "shape_pscho", True)

    # Convert psycho var levels in to indices that start at 0 for each shape-orig.
    df = append_col_with_index_of_level_after_grouping(df, ["shape_orig"], var_psycho_str, var_psycho_idx)


    # Finally, get map from shape to psycho stuff
    map_shape_to_psychoparams = {}
    for i, row in df.iterrows():
        shape = row["shape"]
        params = (row["shape_pscho"], row["shape_orig"], row[var_psycho_unique], row[var_psycho_str], row[var_psycho_idx])
        if shape in map_shape_to_psychoparams:
            assert check_objects_identical(map_shape_to_psychoparams[shape], params)
        else:
            map_shape_to_psychoparams[shape] = params

    # grouping_print_n_samples(df, ["shape", "shape_pscho"])
    # grouping_print_n_samples(df, ["shape_pscho", "shape_hash", "angle_idx_within_shapeorig", "gridloc", "shape"])
    if SANITY:
        grouping_print_n_samples(df, ["shape_pscho", "shape_hash", var_psycho_idx, "gridloc"])
        df.groupby(["shape_pscho", "shape_hash", "shape_orig", "shape", var_psycho_str]).size().reset_index()
    
    if False: # Not needed
        D.tokens_assign_dataframe_back_to_self_mult(df, tk_ver=token_ver)

    ############# EXTRACT DS
    DSlenient = preprocess_dataset_to_datstrokes(D, "singleprim_psycho")
    DS = preprocess_dataset_to_datstrokes(D, "singleprim")

    # Assign columns to DS and D
    names = ["shapeorig_psycho", "shapeorig", var_psycho_unique, var_psycho_str, var_psycho_idx]
    for i, na in enumerate(names):
        D.Dat[f"seqc_0_{na}"] = [map_shape_to_psychoparams[sh][i] for sh in D.Dat["seqc_0_shape"]]
        # DS.Dat[na] = [map_shape_to_psychoparams[sh][i] for sh in DS.Dat["shape"]]
        # DSlenient.Dat[na] = [map_shape_to_psychoparams[sh][i] for sh in DSlenient.Dat["shape"]]
    _apply_psychoparams_to_ds(DS.Dat, map_shape_to_psychoparams, var_psycho)
    _apply_psychoparams_to_ds(DSlenient.Dat, map_shape_to_psychoparams, var_psycho)

    return DS, DSlenient, map_shape_to_psychoparams


def plot_overview(DS, D, SAVEDIR, var_psycho="angle"):
    """
    """
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.pandastools import grouping_print_n_samples

    assert var_psycho == "angle", "code it for others, need to change variable strings below"

    # Only include first stroke in these plots
    DS = DS.copy()
    DS.Dat = DS.Dat[DS.Dat["stroke_index"] == 0].reset_index(drop=True)
    DS.distgood_compute_beh_task_strok_distances()

    ### PLOT DRAWINGS
    savedir = f"{SAVEDIR}/drawings"
    os.makedirs(savedir, exist_ok=True)

    niter = 2
    for _iter in range(niter):

        print("Plotting drawings...")
        figholder = DS.plotshape_multshapes_egstrokes("shapeorig_psycho", 6, ver_behtask="beh");
        for i, (fig, axes) in enumerate(figholder):
            savefig(fig, f"{savedir}/egstrokes-{i}-iter{_iter}.pdf")

        figholder = DS.plotshape_multshapes_egstrokes("shapeorig_psycho", 6, ver_behtask="task_aligned_single_strok");
        for i, (fig, axes) in enumerate(figholder):
            savefig(fig, f"{savedir}/egstrokes-{i}-task-iter{_iter}.pdf")
        plt.close("all")

        figbeh, figtask = DS.plotshape_row_col_vs_othervar("angle_idx_within_shapeorig", "shapeorig", n_examples_per_sublot=8, plot_task=True);
        savefig(figbeh, f"{savedir}/shapeorig-vs-idx-{i}-beh-iter{_iter}.pdf")
        savefig(figtask, f"{savedir}/shapeorig-vs-idx-{i}-task-iter{_iter}.pdf")
        plt.close("all")

    ### PLOT COUNTS
    print("Plotting counts...")
    savedir = f"{SAVEDIR}/counts"
    os.makedirs(savedir, exist_ok=True)

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "angle_idx_within_shapeorig", "shapeorig", ["gridloc"])
    savefig(fig, f"{savedir}/counts-idx-vs-shapeorig.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "angle_str", "shapeorig", ["gridloc"])
    savefig(fig, f"{savedir}/counts-str-vs-shapeorig.pdf")
    
    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "angle_binned", "angle_idx_within_shapeorig", ["shapeorig", "gridloc"])
    savefig(fig, f"{savedir}/counts-angle_binned-vs-idx.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "loc_on_clust", "angle_idx_within_shapeorig", ["shapeorig", "gridloc"])
    savefig(fig, f"{savedir}/counts-loc_on_clust-vs-idx.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "angle_idx_within_shapeorig", "shapeorig_psycho", ["shapeorig", "gridloc"])
    savefig(fig, f"{savedir}/counts-idx-vs-shapeorig_psycho.pdf")

    fig = grouping_plot_n_samples_conjunction_heatmap(DS.Dat, "shape", "shapeorig_psycho", ["gridloc"])
    savefig(fig, f"{savedir}/counts-shape-vs-shapeorig_psycho.pdf")

    savepath = f"{savedir}/groupings.txt"
    grouping_print_n_samples(DS.Dat, 
        ["shapeorig_psycho", "shapeorig", "angle_idx_within_shapeorig", "angle_str", "shape", "character"], 
        savepath=savepath)
    plt.close("all")

    #### ANALYSES (e..,g, timing and velocity)
    savedir = f"{SAVEDIR}/analyses"
    os.makedirs(savedir, exist_ok=True)
    print("Plotting motor analyses...")

    ## First stroke
    dfthis = DS.Dat
    dfthis = dfthis[dfthis["gap_from_prev_dur"]<5].reset_index(drop=True)

    fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="angle", hue="gridloc", col="shapeorig", alpha=0.5)
    savefig(fig, f"{savedir}/angle-vs-idx-1.pdf")
    fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="angle", hue="gridloc", col="shapeorig", kind="point")
    savefig(fig, f"{savedir}/angle-vs-idx-2.pdf")

    fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="loc_on_clust", hue="gridloc", col="shapeorig", jitter=True, alpha=0.5)
    savefig(fig, f"{savedir}/loc_on_clust-vs-idx-1.pdf")
    fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="loc_on_clust", hue="gridloc", col="shapeorig", jitter=True, alpha=0.5)
    savefig(fig, f"{savedir}/loc_on_clust-vs-idx-1.pdf")
    plt.close("all")

    # Slower for psycho?
    for sharey in [True, False]:
        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="velocity", hue="gridloc", col="shapeorig", 
            alpha=0.5, sharey=sharey)
        savefig(fig, f"{savedir}/velocity-vs-idx-sharey={sharey}-1.pdf")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="velocity", hue="gridloc", col="shapeorig", 
            kind="point", sharey=sharey)
        savefig(fig, f"{savedir}/velocity-vs-idx-sharey={sharey}-2.pdf")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="gap_from_prev_dur", hue="gridloc", col="shapeorig", 
            alpha=0.5, sharey=sharey)
        savefig(fig, f"{savedir}/gap_from_prev_dur-vs-idx-sharey={sharey}-1.pdf")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="gap_from_prev_dur", hue="gridloc", col="shapeorig", 
            kind="point", sharey=sharey)
        savefig(fig, f"{savedir}/gap_from_prev_dur-vs-idx-sharey={sharey}-2.pdf")
        plt.close("all")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="dist_beh_task_strok", hue="gridloc", col="shapeorig", 
            alpha=0.5, sharey=sharey)
        savefig(fig, f"{savedir}/dist_beh_task_strok-vs-idx-sharey={sharey}-1.pdf")

        fig = sns.catplot(data=dfthis, x="angle_idx_within_shapeorig", y="dist_beh_task_strok", hue="gridloc", col="shapeorig", 
            kind="point", sharey=sharey)
        savefig(fig, f"{savedir}/dist_beh_task_strok-vs-idx-sharey={sharey}-2.pdf")
        plt.close("all")

    ## Dataset
    savedir = f"{SAVEDIR}/analyses_dataset_lenient_strokes"
    os.makedirs(savedir, exist_ok=True)
    print("Plotting motor analyses...")

    D = D.copy()
    # D.extract_beh_features(["num_strokes_beh"])
    # frac_touched_min = 0.6
    # ft_decim_min = 0.3
    # shortness_min = 0.2
    # D.preprocessGood(params=["beh_strokes_at_least_one",
    #                             "no_supervision",
    #                             "remove_online_abort"],
    #                     frac_touched_min=frac_touched_min,
    #                     ft_decim_min=ft_decim_min,
    #                     shortness_min = shortness_min)
    D.extract_beh_features(["num_strokes_beh"])
    D.preprocessGood(params=["beh_strokes_at_least_one",
                                "no_supervision"])

    # Number of strokes used
    dfthis = D.Dat

    fig = sns.catplot(data=dfthis, x="seqc_0_angle_idx_within_shapeorig", y="FEAT_num_strokes_beh", col="seqc_0_shapeorig", 
        alpha=0.4, jitter=True)
    savefig(fig, f"{savedir}/FEAT_num_strokes_beh-vs-idx-1.pdf")

    fig = sns.catplot(data=dfthis, x="seqc_0_angle_idx_within_shapeorig", y="FEAT_num_strokes_beh", col="seqc_0_shapeorig", kind="box")
    savefig(fig, f"{savedir}/FEAT_num_strokes_beh-vs-idx-2.pdf")
    plt.close("all")

