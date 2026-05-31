"""
Here is just the neural figures in manuscript (not all, those may be in other places)
Linked to notebook: 250925_MANUSCRIPT_FIGURES_2_syntax.ipynb

"""

# from typing import Any
# from pythonlib.dataset.dataset import load_dataset_notdaily_helper, load_dataset_daily_helper
# import pickle
import seaborn as sns
import os
import matplotlib.pyplot as plt
# from pythonlib.tools.snstools import rotateLabel
import pandas as pd
from pythonlib.tools.plottools import savefig
import numpy as np
# import traceback
# from concurrent.futures import ProcessPoolExecutor, as_completed

SAVEDIR_ALL = "/lemur2/lucas/analyses/manuscripts/2_syntax"

def alignmentcompute_prepare_dataset(pa_subspace, var_effect, vars_others, min_n_trials):
    """
    Extract dfdist, after adding in chunk_rank_global to pa
    NOTE: dfdist is missing actual values, just useful for subsequently looking for those pairs.
    RETURNS:
    - dfdist, missing the values. 
    """
    from pythonlib.dataset.dataset_analy.grammar import chunk_rank_global_extract
    from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar

    # Get chunk rank global
    dflab = pa_subspace.Xlabels["trials"]
    dflab["date"] = "dummy"
    chunk_rank_global_extract(dflab, check_low_freq_second_shape=True, shape_ratio_max=0.9)
    
    # Get pairwise euclid distances
    euclidean_label_vars = [var_effect] + vars_others
    # rsa_heatmap_savedir = "/tmp/RSA"
    # os.makedirs(rsa_heatmap_savedir, exist_ok=True)
    rsa_heatmap_savedir = None
    dfdist, _ = timevarying_compute_fast_to_scalar(pa_subspace, label_vars=euclidean_label_vars, 
                                            rsa_heatmap_savedir=rsa_heatmap_savedir, plot_conjunctions_savedir=rsa_heatmap_savedir,
                                            prune_levs_min_n_trials=min_n_trials, get_only_one_direction=False,
                                            skip_computing_dists=True)
    
    dfdist["var_effect"] = var_effect
    dfdist["vars_others"] = [tuple(vars_others) for _ in range(len(dfdist))]

    return dfdist, euclidean_label_vars


def compute_alignment_helper_prune_data(dfdist, var_effect, vars_dont_care, vars_control, vars_others, euclidean_label_vars):
    return alignmentcompute_helper_prune_data(dfdist, var_effect, vars_dont_care, vars_control, vars_others, euclidean_label_vars)


def alignmentcompute_helper_prune_data(dfdist, var_effect, vars_dont_care, vars_control, vars_others, 
    euclidean_label_vars):
    """
    Given dfdist, prune to just pairs which are useful for subsequently computing alginemnt between pairs.
    Things like: each pair must have increaseing values of <var_effect>

    RETURNS:
    - dfdist, pruned.
    """
    # Strict version -- only compare vectors with same controlled for everything
    from pythonlib.tools.vectools import compute_weighted_alignment
    from neuralmonkey.analyses.euclidian_distance import dfdist_variables_generate_constrast_strings, dfdist_variables_generate_var_same
    from neuralmonkey.analyses.euclidian_distance import dfdist_variables_effect_extract_helper

    for v in vars_others:
        assert v in vars_control
    vars_included = [var_effect] + vars_dont_care + vars_control
    if not sorted(vars_included) == sorted(euclidean_label_vars):
        print(sorted(vars_included))
        print(sorted(euclidean_label_vars))
        assert False

    # -- Fixed stuff
    colname_conj_same = dfdist_variables_generate_var_same(euclidean_label_vars)

    # FIRST, Restrict to pairs (rows) that are properly controlled
    levs_sorted = sorted(set(dfdist[f"{var_effect}_1"].unique().tolist() + dfdist[f"{var_effect}_2"].unique().tolist()))
    # e.g.,[0.0, 1.0, 2.0]

    # lev_pairs_allowed = # Get just adjancent levels, so that if the set is 1,2,3,4, then only 12, 23, 34 are allowed.
    # Gets in string format, e.g., "1|2", "2|3", "3|4"
    lev_pairs_allowed = [f"{levs_sorted[i]}|{levs_sorted[i+1]}" for i in range(len(levs_sorted)-1)]
    # e.g., ['0.0|1.0', '1.0|2.0']

    dfdist = dfdist[dfdist[f"{var_effect}_12"].isin(lev_pairs_allowed)]
    assert len(dfdist) > 0, "no data left"

    # 1. Keep only cases where var_effect is increasing
    dfdist = dfdist[dfdist[f"{var_effect}_2"] > dfdist[f"{var_effect}_1"]].reset_index(drop=True)

    # 2. Keep only cases with correct contrast
    # Note: vars_control is all the leftover variables, so this does work.
    dfdist = dfdist_variables_effect_extract_helper(dfdist, colname_conj_same, euclidean_label_vars, 
                                                        [var_effect], vars_dont_care)
    return dfdist

def compute_dot_product_distributions_helper(dfdist, var_effect, vars_dont_care, vars_control, 
        vars_others, euclidean_label_vars):
    """
    GOOD, helper to get pairwise dot products across pairs of var_effect, and also within-level dot products.
    RETURNS:
    - dot_products_all, array of mean dot products, each a mean over multiple
    vector pairs, for a specific pair of levels of var_effect (e.,g, 0|1 vs. 1|2 is one pair)
    - dot_products_1_all, dot_products_2_all, same as above, but for dot products of vectors with themselves
    - n_each_all, array of sample sizes, each a mean over multiple vector pairs, for a specific pair of levels of var_effect
    """
    ### Good version
    from pythonlib.tools.vectools import compute_weighted_alignment, _helper_compute_dot_products
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    # Strict version -- only compare vectors with same controlled for everything

    dfdist_this = compute_alignment_helper_prune_data(dfdist, var_effect, vars_dont_care, vars_control, 
        vars_others, euclidean_label_vars)

    # get vector alignment for var_effect, WITHIN each level of vars_others
    _var_effect = f"{var_effect}_12"
    dot_products_all = []
    dot_products_1_all = []
    dot_products_2_all = []
    n_each_all = []
    effect_lev_pairs_all = []
    grp_others_all = []
    # norms_a_all = []
    # norms_b_all = []

    _vars_others = [f"{v}_1" for v in vars_others] # can use 1 or 2, they are equal
    grpdict_others = grouping_append_and_return_inner_items_good(dfdist_this, _vars_others)
    n_levs = []
    for _grp_others, inds_others in grpdict_others.items():
        dfdist_this_others = dfdist_this.iloc[inds_others]

        levs_exist = sorted(dfdist_this_others[_var_effect].unique())
        n_levs.append(len(levs_exist))

        for _i, lev1 in enumerate(levs_exist):
            for _j, lev2 in enumerate(levs_exist):
                if _j>_i:
                
                    print(lev1, lev2)

                    vectors_1 = np.stack(dfdist_this_others[dfdist_this_others[_var_effect] == lev1]["vector"])
                    vectors_2 = np.stack(dfdist_this_others[dfdist_this_others[_var_effect] == lev2]["vector"])

                    dot_products, dot_products_1, dot_products_2 = _helper_compute_dot_products(vectors_1, vectors_2)

                    # Store sample size for the number of dot products
                    n_each = (len(dot_products), len(dot_products_1), len(dot_products_2))
                    n_each_all.append(n_each)
                    dot_products_all.append(dot_products.mean())
                    dot_products_1_all.append(dot_products_1.mean())
                    dot_products_2_all.append(dot_products_2.mean())
                    effect_lev_pairs_all.append((lev1, lev2))
                    grp_others_all.append(_grp_others)

    if len(dot_products_all) == 0:
        assert all([_x==1 for _x in n_levs]), "figure out why didnt collect any results"
        return None, None, None, None, None, None

    dot_products_all = np.array(dot_products_all)
    dot_products_1_all = np.array(dot_products_1_all)
    dot_products_2_all = np.array(dot_products_2_all)

    return dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all

def compute_dot_product_distributions_helper_across_grpother(dfdist, 
                                                            var_effect, vars_dont_care, vars_control, vars_others,
                                                            euclidean_label_vars):
    """
    Compute alignment for a given var_effect across levels of vars_others.
    E.g., Effect of chunk_within_rank, comparing across chunk_rank

    E.g. compute alignment of <chunk_within_rank> across different <chunk_ranks>.
    """
    # from pythonlib.dataset.scripts.analy_manuscript2_neural_figures import compute_dot_product_distributions_helper_two_vareffects


    # # ***************8 Two options for getting "across"
    # # OPTION 1: Strict. Only use grps with at least two level pairs, just as in the above for "within"
    # # ie this throws out cases where (i) only two chunk_ranks exist; and (ii) for Pancho, two shape sets
    # # computes stuff only within each epoch.
    # var_effect_1 = "chunk_within_rank"
    # vars_dont_care_1 = ["chunk_within_rank_semantic"] # Generally, correlated with var_effect
    # vars_control_1 = [ 'task_kind', 'epoch', 'chunk_rank_global', 'chunk_rank', 'shape', 'gridloc', 
    #                 'CTXT_loc_prev', 'chunk_n_in_chunk'] # These must be identical within each pair (ie each row)
    # vars_others_1 = ["chunk_rank_global"] # Computing vector alignment is done separately for each level of this

    # # ================ (2) Across
    # var_effect_2 = "chunk_within_rank"
    # vars_dont_care_2 = ["chunk_within_rank_semantic"] # Generally, correlated with var_effect
    # vars_control_2 = [ 'task_kind', 'epoch', 'chunk_rank_global', 'chunk_rank', 'shape', 'gridloc', 
    #                 'CTXT_loc_prev', 'chunk_n_in_chunk'] # These must be identical within each pair (ie each row)
    # vars_others_2 = ["chunk_rank_global"] # Computing vector alignment is done separately for each level of this

    only_if_different_grpothers = True # To ensure is across levels of grpothers.
    dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all = compute_dot_product_distributions_helper_two_vareffects(
                                                                dfdist, 
                                                                var_effect, vars_dont_care, vars_control, vars_others,
                                                                var_effect, vars_dont_care, vars_control, vars_others,
                                                                euclidean_label_vars,
                                                                only_if_different_grpothers=only_if_different_grpothers)

    # Becuase this does both directions, its values are reproduced. So take just the first half
    _n = len(dot_products_all)
    dot_products_all = dot_products_all[:int(_n/2)]
    dot_products_1_all = dot_products_1_all[:int(_n/2)]
    dot_products_2_all = dot_products_2_all[:int(_n/2)]
    n_each_all = n_each_all[:int(_n/2)]
    effect_lev_pairs_all = effect_lev_pairs_all[:int(_n/2)]
    grp_others_all = grp_others_all[:int(_n/2)]

    return dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all


def compute_dot_product_distributions_helper_two_vareffects(dfdist, 
                                                            var_effect_1, vars_dont_care_1, vars_control_1, vars_others_1,
                                                            var_effect_2, vars_dont_care_2, vars_control_2, vars_others_2,
                                                            euclidean_label_vars,
                                                            only_keep_if_both_12_have_mult_level_pairs=True,
                                                            only_if_different_grpothers=False,
                                                            debug_print_all_comparisons=False):
    """
    GOOD, like compute_dot_product_distributions_helper, but 
    only gets dot-products "ACROSS" two variables. I used this for comparing angles of <chunk_within_rank> vs.
    <chunk_rank>

    PARAMS:
    - only_keep_if_both_12_have_mult_level_pairs, bool, if True then this is more strict, only keeping cases
    where both var_effect_1 and var_effect_2 would have contributed also to the dot products within them (as they have
    multiple level pairs))

    RETURNS:
    - dot_products_all, array of mean dot products, each a mean over multiple
    vector pairs, for a specific pair of levels of var_effect (e.,g, 0|1 vs. 1|2 is one pair)
    - dot_products_1_all, dot_products_2_all, same as above, but for dot products of vectors with themselves
    - n_each_all, array of sample sizes, each a mean over multiple vector pairs, for a specific pair of levels of var_effect
    """
    ### Good version
    from pythonlib.tools.vectools import _helper_compute_dot_products
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
        
    dfdist_1 = compute_alignment_helper_prune_data(dfdist, var_effect_1, vars_dont_care_1, vars_control_1, vars_others_1, euclidean_label_vars)
    dfdist_2 = compute_alignment_helper_prune_data(dfdist, var_effect_2, vars_dont_care_2, vars_control_2, vars_others_2, euclidean_label_vars)

    if False:
        print(len(dfdist_1), len(dfdist_2))
        display(dfdist_1)
        display(dfdist_2)

    # # get vector alignment for var_effect, WITHIN each level of vars_others
    # _var_effect = f"{var_effect}_12"
    dot_products_all = []
    dot_products_1_all = []
    dot_products_2_all = []
    n_each_all = []
    effect_lev_pairs_all = []
    grp_others_all = []

    _var_effect_1 = f"{var_effect_1}_12"
    _var_effect_2 = f"{var_effect_2}_12"
    _vars_others_1 = [f"{v}_1" for v in vars_others_1] # can use 1 or 2, they are equal
    _vars_others_2 = [f"{v}_1" for v in vars_others_2] # 

    grpdict_others_1 = grouping_append_and_return_inner_items_good(dfdist_1, _vars_others_1)
    grpdict_others_2 = grouping_append_and_return_inner_items_good(dfdist_2, _vars_others_2)

    if only_keep_if_both_12_have_mult_level_pairs:
        min_n_lev_pairs = 2
    else:
        min_n_lev_pairs = 1

    for _grp_others_1, inds_others_1 in grpdict_others_1.items():
        dfdist_1_others = dfdist_1.iloc[inds_others_1]
        levs_exist_1 = sorted(dfdist_1_others[_var_effect_1].unique())

        if len(levs_exist_1) >= min_n_lev_pairs: # Beucase in compute_dot_product_distributions_helper() you only keep a _grp_others_1 if it has more than 1 level
            for lev1 in levs_exist_1:
                
                for _grp_others_2, inds_others_2 in grpdict_others_2.items():
                    dfdist_2_others = dfdist_2.iloc[inds_others_2]  
                    levs_exist_2 = sorted(dfdist_2_others[_var_effect_2].unique())
                    if len(levs_exist_2) >= min_n_lev_pairs:
                        for lev2 in levs_exist_2:

                            if only_if_different_grpothers:
                                if _grp_others_1 == _grp_others_2:
                                    continue
                            
                            if debug_print_all_comparisons:
                                print("(grp, levpair) : ", _var_effect_1, _grp_others_1, "|", lev1, " - vs - ",_var_effect_2, _grp_others_2, "|", lev2)

                            vectors_1 = np.stack(dfdist_1_others[dfdist_1_others[_var_effect_1] == lev1]["vector"])
                            vectors_2 = np.stack(dfdist_2_others[dfdist_2_others[_var_effect_2] == lev2]["vector"])

                            dot_products, dot_products_1, dot_products_2 = _helper_compute_dot_products(
                                vectors_1, vectors_2)

                            # Store sample size for the number of dot products
                            n_each_all.append((len(dot_products), len(dot_products_1), len(dot_products_2)))
                            dot_products_all.append(dot_products.mean())
                            dot_products_1_all.append(dot_products_1.mean())
                            dot_products_2_all.append(dot_products_2.mean())
                            effect_lev_pairs_all.append((lev1, lev2))
                            grp_others_all.append((_grp_others_1, _grp_others_2))
                    else:
                        print("levs_exist_2:", levs_exist_2)
        else:
            print("levs_exist_1:", levs_exist_1)

    dot_products_all = np.array(dot_products_all)
    dot_products_1_all = np.array(dot_products_1_all)
    dot_products_2_all = np.array(dot_products_2_all)

    return dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all

def alignmentcompute_wrapper_single_session(
        animal, date, SAVEDIR, 
        run, n_iter_splits, DEBUG_BREAK=False,
        DEBUG_FORCE_RETURN=False):
    """
    Wrapper, high-level, to go from dim projected neural data (PA) to dot products, and then save them.
    
    PA are taken from previously saved PA.

    PARAMS:
        # run = 30
        # n_iter_splits = 4 # What was used originally

    """
    from glob import glob
    from neuralmonkey.classes.session import _REGIONS_IN_ORDER_COMBINED
    from neuralmonkey.analyses.euclidian_distance import dfdist_extract_label_vars_specific
    from pythonlib.tools.pandastools import replace_None_with_string, stringify_values
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates
    import pickle
    from neuralmonkey.analyses.euclidian_distance import compute_vector_between_conditions
    import numpy as np
    from pythonlib.tools.vectools import compute_weighted_alignment

    ### Where to get the PA
    # save_suffix = "AnBmCk_general" 

    ### Params here
    list_strict = [False]
    # ANIMALS = ["Pancho", "Diego"]
    # ANIMALS = ["Pancho"]
    # DATES = None
    # DATES = [231114, 231116, 220901, 220906, 220907, 220908, 220909]
    _npcs_keep_euclidean = 6
    min_n_trials = 4 # to use a contrast.
    # if DEBUG:
    #     ANIMALS =  ["Diego"]
    #     DATES = [230726]
    #     # date = 230913

    #     # animal = "Pancho"
    #     # date = 220907


    ### Iterate
    print(f"Processing {animal}-{date}")
    RES = []
    RES_DOT = []
    for _iter in range(n_iter_splits):
        for bregion in _REGIONS_IN_ORDER_COMBINED:

            # Load dataset for devo
            path = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{run}/{animal}-{date}-q=RULE_ANBMCK_STROKE/bregion={bregion}/FITTING_subspc=('epoch', 'gridloc', 'DIFF_gridloc', 'chunk_rank', 'shape', 'rank_conj')-iter={_iter}/pa_subspace.pkl"

            with open(path, "rb") as f:
                pa_subspace = pickle.load(f)

            pa_subspace = pa_subspace.slice_by_dim_indices_wrapper("chans", list(range(_npcs_keep_euclidean)))
            pa_subspace = pa_subspace.slice_by_labels_filtdict({"task_kind":["prims_on_grid"]})

            # TODO: Remove effect of first stroke. (DONE)
            # TODO: Do targeted PCA to convert to top 2 PCs... For below code, which uses theta... (IGNORE, targeted PC already done)
            # TODO: Even better, don't convert to theta. Just take angle between vectors directly. (DONE)
            var_effect = "chunk_within_rank"
            vars_others = ["chunk_within_rank_semantic", "task_kind", "epoch", "chunk_rank_global", "chunk_rank", "shape", 
                        "gridloc", "CTXT_loc_prev", "chunk_n_in_chunk"]
            dfdist, euclidean_label_vars = alignmentcompute_prepare_dataset(pa_subspace, var_effect, vars_others, min_n_trials=min_n_trials)

            ### Clean up dfdist
            # from neuralmonkey.analyses.euclidian_distance import dfdist_variables_generate_var_same
            # colname_conj_same = dfdist_variables_generate_var_same(euclidean_label_vars)

            # Clean up
            dfdist["n1"] = [x[0] for x in dfdist["n_1_2"]]
            dfdist["n2"] = [x[1] for x in dfdist["n_1_2"]]
            dfdist = dfdist[(dfdist["n1"] >= min_n_trials) & (dfdist["n2"] >= min_n_trials)].reset_index(drop=True)

            if False:
                # Only
                n1 = sum(dfdist["chunk_rank_global_12"] == dfdist["chunk_rank_12"])
                n2 = len(dfdist["chunk_rank_global_12"] == dfdist["chunk_rank_12"])
                assert n1>0.7*n2, "why so many misaligned? might be fine to continue"
                dfdist = dfdist[dfdist["chunk_rank_global_12"] == dfdist["chunk_rank_12"]].reset_index(drop=True)
            
            ### Get vectors between all conditions
            # Get X for (0,1), (1,2), ... within each (chunk_rank).
            # Get (A, B), (B, C) -- across chunk ranks.
            dfanglevecs = compute_vector_between_conditions(pa_subspace, dfdist, var_effect, vars_others)
            assert np.all(dfanglevecs["labels_1"] == dfdist["labels_1"])
            assert np.all(dfanglevecs["labels_2"] == dfdist["labels_2"])
            dfdist["vector"] = dfanglevecs["vector"]

            # Return dfdist for debugging.
            if DEBUG_FORCE_RETURN:
                return dfdist, euclidean_label_vars

            ### Good version                        
            assert list_strict == [False], "Not using this anymore"
            for strict in list_strict:
                
                if False:
                    assert False, "havent carefully chcekd this recently. Is prob fine, but prob wont work."
                    ####################################################### 
                    ######################## V1 -- USING COSINE DISTANCE
                    ### Within rank
                    var_effect = "chunk_within_rank"
                    vars_dont_care = ["chunk_within_rank_semantic"] # Generally, correlated with var_effect
                    vars_control = [ 'task_kind', 'epoch', 'chunk_rank_global', 'chunk_rank', 'shape', 'gridloc', 
                                    'CTXT_loc_prev', 'chunk_n_in_chunk'] # These must be identical within each pair (ie each row)
                    # vars_others = ["chunk_rank_global", "chunk_n_in_chunk", "gridloc"] # Computing vector alignment is done separately for each level of this
                    # vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
                    if strict:
                        vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
                    else:
                        vars_others = ["chunk_rank_global"] # Computing vector alignment is done separately for each level of this

                    weighted_mean_sim, similarities, weights = compute_alignment_helper(dfdist, var_effect, vars_dont_care, vars_control, vars_others)
                    if similarities is not None:
                        RES.append({
                            "animal": animal,
                            "date": date,
                            "bregion": bregion,
                            "_iter": _iter,
                            "weighted_mean_sim": weighted_mean_sim,
                            "effect_kind": "chunk_within_rank",
                            "similarities":similarities,
                            "weights":weights,
                            "var_effect": var_effect,
                            "vars_dont_care": tuple(vars_dont_care),
                            "vars_control": tuple(vars_control),
                            "vars_others": tuple(vars_others),
                            "strict":strict,
                        })

                    ### Across chunk rank
                    var_effect = "chunk_rank_global"
                    vars_dont_care = ["chunk_rank", "shape"] # Generally, correlated with var_effect
                    vars_control = [ 'task_kind', 'epoch', 'gridloc', 'CTXT_loc_prev', 'chunk_n_in_chunk', 
                        'chunk_within_rank', 'chunk_within_rank_semantic'] # These must be identical within each pair (ie each row)
                    # vars_others = ["chunk_rank_global", "chunk_n_in_chunk", "gridloc"] # Computing vector alignment is done separately for each level of this
                    # vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
                    if strict:
                        vars_others = ["epoch", "chunk_within_rank_semantic"] # Computing vector alignment is done separately for each level of this
                    else:
                        vars_others = ["epoch"] # Computing vector alignment is done separately for each level of this

                    weighted_mean_sim, similarities, weights = compute_alignment_helper(dfdist, var_effect, vars_dont_care, vars_control, vars_others)
                    if similarities is not None:
                        RES.append({
                            "animal": animal,
                            "date": date,
                            "bregion": bregion,
                            "_iter": _iter,
                            "weighted_mean_sim": weighted_mean_sim,
                            "effect_kind": "chunk_rank_global",
                            "similarities":similarities,
                            "weights":weights,
                            "var_effect": var_effect,
                            "vars_dont_care": tuple(vars_dont_care),
                            "vars_control": tuple(vars_control),
                            "vars_others": tuple(vars_others),
                            "strict":strict,
                        })

                    ### Across (within and shapes)
                    # ================ (1) Within
                    var_effect = "chunk_within_rank"
                    vars_dont_care = ["chunk_within_rank_semantic"] # Generally, correlated with var_effect
                    vars_control = [ 'task_kind', 'epoch', 'chunk_rank_global', 'chunk_rank', 'shape', 'gridloc', 
                                    'CTXT_loc_prev', 'chunk_n_in_chunk'] # These must be identical within each pair (ie each row)
                    if strict:
                        vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
                    else:
                        vars_others = ["chunk_rank_global"] # Computing vector alignment is done separately for each level of this
                    dfdist_within = compute_alignment_helper_prune_data(dfdist, var_effect, vars_dont_care, vars_control, vars_others, euclidean_label_vars)

                    # ================ (2) Across
                    var_effect = "chunk_rank_global"
                    vars_dont_care = ["chunk_rank", "shape"] # Generally, correlated with var_effect
                    vars_control = [ 'task_kind', 'epoch', 'gridloc', 'CTXT_loc_prev', 'chunk_n_in_chunk', 'chunk_within_rank', 'chunk_within_rank_semantic'] # These must be identical within each pair (ie each row)
                    if strict:
                        vars_others = ["epoch", "chunk_within_rank_semantic"] # Computing vector alignment is done separately for each level of this
                    else:
                        vars_others = ["epoch"] # Computing vector alignment is done separately for each level of this                    
                    dfdist_across = compute_alignment_helper_prune_data(dfdist, var_effect, vars_dont_care, vars_control, vars_others, euclidean_label_vars)

                    # - Out
                    if len(dfdist_within) > 0 and len(dfdist_across) > 0:

                        vectors_within = np.stack(dfdist_within["vector"])
                        vectors_across = np.stack(dfdist_across["vector"])

                        # ============= COMPUTE
                        weighted_mean_sim, similarities, dot_products, weights, _, _ = compute_weighted_alignment(vectors_within, vectors_across, PLOT=False)
                        if similarities is not None:
                            RES.append({
                                "animal": animal,
                                "date": date,
                                "bregion": bregion,
                                "_iter": _iter,
                                "weighted_mean_sim": weighted_mean_sim,
                                "effect_kind": "across_variables",
                                "similarities":similarities,
                                "weights":weights,
                                "var_effect": "ignore",
                                "vars_dont_care": "ignore",
                                "vars_control": "ignore",
                                "vars_others": "ignore",
                                "strict":strict,
                            })


                ####################################################### 
                ######################## V2 -- USING DOT PRODUCTS DIRECTLY
                ### Within rank
                var_effect = "chunk_within_rank"
                vars_dont_care = ["chunk_within_rank_semantic"] # Generally, correlated with var_effect
                vars_control = [ 'task_kind', 'epoch', 'chunk_rank_global', 'chunk_rank', 'shape', 'gridloc', 
                                'CTXT_loc_prev', 'chunk_n_in_chunk'] # These must be identical within each pair (ie each row)
                # vars_others = ["chunk_rank_global", "chunk_n_in_chunk", "gridloc"] # Computing vector alignment is done separately for each level of this
                # vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
                if strict:
                    vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
                else:
                    vars_others = ["chunk_rank_global"] # Computing vector alignment is done separately for each level of this
                
                dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all = compute_dot_product_distributions_helper(
                        dfdist, var_effect, vars_dont_care, vars_control, vars_others, euclidean_label_vars)
                if dot_products_all is not None:
                    for dot, dot_1, dot_2, n_each, effect_lev_pair, grp_others in zip(dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all):
                        for value_name, value in zip(["dot", "dot_1", "dot_2"], [dot, dot_1, dot_2]):
                            RES_DOT.append({
                                "animal": animal,
                                "date": date,
                                "bregion": bregion,
                                "_iter": _iter,
                                "effect_kind": "chunk_within_rank",
                                "var_effect": var_effect,
                                "vars_dont_care": tuple(vars_dont_care),
                                "vars_control": tuple(vars_control),
                                "vars_others": tuple(vars_others),
                                "strict":strict,
                                "value_name":value_name,
                                "value":value,
                                "n_each":n_each,
                                "effect_lev_pair":effect_lev_pair,
                                "grp_others":grp_others,
                            })

                if DEBUG_BREAK:
                    assert False

                ### Across chunk rank
                var_effect = "chunk_rank_global"
                vars_dont_care = ["chunk_rank", "shape"] # Generally, correlated with var_effect
                vars_control = [ 'task_kind', 'epoch', 'gridloc', 'CTXT_loc_prev', 'chunk_n_in_chunk', 
                    'chunk_within_rank', 'chunk_within_rank_semantic'] # These must be identical within each pair (ie each row)
                # vars_others = ["chunk_rank_global", "chunk_n_in_chunk", "gridloc"] # Computing vector alignment is done separately for each level of this
                # vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
                if strict:
                    vars_others = ["epoch", "chunk_within_rank_semantic"] # Computing vector alignment is done separately for each level of this
                else:
                    vars_others = ["epoch"] # Computing vector alignment is done separately for each level of this

                dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all = compute_dot_product_distributions_helper(
                        dfdist, var_effect, vars_dont_care, vars_control, vars_others, euclidean_label_vars)
                if dot_products_all is not None:
                    for dot, dot_1, dot_2, n_each, effect_lev_pair, grp_others in zip(dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all):
                        for value_name, value in zip(["dot", "dot_1", "dot_2"], [dot, dot_1, dot_2]):
                            RES_DOT.append({
                                "animal": animal,
                                "date": date,
                                "bregion": bregion,
                                "_iter": _iter,
                                "effect_kind": "chunk_rank_global",
                                "var_effect": var_effect,
                                "vars_dont_care": tuple(vars_dont_care),
                                "vars_control": tuple(vars_control),
                                "vars_others": tuple(vars_others),
                                "strict":strict,
                                "value_name":value_name,
                                "value":value,
                                "n_each":n_each,
                                "effect_lev_pair":effect_lev_pair,
                                "grp_others":grp_others,
                            })

                ### Effect of chunk_within_rank, comparing across chunk_rank
                var_effect = "chunk_within_rank"
                vars_dont_care = ["chunk_within_rank_semantic"] # Generally, correlated with var_effect
                vars_control = [ 'task_kind', 'epoch', 'chunk_rank_global', 'chunk_rank', 'shape', 'gridloc', 
                                'CTXT_loc_prev', 'chunk_n_in_chunk'] # These must be identical within each pair (ie each row)
                # vars_others = ["chunk_rank_global", "chunk_n_in_chunk", "gridloc"] # Computing vector alignment is done separately for each level of this
                # vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
                if strict:
                    vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
                else:
                    vars_others = ["chunk_rank_global"] # Computing vector alignment is done separately for each level of this

                dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all = compute_dot_product_distributions_helper_across_grpother(
                                                                            dfdist, 
                                                                            var_effect, vars_dont_care, vars_control, vars_others,
                                                                            euclidean_label_vars)
                if dot_products_all is not None:
                    for dot, dot_1, dot_2, n_each, effect_lev_pair, grp_others in zip(dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all):
                        for value_name, value in zip(["dot", "dot_1", "dot_2"], [dot, dot_1, dot_2]):
                            RES_DOT.append({
                                "animal": animal,
                                "date": date,
                                "bregion": bregion,
                                "_iter": _iter,
                                "effect_kind": "chunk_within_rank_ACROSS",
                                "var_effect": var_effect,
                                "vars_dont_care": tuple(vars_dont_care),
                                "vars_control": tuple(vars_control),
                                "vars_others": tuple(vars_others),
                                "strict":strict,
                                "value_name":value_name,
                                "value":value,
                                "n_each":n_each,
                                "effect_lev_pair":effect_lev_pair,
                                "grp_others":grp_others,
                            })

                # ***************8 Two options for getting "across"
                # OPTION 1: Strict. Only use grps with at least two level pairs, just as in the above for "within"
                # ie this throws out cases where (i) only two chunk_ranks exist; and (ii) for Pancho, two shape sets
                # computes stuff only within each epoch.
                var_effect_1 = "chunk_within_rank"
                vars_dont_care_1 = ["chunk_within_rank_semantic"] # Generally, correlated with var_effect
                vars_control_1 = [ 'task_kind', 'epoch', 'chunk_rank_global', 'chunk_rank', 'shape', 'gridloc', 
                                'CTXT_loc_prev', 'chunk_n_in_chunk'] # These must be identical within each pair (ie each row)
                if strict:
                    vars_others_1 = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
                else:
                    vars_others_1 = ["chunk_rank_global"] # Computing vector alignment is done separately for each level of this

                var_effect_2 = "chunk_rank_global"
                vars_dont_care_2 = ["chunk_rank", "shape"] # Generally, correlated with var_effect
                vars_control_2 = [ 'task_kind', 'epoch', 'gridloc', 'CTXT_loc_prev', 'chunk_n_in_chunk', 'chunk_within_rank', 'chunk_within_rank_semantic'] # These must be identical within each pair (ie each row)
                if strict:
                    vars_others_2 = ["epoch", "chunk_within_rank_semantic"] # Computing vector alignment is done separately for each level of this
                else:
                    vars_others_2 = ["epoch"] # Computing vector alignment is done separately for each level of this                    

                dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all = compute_dot_product_distributions_helper_two_vareffects(
                                                                            dfdist, 
                                                                            var_effect_1, vars_dont_care_1, vars_control_1, vars_others_1,
                                                                            var_effect_2, vars_dont_care_2, vars_control_2, vars_others_2,
                                                                            euclidean_label_vars)
                for dot, dot_1, dot_2, n_each, effect_lev_pair, grp_others in zip(dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all):
                    for value_name, value in zip(["dot", "dot_1", "dot_2"], [dot, dot_1, dot_2]):
                        RES_DOT.append({
                            "animal": animal,
                            "date": date,
                            "bregion": bregion,
                            "_iter": _iter,
                            "effect_kind": "across_variables_strict",
                            "var_effect": "ignore",
                            "vars_dont_care": "ignore",
                            "vars_control": "ignore",
                            "vars_others": "ignore",
                            "strict":strict,
                            "value_name":value_name,
                            "value":value,
                            "n_each":n_each,
                            "effect_lev_pair":effect_lev_pair,
                            "grp_others":grp_others,
                        })

                # ********* Option 2: Lenient, score something even for cases with two chunk ranks
                var_effect_1 = "chunk_within_rank"
                vars_dont_care_1 = ["chunk_within_rank_semantic", "chunk_rank"] # Generally, correlated with var_effect
                vars_control_1 = [ 'task_kind', 'chunk_rank_global', 'gridloc', 
                                'CTXT_loc_prev', 'chunk_n_in_chunk', "epoch", "shape"] # These must be identical within each pair (ie each row)
                vars_others_1 = ["chunk_rank_global"] # Computing vector alignment is done separately for each level of this

                var_effect_2 = "chunk_rank_global"
                vars_dont_care_2 = ["chunk_rank", "shape", "chunk_n_in_chunk", "chunk_within_rank"] # Generally, correlated with var_effect
                vars_control_2 = [ 'task_kind', 'gridloc', 'epoch', 'CTXT_loc_prev', 'chunk_within_rank_semantic'] # These must be identical within each pair (ie each row)
                vars_others_2 = ["task_kind"] # Computing vector alignment is done separately for each level of this                    

                dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all = compute_dot_product_distributions_helper_two_vareffects(dfdist, 
                                                                            var_effect_1, vars_dont_care_1, vars_control_1, vars_others_1,
                                                                            var_effect_2, vars_dont_care_2, vars_control_2, vars_others_2,
                                                                            euclidean_label_vars,
                                                                            only_keep_if_both_12_have_mult_level_pairs=False)
                for dot, dot_1, dot_2, n_each, effect_lev_pair, grp_others in zip(dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all):
                    for value_name, value in zip(["dot", "dot_1", "dot_2"], [dot, dot_1, dot_2]):
                        RES_DOT.append({
                            "animal": animal,
                            "date": date,
                            "bregion": bregion,
                            "_iter": _iter,
                            "effect_kind": "across_variables_lenient",
                            "var_effect": "ignore",
                            "vars_dont_care": "ignore",
                            "vars_control": "ignore",
                            "vars_others": "ignore",
                            "strict":strict,
                            "value_name":value_name,
                            "value":value,
                            "n_each":n_each,
                            "effect_lev_pair":effect_lev_pair,
                            "grp_others":grp_others,
                        })


    dfres_this = pd.DataFrame(RES)
    dfres_this.to_pickle(f"{SAVEDIR}/dfres-{animal}-{date}.pkl")

    dfres_this_dot = pd.DataFrame(RES_DOT)
    dfres_this_dot.to_pickle(f"{SAVEDIR}/dfres_dot-{animal}-{date}.pkl")

    return dfres_this, dfres_this_dot

def alignmentcompute_wrapper(HACK_SKIP_FAILED_DATE_FOR_NOW, 
        run, n_iter_splits, SKIP_IF_DONE = False, DEBUG=False, DEBUG_BREAK=False,
        DEBUG_FORCE_RETURN=False,
        MULTIPROCESS=False):
    """
    Wrapper for first major step in analysis. 
    Here is high-level, to go from dim projected neural data (PA) to dot products, and then save them.
    
    PA are taken from previously saved PA.

    PARAMS:
        # run = 30
        # n_iter_splits = 4 # What was used originally

    """
    from glob import glob
    # from neuralmonkey.classes.session import _REGIONS_IN_ORDER_COMBINED
    # from neuralmonkey.analyses.euclidian_distance import dfdist_extract_label_vars_specific
    # from pythonlib.tools.pandastools import replace_None_with_string, stringify_values
    # from pythonlib.tools.pandastools import aggregGeneral
    # from pythonlib.tools.pandastools import append_col_with_grp_index
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates
    # import pickle
    # from neuralmonkey.analyses.euclidian_distance import compute_vector_between_conditions
    # import numpy as np
    # from pythonlib.tools.vectools import compute_weighted_alignment

    SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good/alignments/using-run={run}-n_iter_splits={n_iter_splits}"
    os.makedirs(SAVEDIR, exist_ok=True)

    ### Where to get the PA
    save_suffix = "AnBmCk_general" 

    ### Params here
    # list_strict = [False]
    ANIMALS = ["Pancho", "Diego"]
    # ANIMALS = ["Pancho"]
    DATES = None
    # DATES = [231114, 231116, 220901, 220906, 220907, 220908, 220909]
    # _npcs_keep_euclidean = 6
    # min_n_trials = 4 # to use a contrast.
    
    if DEBUG:
        ANIMALS =  ["Diego"]
        DATES = [230726]
        # date = 230913

        # animal = "Pancho"
        # date = 220907

    # Get list of animals and dates
    list_animal = []
    list_date = []
    for animal in ANIMALS:
        if DATES is None:
            list_dates, _, _, _ = load_preprocess_get_dates(animal, save_suffix)
        else:
            list_dates = DATES
        for date in list_dates:
            if HACK_SKIP_FAILED_DATE_FOR_NOW:
                if (animal, date) == ("Diego", 250321):
                    # This failed beucase of need to reextract this dataset for run30.
                    print(f"Skipping {animal}-{date} because it failed")
                    continue

            if SKIP_IF_DONE:
                path_check = f"{SAVEDIR}/dfres_dot-{animal}-{date}.pkl"
                if os.path.exists(path_check):
                    print(f"Skipping {animal}-{date} because it already exists")
                    continue
            
            list_animal.append(animal)
            list_date.append(date)
            

    if MULTIPROCESS:
        from multiprocessing import Pool
        from itertools import repeat

        # def runner(_animal, _date):              
        #     alignmentcompute_wrapper_single_session(_animal, _date, 
        #         SAVEDIR, HACK_SKIP_FAILED_DATE_FOR_NOW, 
        #         run, n_iter_splits, SKIP_IF_DONE, DEBUG, DEBUG_BREAK, 
        #         DEBUG_FORCE_RETURN)

        MULTIPROCESS_N_CORES = 8
        with Pool(MULTIPROCESS_N_CORES) as pool:
            pool.starmap(alignmentcompute_wrapper_single_session,
                zip(
                    list_animal,
                    list_date,
                    repeat(SAVEDIR),
                    repeat(HACK_SKIP_FAILED_DATE_FOR_NOW),
                    repeat(run),
                    repeat(n_iter_splits),
                    repeat(SKIP_IF_DONE),
                    repeat(DEBUG),
                    repeat(DEBUG_BREAK),
                    repeat(DEBUG_FORCE_RETURN)))
    else:
        ### Iterate
        # skipped_paths = []
        # error_cases = []
        for animal, date in zip(list_animal, list_date):
        # for animal in ANIMALS:

        #     if DATES is None:
        #         list_dates, _, _, _ = load_preprocess_get_dates(animal, save_suffix)
        #     else:
        #         list_dates = DATES
            
        #     for date in list_dates:

        #         if HACK_SKIP_FAILED_DATE_FOR_NOW:
        #             if (animal, date) == ("Diego", 230915):
        #                 # This failed beucase of need to reextract this dataset for run30.
        #                 print(f"Skipping {animal}-{date} because it failed")
        #                 continue
        #             if (animal, date) == ("Pancho", 230824):
        #                 # This failed beucase of need to reextract this dataset for run30.
        #                 print(f"Skipping {animal}-{date} because it failed")
        #                 continue
        #             if (animal, date) == ("Pancho", 230829):
        #                 # This failed beucase of need to reextract this dataset for run30.
        #                 print(f"Skipping {animal}-{date} because it failed")
        #                 continue

        #         if SKIP_IF_DONE:
        #             path_check = f"{SAVEDIR}/dfres_dot-{animal}-{date}.pkl"
        #             if os.path.exists(path_check):
        #                 print(f"Skipping {animal}-{date} because it already exists")
        #                 continue
                        
            _, _ = alignmentcompute_wrapper_single_session(animal, date, SAVEDIR,
                run, n_iter_splits, DEBUG_BREAK, 
                DEBUG_FORCE_RETURN)

            # ### Do this animal
            # print(f"Processing {animal}-{date}")
            # try:
            #     RES = []
            #     RES_DOT = []
            #     for _iter in range(n_iter_splits):
            #         for bregion in _REGIONS_IN_ORDER_COMBINED:

            #             # Load dataset for devo
            #             path = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{run}/{animal}-{date}-q=RULE_ANBMCK_STROKE/bregion={bregion}/FITTING_subspc=('epoch', 'gridloc', 'DIFF_gridloc', 'chunk_rank', 'shape', 'rank_conj')-iter={_iter}/pa_subspace.pkl"

            #             with open(path, "rb") as f:
            #                 pa_subspace = pickle.load(f)

            #             pa_subspace = pa_subspace.slice_by_dim_indices_wrapper("chans", list(range(_npcs_keep_euclidean)))
            #             pa_subspace = pa_subspace.slice_by_labels_filtdict({"task_kind":["prims_on_grid"]})

            #             # TODO: Remove effect of first stroke. (DONE)
            #             # TODO: Do targeted PCA to convert to top 2 PCs... For below code, which uses theta... (IGNORE, targeted PC already done)
            #             # TODO: Even better, don't convert to theta. Just take angle between vectors directly. (DONE)
            #             var_effect = "chunk_within_rank"
            #             vars_others = ["chunk_within_rank_semantic", "task_kind", "epoch", "chunk_rank_global", "chunk_rank", "shape", 
            #                         "gridloc", "CTXT_loc_prev", "chunk_n_in_chunk"]
            #             dfdist, euclidean_label_vars = alignmentcompute_prepare_dataset(pa_subspace, var_effect, vars_others, min_n_trials=min_n_trials)

            #             ### Clean up dfdist
            #             # from neuralmonkey.analyses.euclidian_distance import dfdist_variables_generate_var_same
            #             # colname_conj_same = dfdist_variables_generate_var_same(euclidean_label_vars)

            #             # Clean up
            #             dfdist["n1"] = [x[0] for x in dfdist["n_1_2"]]
            #             dfdist["n2"] = [x[1] for x in dfdist["n_1_2"]]
            #             dfdist = dfdist[(dfdist["n1"] >= min_n_trials) & (dfdist["n2"] >= min_n_trials)].reset_index(drop=True)

            #             if False:
            #                 # Only
            #                 n1 = sum(dfdist["chunk_rank_global_12"] == dfdist["chunk_rank_12"])
            #                 n2 = len(dfdist["chunk_rank_global_12"] == dfdist["chunk_rank_12"])
            #                 assert n1>0.7*n2, "why so many misaligned? might be fine to continue"
            #                 dfdist = dfdist[dfdist["chunk_rank_global_12"] == dfdist["chunk_rank_12"]].reset_index(drop=True)
                        
            #             ### Get vectors between all conditions
            #             # Get X for (0,1), (1,2), ... within each (chunk_rank).
            #             # Get (A, B), (B, C) -- across chunk ranks.
            #             dfanglevecs = compute_vector_between_conditions(pa_subspace, dfdist, var_effect, vars_others)
            #             assert np.all(dfanglevecs["labels_1"] == dfdist["labels_1"])
            #             assert np.all(dfanglevecs["labels_2"] == dfdist["labels_2"])
            #             dfdist["vector"] = dfanglevecs["vector"]

            #             # Return dfdist for debugging.
            #             if DEBUG_FORCE_RETURN:
            #                 return dfdist, euclidean_label_vars

            #             ### Good version                        
            #             for strict in list_strict:
                            
            #                 if False:
            #                     assert False, "havent carefully chcekd this recently. Is prob fine, but prob wont work."
            #                     ####################################################### 
            #                     ######################## V1 -- USING COSINE DISTANCE
            #                     ### Within rank
            #                     var_effect = "chunk_within_rank"
            #                     vars_dont_care = ["chunk_within_rank_semantic"] # Generally, correlated with var_effect
            #                     vars_control = [ 'task_kind', 'epoch', 'chunk_rank_global', 'chunk_rank', 'shape', 'gridloc', 
            #                                     'CTXT_loc_prev', 'chunk_n_in_chunk'] # These must be identical within each pair (ie each row)
            #                     # vars_others = ["chunk_rank_global", "chunk_n_in_chunk", "gridloc"] # Computing vector alignment is done separately for each level of this
            #                     # vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
            #                     if strict:
            #                         vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
            #                     else:
            #                         vars_others = ["chunk_rank_global"] # Computing vector alignment is done separately for each level of this

            #                     weighted_mean_sim, similarities, weights = compute_alignment_helper(dfdist, var_effect, vars_dont_care, vars_control, vars_others)
            #                     if similarities is not None:
            #                         RES.append({
            #                             "animal": animal,
            #                             "date": date,
            #                             "bregion": bregion,
            #                             "_iter": _iter,
            #                             "weighted_mean_sim": weighted_mean_sim,
            #                             "effect_kind": "chunk_within_rank",
            #                             "similarities":similarities,
            #                             "weights":weights,
            #                             "var_effect": var_effect,
            #                             "vars_dont_care": tuple(vars_dont_care),
            #                             "vars_control": tuple(vars_control),
            #                             "vars_others": tuple(vars_others),
            #                             "strict":strict,
            #                         })

            #                     ### Across chunk rank
            #                     var_effect = "chunk_rank_global"
            #                     vars_dont_care = ["chunk_rank", "shape"] # Generally, correlated with var_effect
            #                     vars_control = [ 'task_kind', 'epoch', 'gridloc', 'CTXT_loc_prev', 'chunk_n_in_chunk', 
            #                         'chunk_within_rank', 'chunk_within_rank_semantic'] # These must be identical within each pair (ie each row)
            #                     # vars_others = ["chunk_rank_global", "chunk_n_in_chunk", "gridloc"] # Computing vector alignment is done separately for each level of this
            #                     # vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
            #                     if strict:
            #                         vars_others = ["epoch", "chunk_within_rank_semantic"] # Computing vector alignment is done separately for each level of this
            #                     else:
            #                         vars_others = ["epoch"] # Computing vector alignment is done separately for each level of this

            #                     weighted_mean_sim, similarities, weights = compute_alignment_helper(dfdist, var_effect, vars_dont_care, vars_control, vars_others)
            #                     if similarities is not None:
            #                         RES.append({
            #                             "animal": animal,
            #                             "date": date,
            #                             "bregion": bregion,
            #                             "_iter": _iter,
            #                             "weighted_mean_sim": weighted_mean_sim,
            #                             "effect_kind": "chunk_rank_global",
            #                             "similarities":similarities,
            #                             "weights":weights,
            #                             "var_effect": var_effect,
            #                             "vars_dont_care": tuple(vars_dont_care),
            #                             "vars_control": tuple(vars_control),
            #                             "vars_others": tuple(vars_others),
            #                             "strict":strict,
            #                         })

            #                     ### Across (within and shapes)
            #                     # ================ (1) Within
            #                     var_effect = "chunk_within_rank"
            #                     vars_dont_care = ["chunk_within_rank_semantic"] # Generally, correlated with var_effect
            #                     vars_control = [ 'task_kind', 'epoch', 'chunk_rank_global', 'chunk_rank', 'shape', 'gridloc', 
            #                                     'CTXT_loc_prev', 'chunk_n_in_chunk'] # These must be identical within each pair (ie each row)
            #                     if strict:
            #                         vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
            #                     else:
            #                         vars_others = ["chunk_rank_global"] # Computing vector alignment is done separately for each level of this
            #                     dfdist_within = compute_alignment_helper_prune_data(dfdist, var_effect, vars_dont_care, vars_control, vars_others, euclidean_label_vars)

            #                     # ================ (2) Across
            #                     var_effect = "chunk_rank_global"
            #                     vars_dont_care = ["chunk_rank", "shape"] # Generally, correlated with var_effect
            #                     vars_control = [ 'task_kind', 'epoch', 'gridloc', 'CTXT_loc_prev', 'chunk_n_in_chunk', 'chunk_within_rank', 'chunk_within_rank_semantic'] # These must be identical within each pair (ie each row)
            #                     if strict:
            #                         vars_others = ["epoch", "chunk_within_rank_semantic"] # Computing vector alignment is done separately for each level of this
            #                     else:
            #                         vars_others = ["epoch"] # Computing vector alignment is done separately for each level of this                    
            #                     dfdist_across = compute_alignment_helper_prune_data(dfdist, var_effect, vars_dont_care, vars_control, vars_others, euclidean_label_vars)

            #                     # - Out
            #                     if len(dfdist_within) > 0 and len(dfdist_across) > 0:

            #                         vectors_within = np.stack(dfdist_within["vector"])
            #                         vectors_across = np.stack(dfdist_across["vector"])

            #                         # ============= COMPUTE
            #                         weighted_mean_sim, similarities, dot_products, weights, _, _ = compute_weighted_alignment(vectors_within, vectors_across, PLOT=False)
            #                         if similarities is not None:
            #                             RES.append({
            #                                 "animal": animal,
            #                                 "date": date,
            #                                 "bregion": bregion,
            #                                 "_iter": _iter,
            #                                 "weighted_mean_sim": weighted_mean_sim,
            #                                 "effect_kind": "across_variables",
            #                                 "similarities":similarities,
            #                                 "weights":weights,
            #                                 "var_effect": "ignore",
            #                                 "vars_dont_care": "ignore",
            #                                 "vars_control": "ignore",
            #                                 "vars_others": "ignore",
            #                                 "strict":strict,
            #                             })


            #                 ####################################################### 
            #                 ######################## V2 -- USING DOT PRODUCTS DIRECTLY
            #                 ### Within rank
            #                 var_effect = "chunk_within_rank"
            #                 vars_dont_care = ["chunk_within_rank_semantic"] # Generally, correlated with var_effect
            #                 vars_control = [ 'task_kind', 'epoch', 'chunk_rank_global', 'chunk_rank', 'shape', 'gridloc', 
            #                                 'CTXT_loc_prev', 'chunk_n_in_chunk'] # These must be identical within each pair (ie each row)
            #                 # vars_others = ["chunk_rank_global", "chunk_n_in_chunk", "gridloc"] # Computing vector alignment is done separately for each level of this
            #                 # vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
            #                 if strict:
            #                     vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
            #                 else:
            #                     vars_others = ["chunk_rank_global"] # Computing vector alignment is done separately for each level of this
                            
            #                 dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all = compute_dot_product_distributions_helper(
            #                         dfdist, var_effect, vars_dont_care, vars_control, vars_others, euclidean_label_vars)
            #                 if dot_products_all is not None:
            #                     for dot, dot_1, dot_2, n_each, effect_lev_pair, grp_others in zip(dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all):
            #                         for value_name, value in zip(["dot", "dot_1", "dot_2"], [dot, dot_1, dot_2]):
            #                             RES_DOT.append({
            #                                 "animal": animal,
            #                                 "date": date,
            #                                 "bregion": bregion,
            #                                 "_iter": _iter,
            #                                 "effect_kind": "chunk_within_rank",
            #                                 "var_effect": var_effect,
            #                                 "vars_dont_care": tuple(vars_dont_care),
            #                                 "vars_control": tuple(vars_control),
            #                                 "vars_others": tuple(vars_others),
            #                                 "strict":strict,
            #                                 "value_name":value_name,
            #                                 "value":value,
            #                                 "n_each":n_each,
            #                                 "effect_lev_pair":effect_lev_pair,
            #                                 "grp_others":grp_others,
            #                             })

            #                 if DEBUG_BREAK:
            #                     assert False

            #                 ### Across chunk rank
            #                 var_effect = "chunk_rank_global"
            #                 vars_dont_care = ["chunk_rank", "shape"] # Generally, correlated with var_effect
            #                 vars_control = [ 'task_kind', 'epoch', 'gridloc', 'CTXT_loc_prev', 'chunk_n_in_chunk', 
            #                     'chunk_within_rank', 'chunk_within_rank_semantic'] # These must be identical within each pair (ie each row)
            #                 # vars_others = ["chunk_rank_global", "chunk_n_in_chunk", "gridloc"] # Computing vector alignment is done separately for each level of this
            #                 # vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
            #                 if strict:
            #                     vars_others = ["epoch", "chunk_within_rank_semantic"] # Computing vector alignment is done separately for each level of this
            #                 else:
            #                     vars_others = ["epoch"] # Computing vector alignment is done separately for each level of this

            #                 dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all = compute_dot_product_distributions_helper(
            #                         dfdist, var_effect, vars_dont_care, vars_control, vars_others, euclidean_label_vars)
            #                 if dot_products_all is not None:
            #                     for dot, dot_1, dot_2, n_each, effect_lev_pair, grp_others in zip(dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all):
            #                         for value_name, value in zip(["dot", "dot_1", "dot_2"], [dot, dot_1, dot_2]):
            #                             RES_DOT.append({
            #                                 "animal": animal,
            #                                 "date": date,
            #                                 "bregion": bregion,
            #                                 "_iter": _iter,
            #                                 "effect_kind": "chunk_rank_global",
            #                                 "var_effect": var_effect,
            #                                 "vars_dont_care": tuple(vars_dont_care),
            #                                 "vars_control": tuple(vars_control),
            #                                 "vars_others": tuple(vars_others),
            #                                 "strict":strict,
            #                                 "value_name":value_name,
            #                                 "value":value,
            #                                 "n_each":n_each,
            #                                 "effect_lev_pair":effect_lev_pair,
            #                                 "grp_others":grp_others,
            #                             })

            #                 ### Effect of chunk_within_rank, comparing across chunk_rank
            #                 var_effect = "chunk_within_rank"
            #                 vars_dont_care = ["chunk_within_rank_semantic"] # Generally, correlated with var_effect
            #                 vars_control = [ 'task_kind', 'epoch', 'chunk_rank_global', 'chunk_rank', 'shape', 'gridloc', 
            #                                 'CTXT_loc_prev', 'chunk_n_in_chunk'] # These must be identical within each pair (ie each row)
            #                 # vars_others = ["chunk_rank_global", "chunk_n_in_chunk", "gridloc"] # Computing vector alignment is done separately for each level of this
            #                 # vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
            #                 if strict:
            #                     vars_others = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
            #                 else:
            #                     vars_others = ["chunk_rank_global"] # Computing vector alignment is done separately for each level of this

            #                 dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all = compute_dot_product_distributions_helper_across_grpother(
            #                                                                             dfdist, 
            #                                                                             var_effect, vars_dont_care, vars_control, vars_others,
            #                                                                             euclidean_label_vars)
            #                 if dot_products_all is not None:
            #                     for dot, dot_1, dot_2, n_each, effect_lev_pair, grp_others in zip(dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all):
            #                         for value_name, value in zip(["dot", "dot_1", "dot_2"], [dot, dot_1, dot_2]):
            #                             RES_DOT.append({
            #                                 "animal": animal,
            #                                 "date": date,
            #                                 "bregion": bregion,
            #                                 "_iter": _iter,
            #                                 "effect_kind": "chunk_within_rank_ACROSS",
            #                                 "var_effect": var_effect,
            #                                 "vars_dont_care": tuple(vars_dont_care),
            #                                 "vars_control": tuple(vars_control),
            #                                 "vars_others": tuple(vars_others),
            #                                 "strict":strict,
            #                                 "value_name":value_name,
            #                                 "value":value,
            #                                 "n_each":n_each,
            #                                 "effect_lev_pair":effect_lev_pair,
            #                                 "grp_others":grp_others,
            #                             })

            #                 # ***************8 Two options for getting "across"
            #                 # OPTION 1: Strict. Only use grps with at least two level pairs, just as in the above for "within"
            #                 # ie this throws out cases where (i) only two chunk_ranks exist; and (ii) for Pancho, two shape sets
            #                 # computes stuff only within each epoch.
            #                 var_effect_1 = "chunk_within_rank"
            #                 vars_dont_care_1 = ["chunk_within_rank_semantic"] # Generally, correlated with var_effect
            #                 vars_control_1 = [ 'task_kind', 'epoch', 'chunk_rank_global', 'chunk_rank', 'shape', 'gridloc', 
            #                                 'CTXT_loc_prev', 'chunk_n_in_chunk'] # These must be identical within each pair (ie each row)
            #                 if strict:
            #                     vars_others_1 = ["chunk_rank_global", "chunk_n_in_chunk"] # Computing vector alignment is done separately for each level of this
            #                 else:
            #                     vars_others_1 = ["chunk_rank_global"] # Computing vector alignment is done separately for each level of this

            #                 # ================ (2) Across
            #                 var_effect_2 = "chunk_rank_global"
            #                 vars_dont_care_2 = ["chunk_rank", "shape"] # Generally, correlated with var_effect
            #                 vars_control_2 = [ 'task_kind', 'epoch', 'gridloc', 'CTXT_loc_prev', 'chunk_n_in_chunk', 'chunk_within_rank', 'chunk_within_rank_semantic'] # These must be identical within each pair (ie each row)
            #                 if strict:
            #                     vars_others_2 = ["epoch", "chunk_within_rank_semantic"] # Computing vector alignment is done separately for each level of this
            #                 else:
            #                     vars_others_2 = ["epoch"] # Computing vector alignment is done separately for each level of this                    

            #                 dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all = compute_dot_product_distributions_helper_two_vareffects(
            #                                                                             dfdist, 
            #                                                                             var_effect_1, vars_dont_care_1, vars_control_1, vars_others_1,
            #                                                                             var_effect_2, vars_dont_care_2, vars_control_2, vars_others_2,
            #                                                                             euclidean_label_vars)
            #                 for dot, dot_1, dot_2, n_each, effect_lev_pair, grp_others in zip(dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all):
            #                     for value_name, value in zip(["dot", "dot_1", "dot_2"], [dot, dot_1, dot_2]):
            #                         RES_DOT.append({
            #                             "animal": animal,
            #                             "date": date,
            #                             "bregion": bregion,
            #                             "_iter": _iter,
            #                             "effect_kind": "across_variables_strict",
            #                             "var_effect": "ignore",
            #                             "vars_dont_care": "ignore",
            #                             "vars_control": "ignore",
            #                             "vars_others": "ignore",
            #                             "strict":strict,
            #                             "value_name":value_name,
            #                             "value":value,
            #                             "n_each":n_each,
            #                             "effect_lev_pair":effect_lev_pair,
            #                             "grp_others":grp_others,
            #                         })

            #                 # ********* Option 2: Lenient, score something even for cases with two chunk ranks
            #                 var_effect_1 = "chunk_within_rank"
            #                 vars_dont_care_1 = ["chunk_within_rank_semantic", "chunk_rank"] # Generally, correlated with var_effect
            #                 vars_control_1 = [ 'task_kind', 'chunk_rank_global', 'gridloc', 
            #                                 'CTXT_loc_prev', 'chunk_n_in_chunk', "epoch", "shape"] # These must be identical within each pair (ie each row)
            #                 vars_others_1 = ["chunk_rank_global"] # Computing vector alignment is done separately for each level of this

            #                 # ================ (2) Across
            #                 var_effect_2 = "chunk_rank_global"
            #                 vars_dont_care_2 = ["chunk_rank", "shape", "chunk_n_in_chunk", "chunk_within_rank"] # Generally, correlated with var_effect
            #                 vars_control_2 = [ 'task_kind', 'gridloc', 'epoch', 'CTXT_loc_prev', 'chunk_within_rank_semantic'] # These must be identical within each pair (ie each row)
            #                 vars_others_2 = ["task_kind"] # Computing vector alignment is done separately for each level of this                    

            #                 dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all = compute_dot_product_distributions_helper_two_vareffects(dfdist, 
            #                                                                             var_effect_1, vars_dont_care_1, vars_control_1, vars_others_1,
            #                                                                             var_effect_2, vars_dont_care_2, vars_control_2, vars_others_2,
            #                                                                             euclidean_label_vars,
            #                                                                             only_keep_if_both_12_have_mult_level_pairs=False)
            #                 for dot, dot_1, dot_2, n_each, effect_lev_pair, grp_others in zip(dot_products_all, dot_products_1_all, dot_products_2_all, n_each_all, effect_lev_pairs_all, grp_others_all):
            #                     for value_name, value in zip(["dot", "dot_1", "dot_2"], [dot, dot_1, dot_2]):
            #                         RES_DOT.append({
            #                             "animal": animal,
            #                             "date": date,
            #                             "bregion": bregion,
            #                             "_iter": _iter,
            #                             "effect_kind": "across_variables_lenient",
            #                             "var_effect": "ignore",
            #                             "vars_dont_care": "ignore",
            #                             "vars_control": "ignore",
            #                             "vars_others": "ignore",
            #                             "strict":strict,
            #                             "value_name":value_name,
            #                             "value":value,
            #                             "n_each":n_each,
            #                             "effect_lev_pair":effect_lev_pair,
            #                             "grp_others":grp_others,
            #                         })


            #     dfres_this = pd.DataFrame(RES)
            #     dfres_this.to_pickle(f"{SAVEDIR}/dfres-{animal}-{date}.pkl")

            #     dfres_this_dot = pd.DataFrame(RES_DOT)
            #     dfres_this_dot.to_pickle(f"{SAVEDIR}/dfres_dot-{animal}-{date}.pkl")

            # except Exception as err:
            #     error_cases.append((animal, date, err))
            #     raise err
            #     if DEBUG:
            #         assert False
            #     continue

    # assert len(skipped_paths)==0, "figure out why skipped"



# def compute_alignment_helper(dfdist, var_effect, vars_dont_care, vars_control, vars_others, euclidean_label_vars):
#     """
#     """

#     ### Good version
#     from pythonlib.tools.vectools import compute_weighted_alignment
#     # Strict version -- only compare vectors with same controlled for everything

#     dfdist_this = compute_alignment_helper_prune_data(dfdist, var_effect, vars_dont_care, vars_control, vars_others, euclidean_label_vars)

#     ######## OLD VERSION, THROWS OUT TOO MUCH DATA
#     #     # For each of (0,1), (1,2), (2,3), ... get a single vector
#     #     def f(x):
#     #         return np.mean(np.stack(x), axis=0)

#     #     _var_effect = f"{var_effect}_12"
#     #     # levs_exist = sorted(dfdist_this_others[_var_effect].unique())

#     #     # for lev in levs_exist:
            
#     #     # vectors_all is (npairs, ndim), where each row of npairs is one of (0,1), (1,2), ..., where each of those have taken mean over all cases
#     #     vectors_all = np.stack((dfdist_this_others.groupby([_var_effect])["vector"].apply(f))) 
    
#     # SECOND, get vector alignment for var_effect, WITHIN each level of vars_others
#     _var_effect = f"{var_effect}_12"
#     similarities_all = []
#     weights_all = []

#     from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
#     _vars_others = [f"{v}_1" for v in vars_others] # can use 1 or 2, they are equal
#     grpdict_others = grouping_append_and_return_inner_items_good(dfdist_this, _vars_others)
#     n_levs = []
#     for _, inds_others in grpdict_others.items():
#         dfdist_this_others = dfdist_this.iloc[inds_others]

#         levs_exist = sorted(dfdist_this_others[_var_effect].unique())
#         n_levs.append(len(levs_exist))
#         for _i, lev1 in enumerate(levs_exist):
#             for _j, lev2 in enumerate(levs_exist):
#                 if _j>_i:

#                     vectors_1 = np.stack(dfdist_this_others[dfdist_this_others[_var_effect] == lev1]["vector"])
#                     vectors_2 = np.stack(dfdist_this_others[dfdist_this_others[_var_effect] == lev2]["vector"])

#                     _, similarities, _, weights, _, _ = compute_weighted_alignment(vectors_1, vectors_2, 
#                                                                                 do_reweight=True, PLOT=False)
#                     similarities_all.append(similarities.flatten())
#                     weights_all.append(weights.flatten())

#     if len(similarities_all) == 0:
#         assert all([_x==1 for _x in n_levs]), "figure out why didnt collect any results"
#         return None, None, None
#     else:
#         similarities = np.concatenate(similarities_all, axis=0) # flat 
#         weights = np.concatenate(weights_all, axis=0) # flat

#         weighted_mean_sim = np.sum(similarities * weights) / np.sum(weights)

#         return weighted_mean_sim, similarities, weights


def alignment_load_extracted_dot_products(SAVEDIR, across_version, prune_trial_version, levels_var_required,
        HACK_SKIP_FAILED_DATE_FOR_NOW=False, ANIMALS=None, DATES=None):
    """
    Load pre-computed dot products using alignmentcompute_wrapper(), and  then preprocess in various ways.
    """
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
    # run = 30

    SAVEDIR_PLOTS = f"{SAVEDIR}/MULT"
    os.makedirs(SAVEDIR_PLOTS, exist_ok=True)

    SAVEDIR_THIS = f"{SAVEDIR_PLOTS}/across_version={across_version}-prune_trial_version={prune_trial_version}-levels_var_required={levels_var_required}"
    os.makedirs(SAVEDIR_THIS, exist_ok=True)
    print("Created dir to save: ", SAVEDIR_THIS)

    ## To use input to debug, taking subset of (animal, date)
    if ANIMALS is None:
        ANIMALS = ["Diego", "Pancho"]

    ###############################
    ### Load saved stuff
    save_suffix = "AnBmCk_general" 
    list_dfres = []
    list_dfres_dot = []
    for animal in ANIMALS:

        if DATES is None:
            list_dates, _, _, _ = load_preprocess_get_dates(animal, save_suffix)
        else:
            list_dates = DATES
        
        for date in list_dates:
            
            if HACK_SKIP_FAILED_DATE_FOR_NOW:
                if (animal, date) == ("Diego", 250321):
                    # This failed beucase of need to reextract this dataset for run30.
                    print(f"Skipping {animal}-{date} because it failed")
                    continue
                # if (animal, date) == ("Pancho", 250322):
                #     # This failed beucase of need to reextract this dataset for run30.
                #     print(f"Skipping {animal}-{date} because it failed")
                #     continue

            path = f"{SAVEDIR}/dfres-{animal}-{date}.pkl"
            path_dot = f"{SAVEDIR}/dfres_dot-{animal}-{date}.pkl"

            assert os.path.exists(path), f"Why missing: {path}"
            
            dfres_this = pd.read_pickle(path)
            dfres_dot_this = pd.read_pickle(path_dot)
            list_dfres.append(dfres_this)
            list_dfres_dot.append(dfres_dot_this)
    DFRES = pd.concat(list_dfres).reset_index(drop=True)
    DFRES_DOT = pd.concat(list_dfres_dot).reset_index(drop=True)

    ### Sanity check
    assert len(DFRES_DOT["strict"].unique())==1, "below may assume this by ignoring strict"

    ##########
    # Choose either across_variables_lenient or across_variables_strict as the new across_variables,
    # depending on what you want.
    if across_version == "strict":
        lev_remove = "across_variables_lenient"
        lev_keep = "across_variables_strict"
    elif across_version == "lenient":
        lev_remove = "across_variables_strict"
        lev_keep = "across_variables_lenient"
    else:
        assert False
    DFRES_DOT = DFRES_DOT[DFRES_DOT["effect_kind"] != lev_remove].reset_index(drop=True)
    def f(x):
        if x==lev_keep:
            return "across_variables"
        else:
            return x
    DFRES_DOT["effect_kind"] = DFRES_DOT["effect_kind"].apply(f)
    DFRES_DOT["effect_kind"].value_counts()

    ### SAnity check that I can ignore the following variables:
    vars_can_ignore = ["var_effect", "vars_dont_care", "vars_control", "vars_others", "strict"]
    def f(x):
        for v in vars_can_ignore:
            assert len(x[v].unique())==1
    DFRES_DOT.groupby(["bregion", "animal", "date", "effect_kind"]).apply(f)

    ##########
    # [OBSOLETE] Stuff related to DFRES. This is the noisy "alignmment" score that is not used anymore.
    if False: 
        def f(x):
            return np.mean(x)
        DFRES["mean_sim"] = DFRES["similarities"].apply(f)

        # Agg over iter
        from pythonlib.tools.pandastools import aggregGeneral
        DFRES = aggregGeneral(DFRES, ["animal", "date", "bregion", "effect_kind", "var_effect", 
                            "vars_dont_care", "vars_control", "vars_others", "strict"], 
                            ["mean_sim", "weighted_mean_sim"])

        ### Prune cases (to just those with enough effect)
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
        DFRES_CLEAN, _ = extract_with_levels_of_conjunction_vars_helper(DFRES, var="effect_kind", vars_others=["animal", "date", "strict"], 
                                                    levels_var=["chunk_within_rank", "across_variables", "chunk_rank_global"])



        fig = sns.catplot(data=DFRES, x="bregion", y="weighted_mean_sim", hue="effect_kind", 
                        col="ani_date_strict", col_wrap=8, alpha=0.5, jitter=True)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)

        for y in ["mean_sim", "weighted_mean_sim"]:
            # Plot summary
            fig = sns.catplot(data=DFRES_CLEAN, x="bregion", y=y, hue="effect_kind", col="animal", row="strict", 
                            jitter=True, alpha=0.5, errorbar="se")
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.5)

            # Plot summary
            fig = sns.catplot(data=DFRES_CLEAN, x="bregion", y=y, hue="effect_kind", col="animal", row="strict", 
                            kind="boxen", errorbar="se")
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.5)    

            # Plot summary
            fig = sns.catplot(data=DFRES_CLEAN, x="bregion", y=y, hue="effect_kind", col="animal", row="strict", 
                            kind="bar", errorbar="se")
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.5)        
        # Scatterplot
        from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
        plot_45scatter_means_flexible_grouping(DFRES, )


    ### Keep only cases that have enough datapoints to compute control dot products (dot1 and dot2) (ie at least 2 trials for each pair)
    print("Starting length: ", len(DFRES_DOT))
    if prune_trial_version == "lowlev_trialcond":
    # if prune_if_too_few_trials:
        # Then remove an entire case (a case = {dot, dot1, dot2}) if any of those three values are na
        def f(x):
            return any([xx==0 for xx in x])
        DFRES_DOT["too_few_trials"] = DFRES_DOT["n_each"].apply(f)
        DFRES_DOT["too_few_trials"].value_counts()

        DFRES_DOT = DFRES_DOT[DFRES_DOT["too_few_trials"]==False].reset_index(drop=True)
    elif prune_trial_version == "highlev_expt":
        # Then remove at level of (animal, date, bregion), if it is missing any of the desired conditions.
        
        # First, do this, remove rows that have na.
        DFRES_DOT = DFRES_DOT[~DFRES_DOT["value"].isna()].reset_index(drop=True)

        # These two steps only keeps experiments that have at least one datapt for each of the 9 cases (3 effect_kind x 3 value_name)
        # (NOTE: couldbe different from 9 if you have <levels_var_required> that is not those 3 effect kind)
        # - First, keep only levels of effect_kind (e.g., "across_variables") that has all three value_name
        # print(len(DFRES_DOT))
        DFRES_DOT, _ = extract_with_levels_of_conjunction_vars_helper(DFRES_DOT, "value_name", 
            ["strict", "date", "animal", "bregion", "effect_kind"], 
            levels_var=["dot", "dot_1", "dot_2"])
        # print(len(DFRES_DOT))
        # - Second, keep only expts which have all required effect_kind
        DFRES_DOT, _ = extract_with_levels_of_conjunction_vars_helper(DFRES_DOT, "effect_kind", 
            ["strict", "date", "animal", "bregion"], 
            levels_var=levels_var_required)
        # print(len(DFRES_DOT))

    elif prune_trial_version is None:
        # Then remove only the specific value (e.g., dot, dot1, or dot2). This leads to imbalance.
        DFRES_DOT = DFRES_DOT[~DFRES_DOT["value"].isna()].reset_index(drop=True)
    else:
        print(prune_trial_version)
        assert False
    print("Ending length: ", len(DFRES_DOT))

    if False:
        # NOTE: Not doing this anymore since it is too strict.
        
        # Sanity checks
        # (1) All (var, varsothers) extracted for cross-vectors match exactly those for within-vectors
        only_include_chunkrank_that_contribute_chunkwithinrank_pair = False

        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
        grpdict = grouping_append_and_return_inner_items_good(DFRES_DOT, ["animal", "date", "bregion", "_iter", "strict"])


        # effect_kind\
        for grp, inds in grpdict.items():
            dfres_dot_this = DFRES_DOT.iloc[inds]
            dfres_dot_this_1 = dfres_dot_this[dfres_dot_this["effect_kind"] == "chunk_within_rank"]
            dfres_dot_this_2 = dfres_dot_this[dfres_dot_this["effect_kind"] == "chunk_rank_global"]
            dfres_dot_this_3 = dfres_dot_this[dfres_dot_this["effect_kind"] == "across_variables"]
            
        # Make sure that the only cases that have nan values are those that have 0 coutns
        for _, row in DFRES_DOT.iterrows():
            if np.isnan(row["value"]):
                if row["value_name"] == "dot":
                    assert row["n_each"][0] == 0
                elif row["value_name"] == "dot_1":
                    assert row["n_each"][1] == 0
                elif row["value_name"] == "dot_2":
                    assert row["n_each"][2] == 0
            else:
                if row["value_name"] == "dot":
                    assert row["n_each"][0] > 0
                elif row["value_name"] == "dot_1":
                    assert row["n_each"][1] > 0
                elif row["value_name"] == "dot_2":
                    assert row["n_each"][2] > 0

        chunk_rank_list = []
        chunk_within_rank_list = []
        for _, row in dfres_dot_this_1.iterrows():

            # Include all chunk_rank_global values
            chunk_rank = row["grp_others"][0]
            chunk_rank_list.append(chunk_rank)

            for pair_chunkwithinrank in row["effect_lev_pair"]:
                chunk_within_rank_list.append(pair_chunkwithinrank)

        chunk_rank_list = sorted(set(chunk_rank_list))
        chunk_within_rank_list = sorted(set(chunk_within_rank_list))

        chunk_rank_pairs = []
        for cr1 in chunk_rank_list[:-1]:
            for cr2 in chunk_rank_list[1:]:
                chunk_rank_pairs.append(f"{cr1}|{cr2}")

        tmp = []
        for crpair in chunk_rank_pairs:
            for cwrpair in chunk_within_rank_list:
                val = (cwrpair, crpair)
                print(val)

                tmp.append(val)
                # Make sure this exists
        print("The cases that WOULD contribute to computing alignment between within_chunk_rank axis vs. across_chunk_rank axis, if you only include chunk_within_rank pairs and chunk_rank pairs that contribute to computing chunk_within_rank axes:")
        print("(chunk_within_rank pair), (chunk_rank pair)")
        print(tmp)
        print()
        print("The cases that contribute to computing alignment between within_chunk_rank axis vs. across_chunk_rank axis:")
        print("(chunk_within_rank pair), (chunk_rank pair)")
        print(dfres_dot_this_3["effect_lev_pair"].unique().tolist())

        if only_include_chunkrank_that_contribute_chunkwithinrank_pair:
            assert tmp == dfres_dot_this_3["effect_lev_pair"].unique().tolist()   

    ### Aggregate over iter (always do this)
    from pythonlib.tools.pandastools import aggregGeneral
    DFRES_DOT = aggregGeneral(DFRES_DOT, ["animal", "date", "bregion", "effect_kind", "var_effect", 
                        "vars_dont_care", "vars_control", "vars_others", "strict", "value_name", 
                        "effect_lev_pair", "grp_others"], 
                        ["value"])
    print("Afteragg over iter: ", len(DFRES_DOT))

    ### Only keep (animal, bregion) that have all cases of desired effect kind, 
    # such as (within_rank, chunk_rank, across)
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
    n1 = len(DFRES_DOT)
    DFRES_DOT, _ = extract_with_levels_of_conjunction_vars_helper(DFRES_DOT, "effect_kind", 
        ["animal", "date", "bregion", "strict"], 1, "/tmp/counts.pdf", levels_var= levels_var_required, 
        plot_counts_also_before_prune_path="/tmp/counts_pre.pdf")
    n2 = len(DFRES_DOT)
    print(n1, n2)
    assert n2/n1 > 0.2  
    
    #### Agg so that each expt is a single point (ie agg over var_effect pairs)
    # print(len(DFRES_DOT))
    DFRES_DOT_AGG = aggregGeneral(DFRES_DOT, ["animal", "date", "bregion", "effect_kind", "var_effect", 
                        "vars_dont_care", "vars_control", "vars_others", "strict", "value_name"], 
                        ["value"])
    # print(len(DFRES_DOT_AGG))

    return DFRES_DOT, DFRES_DOT_AGG, SAVEDIR_PLOTS, SAVEDIR_THIS

def alignment_preprocess_hacky_good_cases_for_across_variables(DFRES_DOT):
    """
    Prune DFRES_DOT: This only affects cases with effect_kind==across_variables. For these
    cases, only keep if the chunk_rank used to compute ordinal axis (ie chunk_rank_within vectors) 
    is a chunk_rank that is used in the comptuation of shape axis. If not, then remove the row.

    The prupose is more matched/rigorous matching of these two sets of vectors.
    """

    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good

    def _toint(x):
        if x == "0.0":
            return 0
        elif x=="0":
            return 0
        elif x == "1.0":
            return 1
        elif x=="1":
            return 1
        elif x == "2.0":
            return 2
        elif x=="2":
            return 2
        elif x == "3.0":
            return 3
        elif x=="3":
            return 3
        else:
            print(x)
            assert False
            
    # Iterate over each effect_kind
    grpdict = grouping_append_and_return_inner_items_good(DFRES_DOT, ["effect_kind"])
    list_df = []
    for grp, inds in grpdict.items():
        
        dfres = DFRES_DOT.iloc[inds].reset_index(drop=True)

        if grp[0] == "across_variables":
            # Then add a row with the useful info.

            dfres["cr_for_ordinal"] = [x[0] for x in dfres["grp_others"]] # e.g, (1.0,)
            dfres["crpair_for_shape"] = [x[1] for x in dfres["effect_lev_pair"]] # eg '1.0|2.0' or '0|1'

            # from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
            # grouping_plot_n_samples_conjunction_heatmap(dfres, "cr_for_ordinal", "crpair_for_shape", ["animal", "date"]);
            list_cr1 = []
            list_cr2 = []
            list_good = []
            for _, row in dfres.iterrows():
                cr1 = int(row["cr_for_ordinal"][0]) # eg 1
                
                cr2 = row["crpair_for_shape"]
                ind = cr2.find("|")
                cr2 = (_toint(cr2[:ind]), _toint(cr2[ind+1:])) # converst '1.0|2.0' to (1,2)

                good = cr1 in cr2
                # print(cr1, cr2, good)

                list_cr1.append(cr1)
                list_cr2.append(cr2)
                list_good.append(good)

            dfres["cr_for_ordinal"] = list_cr1
            dfres["crpair_for_shape"] = list_cr2
            dfres["cr1_in_cr2"] = list_good

        else:
            # Just collect it, adding dummy variables.
            dfres["cr_for_ordinal"] = "ignore"
            dfres["crpair_for_shape"] = "ignore"
            dfres["cr1_in_cr2"] = "ignore"

        list_df.append(dfres)

    DFRES_DOT = pd.concat(list_df).reset_index(drop=True)

    ### PRUNE
    # Now, for those that are effect_kind==across_variables, only keep if cr1_in_cr2
    DFRES_DOT = DFRES_DOT[(DFRES_DOT["cr1_in_cr2"]==True) | (DFRES_DOT["cr1_in_cr2"]=="ignore")].reset_index(drop=True)

    return DFRES_DOT

def alignment_plot_overview_1(DFRES_DOT, DFRES_DOT_AGG, SAVEDIR_THIS):
    """
    Wrapper to plot various overview plots related to alingmnets, which are summarized
    in DFRES_DOT.
    """
    
    #### PRINT low-level data BEFORE agging
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper, grouping_print_n_samples, stringify_values, aggregGeneral, append_col_with_grp_index
    from pythonlib.tools.snstools import rotateLabel

    DFRES_DOT_STR = stringify_values(DFRES_DOT)

    savepath = f"{SAVEDIR_THIS}/final_counts_lowlevel_conditions-1.txt"
    grouping_print_n_samples(DFRES_DOT_STR, ["animal", "date", "bregion", "effect_kind", "value_name"], savepath=savepath)

    savepath = f"{SAVEDIR_THIS}/final_counts_lowlevel_conditions-2.txt"
    grouping_print_n_samples(DFRES_DOT_STR, ["animal", "date", "bregion", "effect_kind", "value_name", "grp_others", "effect_lev_pair"], savepath=savepath)

    if False:
        DFRES = append_col_with_grp_index(DFRES, ["animal", "date"], "ani_date")
        DFRES = append_col_with_grp_index(DFRES, ["animal", "date", "strict"], "ani_date_strict")

    # Add some labels.
    DFRES_DOT = append_col_with_grp_index(DFRES_DOT, ["animal", "date"], "ani_date")
    DFRES_DOT = append_col_with_grp_index(DFRES_DOT, ["animal", "date", "strict"], "ani_date_strict")

    ### Prune cases (to just those with enough effect)
    if False:
        DFRES_DOT_CLEAN, _ = extract_with_levels_of_conjunction_vars_helper(DFRES_DOT, var="effect_kind", vars_others=["animal", "date", "strict"], 
                                                    levels_var=["chunk_within_rank", "across_variables", "chunk_rank_global"])
    
    ##### PLOTS
    hue_order = ["dot", "dot_1", "dot_2"]
    x_order = ["chunk_within_rank", "chunk_rank_global", "across_variables", "chunk_within_rank_ACROSS"]

    # (1) Each individual
    for ani_date_strict in DFRES_DOT["ani_date_strict"].unique():
        dfres_dot_this = DFRES_DOT[DFRES_DOT["ani_date_strict"] == ani_date_strict]
        fig = sns.catplot(data=dfres_dot_this, x = "effect_kind", y="value", hue="value_name", col="bregion", 
            alpha=0.85, jitter=True, sharey=False, hue_order=hue_order, order=x_order)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
        rotateLabel(fig)

        savefig(fig, f"{SAVEDIR_THIS}/EACH-bar-{ani_date_strict}.pdf")
        plt.close("all")

    # (2) Summary
    for sharey in [False, True]:
        fig = sns.catplot(data=DFRES_DOT, x = "effect_kind", y="value", hue="value_name", col="bregion", 
            kind="bar", sharey=sharey, row="animal", hue_order=hue_order, order=x_order, errorbar="se")
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
        rotateLabel(fig)
        savefig(fig, f"{SAVEDIR_THIS}/ANIMALS-bar-sharey={sharey}-1.pdf")

        fig = sns.catplot(data=DFRES_DOT, x = "effect_kind", y="value", hue="value_name", col="bregion", 
            alpha=0.5, jitter=True, sharey=sharey, row="animal", hue_order=hue_order, order=x_order, errorbar="se")
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
        rotateLabel(fig)
        savefig(fig, f"{SAVEDIR_THIS}/ANIMALS-bar-sharey={sharey}-2.pdf")
        
        fig = sns.catplot(data=DFRES_DOT_AGG, x = "effect_kind", y="value", hue="value_name", col="bregion", 
            kind="bar", sharey=sharey, row="animal", hue_order=hue_order, order=x_order, errorbar="se")
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
        rotateLabel(fig)
        savefig(fig, f"{SAVEDIR_THIS}/ANIMALS-bar-sharey={sharey}-1-agg.pdf")

        fig = sns.catplot(data=DFRES_DOT_AGG, x = "effect_kind", y="value", hue="value_name", col="bregion", 
            alpha=0.5, jitter=True, sharey=sharey, row="animal", hue_order=hue_order, order=x_order, errorbar="se")
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
        rotateLabel(fig)
        savefig(fig, f"{SAVEDIR_THIS}/ANIMALS-bar-sharey={sharey}-2-agg.pdf")

        plt.close("all")

def alignment_score_cosine_sim_wrapper(DFRES_DOT, PLOT=True, 
        effect_1 = "chunk_within_rank", effect_2 = "across_variables"):
    """
    Low-level function for scoring cosine similarity using the mean dot products
    across all cases for a given (animal). 

    Will also compute <effect_2> - <effect_1>
    PARAMS:
    - 
    """
    from pythonlib.tools.vectools import cosine_similarity_from_dot_products
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.pandastools import pivot_table
    from pythonlib.tools.pandastools import summarize_featurediff

    # (1) Agg so that each expt is a single point
    DFRES_DOT_AGG = aggregGeneral(DFRES_DOT, ["animal", "date", "bregion", "effect_kind", "strict", "value_name"], ["value"])
    if False:
        print(len(DFRES_DOT))
        print(len(DFRES_DOT_AGG))

    # (2) For each (bregion", "effect_kind", "animal) get its effect kinds along columns. 
    assert all(DFRES_DOT_AGG["strict"] == False), "assuming so. oterwise need to add 'strict' to list of index below"
    DFRES_DOT_AGG_FINAL_PIVOT = pivot_table(DFRES_DOT_AGG, ["bregion", "effect_kind", "animal"], ["value_name"], ["value"], 
        flatten_col_names=True)
    DFRES_DOT_AGG_FINAL_PIVOT = DFRES_DOT_AGG_FINAL_PIVOT.dropna().reset_index(drop=True) # Clean up.

    # (3) Compute angles
    def f(x):
        _, cosine_sim = cosine_similarity_from_dot_products(x["value-dot"], x["value-dot_1"], x["value-dot_2"])
        return cosine_sim
    DFRES_DOT_AGG_FINAL_PIVOT["cosine_sim"] = DFRES_DOT_AGG_FINAL_PIVOT.apply(f, axis=1)

    # (4) Compute a single scalar value, the diference between effect kinds
    _, dfsummaryflat, _, _, _ = summarize_featurediff(
        DFRES_DOT_AGG_FINAL_PIVOT, "effect_kind", [effect_2, effect_1], 
        ["cosine_sim"], ["bregion", "animal"], return_dfpivot=False)

    if PLOT:
        fig1 = sns.catplot(data=DFRES_DOT_AGG_FINAL_PIVOT, x="bregion", y="cosine_sim", hue="effect_kind", 
            col="animal", kind="bar", errorbar="se")
        for ax in fig1.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
            ax.set_ylim([-1, 1])

        fig2 = sns.catplot(data=DFRES_DOT_AGG_FINAL_PIVOT, x="bregion", y="cosine_sim", hue="effect_kind", kind="bar", 
            errorbar="se")
        for ax in fig2.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
            ax.set_ylim([-1, 1])

        fig3 = sns.catplot(data=dfsummaryflat, x="bregion", y="value", hue="animal", kind="bar", errorbar="se")
        for ax in fig3.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
            ax.axhline(1, color="k", alpha=0.5)
            ax.set_ylim([-1, 1])

        # fig4 = sns.catplot(data=DFRES_DOT_AGG_FINAL_PIVOT, x="bregion", y="cosine_sim", hue="effect_kind", 
        #     col="animal", kind="bar", errorbar="se")
        # for ax in fig1.axes.flatten():
        #     ax.axhline(0, color="k", alpha=0.5)
        #     ax.set_ylim([-1, 1])

    if PLOT:
        return DFRES_DOT_AGG_FINAL_PIVOT, dfsummaryflat, fig1, fig2, fig3
    else:
        return DFRES_DOT_AGG_FINAL_PIVOT, dfsummaryflat


def alignment_score_cosine_sim_permutation_test(DFRES_DOT, N = 10, animal=None, effect_1 = "chunk_within_rank", 
        effect_2 = "across_variables", bregion = "preSMA", side="two"):
    """
    Permutation test to test signifiicantn of differnces in cosine alignment across effect_kind.
    Shuffling is done by this: for each (pair of pairs), shuffle what its <effect_kind> is.
    PARAMS:
    - animal, whether to prune to specific animal, or to take average over both animals (None)

    effect_1 = "chunk_within_rank"
    # effect_2 = "across_variables"
    effect_2 = "chunk_rank_global"
    bregion = "preSMA"

    """
    # Plot summary of permutation test
    from pythonlib.tools.statstools import permutationTest
    from pythonlib.tools.pandastools import shuffle_dataset_hierarchical, shuffle_dataset_hierarchical_remap, grouping_print_n_samples
    from pythonlib.tools.pandastools import append_col_with_grp_index

    # Some preprocessing
    DFRES_DOT = DFRES_DOT[DFRES_DOT["bregion"] == bregion].reset_index(drop=True)
    if isinstance(animal, str):
        # Prune to a single animal
        DFRES_DOT = DFRES_DOT[DFRES_DOT["animal"] ==  animal].reset_index(drop=True)
    else:
        # Get both animals.
        assert animal is None

    # Create new variable useful for shuffling.
    # the logic here is that each level of effect_kind needs to keep its (dot, dot1, dot2) together. So this
    # does thist (ie you shuffle each effect_kind case (var, var_other)
    _vars_datapt = ["bregion", "grp_others", "var_effect", "date", "strict", "effect_lev_pair", "effect_kind", "animal"]
    DFRES_DOT = append_col_with_grp_index(DFRES_DOT, _vars_datapt, "_var_datapt", use_strings=False)

    # This makes sure shuffleing is done only within each level of _var_grp
    _vars_grp = ["bregion", "strict", "date", "animal"]
    DFRES_DOT = append_col_with_grp_index(DFRES_DOT, _vars_grp, "_var_grp")

    if False:
        from pythonlib.tools.pandastools import stringify_values
        DFRES_DOT_STR = stringify_values(DFRES_DOT)
        grouping_print_n_samples(DFRES_DOT_STR, ["_var_datapt"])

    # Shuffle function
    assert _vars_datapt[-2] == "effect_kind", "the function below is wrong."
    def funshuff(df):
        df_shuff = shuffle_dataset_hierarchical_remap(df, "_var_datapt", "_var_grp")
        # Extract the new effect kind
        df_shuff["effect_kind"] = [x[-2] for x in df_shuff["_var_datapt_remapped"]]
        return df_shuff

    # def funstat(df):

    #     _, dfsummaryflat = alignment_score_cosine_sim_wrapper(df, PLOT=False, effect_1=effect_1, effect_2=effect_2)

    #     val_Diego = dfsummaryflat[(dfsummaryflat["bregion"] == bregion) & (dfsummaryflat["animal"] == "Diego")]["value"].values[0]
    #     val_Pancho = dfsummaryflat[(dfsummaryflat["bregion"] == bregion) & (dfsummaryflat["animal"] == "Pancho")]["value"].values[0]
    #     val = (val_Diego, val_Pancho)

    #     val_both = np.mean(val)

    #     return val_both, val

    def funstat(df):
        _, dfsummaryflat = alignment_score_cosine_sim_wrapper(df, PLOT=False, effect_1=effect_1, effect_2=effect_2)
        # assert len(dfsummaryflat[(dfsummaryflat["bregion"] == bregion)]["value"])==1

        val = dfsummaryflat[(dfsummaryflat["bregion"] == bregion)]["value"].mean()
        # display(dfsummaryflat)
        # print(val)
        # asd
        return val

    if False:
        # Plot cosine similarity (for actual data)
        _, dfsummaryflat = alignment_score_cosine_sim_wrapper(DFRES_DOT, PLOT=True, effect_1=effect_1, effect_2=effect_2)

    p, fig = permutationTest(DFRES_DOT, funstat, funshuff, N, side=side)        

    return p, fig

def alignment_dot_products_scatterplot(DFRES_DOT_AGG, savedir):
    """
    Plot scatter (using dot products), where each dot shows dot product also with dot1 and dot2 (within dataset)
    """
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping_color_mapper, plot_45scatter_means_flexible_grouping
    # plot_45scatter_means_flexible_grouping_color_mapper(DFRES_DOT_AGG, "date", "effect_kind")


    # Take mean of dot1 and dot2
    def _map(x):
        if x=="dot":
            return x
        elif x in ["dot_1", "dot_2"]:
            return "dot_self"
        else:
            assert False
    DFRES_DOT_AGG["value_name_clean"] = [_map(x) for x in DFRES_DOT_AGG["value_name"]]

    # Plot
    for effect_kind in DFRES_DOT_AGG["effect_kind"].unique():
        dfres = DFRES_DOT_AGG[(DFRES_DOT_AGG["effect_kind"] == effect_kind)]

        for shareaxes in [False, True]:

            # Figure out how to color points by animal
            map_datapt_lev_to_colorlev, colorlevs_that_exist = plot_45scatter_means_flexible_grouping_color_mapper(dfres, "date", "animal")

            # Plot
            _, fig = plot_45scatter_means_flexible_grouping(dfres, "value_name_clean", "dot_self", "dot", 
                "bregion", "value", "date", False, shareaxes=shareaxes, plot_error_bars=False, SIZE=3,
                map_datapt_lev_to_colorlev=map_datapt_lev_to_colorlev, 
                colorlevs_that_exist=colorlevs_that_exist, alpha=0.6)
            
            print(" ******* ", effect_kind)
            savefig(fig, f"{savedir}/scatter-effect_kind={effect_kind}-shareaxes={shareaxes}.pdf")

            plt.close("all")

if __name__=="__main__":
    
    pass
    # import sys        


    # PLOTS_DO = [1]
    # # PLOTS_DO = [5.1]

    # ###
    # for plot_do in PLOTS_DO:
    #     if plot_do==1:
    #         animal = sys.argv[1]
    #         date = sys.argv[2]
    #         fig1_generalize_wrapper(animal, date)
    #     else:
    #         assert False