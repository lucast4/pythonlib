"""Analhysis of beh effects of microstim
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pythonlib.tools.plottools import savefig
import seaborn as sns
import pandas as pd
import numpy as np

def plot_all_wrapper(D):
    """ 
    """

    # Grammar
    from pythonlib.dataset.dataset_analy.grammar import pipeline_generate_and_plot_all
    pipeline_generate_and_plot_all(D)

    # primitiveness
    from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_plot_pipeline
    preprocess_plot_pipeline(D)

    # single prims.
    from pythonlib.dataset.dataset_analy.singleprims import preprocess_dataset
    preprocess_dataset(D, PLOT=True)

    # motor timing
    plot_motortiming(D)

def plot_overview_behcode_timings(D, sdir, STIM_DUR = 0.5):
    """ Wuick plot , for each stim trial, of 
    of timings of behe vents, strokes, and stim onsets.
    Based on beh codes extracted.
    Only plots for trials with detercted stim, based on beh codes.
    PARAMS:
    - STIM_DUR, num in sec, duiratin of stim, for use in plotting
    """  
    
    from pythonlib.tools.plottools import plot_beh_codes

    D.ml2_extract_raw()
    PRINT=False

    for ind in range(len(D.Dat)):

        # get times of stim
        # codes_keep = [141, 142, 151, 152] # 152 ignore. is TTL off
        codes_keep = [141, 142, 151] 
        codes, times = D.ml2_utils_getTrialsBehCodes(ind, codes_keep, PRINT=PRINT)
        
        ms_fix = D.blockparams_extract_single_taskparams(ind)["microstim_fix"]
        ms_stroke = D.blockparams_extract_single_taskparams(ind)["microstim_stroke"]
        
        if len(codes)>0:
                    
            assert ms_fix["on"]==1
            assert ms_stroke["on"]==1
            
            fig, ax = plt.subplots(figsize=(12,4))

            # Plot times of stim
            plot_beh_codes(codes, times, ax=ax, color="r", yval=2)
            # Plot offsets
            times_offset = [t+STIM_DUR for t in times]
            plot_beh_codes(codes, times_offset, ax=ax, color="m", yval=2)

            # get times of strokes and trial events
            codes, times = D.ml2_utils_getTrialsBehCodes(ind, keep_only_codes_standard=True, PRINT=PRINT)
            plot_beh_codes(codes, times, ax=ax, color="k", yval=0)

            # get times of strokes
            strokes = D.Dat.iloc[ind]["strokes_beh"]

            ons = [s[0,2] for s in strokes]
            offs = [s[-1,2] for s in strokes]

            plot_beh_codes(["on" for _ in range(len(ons))], ons, ax=ax, color="b", yval=1)
            plot_beh_codes(["off" for _ in range(len(offs))], offs, ax=ax, color="k", yval=1)

            ax.set_ylim([-3, 5])
            ax.set_ylabel("0:trial events; 1:strokes: 2: stim")
            fig.savefig(f"{sdir}/trial_{ind}.png")
        else:
            assert ms_fix["on"]==0
            assert ms_stroke["on"]==0
            
        plt.close("all")    

def preprocess_assign_stim_code(D, map_ttl_region):
    """
    Give each trial a string code for its stim params, which can vary across 
    expts.
    PARAMS:
    - map_ttl_region, dict mapping from int ttl values to string , usually
    breian regions, which will be the code.
    """

    # if HACK:
    #   map_ttl_region = {
    #       3:"M1",
    #       4:"pSMA"
    #   }
    # else:
    #   assert False, "code this input"

    print(map_ttl_region)
    list_stim_code = []
    for ind in range(len(D.Dat)):
        ms_fix = D.blockparams_extract_single_taskparams(ind)["microstim_fix"]
        ms_str = D.blockparams_extract_single_taskparams(ind)["microstim_stroke"]

        stim_code = []

        ttl_codes = [] #collect across all stims.

        # FIxation
        stim_code.append(ms_fix["on"]==1)
        if ms_fix["on"]==1:
            ttls = [int(x[0]) for x in ms_fix["stimlist"] if int(x[0]) in map_ttl_region.keys()]
            assert len(set(ttls))==1
            stim_code.append(ttls[0])
            ttl_codes.append(ttls[0])

        # Strokes
        stim_code.append(ms_str["on"]==1)
        if ms_str["on"]==1:
            ttls = [int(x[2]) for x in ms_str["stimlist"]  if int(x[2]) in map_ttl_region.keys()]
            assert len(set(ttls))==1
            stim_code.append(ttls[0])
            ttl_codes.append(ttls[0])
            
        # print(ms_fix)
        # print(ms_str)
        # print(stim_code)

        # SHORTHANDS
        if all([x==False for x in stim_code]):
            # Stim off
            sc = "off"
        else:
            # Code it by its ttl
            # ttls = stim_code[1::2]
            assert len(set(ttl_codes))==1
            sc = map_ttl_region[ttl_codes[0]]

        list_stim_code.append(sc)

    # place stim code back into data
    D.Dat["microstim_epoch_code"] = list_stim_code

    print("New column: microstim_epoch_code")

# def primitiveness_preprocess_plot(D):
#     """
#     """
#     from pythonlib.dataset.dataset_analy.primitivenessv2 import plot_timecourse_results, plot_drawings_results, preprocess, extract_grouplevel_motor_stats, plot_grouplevel_results, plot_triallevel_results

#     grouping = ["shape", "gridloc", "epoch"]

#     ############### Extract data
#     DS, SAVEDIR = preprocess(D, True)
#     dfres, grouping = extract_grouplevel_motor_stats(DS, D, grouping)

#     ############### PLOTS

#     # Plot, comparing mean across levels of contrast variable.
#     # Each datapt is a single level of grouping.
#     savedir = f"{SAVEDIR}/grouplevel"
#     os.makedirs(savedir, exist_ok=True)
#     contrast = "epoch"
#     plot_grouplevel_results(dfres, DS, D, grouping, contrast, savedir)

#     # Plot, each datapt a single trial.
#     savedir = f"{SAVEDIR}/triallevel"
#     os.makedirs(savedir, exist_ok=True)
#     contrast = "epoch"
#     plot_triallevel_results(DS, contrast, savedir)

#     # Plot drawings
#     savedir = f"{SAVEDIR}/drawings"
#     os.makedirs(savedir, exist_ok=True)
#     plot_drawings_results(DS, savedir)

#     # Plot timecourses
#     savedir = f"{SAVEDIR}/timecourse"
#     os.makedirs(savedir, exist_ok=True)
#     plot_timecourse_results(DS, savedir, "epoch")




def plot_motortiming(D):
    """
    Plots of timing (gaps and strokes) effects of stimulation, controlling for
    stroke and seuqence context (and stroke index).
    RETURNS:
    - DS, DFTHIS
    And makes plots.
    """
    from pythonlib.dataset.dataset_analy.motortiming import gapstroke_timing_compare_by_variable
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import assign_epochsets_group_by_matching_levels_of_var

    # Prep dataset
    D.grouping_get_inner_items("block", "epoch")
    SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("microstim/motortiming")
    
    if False:
        for ind in range(len(D.Dat)):
            block = D.Dat.iloc[ind]["block"]
            rule = D.blockparams_extract_single_combined_task_and_block(ind)["GENERAL"]["block_transition_rule"]
            print(ind, block, rule)

    # Extract Strokes data
    VARS_CONTEXT = ["CTXT_shape_prev", "shape", "CTXT_loc_prev", "gridloc"]
    params_preprocess = ["remove_baseline", "no_supervision", "only_blocks_with_n_min_trials"]
    VAR = "epoch"
    DS, DFTHIS = gapstroke_timing_compare_by_variable(D, VAR, VARS_CONTEXT, params_preprocess);

    # Take controlled data --> plot a single distribution, one for each (context, index) combo
    DFTHIS = append_col_with_grp_index(DFTHIS, ["stroke_index", "context"], "strk_idx_ctxt", use_strings=False)

    # Remove strk_idx_ctxt that have too few data, or else makes plot harder to interpret.
    n_min_each_conj_outer_inner = 3
    DFTHIS, list_epochsets_unique= assign_epochsets_group_by_matching_levels_of_var(DFTHIS, 
                                                                                    var_outer_trials="strk_idx_ctxt", 
                                                                                    var_inner="epoch",
                                                                                    epochset_col_name="epochset", 
                                                                                    PRINT=True, n_min_each_conj_outer_inner=n_min_each_conj_outer_inner)
    # Plot as function of storke index
    n_min_each_conj_outer_inner = 3
    DFTHIS, list_epochsets_unique= assign_epochsets_group_by_matching_levels_of_var(DFTHIS, 
                                                                                    var_outer_trials="locshape_pre_this", 
                                                                                    var_inner="stroke_index",
                                                                                    epochset_col_name="stroke_index_set", 
                                                                                    PRINT=True,
                                                                                   n_min_each_conj_outer_inner=n_min_each_conj_outer_inner)

    print("SAVING FIGURES AT: ", SAVEDIR)
    for y in ["gap_from_prev_dur", "time_duration"]:
        fig = sns.catplot(data=DFTHIS, x="strk_idx_ctxt", y=y, hue="epoch", kind="point", aspect=2)
        rotateLabel(fig)    
        savefig(fig, f"{SAVEDIR}/{y}-vs-strk_idx_ctxt.pdf")

        fig = sns.catplot(data=DFTHIS, x="strk_idx_ctxt", y=y, hue="epoch", row="epochset", kind="point")
        rotateLabel(fig)    
        savefig(fig, f"{SAVEDIR}/{y}-vs-strk_idx_ctxt-grp_by_epochset.pdf")

        plt.close("all")
        
        fig = sns.catplot(data=DFTHIS, x="stroke_index", y=y, hue="epoch", row="locshape_pre_this", col="stroke_index_set", kind="point")
        rotateLabel(fig)    
        savefig(fig, f"{SAVEDIR}/{y}-vs-stroke_index-grp_by_stroke_index_set.pdf")

        plt.close("all")

    return DS, DFTHIS