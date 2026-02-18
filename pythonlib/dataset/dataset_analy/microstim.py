"""Analhysis of beh effects of microstim
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pythonlib.tools.plottools import savefig
import seaborn as sns
import pandas as pd
import numpy as np
from pythonlib.tools.exceptions import NotEnoughDataException


def _convert_mec_to_bregion(mec, date, map_date_bregion):
    """
    Helper to convert from mec ("microstim epoch code"?), which
    is the code for which TTL it is, to a brain region, given a hand-coded map.

    assumes ordering of regions is always (TTL3, TTL4)
    """
    regions = map_date_bregion[date]
        
    off = mec.find("off")>-1
    ttl3 = mec.find("TTL3")>-1    
    ttl4 = mec.find("TTL4")>-1
    
    assert sum([off, ttl3, ttl4])==1
    
    if off:
        return "off"
    elif ttl3:
        return regions[0]
    elif ttl4:
        return regions[1]
    else:
        print(off, ttl3, ttl4)
        assert False   

def postprocess_condition_df_multdates(DFALL):
    """
    Adds things to dataframe that are useful, after loading dataset (baseically this is
    D.Dat, but concatenated across days, potnetialy, for summary analyses).

    Also, if the day has multple bregions stimmed, then duplicates "off"
    trials, so have one set of trials per bregion. Useful for 45 degree scatter plots.
    """

    # Collect mappings from date to bregion
    MAP_DATE_BREGION_EACH_ANIMAL = {}
    for animal in ["Diego", "Pancho"]:
        _, _, map_date_bregion, _ = multday_dates_extract(animal)
        MAP_DATE_BREGION_EACH_ANIMAL[animal] = map_date_bregion    

    # Convert from TTL to actual brain region name.
    list_bregion = []
    for _, row in DFALL.iterrows():
        mec = row["microstim_epoch_code"]
        date = row["date"]
        animal = row["animal"]
        map_date_bregion = MAP_DATE_BREGION_EACH_ANIMAL[animal]
        
        bregion = _convert_mec_to_bregion(mec, date, map_date_bregion)
        list_bregion.append(bregion)
    DFALL["microstim_bregion"] = list_bregion

    # Was this stim or not
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    def F(x):
        if x["microstim_epoch_code"]=="off":
            return "off"
        else:
            return "on"
    DFALL = applyFunctionToAllRows(DFALL, F, "microstim_status")

    # Duplicate "off" trials for each day, one for each bregion that is stimmed that day.
    list_df = []
    list_date = sorted(DFALL["date"].unique().tolist())
    for date in list_date:
        dfthis = DFALL[DFALL["date"]==date]
        regions = [r for r in dfthis["microstim_bregion"].unique().tolist() if not r=="off"]
        for r in regions:
            # Get a copy of the stim "off" trials for this (day, region)
            dftmp = dfthis[dfthis["microstim_bregion"]=="off"].copy()
            assert dftmp["microstim_status"].unique().tolist()[0]=="off"
            dftmp["bregion_expt"] = r
            list_df.append(dftmp)
            
            # Same, but stim on:
            dftmp = dfthis[dfthis["microstim_bregion"]==r].copy()
            assert dftmp["microstim_status"].unique().tolist()[0]=="on"
            dftmp["bregion_expt"] = r
            list_df.append(dftmp)
    DFALL = pd.concat(list_df).reset_index(drop=True)

    return DFALL

def multday_dates_extract(ANIMAL):
    """
    Repo of all dates and their bregions stimmed.

    NOTE: 100% confirmed that the map from dates to bregions are correct (5/14/25)
    NOTE: 100% confirmed that added all dates to the lists, except syntax_TI dates (both animals)
    and dir-vs-dir (Diego).
    """
    if ANIMAL=="Pancho":
        
        dates_single = [231011, 231101, 231102, 231103, 231106, 231109, 231110]
        dates_multi = [231013, 231018, 231023, 231024, 231026, 231118, 231120, 
                 231121, 231122, 231128, 231129, 231201]
        # This includes all except:
        # (i) AnBm vs. RIGHT 
        # (ii) shapeseqsupstim 

        # for each date, note which brain regions.
        map_date_bregion = {
            231011: ("M1", "preSMA"),
            231013: ("M1", "preSMA"),
            231018: ("dlPFC", "vlPFC"),
            231023: ("FP", "SMA"),
            231024: ("PMv", "PMd"),
            231026: ("M1", "preSMA"),
            231101: ("M1", "preSMA"),
            231102: ("vlPFC", "preSMA"),
            231103: ("dlPFC", "PMd_a"),
            231106: ("preSMA", "M1"),
            231109: ("PMv", "SMA"),
            231110: ("PMd_p", "FP"),
            231118: ("dlPFC",),
            231120: ("vlPFC",),
            231121: ("preSMA",),
            231122: ("PMd_a",),
            231128: ("FP",),
            231129: ("FP",),
            231201: ("dlPFC",),
        } # Confirmed
        
        DATES_TO_SKIP = [
            231011, # AnBm too easy
            231026, # bad beahvior
            231118, # too strongly biased to DIR rule.
            231128, # didnt work much
        ] # (CONFIRMED these are fair)

    elif ANIMAL=="Diego":
        
        dates_single = [
            231101, 231102, 231103, 231106,
            231107, 231108, 231109, 231110, 231113, 231114, 231115,
            250421, 250422, 250423, 250424, 250425,
        ]
        dates_multi = [
            231025, 231026, 231027, 231029,
        ] 
        # This includes all except:
        # (i) DIR vs. DIR 
        # (ii) shapeseqsupstim and 
        # (iii) AnBmCk (n=6) v. RIGHT
        # [Confirmed]
        
        # for each date, note which brain regions.
        map_date_bregion = {
            231025: ("vlPFC", "preSMA"),
            231026: ("M1", "dlPFC"),
            231027: ("PMv", "FP"),
            231029: ("SMA", "PMd"),
            231101: ("M1", "preSMA"),
            231102: ("vlPFC", "preSMA"),
            231103: ("dlPFC", "PMd"),
            231106: ("preSMA",),
            231107: ("dlPFC", "PMd_a"), 
            231108: ("vlPFC",),
            231109: ("preSMA",),
            231110: ("dlPFC", "vlPFC"),
            231113: ("PMd_a", "PMv"),
            231114: ("SMA", "FP"),
            231115: ("M1-PMd_p", "preSMA"),
            # 250418: ("M1", "preSMA"),
            250421: ("M1", "preSMA"),
            250422: ("dlPFC", "vlPFC"),
            250423: ("PMv", "SMA"),
            250424: ("vlPFC", "FP"),
            250425: ("PMd_a", "dlPFC"),
        } # CONFIRMED list
        
        DATES_TO_SKIP = [231101, 231102, 231103, 231106] # all too easy (CONFIRMED these are fair)
    else:
        assert False
    for v in map_date_bregion.values():
        assert isinstance(v, tuple)

    # if grammar_version=="all":
    #     dates = dates_single + dates_multi
    # elif grammar_version=="single":
    #     dates = dates_single
    # elif grammar_version=="multi":
    #     dates = dates_multi
    # else:
    #     assert False

    assert not any([x in dates_multi for x in dates_single]), "you made mistake in inputting the dates"

    return dates_single, dates_multi, map_date_bregion, DATES_TO_SKIP


def mult_load_all_days(animal):
    """
    Helper to load all dates, already pre-saved (see notebook)
    """
    from glob import glob

    # animal = "Pancho"
    dates_single, dates_multi, _, DATES_TO_SKIP = multday_dates_extract(animal)
    list_dates = dates_single + dates_multi

    # SAVEDIR = /lemur2/lucas/analyses/main/grammar/Diego_gramstimdiego3_250421/df_trial.pkl
    SAVEDIR = "/lemur2/lucas/analyses/main/grammar"

    list_df =[]
    for date in list_dates:

        # Find the dataset path
        path = f"{SAVEDIR}/{animal}_*_{date}/df_trial.pkl"
        paths = glob(path)
        if len(paths)==0:
            print("SKIPPED: ", date)
            continue

        assert len(paths)==1
        pathfinal = paths[0]
        
        print(pathfinal)

        # Load it and store it
        df = pd.read_pickle(pathfinal)
        df["animal"] = animal
        df["date"] = date
        if date in dates_single:
            df["num_grammars"]=1
        elif date in dates_multi:
            df["num_grammars"]=2
        else:
            assert False

        list_df.append(df)

    DFALL = pd.concat(list_df).reset_index(drop=True)

    DFALL = postprocess_condition_df_multdates(DFALL)

    return DFALL, DATES_TO_SKIP

def mult_stats_plots_scatter(DFALL, DFALL_AGG, DATES_TO_SKIP, savedir):
    """
    Using data across dates,
    Do scatter plots of stim vs. no stim.
    """

    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from pythonlib.tools.plottools import  savefig
    import matplotlib.pyplot as plt

    # each trial is a single datapt - correct vs failure.
    var_manip = "microstim_status"
    x_lev_manip = "off"
    y_lev_manip = "on"
    var_subplot = "bregion_expt"
    var_value = "success_binary_quick"
    var_datapt = "date"
    shareaxes = True

    # Check that this is correct subset of data
    assert len(DFALL["epoch_orig"].unique())==1
    assert len(DFALL_AGG["epoch_orig"].unique())==1

    for do_agg in [False, True]:
        if do_agg:
            dfthis = DFALL_AGG
        else:
            dfthis = DFALL

        for skip in [False, True]:

            for n_grammars in [1, 2, None]:

                ##### Skip bad dates (based on not working much)
                if skip:
                    dates_to_skip = DATES_TO_SKIP
                else:
                    dates_to_skip = []

                if n_grammars is None:
                    dfall = dfthis[(~dfthis["date"].isin(dates_to_skip))].reset_index(drop=True)
                else:
                    dfall = dfthis[(~dfthis["date"].isin(dates_to_skip)) & (dfthis["num_grammars"]==n_grammars)].reset_index(drop=True)

                if len(dfall)>0:
                    # list_epoch_orig = dfall["epoch_orig"].unique().tolist()
                    # list_epochs_today = df_trials_all["epochs_today"].unique().tolist()

                    # SDIR_SAVE = f"/gorilla1/analyses/main/stepwise/MULT_DATES/{ANIMAL}/{WHICH_DATA}/TRIAL_LEVEL"
                    # import os
                    # os.makedirs(SDIR_SAVE, exist_ok=True)

                    _, fig = plot_45scatter_means_flexible_grouping(dfall, var_manip, x_lev_manip, y_lev_manip,
                                                            var_subplot, var_value, var_datapt, shareaxes=shareaxes)

                    savefig(fig, f"{savedir}/scatter-agg={do_agg}-ngrams={n_grammars}-skipbadday={skip}.pdf")
                    plt.close("all")    

def mult_stats_permutation_test(DFALL, nperms, savedir):
    """
    [GOOD] Final stats, testing effect of stim, across all dates.

    Permutation test, which shuffles whether trial is stim or no stim, within each (date, character) level, 
    and only for a single bregion. Does separately for each region.

    """
    ### scoring function
    from pythonlib.tools.pandastools import aggregGeneral, grouping_print_n_samples
    from pythonlib.tools.pandastools import shuffle_dataset_singlevar, shuffle_dataset_hierarchical
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    from pythonlib.tools.statstools import permutationTest
    from pythonlib.tools.plottools import savefig

    cols_keep = ["epoch_orig", "date", "microstim_status", "character", "success_binary_quick", "bregion_expt"]
    DFALL = DFALL.loc[:, cols_keep]

    if False:
        # dfthis_agg = aggregGeneral(dfthis, ["date", "epoch_orig", "bregion_expt", "microstim_status", "character"])
        # dfthis_agg = aggregGeneral(dfthis, ["date", "epoch_orig", "bregion_expt", "microstim_status", "character"])
        grouping_print_n_samples(dfthis, ["date", "epoch_orig", "bregion_expt", "microstim_status"]);

    def score_fun(dfthis):
        import numpy as np

        # First aggregate so that each (date, character) has one datapt per on and off
        if False:
            # Avoid this, since it reduces the sample size effectively.
            dfthis_agg = aggregGeneral(dfthis, ["date", "microstim_status", "character"], ["success_binary_quick"])
        else:
            dfthis_agg = dfthis
        score_stim = np.mean(dfthis_agg[dfthis_agg["microstim_status"]=="on"]["success_binary_quick"])
        score_off = np.mean(dfthis_agg[dfthis_agg["microstim_status"]=="off"]["success_binary_quick"])
        score_diff = score_stim - score_off
        return score_diff, score_diff

    ### shuffling function
    def shuff_fun(dfthis):
        # only shuffle within a (date, character)   
        list_var_shuff = ["microstim_status"] # shuffles these values
        # list_var_noshuff = ["date", "epoch_orig", "bregion_expt", "character"]
        list_var_noshuff = ["date", "character"] # controls for this (ie shuffle within each level of this)
        if False:
            print(grouping_print_n_samples(dfthis, ["date", "epoch_orig", "bregion_expt", "character", "microstim_status"]))
            assert False
        dfthis_shuff = shuffle_dataset_hierarchical(dfthis, list_var_shuff, list_var_noshuff)
        return dfthis_shuff


    # Sanity check that you have inputted a correctly subsampled dataframe
    assert len(DFALL["epoch_orig"].unique())==1

    ### Collect stats, running separately for each bregion
    grpdict = grouping_append_and_return_inner_items_good(DFALL, ["bregion_expt"])
    res = []
    for grp, inds in grpdict.items():

        print("Running perm for: ", grp)

        dfthis = DFALL.iloc[inds].reset_index(drop=True)

        p, stat_actual_collected, stats_shuff_collected, fig = permutationTest(dfthis, score_fun, shuff_fun, 
                                                                            nperms, True, side="left")

        savefig(fig, f"{savedir}/shuffle_distro-{grp}.pdf")

        res.append(
            {
                "bregion_expt":grp[0],
                "p":p,
                "stat_actual_collected":stat_actual_collected,
                "stats_shuff_collected":stats_shuff_collected,
            }
            )

        plt.close("all")

    ### Collect all stats (flattened)
    res_flat = []
    for dat in res:

        # Colect shuffles
        for val in dat["stats_shuff_collected"]:
            res_flat.append({
                "bregion_expt":dat["bregion_expt"],
                "val":val,
                "shuffled":True
            })

        # Collect the actual
        res_flat.append({
            "bregion_expt":dat["bregion_expt"],
            "val":dat["stat_actual_collected"],
            "shuffled":False
        })

    dfres = pd.DataFrame(res)
    dfres_flat = pd.DataFrame(res_flat)

    # SAVe data
    dfres.to_pickle(f"{savedir}/dfres.pkl")
    dfres_flat.to_pickle(f"{savedir}/dfres_flat.pkl")

    return dfres, dfres_flat

def mult_stats_permutation_test_plot(dfres, dfres_flat, savedir):
    """
    Given results of permutation test, 
    make summary plots of permutation test data. e.g, plot shuffled, and overlay the actual data.
    """
    import seaborn as sns
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.statstools import plotmod_pvalues
    
    bregions = sorted(dfres_flat["bregion_expt"].unique().tolist())

    def _mod_plot(fig):
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)

            # Overlay pvals
            x = dfres["bregion_expt"]
            p = dfres["p"]
            plotmod_pvalues(ax, x, p)

        rotateLabel(fig)

    ### (1) Single plot combining 
    fig = sns.catplot(data=dfres_flat, x="bregion_expt", y="val", hue="shuffled", jitter=True, alpha=0.2,
                    order=bregions)
    _mod_plot(fig)
    savefig(fig, f"{savedir}/catplot-all-1.pdf")
    

    ### (2) Plot separatley shuff vs. no shuff
    dfres_flat_this = dfres_flat[dfres_flat["shuffled"] == True]
    fig = sns.catplot(data=dfres_flat_this, x="bregion_expt", y="val", hue="shuffled", jitter=True, alpha=0.1,
                    order=bregions)

    for ax in fig.axes.flatten():
        dfres_flat_this = dfres_flat[dfres_flat["shuffled"] == False]
        sns.scatterplot(dfres_flat_this, x="bregion_expt", y="val", marker="o", color="r", alpha=0.5, ax=ax)

    _mod_plot(fig)
    savefig(fig, f"{savedir}/catplot-all-2.pdf")

    ### (2) Plot separatley shuff vs. no shuff
    dfres_flat_this = dfres_flat[dfres_flat["shuffled"] == True]
    fig = sns.catplot(data=dfres_flat_this, x="bregion_expt", y="val", hue="shuffled", kind="violin", order=bregions)

    for ax in fig.axes.flatten():
        dfres_flat_this = dfres_flat[dfres_flat["shuffled"] == False]
        sns.scatterplot(dfres_flat_this, x="bregion_expt", y="val", marker="o", color="r", alpha=0.9, ax=ax)

    _mod_plot(fig)
    savefig(fig, f"{savedir}/catplot-all-2.pdf")

    ### CLEAN PLOT of shuffle distribution
    # import numpy as np
    from pythonlib.tools.plottools import plot_patch_rectangle_filled
    from pythonlib.tools.plottools import rotate_x_labels

    def f_low(x):
        return np.percentile(x["val"], [2.5])[0]
    def f_high(x):
        return np.percentile(x["val"], [97.5])[0]

    dflow = dfres_flat.groupby(["bregion_expt"]).apply(f_low).reset_index()
    dfhigh = dfres_flat.groupby(["bregion_expt"]).apply(f_high).reset_index()
    # dftmp[""]

    # pd.merge(dfres_flat, dftmp, on="bregion_expt")
    dflowhigh = pd.merge(dflow, dfhigh, on="bregion_expt") 

    fig, ax = plt.subplots()
    YLIM = [-0.25, 0.25]
    # x1 = 0.4
    # x2 = 0.5
    # ylim = [0.2, 0.3]

    delt = 0.2
    list_bregion = []
    for i, row in dflowhigh.iterrows():
        br = row["bregion_expt"]
        x1 = i-delt
        x2 = i+delt
        ylim = [row["0_x"], row["0_y"]]

        color = "k"
        plot_patch_rectangle_filled(ax, x1, x2, color, alpha=0.3, YLIM=ylim, linewidth=0)

        list_bregion.append(br)

    ax.set_xlim([-1, len(list_bregion)])
    ax.set_ylim(YLIM)
    ax.axhline(0, color="k", alpha=0.5)

    assert dfres["bregion_expt"].tolist() == list_bregion

    ax.plot(dfres["bregion_expt"], dfres["stat_actual_collected"], "or")
    
    rotate_x_labels(ax)

    plotmod_pvalues(ax, dfres["bregion_expt"], dfres["p"])

    ax.set_ylabel("stim_on - stim_off")

    savefig(fig, f"{savedir}/summary_025_975_bounds.pdf")    

    plt.close("all")

def plot_all_wrapper(D):
    """ 
    """

    # primitiveness
    from pythonlib.dataset.dataset_analy.primitivenessv2 import preprocess_plot_pipeline
    Dcopy = D.copy()
    preprocess_plot_pipeline(Dcopy, microstim_version=True)

    # single prims.
    from pythonlib.dataset.dataset_analy.singleprims import preprocess_dataset
    Dcopy = D.copy()
    preprocess_dataset(Dcopy, PLOT=True)

    # motor timing
    if False:
        # all folded into primitiveness now
        Dcopy = D.copy()
        plot_motortiming(Dcopy)

    # Grammar
    date = D.dates(True)[0]
    is_grammar = (sum(D.Dat["epoch_orig"] != date)>0) and (np.any(D.Dat["task_kind"]=="prims_on_grid"))

    # Example values:
    # print(D.Dat["epoch"].unique())
    # print(D.Dat["epoch_orig"].unique())
    # print(D.Dat["task_kind"].unique())
        # ['240611|off' '240611|TTL3-fg']
        # ['240611']
        # ['prims_single' 'prims_on_grid']
    if is_grammar:
        try:
            from pythonlib.dataset.dataset_analy.grammar import pipeline_generate_and_plot_all
            Dcopy = D.copy()
            pipeline_generate_and_plot_all(Dcopy)
        except NotEnoughDataException as err:
            # skip, since rules are not defined
            pass
        except Exception as err:
            raise err

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
            tc = D.Dat.iloc[ind]["trialcode"]
            tr = D.Dat.iloc[ind]["trial"]
            ax.set_title(f"tc_{tc}|trial_{tr}")
            print(f"{sdir}/trial_{tr}.png")
            fig.savefig(f"{sdir}/trial_{tr}.png")
            # assert False
            plt.close("all")    
        else:
            assert ms_fix["on"]==0
            assert ms_stroke["on"]==0
            

def preprocess_assign_stim_code(D, map_ttl_region, code_ignore_within_trial_time=False):
    """
    Give each trial a string code for its stim params, which can vary across 
    expts. Code indicates both the map_ttl_region, and which time windows
    within trial are being stimmed (as a string code).
    PARAMS:
    - map_ttl_region, dict mapping from int ttl values to string , usually
    breian regions, which will be the code.
    - code_ignore_within_trial_time, then will code simply by ttl, not by when
    in trial it stimmed.
    EXAMPLE:
    - 
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

        ## Parse when stim occured on this trial, abstractly.
        def _parse_stim_strokes(ms_str):
            """ Return a list of stims that are assigned to this trial, such as
            e./.g, [('go_cue',), ('on', 1), ('on', 2), ('on', 3), ('on', 4), ('off', 1), ('off', 2), ('off', 3)]
            """
            _list_stims =[]
            for x in ms_str["stimlist"]:
                if x[0]=="go_cue":
                    _list_stims.append(("go_cue",))
                elif x[0] in ["on", "off"]:
                    indstroke = int(x[1])
                    _list_stims.append((x[0], indstroke))
                else:
                    print(x[0])
                    print(x)
                    assert False, "dont know this"
            return _list_stims

        # Collect a string code for which time windows wihtin trial.
        stim_window_code = ""
        if ms_fix["on"]==1:
            stim_window_code+="f"
        if ms_str["on"]==1:
            # could be go_cue, stroke onsets, or stroke offsets
            _list_stims = _parse_stim_strokes(ms_str)
            _list_stim_kinds_unique = sorted(list(set([x[0] for x in _list_stims]))) # ['on', 'go_cue', 'off']
            # print(_list_stims)
            # print(_list_stim_kinds_unique)

            for _stim in _list_stim_kinds_unique:
                if _stim=="go_cue":
                    stim_window_code+="g"
                elif _stim=="on":
                    stim_window_code+="n"
                elif _stim=="off":
                    stim_window_code+="o"
                else:
                    print(_stim)
                    assert False

        # print(" --------------- ", ind)
        # print(ms_fix)
        # print(ms_str)
        # print(stim_window_code)

        # try: 
        # FIxation
        stim_code = []
        ttl_codes = [] #collect across all stims.
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
            # if len(set(ttls))!=1:
            #     print(ttls)
            #     print(ms_str)
            #     assert False
            stim_code.append(ttls[0])
            ttl_codes.append(ttls[0])
        
        stim_code = sorted(stim_code) # so it is unqiue
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

            # and append the code indicating WHEN in trial it is stimmed
            if not code_ignore_within_trial_time:
                sc = f"{sc}-{stim_window_code}"
        # except Exception as err:
        #     # VEry hacky placeholder, avoiding fixing code until I need to
        #     sc = "DUMMY"

        list_stim_code.append(sc)

    # place stim code back into data
    D.Dat["microstim_epoch_code"] = list_stim_code

    print("New column: microstim_epoch_code")
    

def plot_motortiming(D, PLOT=True, microstim_version=True):
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
    from pythonlib.stats.lme import lme_categorical_fit_plot
    from pythonlib.tools.pandastools import datamod_normalize_row_after_grouping

    PLOT_BY_STROKE_INDEX_SET = False

    # Prep dataset
    D.grouping_get_inner_items("block", "epoch") 
    SAVEDIR_BASE = D.make_savedir_for_analysis_figures_BETTER("microstim/motortiming")
    
    if False:
        for ind in range(len(D.Dat)):
            block = D.Dat.iloc[ind]["block"]
            rule = D.blockparams_extract_single_combined_task_and_block(ind)["GENERAL"]["block_transition_rule"]
            print(ind, block, rule)

    # Extract Strokes data
    VARS_CONTEXT = ["CTXT_shape_prev", "shape", "CTXT_loc_prev", "gridloc"]
    params_preprocess = ["remove_baseline", "no_supervision", "only_blocks_with_n_min_trials"]
    VAR = "epoch"
    DS, DFTHIS_ORIG = gapstroke_timing_compare_by_variable(D, VAR, VARS_CONTEXT, 
        params_preprocess, PLOT=False, microstim_version=microstim_version)
    # Take controlled data --> plot a single distribution, one for each (context, index) combo
    DFTHIS_ORIG = append_col_with_grp_index(DFTHIS_ORIG, ["stroke_index", "context"], "strk_idx_ctxt", use_strings=False)

    if len(DFTHIS_ORIG)>0 and PLOT:

        for context_var in ["context", "strk_idx_ctxt"]:

            DFTHIS = DFTHIS_ORIG.copy()

            SAVEDIR = f"{SAVEDIR_BASE}/contextvar_{context_var}"
            os.makedirs(SAVEDIR, exist_ok=True)

            # Remove strk_idx_ctxt that have too few data, or else makes plot harder to interpret.
            n_min_each_conj_outer_inner = 3
            # print("**** len DS (3)", len(DFTHIS))
            DFTHIS, list_epochsets_unique= assign_epochsets_group_by_matching_levels_of_var(DFTHIS, 
                                                                                            var_outer_trials=context_var, 
                                                                                            var_inner="epoch",
                                                                                            epochset_col_name="epochset", 
                                                                                            PRINT=True, n_min_each_conj_outer_inner=n_min_each_conj_outer_inner)
            #print("**** len DS (4)", len(DFTHIS))
            #assert False
            # Plot as function of storke index
            if PLOT_BY_STROKE_INDEX_SET:
                n_min_each_conj_outer_inner = 3
                DFTHIS, list_epochsets_unique= assign_epochsets_group_by_matching_levels_of_var(DFTHIS, 
                                                                                                var_outer_trials="locshape_pre_this", 
                                                                                                var_inner="stroke_index",
                                                                                                epochset_col_name="stroke_index_set", 
                                                                                                PRINT=True,
                                                                                               n_min_each_conj_outer_inner=n_min_each_conj_outer_inner)

            print("SAVING FIGURES AT: ", SAVEDIR)
            for y in ["gap_from_prev_dur", "gap_from_prev_dist", "gap_from_prev_vel", "time_duration", "distcum", "velocity"]:
                
                fig = sns.catplot(data=DFTHIS, x=context_var, y=y, hue="epoch", kind="point", aspect=3, row="epoch_orig")
                rotateLabel(fig)    
                savefig(fig, f"{SAVEDIR}/{y}-vs-{context_var}.pdf")

                fig = sns.catplot(data=DFTHIS, x=context_var, y=y, hue="epoch", row="epochset", kind="point", col="epoch_orig")
                rotateLabel(fig)    
                savefig(fig, f"{SAVEDIR}/{y}-vs-{context_var}-grp_by_epochset.pdf")

                plt.close("all")
                
                if PLOT_BY_STROKE_INDEX_SET:
                    # Too large, and sparse, I dont use anwyay
                    fig = sns.catplot(data=DFTHIS, x="stroke_index", y=y, hue="epoch", row="locshape_pre_this", col="stroke_index_set", kind="point")
                    rotateLabel(fig)    
                    savefig(fig, f"{SAVEDIR}/{y}-vs-stroke_index-grp_by_stroke_index_set.pdf")

                plt.close("all")

                ####### PLOTS OF CONTRAST ACROSS LEVELS.
                INDEX = [context_var, "epoch_orig", "block"]
                if "microstim_epoch_code" in DFTHIS.columns:
                    fixed_treat = "microstim_epoch_code"
                    lev_treat_default = "off"
                else:
                    fixed_treat = "epoch"
                    lev_treat_default = None

                if True:
                    # Linear mixed effects
                    # This doesnt make sense, since there is only one datapt per group
                    RES, fig, ax = lme_categorical_fit_plot(DFTHIS, y=y, fixed_treat=fixed_treat, 
                            lev_treat_default=lev_treat_default, 
                            rand_grp_list=INDEX, PLOT=True)
                    savefig(fig, f"{SAVEDIR}/LME-{fixed_treat}-{y}.pdf")

                # Plot normalized to the default level.
                _, _, _, _, fig = datamod_normalize_row_after_grouping(DFTHIS, 
                                                                      fixed_treat, 
                                                                      INDEX, 
                                                                      y,
                                                                      lev_treat_default,
                                                                      PLOT=True
                                                                     )
                savefig(fig, f"{SAVEDIR}/NORM-{fixed_treat}-{y}.pdf")

                # ## ALso seaprately for each epoch-orig
                # for epoch_orig in list_epoch_orig:

                #     _, _, _, _, fig = datamod_normalize_row_after_grouping(DFTHIS, 
                #                                                           fixed_treat, 
                #                                                           INDEX, 
                #                                                           y,
                #                                                           lev_treat_default,
                #                                                           PLOT=True
                #                                                          )
                #     savefig(fig, f"{SAVEDIR}/NORM-{fixed_treat}-{y}.pdf")



    return DS, DFTHIS