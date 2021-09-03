import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..drawmodel.strokePlots import plotDatStrokes
from pythonlib.tools.pandastools import applyFunctionToAllRows


def _groupingParams(D, expt):
    """ Filter and grouping variable to apply to 
    Dataset"""

    grouping_levels = ["short", "long"]
    vals = ["nstrokes", "hausdorff", "time_go2raise", "time_raise2firsttouch", 
        "dist_raise2firsttouch", "sdur", "isi", 
        "time_touchdone", "dist_touchdone", "dist_strokes", 
        "dist_gaps", "sdur_std", "gdur_std", "hdoffline", "alignment"]

    feature_names = vals + ["stroke_speed", "gap_speed", "onset_speed", "offset_speed", "total_distance",
                    "total_time", "total_speed", "dist_per_gap", "dist_per_stroke"]

    features_to_remove_nan =  ["dist_strokes", "sdur", "dist_gaps" , "isi", "dist_raise2firsttouch", "time_raise2firsttouch", "nstrokes"]
    features_to_remove_outliers = ["dist_strokes", "sdur", "dist_gaps" , "isi", "dist_raise2firsttouch", "time_raise2firsttouch"] # distnaces and times.


    ### FILTER BLOCKS
    if expt == "neuralprep2":
        F = {
            "block":[17, 18]
        }
        grouping = "block"
        assert False, "get plantime_cats"
    elif expt=="neuralprep3":
        F = {
            "block":[9],
            "hold_time_exists":[True]
        }
        grouping = "hold_time_string"
        assert False, "get plantime_cats"
    elif expt in ["neuralprep4", "neuralprep5", "neuralprep6", "neuralprep8"]:
        F = {
            "block":[16],
            "plan_time":[0., 1000.]
        }
        grouping = "plan_time_cat"
        plantime_cats = {0.: "short", 1000.: "long"}
    elif expt in ["neuralprep7"]:
        F = {
            "block":[16],
            "plan_time":[150., 1000.]
        }
        grouping = "plan_time_cat"
        plantime_cats = {150.: "short", 1000.: "long"}
    elif expt in ["plan3"]:
        F = {
            "block":[11],
            "plan_time":[0., 1200.]
        }
        grouping = "plan_time_cat"
        plantime_cats = {0.: "short", 1200.: "long"}
        feature_names = [f for f in feature_names if f not in 
        ["time_touchdone", "dist_touchdone", "offset_speed"]]
    elif expt in ["plan4", "plan5"]:
        F = {
            "block":[11],
            "plan_time":[0., 1000.]
        }
        grouping = "plan_time_cat"
        plantime_cats = {0.: "short", 1000.: "long"}
        feature_names = [f for f in feature_names if f not in 
            ["time_touchdone", "dist_touchdone", "offset_speed"]]
    elif expt in ["plandir1"]:
        print("ONLY FOCUS ON l-->r BLOCK FOR NOW!! NEED TO ALSO ADD BLOCK 1")
        F = {
            "block":[11],
            "plan_time":[0., 1000.]
        }
        grouping = "plan_time_cat"
        plantime_cats = {0.: "short", 1000.: "long"}
        feature_names = [f for f in feature_names if f not in 
            ["time_touchdone", "dist_touchdone", "offset_speed", "alignment"]] # exclude alignment if there is only one trial per task.
    elif expt in ["plandir2"]:
        print("ONLY FOCUS ON r-->l BLOCK FOR NOW!! NEED TO ALSO ADD BLOCK 1")
        F = {
            "block":[18],
            "plan_time":[0., 1000.]
        }
        grouping = "plan_time_cat"
        plantime_cats = {0.: "short", 1000.: "long"}
        feature_names = [f for f in feature_names if f not in 
            ["time_touchdone", "dist_touchdone", "offset_speed", "alignment"]] # exclude alignment if there is only one trial per task.
    elif expt=="lines5":
        F = {}
        grouping = "epoch"
        plantime_cats = {}
        features_to_remove_nan =  []
        features_to_remove_outliers = []
        grouping_levels = ["straight", "bent"]
        feature_names = ["hdoffline", "num_strokes", "circ", "dist"]
    elif expt=="linecircle":
        F = {}
        grouping = "epoch"
        plantime_cats = {}
        features_to_remove_nan =  []
        features_to_remove_outliers = []
        grouping_levels = ["null"]
        feature_names = ["hdoffline"]
    elif expt=="figures9":
        F = {}
        grouping = "epoch"
        plantime_cats = {}
        features_to_remove_nan =  []
        features_to_remove_outliers = []
        grouping_levels = ["straight", "bent"]
        feature_names = ["hdoffline", "num_strokes", "circ", "dist"]        
    elif expt == "gridlinecircle":
        animal = D.animals()
        if len(animal)>1:
            assert False, "params different for each animal.."
        else:
            animal = animal[0]
        F = {}
        grouping = "epoch"
        plantime_cats = {}
        features_to_remove_nan =  []
        features_to_remove_outliers = []
        if animal in ["Diego", "Pancho"]:
            grouping_levels = ["baseline", "linetocircle", "circletoline", "lolli"]
        elif animal in ["Red"]:
            grouping_levels = ["baseline", "Ltoline"]
        else:
            assert False

        feature_names = ["hdoffline", "num_strokes", "circ", "dist"]                
    else:
        print(expt)
        assert False

    if len(F)>0:
        D = D.filterPandas(F, return_ver="dataset")

    # classify based on plan times
    if len(plantime_cats)>0:
        F = lambda x: plantime_cats[x["plan_time"]]
        D.Dat = applyFunctionToAllRows(D.Dat, F, "plan_time_cat")

    return D, grouping, grouping_levels, feature_names, features_to_remove_nan, features_to_remove_outliers


def preprocessDat(D, expt, get_sequence_rank=False, sequence_rank_confidence_min=None,
    remove_outliers=False, sequence_match_kind=None, extract_motor_stats=False,
    score_all_pairwise_within_task=False, extract_features = False, only_keep_trials_across_groupings=False):
    """ wrapper for preprocessing, can differ for each expt, includes
    both general and expt-specific stuff.
    INPUT:
    - get_sequence_rank, then gets rank of beh sequence using parses (rank of efficiency 
    of best-matching parse). NOTE: this removes all trials where strokes_beh cannot be compared
    to strokes_task - by defualt removes cases where num strokes not equal. Should be qwuick, since
    requires preprocessing with D.planner... methods.
    - only_if_sequence_different_across_grouping, then only trials where sequence is not used in other groups.
    THis ideally goes with get_sequence_rank=True, sequence_rank_confidence_min=0.1, as only make sense if
    sequence assignment is accurate/confident.
    NOTE:
    - if D.Dat ends up being empty, then returns None
    - if all flags False, then doesnt do any mods to D, just returns groupings, etc.
    """
    from pythonlib.tools.pandastools import filterPandas, aggregGeneral, applyFunctionToAllRows
    from pythonlib.drawmodel.strokedists import distscalarStrokes

    if hasattr(D, "_analy_preprocess_done"):
        if D._analy_preprocess_done:
            assert False, "already done preprocess!!"

    # (1) Warnings
    if expt=="neuralprep4":
        print("First day (210427) need to take session 5. Second day take all. ..")
        assert False

    # (2) Extract new derived varaibles
    # -- Plan time
    if "holdtime" in D.Dat.columns and "delaytime" in D.Dat.columns:
        tmp = D.Dat["holdtime"] - D.Dat["delaytime"]
        tmp[tmp<0.] = 0.
        D.Dat["plan_time"] = tmp

    # (3) Apply grouping variabples + prune dataset
    print("- starting/ending len (grouping params):")
    print(len(D.Dat))
    D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, features_to_remove_nan, features_to_remove_outliers \
        = _groupingParams(D, expt)
    print(len(D.Dat))


    # Only keep characters that have at lesat one trial across all grouping levels.
    if only_keep_trials_across_groupings:
        D.removeTrialsExistAcrossGroupingLevels(GROUPING, GROUPING_LEVELS)

    # -- std of stroke and gaps
    if extract_motor_stats:
        if "motorevents" in D.Dat.columns:
            def F(x, ver):
                ons = x["motorevents"]["ons"]
                offs = x["motorevents"]["offs"]
                if len(ons)==1:
                    return np.nan
                strokedurs = np.array(offs) - np.array(ons)
                gapdurs = np.array(ons)[1:] - np.array(offs)[:-1]
                if ver=="stroke":
                    return np.std(strokedurs)
                elif ver=="gap":
                    return np.std(gapdurs)
            D.Dat = applyFunctionToAllRows(D.Dat, lambda x: F(x, "stroke"), "sdur_std")
            D.Dat = applyFunctionToAllRows(D.Dat, lambda x: F(x, "gap"), "gdur_std")

    ### EXTRACT FEATURES
    # - hausdorff, offline score
    if extract_features:
        D.extract_beh_features(feature_list = FEATURE_NAMES)
        D.score_visual_distance()

    # -- pull out variables into separate columns
    if extract_motor_stats:
        for col in ["motortiming", "motorevents"]:
            keys = D.Dat[col][0].keys()
            for k in keys:
                def F(x):
                    return x[col][k]
                D.Dat = applyFunctionToAllRows(D.Dat, F, k)

        # - derived motor stats
        D.Dat["stroke_speed"] = D.Dat["dist_strokes"]/D.Dat["sdur"]
        D.Dat["gap_speed"] = D.Dat["dist_gaps"]/D.Dat["isi"]
        D.Dat["onset_speed"] = D.Dat["dist_raise2firsttouch"]/D.Dat["time_raise2firsttouch"]
        D.Dat["offset_speed"] = D.Dat["dist_touchdone"]/D.Dat["time_touchdone"]

        # D.Dat["total_distance"] = D.Dat["dist_strokes"] + D.Dat["dist_gaps"] + D.Dat["dist_raise2firsttouch"] + D.Dat["dist_touchdone"]
        # D.Dat["total_time"] = D.Dat["sdur"] + D.Dat["isi"] + D.Dat["time_raise2firsttouch"] + D.Dat["time_touchdone"]
        # D.Dat["total_speed"] = D.Dat["total_distance"]/D.Dat["total_time"]
        D.Dat["total_distance"] = D.Dat["dist_strokes"] + D.Dat["dist_gaps"]
        D.Dat["total_time"] = D.Dat["sdur"] + D.Dat["isi"]
        D.Dat["total_speed"] = D.Dat["total_distance"]/D.Dat["total_time"]

        D.Dat["dist_per_gap"] = D.Dat["dist_gaps"]/(D.Dat["nstrokes"]-1)
        D.Dat["dist_per_stroke"] = D.Dat["dist_strokes"]/(D.Dat["nstrokes"])


    # (4) Sequences more similar within group than between?
    if score_all_pairwise_within_task:
        from pythonlib.dataset.analy import score_all_pairwise_within_task
        from pythonlib.dataset.analy import score_alignment
        DIST_VER = "dtw_split_segments"

        # - score all pairwise, trials for a given task
        SCORE_COL_NAMES = score_all_pairwise_within_task(D, GROUPING, GROUPING_LEVELS,
            DIST_VER, DONEG=True)

        # - score alignment
        score_alignment(D, GROUPING, GROUPING_LEVELS, SCORE_COL_NAMES)
    else:
        SCORE_COL_NAMES = []


    # - score beh sequence rank relative to parses
    if get_sequence_rank:
        sequence_get_rank_vs_task_permutations_quick(D)
        FEATURE_NAMES = sorted(set(FEATURE_NAMES + ["effic_rank", "effic_summary", "effic_confid"]))

    # print("pruning by confidence of rank")
    # print(len(D.Dat))
    # print(D.Dat["effic_confid"])
    # print(len(np.isnan(D.Dat["effic_confid"])))
    print("- starting/ending len (getting sequence):")
    print(len(D.Dat))
    if get_sequence_rank and sequence_rank_confidence_min is not None:
        D.Dat = D.Dat[D.Dat["effic_confid"]>=sequence_rank_confidence_min]
        D.Dat = D.Dat.reset_index(drop=True)
    if len(D.Dat)==0:
        return None
    print(len(D.Dat))

    # =========
    if sequence_match_kind in ["same", "diff"]:
        print("-- Doing only_if_sequence_different_across_grouping")
        print(len(D.Dat))
        D.analy_match_sequence_discrete_per_task(groupby=GROUPING, 
            grouping_levels=GROUPING_LEVELS, ver = sequence_match_kind, 
            print_summary=True)
        print(len(D.Dat))      
    else:
        assert sequence_match_kind is None

    # ======== CLEAN, REMOVE NAN AND OUTLIERS
    # - Remove nans
    D.removeNans(columns=features_to_remove_nan) 
    # - Replace outliers with nans
    for F in features_to_remove_outliers:
        D.removeOutlierRowsTukey(F, niqr=2.5, replace_with_nan=True)
    if remove_outliers:
        D.removeOutlierRows(FEATURE_NAMES, [0.1, 99.9])

    # () Note that preprocess done
    D._analy_preprocess_done=True

    # Print outcomes
    print("GROUPING", GROUPING)
    print("GROUPING_LEVELS", GROUPING_LEVELS)
    print("FEATURE_NAMES", FEATURE_NAMES)
    print("SCORE_COL_NAMES", SCORE_COL_NAMES)

    return D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES


def preprocess_dates(D, groupby, animal, expt, nmin = 5, return_filtered_dataset=False,
    SAVEDIR=None):
    """ Figure out good dates for summary analyses based on flexible criteria
    - groupby, variable for which you desire there to exist trials for all levels in all date
    epochs. e..g, if groupby is "character" then extracts dates that have at least one trial for each char.
    NOTE: throws out levels that are not present across all datse.
    INPUT:
    - groupby, col name
    - nmin = 5, at lesat this many trials to use a day.
    - return_filtered_dataset, then returns dataset pruned to only have the levels of groupby that are kept.
    - SAVEDIR, if provided, then will save date information as yaml
    NOTE:
    - default criteria is to find dates that have at lesat nmin trials for all levels of groupby. finds the earlierst
    and latest within the window hand coded inside. throws out any levels which fail.
    """

    # Split all into early and late
    # for each prim, determine its early and late
    day_window = None
    if expt=="primitivesall":
        if animal=="Pancho":
            day_window = list(range(210605, 210613+1)) # the days in which to look.
        elif animal=="Diego":
            day_window = [210610, 210611, 210612, 210613] # the days in which to look.
        else:
            assert False
    elif expt=="primcat12":
        day_window = list(range(210731, 210804+1)) # the days in which to look.
    assert day_window is not None


    list_levels = D.Dat[groupby].unique()
    list_days = D.Dat["date"].unique()

    # For each primtuple, check if has enough trials early and late
    #### Output is, for each primtuple, all the inds present on each date.
    DictDateInds = {}
    for level in list_levels:

        DictDateInds[level] = {}
        for day in list_days:
            # save the indices and dates for this primtuple
            inds = D.filterPandas({"date":[day], groupby:[level]})
            DictDateInds[level][str(day)] = inds

    #### For each day, find the levels which have at least a min num trials.
    DictDay = {}
    for day in list_days:
        list_levels_this = []
        for level, val in DictDateInds.items():
            if len(val[day])>nmin:
                list_levels_this.append(level)

        DictDay[day] = list_levels_this

    for day, prims in DictDay.items():
        print(day, len(prims))

    # given two days, find set of overlapping prims
    def _olap(day1, day2):
        return [this for this in DictDay[str(day1)] if this in DictDay[str(day2)]]

    def _intersect(list_of_lists):
        result = set(list_of_lists[0])
        for s in list_of_lists[1:]:
            result.intersection_update(s)
        return result

    def intersect_days(list_days):
        list_of_l = [DictDay[str(d)] for d in list_days]
        return _intersect(list_of_l)


    # will take the earliers and latest which exist for each prim.
    DictDateEarlyLate = {}
    for prim, days in DictDateInds.items():
        # ealiest day in this window
        day_candidates = [d for d, inds in days.items() if len(inds)>nmin and int(d) in day_window] # days with at least nmin trials
        if len(day_candidates)<2:
            continue
    #     print(sorted(day_candidates))
        # take the first day 
        firstday = day_candidates[0]
        lastday =  day_candidates[-1]

        DictDateEarlyLate[prim] = [firstday, lastday]

    if SAVEDIR is not None:
        from pythonlib.tools.expttools import writeDictToYaml, makeTimeStamp
        writeDictToYaml(DictDateEarlyLate, f"{SAVEDIR}/DictDateEarlyLate.yaml")
        writeDictToYaml(DictDay, f"{SAVEDIR}/DictDay.yaml")
        writeDictToYaml(DictDateInds, f"{SAVEDIR}/DictDateInds.yaml")


    if return_filtered_dataset:
        Dfilt = D.filterPandas({groupby:list(DictDateEarlyLate.keys())}, "dataset")
        return DictDateEarlyLate, DictDay, DictDateInds, Dfilt
    else:
        return DictDateEarlyLate, DictDay, DictDateInds



def sequence_get_rank_vs_task_permutations_quick(D, permver = "all_orders_directions"):
    from pythonlib.drawmodel.efficiencycost import rank_beh_out_of_all_possible_sequences
    """ For each behavhora trial, align its sequence vs. task, then compute the rank
    of the saeuqnece based on efficiency cost. can only compute rank if have same lenth
    otherwise will return nan.
    INPUTS:
    - permver, think of this as index into a particular planner expt/analysis.
    RETURNS:
    - D.Dat modified to have 5 new columns. see code.
    NOTE:
    - "quick" means that parses, parse-beh dist, and parse-scores are all precomputed and saved,
    using the appropriate Dataset methods.
    """
    from pythonlib.drawmodel.efficiencycost import rank_beh_out_of_all_possible_sequences_quick

    # load pre-computed planner stuff
    D.planner_load_everything(permver = "all_orders_directions")

    def F(x):
        strokes_beh = x["strokes_beh"]
        strokes_task_perms = x["parses_planner"]
        beh_task_distances = x["parses_planner_behtaskdist"]
        task_inefficiency = x["parses_planner_taskscore"]

        out = rank_beh_out_of_all_possible_sequences_quick(
            strokes_beh, strokes_task_perms, beh_task_distances, 
            task_inefficiency, efficiency_score_ver="weighted_avg", 
            confidence_ver="diff_relative_all")
        # print("this")
        # print(out)
        # print("th")
        if out is None:
            return np.nan
        out = list(out) # dont inlcude the last element, which is strokes task picked.
        out[3] = len(strokes_task_perms) # num perms.
        return out

    # def F(x):
    #     sb = x["strokes_beh"]
    #     st = x["strokes_task"]
    #     if len(sb)!=len(st):
    #         return np.nan
    #     out = rank_beh_out_of_all_possible_sequences(sb, st, return_chosen_task_strokes=False,
    #                                                 plot_rank_distribution=False, 
    #                                                 return_num_possible_seq=True)
    # #     rank, confidence, summaryscore, numseq = out
    #     return out
    # #     return rank_beh_out_of_all_possible_sequences(x["strokes_beh"], x["strokes_task"])

    # print(len(D.Dat))
    # for row in D.Dat.iterrows():
    #     row[1]["rankeffic"] = F(row[1])
    D.Dat = applyFunctionToAllRows(D.Dat, F, "rankeffic")

    # Expand out efficiency scores
    def F(x, ind):
        if np.all(np.isnan(x["rankeffic"])):
            return np.nan
        return x["rankeffic"][ind]

    # for ind in range(4):
    #     D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,ind), "effic_rank")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,0), "effic_rank")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,1), "effic_confid")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,2), "effic_summary")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,3), "effic_nperms")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,4), "effic_taskscore")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,5), "effic_behtaskdist")
    del D.Dat["rankeffic"]

    def F(x):
        if np.isnan(x["effic_rank"]):
            return False
        else:
            return True
    D.Dat = applyFunctionToAllRows(D.Dat, F, "has_effic_rank")



def sequence_get_rank_vs_task_permutations(D):
    from pythonlib.drawmodel.efficiencycost import rank_beh_out_of_all_possible_sequences
    """ For each behavhora trial, align its sequence vs. task, then compute the rank
    of the saeuqnece based on efficiency cost. can only compute rank if have same lenth
    otherwise will return nan.
    RETURNS:
    - D.Dat modified to have 5 new columns. see code.
    """
    def F(x):
        sb = x["strokes_beh"]
        st = x["strokes_task"]
        if len(sb)!=len(st):
            return np.nan
        out = rank_beh_out_of_all_possible_sequences(sb, st, return_chosen_task_strokes=False,
                                                    plot_rank_distribution=False, 
                                                    return_num_possible_seq=True)
    #     rank, confidence, summaryscore, numseq = out
        return out
    #     return rank_beh_out_of_all_possible_sequences(x["strokes_beh"], x["strokes_task"])

    D.Dat = applyFunctionToAllRows(D.Dat, F, "rankeffic")

    # Expand out efficiency scores
    def F(x, ind):
        if np.all(np.isnan(x["rankeffic"])):
            return np.nan
        return x["rankeffic"][ind]

    # for ind in range(4):
    #     D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,ind), "effic_rank")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,0), "effic_rank")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,1), "effic_confid")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,2), "effic_summary")
    D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x,3), "effic_nperms")
    del D.Dat["rankeffic"]

    def F(x):
        if np.isnan(x["effic_rank"]):
            return False
        else:
            return True
    D.Dat = applyFunctionToAllRows(D.Dat, F, "has_effic_rank")



def get_task_pairwise_metrics(D, grouping, func):
    """ groups data bsed on tasks, then runs thru all pairs of trials (same task) and computs
    func. 
    INPUT:
    - grouping, will save the value of this variable, in the output, for each item in each pair.
    - func(x,y), where x and y are two trials, row of dataframe.
    OUTPUT:
    - list of dict, where each dict holds a unique pair of trials
    """

    # Compute all pairwise between all trials (conditions on pairs being from the same task).
    tasklist = set(D.Dat["character"])
    out = []
    for task in tasklist:
        dfthis = D.Dat[D.Dat["character"]==task]

        # get all pairwise scores between all rows.
        c = 0
        for i, row1 in enumerate(dfthis.iterrows()):
            for ii, row2 in enumerate(dfthis.iterrows()):
                if ii>i:
                    # print(i, ii)
                    # print(row1[0])
                    # print(row2[0])
                    # assert False
                    value = func(row1[1], row2[1])
                    # strokes1 = row1[1]["strokes_beh"]
                    # strokes2 = row2[1]["strokes_beh"]
                    # d = get_dist(strokes1, strokes2)

                    monkey_priors = (row1[1][grouping], row2[1][grouping])
                    out.append({
                        "value":value,
                        "task":task,
                        "groupings":sorted(monkey_priors),
                        "indexes":(row1[0], row2[0])})
                    c+=1

        # print(task, c)
    return out




def score_all_pairwise_within_task(D,  GROUPING, 
    GROUPING_LEVELS, DIST_VER = "dtw_segments", DONEG=False):
    """
    for each unique task, gets all pairwise distances for same level, diff level,
    where levels defined by grouping variable (e..g, monkey_prior, epoch etc).
    INPUTS:
    - DIST_VER, str, which kidn of score to use.
    - GROUPING, dataframe column, whose levels are used to split data.
    - DONEG, if True, then will flip sign of distance, so that more neg is more worse.
    - GROUPING_LEVELS, the levels to care about (i.e., within level vs. across level scoring)
    RETURNS:
    - modifies D.Dat, modified to have new columns, where for each trial give score reltaive to 
    all other trials, separated by how those other trials are grouped. if multipel trials 
    compared to it, then takes mean.
    - colnames, the names of new cols. **Will be in same order as GROUPING_LEVELS input!!
    NOTE:
    - Useful, e.g, for positive control based on across-trial varibility given
    the same task(stimulus)
    - NOTE: grouping is synonymous with monkey prior in this code.
    - 
    """
    
    # Distancd function defined.
    def get_dist(strokes1, strokes2):
        """ return scalar dist between strokes"""
        from pythonlib.drawmodel.strokedists import distscalarStrokes
        return distscalarStrokes(strokes1, strokes2, ver=DIST_VER)

    def func(x, y):
        strokes1 = x["strokes_beh"]
        strokes2 = y["strokes_beh"]
        return get_dist(strokes1, strokes2)


    out = get_task_pairwise_metrics(D, GROUPING, func)
    # # Compute all pairwise between all trials (conditions on pairs being from the same task).
    # tasklist = set(D.Dat["character"])
    # out = []
    # for task in tasklist:
    #     dfthis = D.Dat[D.Dat["character"]==task]

    #     # get all pairwise scores between all rows.
    #     c = 0
    #     for i, row1 in enumerate(dfthis.iterrows()):
    #         for ii, row2 in enumerate(dfthis.iterrows()):
    #             if ii>i:
    #                 strokes1 = row1[1]["strokes_beh"]
    #                 strokes2 = row2[1]["strokes_beh"]
    #                 d = get_dist(strokes1, strokes2)

    #                 monkey_priors = [row1[1][GROUPING], row2[1][GROUPING]]
    #                 out.append({
    #                     "score":d,
    #                     "task":task,
    #                     "groupings":sorted(monkey_priors)})
    #                 c+=1

    #     # print(task, c)


    # Repopulate into main DF
    def _extract(task, monkey_prior_this, monkey_prior_other):
        """ extract list of scores for this task, done while under
        monkey_prior_this, compared to when monkey was under monkey_prior_other
        RETURNS:
        - scores, list of scores.
        """

        mkp_index = sorted([monkey_prior_this, monkey_prior_other])
        outthis = [o for o in out if o["task"]==task and o["groupings"]==mkp_index]
        scores = [o["value"] for o in outthis]
        return scores

    colnames = []
    for group_other in GROUPING_LEVELS:
        def F(x):
            scores = _extract(x["character"], x[GROUPING], group_other)
            if len(scores)==0:
                return np.nan
            if any(np.array(scores)<0.001):
                print(scores)
                assert False, "this means there is task compared to itself..."
            s = np.nanmean(scores) # negative to conver to score (from distance)
            if DONEG:
                s = -s
            return s
         
        D.Dat = applyFunctionToAllRows(D.Dat, F, newcolname=f"vs-selfsametask-{group_other}")    
        colnames.append(f"vs-selfsametask-{group_other}")

    return colnames


def score_alignment(D, monkey_prior_col, monkey_prior_levels, score_name_list, suffix=""):
    """ 2 x 2 alignment, where there are two monkey priors, and two scores (model scores).
    each row is one particular level, but has a score for each score_name_list.
    INPUTS:
    - monkey_prior_levels, list of levels.
    - score_name_list, list of names (coilumns)
    OUT:
    - modifies D in place.
    NOTE:
    ** ORDER matters, monkey_prior and score_name must be aligned.
    - score is assumed to be more positive --> better score.
    """

    for prior, score in zip(monkey_prior_levels, score_name_list):
        print(f"{prior} - aligned with - {score}")
        assert prior in score, "are they not aligned?"

    # 1) get difference of scores
    if len(suffix)>0:
        suffix = "_" + suffix
    colname_m2_m1 = f"mod2_minus_mod1{suffix}"
    D.Dat[colname_m2_m1] = D.Dat[score_name_list[1]] - D.Dat[score_name_list[0]]

    # 2) for each row, its alignemnet 
    def F(x):
        if x[monkey_prior_col]==monkey_prior_levels[0]:
            return -x[colname_m2_m1]
        elif x[monkey_prior_col]==monkey_prior_levels[1]:
            return x[colname_m2_m1]
        else:
            print(x)
            assert False

    newcol = f"alignment_trials{suffix}"
    print("New alignmment col: ", newcol)
    D.Dat = applyFunctionToAllRows(D.Dat, F, newcol)

    return colname_m2_m1, newcol

def taskmodel_assign_score(D, expt="lines5"):
    """ Quick and dirty, to replicate what did in Probedat, for scoring datsate.
    Uses probedat here just to extract Params.
    Note: only works for lines5 currently - the chunkmodel functin.
    OUTPUT:
    - new columnds in D.Dat, with names as in model_score_names
    - model_score_names, list of strings.
    """
    from pythonlib.drawmodel.taskmodel import Model, makeParseFunction, getParamsWrapper
    from pythonlib.drawmodel.taskmodel import Dataset as DatasetModel

    assert expt=="lines5", "otherwise make this more general purpose params"

    # -- 2) Fit different models.
    model_score_names = []
    for likeliver, priorver, parse_ver, chunkmodel, posterior_ver, name in zip(
        ["segments", "segments"],
        ["uniform","uniform"], 
        ["chunks", "chunks"],
        ["3line", None],
        ["maxlikeli", "maxlikeli"],
        ["3line", "linePlusL"]):

#         PD = ProbedatTaskmodel([])

        # 3) Buidl model
        if name in ["linePlusL", "linePlusL_combine", "onechunk"]:
            chunkmodel = makeParseFunction(name)
            
        assert chunkmodel is not None, "need to replace this with ParseFunction.."
        PARAMS, PARAMS_MODEL = getParamsWrapper(priorver=priorver, parse_ver=parse_ver, 
                                            chunkmodel=chunkmodel, name=name,
                                            posterior_ver=posterior_ver, 
                                            likeliver=likeliver)

        PARAMS_DATA, PARAMS_MODEL = getParamsWrapper(priorver, parse_ver, chunkmodel,
            name, posterior_ver, likeliver)

        # Prepare dataset
        strokes = D.Dat["strokes_beh"].to_list()
        strokes_task = D.Dat["strokes_task"].to_list()
        fix = D.Dat["origin"].to_list()
        tasks = [D.Dat["Task"].iloc[i].Params["input_params"].Task for i in range(len(D.Dat))]
        for stask, f, T in zip(strokes_task, fix, tasks):
            T["strokes"] = stask
            T["fixpos"] = f
    
        # Score using this model.
        mod = Model(PARAMS_MODEL)
        data = DatasetModel(strokes, tasks, PARAMS=PARAMS)
        data.applyModel(mod)

        # Assign scores back into dataset
        assert len(D.Dat)==len(data.trials)
        namethis = f"MOD_{name}"
        D.Dat[namethis] = [t["posterior"] for t in data.trials]
        model_score_names.append(namethis)


    return model_score_names




def extract_strokes_monkey_vs_self(Dlist, GROUPING, GROUPING_LEVELS):
    """
    populate new column with list of strokes for all other trials
    IN:
    - Dlist, list of D, where each D is a single dataset holding a single epoch (i.e)
    i.e., grouping level. doesnt' have to be, but that is how I did.
    NOTES:
    - each D in Dlist will have one new columns per grouping level, which will be all strokes
    across all D in Dlist which are from that grouping_level (escept your own trial's stroke)
    """
    from .dataset import concatDatasets

    # 1) Concat all datasets of interest, so can get all cross-parses
    Dall = concatDatasets(Dlist) # holds pool of strokes
    for D in Dlist:
        for group in GROUPING_LEVELS:
#             group = "straight" # which group's strokes to keep
#             Dthis = D # which one to modify
            
            def _get_strokes(D, task, group):
                """ returns all strokes for this task and this group, given a dataset D
                """
                dfthis = D.Dat[(D.Dat["unique_task_name"]==task) & (D.Dat[GROUPING]==group)]
                inds = dfthis.index.to_list()
                trialcodes = dfthis["trialcode"]
                strokes = dfthis["strokes_beh"]
                return trialcodes, strokes

            list_list_strokes = []
            for i in range(len(D.Dat)):
                task = D.Dat.iloc[i]["unique_task_name"]
                tcthis = D.Dat.iloc[i]["trialcode"]

                trialcodes, strokes = _get_strokes(Dall, task, group)

                # only keep strokes that are not from the current trial
                list_strokes = []
                for tc, s in zip(trialcodes, strokes):
                    if tcthis!=tc:
                        list_strokes.append(s)

                list_list_strokes.append(list_strokes)

            parsesname = f"strokes_beh_group_{group}"
            D.Dat[parsesname] = list_list_strokes
    
    # to confirm concat didnt modify
    for D in Dlist:
        D._check_consistency()
        
    # Only keep rows which have strokes across all models
    list_col_names = [f"strokes_beh_group_{group}" for group in GROUPING_LEVELS]
    for D in Dlist:
        def remove(D, i):
            # True if any of cols are empty
            x = [len(D.Dat.iloc[i][col])==0 for col in list_col_names]
            if any(x):
                return True
            else:
                return False
        inds_remove = [i for i in range(len(D.Dat)) if remove(D, i)]
        D.Dat = D.Dat.drop(inds_remove).reset_index(drop=True)
        
