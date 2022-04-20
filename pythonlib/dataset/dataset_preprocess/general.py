import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pythonlib.drawmodel.strokePlots import plotDatStrokes
from pythonlib.tools.pandastools import applyFunctionToAllRows
from ..analy_dlist import extract_strokes_monkey_vs_self

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

    def _get_defaults(D):
        F = {}
        grouping = "epoch"
        plantime_cats = {}
        features_to_remove_nan =  []
        features_to_remove_outliers = []
        grouping_levels = D.Dat[grouping].unique().tolist() # note, will be in order in dataset (usually chron)
        feature_names = ["hdoffline", "num_strokes", "circ", "dist"]     
        return F, grouping, plantime_cats, features_to_remove_nan, \
            features_to_remove_outliers, grouping_levels, feature_names

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
    elif expt in ["chunkbyshape1", "chunkbyshape1b", "chunkbyshape2", "chunkbyshape2b"]:
        F = {}
        grouping = "epoch"
        plantime_cats = {}
        features_to_remove_nan =  []
        features_to_remove_outliers = []
        grouping_levels = ["horiz", "vert"]
        feature_names = ["hdoffline", "num_strokes", "circ", "dist"]     
    elif expt=="character34":
        assert False, "fix this, see here"
        # epoch 1 (line) the test tasks were not defined as probes. Add param here , which
        # should have affect in subsewuen code redefining monkye train test.
    else:
        # Get defaults
        F, grouping, plantime_cats, features_to_remove_nan, \
         features_to_remove_outliers, grouping_levels, feature_names = _get_defaults(D)

    if len(F)>0:
        D = D.filterPandas(F, return_ver="dataset")

    # classify based on plan times
    if len(plantime_cats)>0:
        F = lambda x: plantime_cats[x["plan_time"]]
        D.Dat = applyFunctionToAllRows(D.Dat, F, "plan_time_cat")

    return D, grouping, grouping_levels, feature_names, features_to_remove_nan, features_to_remove_outliers


def preprocessDat(D, expt, get_sequence_rank=False, sequence_rank_confidence_min=None,
    remove_outliers=False, sequence_match_kind=None, extract_motor_stats=False,
    score_all_pairwise_within_task=False, extract_features = False, 
    only_keep_trials_across_groupings=False):
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

    ### Rename things as monkey train test depenidng on expt
    preprocess_task_train_test(D, expt)

    # () Note that preprocess done
    D._analy_preprocess_done=True

    # Print outcomes
    print("GROUPING", GROUPING)
    print("GROUPING_LEVELS", GROUPING_LEVELS)
    print("FEATURE_NAMES", FEATURE_NAMES)
    print("SCORE_COL_NAMES", SCORE_COL_NAMES)

    return D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES


def preprocess_task_train_test(D, expt):
    """ Clean up naming of tasks as train or test
    in the monkey_train_or_test key of D.Dat. Makes sure that all rows have either
    "train" or "test" for this column.
    RETURNS:
    - modifies D.Dat
    """

    probe_list = []
    for i in range(0, len(D.Dat["Task"])):
        t = D.Dat.iloc[i]["Task"]
        t_p = t.Params["input_params"]
        probe_val = t_p.info_summarize_task()["probe"]["probe"]
        probe_list.append(probe_val)
    D.Dat["probe"] = probe_list

    if expt in ["gridlinecircle", "chunkbyshape1", "resize1"]:
        # train were all random tasks, test were all fixed.
        key = "random_task"
        list_train = [True]
        list_test = [False]
        D.analy_reassign_monkeytraintest(key, list_train, list_test)
    else:
        D.analy_reassign_monkeytraintest("probe", [0], [1])


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


def extract_expt_metadat(expt=None, animal=None, rule=None, metadat_dir = "/home/lucast4/drawmonkey/expt_metadat"):
    """ Get matadata for this expt, without having to load a Dataset first. useful if want to
    know what rules exits (for exampel) before extracting those datsaets.
    This is looks for expt metadat yaml files defined usually in drawmonkey repo.
    PARAMS:
    - expt, animal, rule, all strings or None. If None, then gets ignores this feature (e.g., animal).
    e.g., expt="lines5" gets all rules and animals under lines5.
    - metadat_dir, string name of path where yaml files saved. 
    RETURNS:
    - list_expts, list of lists of strings, where inner lists are like ['chunkbyshape1', 'rule2', 'Pancho']
    """
    from pythonlib.tools.expttools import findPath, extractStrFromFname

    # construct the path wildcard
    pathwildcard = []
    if expt is not None:
        pathwildcard.append(expt)
    if rule is not None:
        pathwildcard.append(rule)
    if animal is not None:
        pathwildcard.append(animal)

    list_path = findPath(metadat_dir, [pathwildcard], None)
    list_expts = [(extractStrFromFname(path, "-", "all")) for path in list_path]
    return list_expts