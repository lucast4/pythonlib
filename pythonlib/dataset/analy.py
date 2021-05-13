import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..drawmodel.strokePlots import plotDatStrokes
from pythonlib.tools.pandastools import applyFunctionToAllRows


def _groupingParams(D, expt):
    """ Filter and grouping variable to apply to 
    Dataset"""

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
    else:
        assert False
    D = D.filterPandas(F, return_ver="dataset")

    # classify based on plan times
    F = lambda x: plantime_cats[x["plan_time"]]
    D.Dat = applyFunctionToAllRows(D.Dat, F, "plan_time_cat")

    grouping_levels = sorted([v for v in plantime_cats.values()])

    vals = ["nstrokes", "hausdorff", "time_go2raise", "time_raise2firsttouch", 
        "dist_raise2firsttouch", "sdur", "isi", 
        "time_touchdone", "dist_touchdone", "dist_strokes", 
        "dist_gaps", "sdur_std", "gdur_std", "hdoffline", "alignment"]

    feature_names = vals + ["stroke_speed", "gap_speed", "onset_speed", "offset_speed", "total_distance",
                   "total_time", "total_speed", "dist_per_gap", "dist_per_stroke"]


    return D, grouping, grouping_levels, feature_names


def preprocessDat(D, expt):
    """ wrapper for preprocessing, can differ for each expt, includes
    both general and expt-specific stuff.
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

    # -- std of stroke and gaps
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

    # -- hausdorff, offline score
    def F(x):
        return distscalarStrokes(x["strokes_beh"], x["strokes_task"], "position_hd")
    D.Dat = applyFunctionToAllRows(D.Dat, F, "hdoffline")

    # -- pull out variables into separate columns
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

    D.Dat["total_distance"] = D.Dat["dist_strokes"] + D.Dat["dist_gaps"] + D.Dat["dist_raise2firsttouch"] + D.Dat["dist_touchdone"]
    D.Dat["total_time"] = D.Dat["sdur"] + D.Dat["isi"] + D.Dat["time_raise2firsttouch"] + D.Dat["time_touchdone"]
    D.Dat["total_speed"] = D.Dat["total_distance"]/D.Dat["total_time"]

    D.Dat["dist_per_gap"] = D.Dat["dist_gaps"]/(D.Dat["nstrokes"]-1)
    D.Dat["dist_per_stroke"] = D.Dat["dist_strokes"]/(D.Dat["nstrokes"])

    # (3) Apply grouping variabples + prune dataset
    D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES = _groupingParams(D, expt)

    
    # (4) Sequences more similar within group than between?
    DIST_VER = "dtw_split_segments"
    from pythonlib.dataset.analy import score_all_pairwise_within_task
    from pythonlib.dataset.analy import score_alignment

    # - score all pairwise, trials for a given task
    SCORE_COL_NAMES = score_all_pairwise_within_task(D, DIST_VER, GROUPING, DONEG=True)
    # - score alignment
    # score_name_list = [f"vs-selfsametask-{x}" for x in GROUPING_LEVELS]
    score_alignment(D, GROUPING, GROUPING_LEVELS, SCORE_COL_NAMES)
   
    # ======== CLEAN, REMOVE NAN AND OUTLIERS
    D.removeNans(columns=FEATURE_NAMES)
    D.removeOutlierRows(FEATURE_NAMES, [0.1, 99.9])

    if False:
        def prep_data(D):
            """ D --> df"""

            ### aggregrate over unique tasks
            dfthis = aggregGeneral(D.Dat, [condition, "character"], values=vals)
            # dfthis = aggregGeneral(dfthis, ["hold_time_string", "unique_task_name"], values=vals)

            ### Derive new features, using the trial-averaged features

            # ADD THINGS
            dfthis["stroke_speed"] = dfthis["dist_strokes"]/dfthis["sdur"]
            dfthis["gap_speed"] = dfthis["dist_gaps"]/dfthis["isi"]
            dfthis["onset_speed"] = dfthis["dist_raise2firsttouch"]/dfthis["time_raise2firsttouch"]
            dfthis["offset_speed"] = dfthis["dist_touchdone"]/dfthis["time_touchdone"]

            dfthis["total_distance"] = dfthis["dist_strokes"] + dfthis["dist_gaps"] + dfthis["dist_raise2firsttouch"] + dfthis["dist_touchdone"]
            dfthis["total_time"] = dfthis["sdur"] + dfthis["isi"] + dfthis["time_raise2firsttouch"] + dfthis["time_touchdone"]
            dfthis["total_speed"] = dfthis["total_distance"]/dfthis["total_time"]

            dfthis["dist_per_gap"] = dfthis["dist_gaps"]/(dfthis["nstrokes"]-1)
            dfthis["dist_per_stroke"] = dfthis["dist_strokes"]/(dfthis["nstrokes"])

            return dfthis

        dfthis = prep_data(D)
        dfAgg = dfthis
        
        ###### OUTPUT
        feature_names = vals + ["stroke_speed", "gap_speed", "onset_speed", "offset_speed", "total_distance",
                               "total_time", "total_speed", "dist_per_gap", "dist_per_stroke"]
    
    # () Note that preprocess done
    D._analy_preprocess_done=True

    return D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES




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




def score_all_pairwise_within_task(D, DIST_VER = "dtw_segments", GROUPING = "plan_time_cat", 
    DONEG=False):
    """
    for each unique task, gets all pairwise distances for same level, diff level,
    where levels defined by grouping variable (e..g, monkey_prior, epoch etc).
    INPUTS:
    - DIST_VER, str, which kidn of score to use.
    - GROUPING, dataframe column, whose levels are used to split data.
    - DONEG, if True, then will flip sign of distance, so that more neg is more worse.
    RETURNS:
    - modifies D.Dat, modified to have new columns, where for each trial give score reltaive to 
    all other trials, separated by how those other trials are grouped. if multipel trials 
    compared to it, then takes mean.
    - colnames, the names of new cols.
    NOTE:
    - Useful, e.g, for positive control based on across-trial varibility given
    the same task(stimulus)
    - NOTE: grouping is synonymous with monkey prior in this code.
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

    monkey_prior_list = sorted(D.Dat[GROUPING].unique().tolist())

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
    for MONKEY_PRIOR_OTHER in monkey_prior_list:
        def F(x):
            scores = _extract(x["character"], x[GROUPING], MONKEY_PRIOR_OTHER)
            if len(scores)==0:
                return np.nan
            if any(np.array(scores)<0.001):
                print(scores)
                assert False, "this means there is task compared to itself..."
            s = np.nanmean(scores) # negative to conver to score (from distance)
            if DONEG:
                s = -s
            return s
         
        D.Dat = applyFunctionToAllRows(D.Dat, F, newcolname=f"vs-selfsametask-{MONKEY_PRIOR_OTHER}")    
        colnames.append(f"vs-selfsametask-{MONKEY_PRIOR_OTHER}")

    return colnames


def score_alignment(D, monkey_prior_col, monkey_prior_levels, score_name_list):
    """ 2 x 2 alignment, where there are two monkey priors, and two scores (model scores).
    each row is one particular level, but has a score for each score_name_list.
    INPUTS:
    - monkey_prior_levels, list of levels.
    - score_name_list, list of names (coilumns)
    NOTE:
    ** ORDER matters, monkey_prior and score_name must be aligned.
    - score is assumed to be more positive --> better score.
    """


    # 1) get difference of scores
    D.Dat["mod2_minus_mod1"] = D.Dat[score_name_list[1]] - D.Dat[score_name_list[0]]

    # 2) for each row, its alignemnet 
    def F(x):
        if x[monkey_prior_col]==monkey_prior_levels[0]:
            return -x["mod2_minus_mod1"]
        elif x[monkey_prior_col]==monkey_prior_levels[1]:
            return x["mod2_minus_mod1"]
        else:
            print(x)
            assert False
    D.Dat = applyFunctionToAllRows(D.Dat, F, "alignment")