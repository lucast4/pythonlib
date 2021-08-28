""" 
Analysis of primitives, categorization, etc. 
Related to nbotebook dataset --> analy_primitives_081721
"""

from pythonlib.tools.pandastools import applyFunctionToAllRows
import os
import numpy as np
import matplotlib.pyplot as plt

def preprocess(D):
    """ for preparaing for prim anslsyus.
    """

    #### First preprocess dataset so that strokes are all centerized
    list_sb, list_st, D = D.extractStrokeLists(recenter=True)
    D.Dat["strokes_beh"] = list_sb

    # Expose each tasks params
    def F(x, ver):
        """ expose the task params"""
        T = x["Task"]
        
        if "circle" in x["character"] and ver=="theta":
            # circles are symmetric circluar.
            return 0.
        else:
        #     sx = T.Shapes[0][1]["sx"]
        #     sy = T.Shapes[0][1]["sy"]
        #     xpos = T.Shapes[0][1]["x"]
        #     ypos = T.Shapes[0][1]["y"]
        #     th = T.Shapes[0][1]["theta"]
            return T.Shapes[0][1][ver]
        
    for ver in ["sx", "sy", "x", "y", "theta"]:
        D.Dat = applyFunctionToAllRows(D.Dat, lambda x:F(x, ver), ver)

    return D


def extract_primlist(D):
    """ helper to pull out list of prim names (strings)
    """
    prim_list = D.Dat["character"].unique()
    return prim_list

def extract_list_features(D, PRIM):
    list_sx = np.unique(D.Dat[D.Dat["character"]==PRIM]["sx"].to_list())
    list_sy = np.unique(D.Dat[D.Dat["character"]==PRIM]["sy"].to_list())
    list_theta = np.unique(D.Dat[D.Dat["character"]==PRIM]["theta"].to_list())
    return list_sx, list_sy, list_theta


def extract_primtuple_list(D, Nmin = 8, list_prims_exclude = ["dot", "line"]):
    """ 
    GET list of primtuples that have more than N trials over all days.
    primtuple is (prim, sx, sy, theta)
    """
    
    from itertools import product

    prim_list = extract_primlist(D)

    # 1) collect list of primitives
    list_primtuple = []
    for PRIM in prim_list:
        list_sx, list_sy, list_theta = extract_list_features(D, PRIM)

        # Dprim = D.filterPandas({"character":[PRIM]}, "dataset")
        # list_sx = np.unique(Dprim.Dat["sx"])
        # list_sy = np.unique(Dprim.Dat["sy"])
        # list_theta = np.unique(Dprim.Dat["theta"])

        # For each combo of params, plot sample size
        for sx, sy, th in product(list_sx, list_sy, list_theta):
            DprimThis = D.filterPandas({"character":[PRIM], "sx":[sx], "sy":[sy], "theta":[th]}, "dataset")

            n =  len(DprimThis.Dat)
            if n>Nmin:
                primtuple = (PRIM, sx, sy, th)
                list_primtuple.append(primtuple)
                
    for prim_exclude in list_prims_exclude:
        list_primtuple = [this for this in list_primtuple if f"{prim_exclude}-" not in this[0]]

    # Collect list of days
    list_days = D.Dat["date"].unique()

    return list_primtuple, list_days



def plot_beh_day_vs_trial(D, prim_list, SAVEDIR, Nmin_toplot=5, max_cols=40):
    """ Plot raw behavior across days (rows) and trials (columns). 
    Useful for seeing progression over learning (rows).
    Each figure a unique scale and orientation
    INPUT:
        SAVEDIR, base dir. each prim is a subdir.
        Nmin_toplot = 5 # only ploit if > this many trials across all days.
        max_cols = 40 # max to plot in single day
    """
    from pythonlib.dataset.plots import plot_beh_grid_singletask_alltrials
    from itertools import product

    # pick out all trials for a combo of (scale, orientation)
    for PRIM in prim_list:

        SAVEDIR_THIS = f"{SAVEDIR}/{PRIM}"
        os.makedirs(SAVEDIR_THIS, exist_ok=True)
                
        list_sx, list_sy, list_theta = extract_list_features(D, PRIM)

        # list_sx = np.unique(D.Dat[D.Dat["character"]==PRIM]["sx"].to_list())
        # list_sy = np.unique(D.Dat[D.Dat["character"]==PRIM]["sy"].to_list())
        # list_theta = np.unique(D.Dat[D.Dat["character"]==PRIM]["theta"].to_list())
        
    # #     Dprim = D.filterPandas({"character":[PRIM]}, "dataset")
    # #     list_sx = np.unique(Dprim.Dat["sx"])
    # #     list_sy = np.unique(Dprim.Dat["sy"])
    # #     list_theta = np.unique(Dprim.Dat["theta"])
    #     print(list_sx)
    #     print(list_sy)
    #     print(list_theta)

        # For each combo of params, plot sample size
        for sx, sy, th in product(list_sx, list_sy, list_theta):
            DprimThis = D.filterPandas({"character":[PRIM], "sx":[sx], "sy":[sy], "theta":[th]}, "dataset")
            n =  len(DprimThis.Dat)

            if n>Nmin_toplot:
                print("** SAVING", sy, sy, th, "N: ", n)
                figb, figt = plot_beh_grid_singletask_alltrials(DprimThis, PRIM, "date", plotkwargs={"max_cols":max_cols})

                fname = f"sx{sx:.2f}_sy{sy:.2f}_th{th:.2f}"

                figb.savefig(f"{SAVEDIR_THIS}/{fname}-BEH.pdf")
                figt.savefig(f"{SAVEDIR_THIS}/{fname}-TASK.pdf")
            else:
                print("** SKIPPING (low n)", sy, sy, th, "N: ", n)

        plt.close("all")



def plot_all(D, SAVEDIR, animal, expt):
    """ Quick and dirty, all plots
    """

    SDIRTHIS = f"{SAVEDIR}/quantification_primitiveness"
    import os 
    os.makedirs(SDIRTHIS, exist_ok=True)
    print(SDIRTHIS)


    list_primtuple, list_days = extract_primtuple_list(D)

    # For each primtuple, check if has enough trials early and late

    def _hash(primtuple):
        return "_".join([str(p) for p in primtuple])

    #### Output is, for each primtuple, all the inds present on each date.
    DictDateInds = {}
    for primtuple in list_primtuple:

        DictDateInds[_hash(primtuple)] = {}
        print(primtuple)
        for day in list_days:
            # save the indices and dates for this primtuple
            inds = D.filterPandas({"date":[day], "character":[primtuple[0]], "sx":[primtuple[1]], 
                                            "sy":[primtuple[2]], "theta":[primtuple[3]]})

            DictDateInds[_hash(primtuple)][str(day)] = inds


    from pythonlib.drawmodel.strokedists import distscalarStrokes, distanceStroksMustBePaired
    def dfunc(strokes1, strokes2):
        def _pick_single_strok(strokes):
            # If strokes is len>1, then returns the longest strok
            # takes most pts as proxy (not realy longest)
            if len(strokes)==1:
                return strokes
            list_lens = [s.shape[0] for s in strokes]
            ind = np.argmax(list_lens)
            return [strokes[ind]]

        # if any are >1 strokes, just take the first
        strokes1 = _pick_single_strok(strokes1)
        strokes2 = _pick_single_strok(strokes2)

    #     return distanceStroksMustBePaired(strokes1, strokes2, ver="euclidian")
        return distanceStroksMustBePaired(strokes1, strokes2, ver="euclidian_diffs")



    #### For each day, compute distance across categories
    # THIS more flexible than above, allows each prim to have diff days which call early and late
    # Allows replicating above as well.

    nmin = 5
    DictDay = {}
    for day in list_days:
        list_ptuples = []
        for ptuple, val in DictDateInds.items():
            if len(val[day])>nmin:
                list_ptuples.append(ptuple)

        DictDay[day] = list_ptuples

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
        list_of_ptuples = [DictDay[str(d)] for d in list_days]
        return _intersect(list_of_ptuples)



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

    # save this dict
    from pythonlib.tools.expttools import writeDictToYaml, makeTimeStamp

    writeDictToYaml(DictDateEarlyLate, f"{SDIRTHIS}/DictDateEarlyLate.yaml")
    writeDictToYaml(DictDay, f"{SDIRTHIS}/DictDay.yaml")
    writeDictToYaml(DictDateInds, f"{SDIRTHIS}/DictDateInds.yaml")


    # compute across-category for early and late days
    list_ptuple_this = list(DictDateEarlyLate.keys())
    def _get_inds(ptuple, inddate):
        date = DictDateEarlyLate[ptuple][inddate]
        inds = DictDateInds[ptuple][date]
        assert len(inds)>0
        return inds

    SDIRTMP = f"{SAVEDIR}/ALL_PRIMS_COMBINED"
    os.makedirs(SDIRTMP, exist_ok=True)
    import random

    for early_or_late in ["early", "late"]:

        for i in range(5):

            if early_or_late=="early":
                inddate1 = 0 # first day
                inddate2 = 0 # last day
            elif early_or_late=="late":
                inddate1 = -1 # first day
                inddate2 = -1 # last day
            else:
                assert False

            # Collect one trial for each primimtive
            list_inds =[]
            for ptuple1 in list_ptuple_this:
                inds1 = _get_inds(ptuple1, inddate1)

                # get a random ind
                indthis = random.choice(inds1)
                list_inds.append(indthis)

            print("Got these inds, one per prim:", list_inds)
            figb = D.plotMultTrials(list_inds, titles=list_ptuple_this);
            figt = D.plotMultTrials(list_inds, "strokes_task");

            figb.savefig(f"{SDIRTMP}/prims_grid_iter-{early_or_late}-iter{i}-BEH.pdf")
            figt.savefig(f"{SDIRTMP}/prims_grid_iter-{early_or_late}-iter{i}-TASK.pdf")
        plt.close("all")

    OUT_cross = []
    Niter = 20
    for _ in range(Niter):

        for self_or_cross in ["self", "cross"]:
            for early_or_late in ["early", "late"]: # affects both prim1 and 2

                print(self_or_cross, early_or_late)

                if early_or_late=="early":
                    inddate1 = 0 # first day
                    inddate2 = 0 # last day
                elif early_or_late=="late":
                    inddate1 = -1 # first day
                    inddate2 = -1 # last day
                else:
                    assert False

                for ptuple1 in list_ptuple_this:
                    for ptuple2 in list_ptuple_this:

                        if self_or_cross=="cross":
                            # then want them diff
                            if ptuple1==ptuple2:
                                continue
                        elif self_or_cross=="self":
                            # then want same
                            if ptuple1!=ptuple2:
                                continue
                        else:
                            assert False

                        inds1 = _get_inds(ptuple1, inddate1)
                        inds2 = _get_inds(ptuple2, inddate2)

                        # Pick a single random trial for each, then compute distance
                        assert len(inds1)>1
                        assert len(inds2)>1

                        i1 = random.choice(inds1)
                        i2 = random.choice(inds2)
                        while i1==i2:
                            i1 = random.choice(inds1)
                            i2 = random.choice(inds2)                

                        strokes1 = D.Dat.iloc[i1]["strokes_beh"]
                        strokes2 = D.Dat.iloc[i2]["strokes_beh"]

                        dist = dfunc(strokes1, strokes2)

                        OUT_cross.append({
                            "ptuple1":ptuple1,
                            "ptuple2":ptuple2,
                            "dist":dist,
                            "inddate1":inddate1,
                            "inddate2":inddate2,
                            "early_or_late":early_or_late,
                            "self_or_cross":self_or_cross
                        })


    import pandas as pd
    import seaborn as sns
    DF_cross = pd.DataFrame(OUT_cross)

    fig = sns.catplot(data=DF_cross, x="early_or_late", row="ptuple1", col="self_or_cross", y="dist", kind="point", ci=68)
    # sns.catplot(data=DF_cross, x="early_or_late", hue="self_or_cross", y="dist")
    fig.savefig(f"{SDIRTHIS}/eachprim_overview_selfandcross.pdf")

    # ## aggregate, so that each unique primitive gets one value for self and one for cross.
    from pythonlib.tools.pandastools import summarize_feature, summarize_featurediff
    # dfagg, dfaggflat = summarize_feature(DF_cross, GROUPING=["self_or_cross", "early_or_late"], FEATURE_NAMES=["dist"], INDEX=["ptuple1"])
    # # dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF = 

    # sns.catplot(data=dfagg, x="early_or_late", col="self_or_cross", y="dist")

    #### SUMMARY PLOST

    dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF, dfpivot = summarize_featurediff(DF_cross, GROUPING="early_or_late", GROUPING_LEVELS=["early", "late"], 
                          FEATURE_NAMES=["dist"], INDEX=["ptuple1", "self_or_cross"], return_dfpivot=True)

    # PRint results, to compare by eye to raw datya
    # for row in dfsummary.iterrows():
    #     print([row[1]["ptuple1"], row[1]["self_or_cross"], row[1]["dist-lateminearly"], DictDateEarlyLate[row[1]["ptuple1"]]])

    fig = sns.catplot(data=dfsummaryflat, x="self_or_cross", row="variable", y="value")
    fig.savefig(f"{SDIRTHIS}/summary_eachprim_crossvsself_1.pdf")
    fig = sns.catplot(data=dfsummaryflat, x="self_or_cross", row="variable", y="value", kind="bar", ci=68)
    fig.savefig(f"{SDIRTHIS}/summary_eachprim_crossvsself_2..pdf")

    # from pythonlib.tools.snstools import relplotOverlaid, relPlotOverlayLineScatter

    # relPlotOverlayLineScatter(dfsummaryflat, x="self_or_cross", y="value", hue="ptuple1",
    #                          height=3, aspect=2)

    # relplotOverlaid(dfsummaryflat, line_category="ptuple1", color="k", x="self_or_cross", y="value")

    #### CATEGORIZATION SCORE (KIND OF LIKE R^2)
    # For each "early" and "late", take ratio of within-cat distances to across-category distances
    _, _, _, _, _, dfpivotthis = summarize_featurediff(DF_cross, GROUPING="self_or_cross", GROUPING_LEVELS=["self", "cross"], 
                          FEATURE_NAMES=["dist"], INDEX=["ptuple1", "early_or_late"], return_dfpivot=True)

    # compute self/cross
    dfpivotthis["self_div_cross"] = dfpivotthis["dist"]["self"] / dfpivotthis["dist"]["cross"]

    # Melt pivot into long form
    dfthis = pd.melt(dfpivotthis, id_vars = ["ptuple1", "early_or_late"], var_name="variable", value_name="value")
    # dfthis = dfthis[dfthis["test"]=="self_div_cross"].reset_index(drop=True)

    import seaborn as sns
    fig = sns.catplot(data=dfthis, x="early_or_late", y="value",  kind="bar", col="variable", ci=68)
    fig.savefig(f"{SDIRTHIS}/selfdist_div_crossdist-1.pdf")
    fig = sns.catplot(data=dfthis, x="early_or_late", y="value",  kind="point", col="variable", ci=68)
    fig.savefig(f"{SDIRTHIS}/selfdist_div_crossdist-2.pdf")
    fig = sns.catplot(data=dfthis, x="early_or_late", y="value",  col="variable")
    fig.savefig(f"{SDIRTHIS}/selfdist_div_crossdist-3.pdf")


    ##### Visual accuracy improve for primtiives?

    # score everuthing
    D.score_visual_distance()

    # compute across-category for early and late days
    list_ptuple_this = DictDateEarlyLate.keys()

    def _get_inds(ptuple, inddate):
        date = DictDateEarlyLate[ptuple][inddate]
        inds = DictDateInds[ptuple][date]
        assert len(inds)>0
        return inds

    OUT_visual = []
    for early_or_late in ["early", "late"]: # affects both prim1 and 2

        print(early_or_late)

        if early_or_late=="early":
            inddate1 = 0 # first day
        elif early_or_late=="late":
            inddate1 = -1 # first day
        else:
            assert False

        for ptuple1 in list_ptuple_this:

            inds1 = _get_inds(ptuple1, inddate1)

            list_dist = D.Dat.iloc[inds1]["hdoffline"].to_list()
            for dist in list_dist:

                OUT_visual.append({
                    "ptuple1":ptuple1,
                    "dist":dist,
                    "early_or_late":early_or_late,
                })


    DF_visual = pd.DataFrame(OUT_visual)


    dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF, dfpivot = \
        summarize_featurediff(DF_visual, GROUPING="early_or_late", GROUPING_LEVELS=["early", "late"], 
                          FEATURE_NAMES=["dist"], INDEX=["ptuple1"], return_dfpivot=True)

    # dfagg, dfaggflat = summarize_feature(DF_visual, GROUPING="early_or_late", FEATURE_NAMES=["dist"], 
    #                   INDEX=["ptuple1"])


    # PRint results, to compare by eye to raw datya
    # for row in dfsummary.iterrows():
    #     print([row[1]["ptuple1"], row[1]["self_or_cross"], row[1]["dist-lateminearly"], DictDateEarlyLate[row[1]["ptuple1"]]])

    fig = sns.catplot(data=dfsummaryflat, x="variable", y="value")
    fig.savefig(f"{SDIRTHIS}/visual_error_eachprim.pdf")


    # sns.lineplot(data=DF_visual, x="early_or_late", y="dist", hue="ptuple1")    

