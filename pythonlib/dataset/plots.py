""" plots, takles in dataframe, Dataset().Dat,
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..drawmodel.strokePlots import plotDatStrokes
from pythonlib.tools.pandastools import applyFunctionToAllRows

def plot_dat_grid_inputrowscols(df, strokes_ver="strokes_beh", max_n_per_grid=None,
    col_labels = None, row_labels=None, strokes_by_order=False, plotfuncbeh=None, 
    max_cols = 40, max_rows=40):
    """ in each grid position, plots a single trial; (e.. strokes)
    df must have two columns, row and col, which indicate where to plot each
    trial.
    by default, top-left is 0,0
    INPUTS:
    - strokes_by_order, then colors strokes by ordinal (from blue to pink, based on "cool")
    RETURNS:
    - figbeh, figtask
    NOTES:
    - plots both strokes_ver and "strokes_task"
    """


    from pythonlib.tools.plottools import plotGridWrapper

    # only keep trials that are within the col range
    a = len(df)
    df = df[df["col"]<max_cols]
    df = df[df["row"]<max_rows].reset_index(drop=True)
    if len(df)<a:
        print("Old len(df):", a)
        print("New, removing if too many col or row:", len(df))

    # extract data to plot
    strokeslist = df[strokes_ver].values
    strokestasklist = df["strokes_task"].values
    rowlist = df["row"].values
    collist = df["col"].values

    if plotfuncbeh is None:
        if strokes_by_order:
            plotfuncbeh = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_ordered_ordinal=True, alpha=0.4, add_stroke_number=False, 
                force_onsets_same_col_as_strokes=True, naked=True)
        else:
            plotfuncbeh = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_ordered=True, alpha=0.4, add_stroke_number=False,
                force_onsets_same_col_as_strokes=True, naked=True)
        
    figbeh = plotGridWrapper(strokeslist, plotfuncbeh, collist, rowlist, origin="top_left", max_n_per_grid=max_n_per_grid,
        col_labels = col_labels, row_labels=row_labels)


    # plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_unordered=True, naked=True)
    plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_task=True, naked=True)
    figtask = plotGridWrapper(strokestasklist, plotfunc, collist, rowlist, origin="top_left", max_n_per_grid=max_n_per_grid,
        col_labels = col_labels, row_labels=row_labels)

    return figbeh, figtask

############## HELPERS THAT CALL plot_dat_grid_inputrowscols
def plot_beh_grid_grouping_vs_task(df, row_variable, tasklist, row_levels=None, plotkwargs = {},
    plotfuncbeh=None, max_n_per_grid=1):
    """
    Helper (USEFUL) for plotting 2d grid of beh (strokes), with columns as unique tasks, and
    rows as flexible grouping variable.
    INPUTS:
    - df (from D.Dat)
    - row_variable, str that is a col name in df, which will dictate row for each datapont.
    - row_levels, list of levels of row)variable. Optional. if pass in, then will only
    have this many rows (each corresponding to one level). Otherwise will have as many rows
    as there are unique levels
    RETURNS:
    - figbeh, figtask
    """
    from pythonlib.dataset.plots import plot_dat_grid_inputrowscols
    dfthis = df[df["character"].isin(tasklist)]

    # get levels for rows
    row_levels = sorted(dfthis[row_variable].unique().tolist())
    # print("row levels:")
    # print(row_levels)
    assert len(row_levels)<40, "40+ row levesls, are you sure"

    # Row mapper
    print("Row levels, mapped to column indices:")
    row_mapper_ = {row_levels[i]:i for i in range(len(row_levels))}
    print(row_mapper_)
    def row_mapper(x):
        """ x is a df row --> index which row to plot"""
        return row_mapper_[x[row_variable]]
    def col_mapper(x):
        tmp = [i for i, t in enumerate(tasklist) if t==x["character"]]
        assert len(tmp)==1
        return tmp[0]

    # Assign row and col varibles
    dfthis = applyFunctionToAllRows(dfthis, row_mapper, "row")
    dfthis = applyFunctionToAllRows(dfthis, col_mapper, "col")


    # tasknames too long, so prune for plotting
    tasklist_titles = [t[:10] + ".." + t[-5:] for t in tasklist]
    # Plot
    figbeh, figtask = plot_dat_grid_inputrowscols(dfthis, max_n_per_grid=max_n_per_grid, 
        col_labels = tasklist_titles, row_labels=row_levels, plotfuncbeh=plotfuncbeh, 
        **plotkwargs)

    return figbeh, figtask


def plot_beh_grid_singletask_alltrials(D, task, row_variable, row_levels=None, plotkwargs = {},
    plotfuncbeh=None):
    """ given a singel task, plots all trials in df in a grid, where rows split into row_levels (grouped by row_variable)
    and columns are trials, in order encountered in df.
    INPUTS:
    - df, usually make this D.Dat, or related
    """

    # df = D.Dat[D.Dat["character"]==task]
    import pandas as pd

    dfplot, row_levels = D.analy_singletask_df(task, row_variable, row_levels=row_levels, return_row_levels=True)

    # df = df[df["character"] == task]

    # # what are rows
    # if row_levels is None:
    #     row_levels = sorted(df[row_variable].unique().tolist())

    # # Skip if dont have data across all rows.
    # if not all([l in df[row_variable].unique().tolist() for l in row_levels]):
    #     print("SKIPPING, since not have data across levels:", task)
    #     return

    # out = []
    # for i, lev in enumerate(row_levels):
    #     dfthis = df[df[row_variable]==lev]
    #     dfthis = dfthis.reset_index(drop=True)
    #     dfthis["col"] = dfthis.index.tolist()
    #     dfthis["row"] = i
    #     out.append(dfthis)

    # dfplot = pd.concat(out)

    # PLOT
    figb, figt = plot_dat_grid_inputrowscols(dfplot, row_labels=row_levels, plotfuncbeh=None, **plotkwargs)

    return figb, figt

def plot_beh_waterfall_singletask_alltrials(D, task, row_variable, row_levels=None, plotkwargs = {},
    plotfuncbeh=None):
    """ given a singel task, plots all trials in df in a waterfall, where organizes them based on row_levels, in order encountered in df.
    this will dictate order, and label in plot.
    INPUTS:
    - df, usually make this D.Dat, or related
    """
    from pythonlib.drawmodel.strokePlots import plotDatWaterfallWrapper
    from pythonlib.tools.stroketools import strokesVelocity

    dfplot, row_levels = D.analy_singletask_df(task, row_variable, row_levels=row_levels, return_row_levels=True)

    # PLOT WATERFALL
    strokes_list = dfplot["strokes_beh"].values
    onsets_list = dfplot["go_cue"].values
    row_list = dfplot["row"].values
    row_list = [row_levels[i] for i in row_list]
    col_list = dfplot["col"].values
    labels = [f"c_{c}|r_{r}" for c,r in zip(col_list, row_list)]

    # Get velocities
    # TODO: assuming that fs for first params will apply across all.
    fs = D.Metadats[0]["filedata_params"]["sample_rate"]
    strokespeed_list = [strokesVelocity(strokes, fs, clean=True)[1] for strokes in strokes_list]
    strokespeed_list = [[ss[:,0] for ss in s] for s in strokespeed_list] # only keep the speed, not the time.

    n = len(strokes_list)
    fig, ax = plt.subplots(1, 1, figsize=(n*1.5, 5))
    # plotDatWaterfallWrapper(strokes_list, onset_time_list=onsets_list, ax=ax, ylabels=labels)
    plotDatWaterfallWrapper(strokes_list, onset_time_list=onsets_list, strokes_ypos_list=strokespeed_list, ax=ax, ylabels=labels)
    return fig


def plot_timecourse_overlaid(D, features_list, xval="tvalfake", grouping=None, doscatter=True, domean=True):
    """
    Plot timecourse, (feature vs. tval) to summarize an expt. Separate 
    columns (character) and rows (taskgroup) and colors (epochs).
    Overlays indivitual trials and within day mean. 
    - xval, string, for what to use on x axis:
    --- "tvalfake", then is time
    --- "epoch" then is grouping across trials, based on epoch.
    - features_list, list of strings, columns of D.Dat, where values are scalars
    - grouping, string, where defines levels (for colored epochs)
    NOTE:
    - this used in model expt timecurse analyses
    """
    from pythonlib.tools.snstools import timecourse_overlaid
    DF = D.Dat
    # YLIM = [0, 6]
    row = "task_stagecategory"
    col = "taskgroup"
    figlist = []
    for f in features_list:
        # feat = f"FEAT_{f}"
        feat = f
        fig = timecourse_overlaid(DF, feat, xval=xval, row=row, col=col, grouping=grouping,
         doscatter=doscatter, domean=domean)
        figlist.append(fig)
    return figlist
