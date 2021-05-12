""" plots, takles in dataframe, Dataset().Dat,
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..drawmodel.strokePlots import plotDatStrokes
from pythonlib.tools.pandastools import applyFunctionToAllRows

def plot_dat_grid_inputrowscols(df, strokes_ver="strokes_beh", max_n_per_grid=None,
    col_labels = None, row_labels=None, strokes_by_order=False, plotfuncbeh=None):
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


    plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_unordered=True, naked=True)
    figtask = plotGridWrapper(strokestasklist, plotfunc, collist, rowlist, origin="top_left", max_n_per_grid=max_n_per_grid,
        col_labels = col_labels, row_labels=row_labels)

    return figbeh, figtask


def plot_beh_grid_grouping_vs_task(df, row_variable, tasklist, row_levels=None, plotkwargs = {},
    plotfuncbeh=None):
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
    tasklist_titles = [t[:18] + "..." for t in tasklist]
    # Plot
    figbeh, figtask = plot_dat_grid_inputrowscols(dfthis, max_n_per_grid=1, 
        col_labels = tasklist_titles, row_labels=row_levels, plotfuncbeh=plotfuncbeh, 
        **plotkwargs)

    return figbeh, figtask
