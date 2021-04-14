""" plots, takles in dataframe, Dataset().Dat,
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..drawmodel.strokePlots import plotDatStrokes

def plot_dat_grid_inputrowscols(df, strokes_ver="strokes_beh", max_n_per_grid=None):
    """ in each grid position, plots a single trial; (e.. strokes)
    df must have two columns, row and col, which indicate where to plot each
    trial.
    by default, top-left is 0,0
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

    plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_ordered=True, alpha=0.4, add_stroke_number=False)
    figbeh = plotGridWrapper(strokeslist, plotfunc, collist, rowlist, origin="top_left", max_n_per_grid=max_n_per_grid)


    plotfunc = lambda strokes, ax: plotDatStrokes(strokes, ax, clean_unordered=True)
    figtask = plotGridWrapper(strokestasklist, plotfunc, collist, rowlist, origin="top_left", max_n_per_grid=max_n_per_grid)

    return figbeh, figtask
