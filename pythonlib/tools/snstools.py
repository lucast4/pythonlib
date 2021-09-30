""" for seaborn plotting"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def rotateLabel(ax, rotation=45, horizontalalignment="right"):
    """ seaborn, maek sure to add labels for catplot
    ax = sns.catplot(...)
    """
    for a in ax.axes.flat:
        a.set_xticklabels(a.get_xticklabels(), rotation=rotation, 
            horizontalalignment=horizontalalignment)


def addLabel(ax):
    """ seaborn, maek sure to add labels for catplot"""
    for a in ax.axes.flat:
        a.set_xticklabels(a.get_xticklabels(), rotation=45, 
            horizontalalignment="right")


def addTextLabelToPoints(ax, df, x, y, name):
    """ 
    adds on top of points labels in text.
    e.g., ax = sns.scatterplot(x="x", y="y", hue="category", data=df, hue_order=sorted(list(set(tasks_all_categories))))
    addTextLabelToPoints(ax, df, "x", "y", "name"),
    i.e,., tell me the x, y, and text strings to use 
    """
    
    # add annotations one by one with a loop
    for line in range(df.shape[0]):
        ax.text(df[x].values[line], df[y].values[line], df[name].values[line], 
                 horizontalalignment='left', size='medium', color='black',
                alpha=0.6)

def relplotOverlaid(df, line_category, color, **relplotkwargs):
    """ if want to plot single lines for each cataegory in 
    line_category, and all the same color. sns I think forces you to 
    either average over all categories in line_category (i.e., one
    output per facet, or hue, etc, or they will be different colors if
    use hue=line_category. Here can make all the same color. 
    """
    # print(relplotkwargs)
    # assert False
    catlist = set(df[line_category])
    palette = {n:color for n in catlist}
    relplotkwargs["legend"]= False # since all same color, legend is useulse..
    relplotkwargs["palette"]= palette
    relplotkwargs["data"] = df
    return sns.relplot(**relplotkwargs)


def relPlotOverlayLineScatter(data, x, y, hue=None, row=None, col=None, palette=None,
    height=3, aspect=3):
    """
    row="block"
    """

    # g = sns.FacetGrid(data, row=row, sharex=True, sharey=True, height=3, aspect=3,
    #                  legend_out=True)
    g = sns.FacetGrid(data, row=row, hue=hue, col=col, sharex=True, sharey=True, height=height, aspect=aspect,
                     legend_out=True)

    catlist = set(data[hue])
    palette = {n:"k" for n in catlist}

    # print(x)
    # print(y)
    # print(hue)
    # print(palette)
    # assert False
    # g = g.map(sns.scatterplot, x, y, hue=hue, palette=palette)
    # g = g.map(sns.lineplot, x, y, hue=hue, palette=palette, legend="full")
    g = g.map(sns.scatterplot, x, y, palette=palette)
    g = g.map(sns.lineplot, x, y,  palette=palette, legend="full")
    return g


def pairplot_corrcoeff(data, x_vars=None, y_vars = None, hue=None, vars=None, aspect=1,
    corrver="spearman"):
    """ like sns.pairplot, but overlaying pearsons r and p.
    """
    from scipy import stats
    def corrfunc(x, y, **kws):
        if corrver=="pearson":
            r, p = stats.pearsonr(x, y)
        elif corrver=="spearman":
            r, p = stats.spearmanr(x, y)
        else:
            assert False

        ax = plt.gca()
        ax.annotate(f"r={r:.2f}|p={p:.4f}", xy=(.1, .9), xycoords=ax.transAxes)

    g = sns.pairplot(data=data, x_vars = x_vars, y_vars = y_vars, aspect=aspect,
                      kind="reg", hue=hue)
    # g.map_upper(plt.scatter, s=10)
    # g.map_diag(sns.distplot, kde=False)
    # g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map(corrfunc)
    return g

def map_function_tofacet(fig, func):
    """
    fig is a facetgrid plot, e.g., catplot
    func(ax), does stuff with ax,applies to each facet
    """

    def F(*pargs, **kws):
        ax = plt.gca()
        func(ax)
        # ax.axhline(0, alpha=0.2, **kws)

    fig.map(F)


def timecourse_overlaid(df, feat, xval="tvalfake", YLIM=None, row=None, col=None, grouping=None,
    ALPHA = 0.5, doscatter=True, domean=True, squeeze_ylim=True):
    """ plot timecourse, both scatter of pts and overlay means"""
    
    if grouping:
        tmp = df[grouping].unique().tolist()
        colors = ["r", "b", "g", "k"]
        if len(tmp)<=len(colors):
            PALLETE = {t:c for t, c in zip(tmp, colors)}
        else:
            PALLETE=None
    else:
        PALLETE = None

    if YLIM is None and squeeze_ylim and doscatter:
        from .plottools import get_ylim
        YLIM = get_ylim(df[feat])

    g = sns.FacetGrid(df, row=row, col=col, height=4, aspect=2, 
                      sharex=True, sharey=True, ylim=YLIM)

    # Different plots, depening on if timecourse, or summaries.
    if xval=="tvalfake":
        xvalmean = "tvalday"
        scatplot = sns.scatterplot
        meanplot = sns.lineplot
    elif xval=="epoch":
        xvalmean = "epoch"
        scatplot = sns.swarmplot
        meanplot = sns.pointplot
    else:
        print(xval)
        assert False, "not sure what is mean summary for this xval"

    if domean:
        g.map(meanplot, xvalmean, feat, **{"err_style":"bars", "ci":68, "color":"k", "linewidth":2})
    if doscatter:
        if xval=="tvalfake":
            g.map(sns.scatterplot, xval, feat, "epoch", **{"marker":"x", 
                                                          "alpha":ALPHA,
                                                                  "s":40, 
                                                                  "palette":PALLETE})
        elif xval=="epoch":
            g.map(sns.swarmplot, xval, feat, "epoch", **{"alpha":ALPHA,
                                                                  "s":4, 
                                                                  "palette":PALLETE})
    g.map(plt.axhline, **{"color":[0.7, 0.7, 0.7]})

    return g
