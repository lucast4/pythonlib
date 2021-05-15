""" for seaborn plotting"""
import seaborn as sns
import matplotlib.pyplot as plt

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
    catlist = set(df[line_category])
    palette = {n:color for n in catlist}
    relplotkwargs["legend"]= False # since all same color, legend is useulse..
    relplotkwargs["palette"]= palette
    return sns.relplot(**relplotkwargs)


def relPlotOverlayLineScatter(data, x, y, hue=None, row=None, col=None, palette=None):
    g = sns.FacetGrid(data, row="block", sharex=True, sharey=True, height=3, aspect=3,
                     legend_out=True)
    g = g.map(sns.scatterplot, x, y, hue, palette=palette)
    g = g.map(sns.lineplot, x, y, hue, palette=palette, legend="full")
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