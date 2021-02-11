""" for seaborn plotting"""
import seaborn as sns

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
        ax.text(df[x][line], df[y][line], df[name][line], 
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
