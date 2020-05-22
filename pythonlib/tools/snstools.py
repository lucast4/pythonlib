""" for seaborn plotting"""


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
