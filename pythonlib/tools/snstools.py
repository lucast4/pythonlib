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
