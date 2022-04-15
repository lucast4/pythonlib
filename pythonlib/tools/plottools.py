import numpy as np
import matplotlib.pyplot as plt

def radar_plot(ax, thetas, values, color="k", fill=True):
    """ polar radar plot 
    assumes that theta[-1] immedaitely precedes theta[0]. will connect them
    assumes that thetas are evenly spaced 
    assumes that ax craeted using subplot_kw=dict(projection='polar')"""
    
    values = np.concatenate([values, values[0].reshape(1,)])
    thetas = np.concatenate([thetas, thetas[0].reshape(1,)])
    
    width = np.mean(np.diff(thetas[:-1]))
    thetas = thetas + width/2
    ax.plot(thetas, values, '-k', color=color, zorder=3)
    if fill:
        ax.fill(thetas, values, '-k', color=color, alpha=0.5)

    
def rose_plot(ax, angles, bins=16, density="frequency_radius", offset=0, lab_unit="degrees",
              start_zero=True, fill=False, skip_plot=False, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    from: https://stackoverflow.com/questions/22562364/circular-histogram-for-python
    """
    # Wrap angles to [-pi, pi)
    angles = (angles + np.pi) % (2*np.pi) - np.pi

    # Set bins symetrically around zero
    if start_zero:
        # To have a bin edge at zero use an even number of bins
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    count, bin = np.histogram(angles, bins=bins)

    # Compute width of each bin
    widths = np.diff(bin)

    # By default plot density (frequency potentially misleading)
    if density=="proportion_area":
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / np.pi)**.5
    elif density=="frequency_area":
#         count = count/np.sum(count)
        radius = (count)**.5
    elif density=="proportion_radius":
        radius = count / angles.size
    elif density=="frequency_radius":
#         count = count/np.sum(count)
        radius = count
    else:
        assert False, "not coded"

    if not skip_plot:
        ####################### PLOTS
        # Plot data on ax
        if not fill:
            ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
                   edgecolor='k', fill=fill, linewidth=1)
        else:
            ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
               edgecolor='k', color=[0.7, 0.7, 0.7], fill=fill, linewidth=1)
        # Set the direction of the zero angle
        ax.set_theta_offset(offset)

        # Remove ylabels, they are mostly obstructive and not informative
        ax.set_yticks([])

        # -- plot each line?
        ax.plot(angles, 0.9*np.max(radius)*np.ones(angles.size), 'xk')
        # c = np.vstack([count, count[0]])
        # print(bins)

        ax.set_thetagrids([])

        if False:
            # works, but doesn't look great unless havea lotns of data. 
            if start_zero:
                r = np.concatenate([radius, radius[0].reshape(1,)])
                # b = bins[:-1] + widths/2
                # b = np.concatenate([b, bins[-1].reshape(1,)])
                ax.plot(bins+np.mean(widths)/2, r, '-k', color='k')
                ax.fill(bins+np.mean(widths)/2, r, color='k')

        if lab_unit == "radians":
            label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                      r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
            ax.set_xticklabels(label)
        elif lab_unit is None:
            ax.set_xticklabels([])
        
    return radius, bins
#     # Fiddle with labels and limits
#     polar_ax.set_xticks([0, np.pi/4, 2*np.pi - np.pi/4])
#     polar_ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$-\pi/4$'])
#     polar_ax.set_rlabel_position(90)


def annotate(s, ax=None, color="k"):
    """ puts on top left corner"""
    if ax is None:
        plt.annotate(s, (0.05, 0.9), color=color, size=12, xycoords="axes fraction")
    else:
        ax.annotate(s, (0.05, 0.9), color=color, size=12, xycoords="axes fraction")


def makeColors(numcol, alpha=1, cmap="plasma"):
    """ gets evensly spaced colors. currntly uses plasma map"""
    import matplotlib.pylab as pl
    import matplotlib.cm as cm
    if True:
        pcols = getattr(cm, cmap)(np.linspace(0,1, numcol), alpha=alpha)
    else:
        if cmap=="plasma":
            pcols = pl.cm.plasma(np.linspace(0,1, numcol), alpha=alpha)
        elif cmap=="jet":
            pcols = pl.cm.jet(np.linspace(0,1, numcol), alpha=alpha)
        else:
            print(cmaps)
            assert False, "dont know this CMAP"
    # cool
    return pcols

def colorGradient(pos, col1=None, col2=None, cmap="plasma"):
    """ pos between 0 and 1, gets a value between gradient btw
    col1 and col2. altenratively if leave both col1 and 2 None, 
    then will use camp, if pos <0 or >1 wil clamp to this range"""

    if pos<0:
        pos=0.
    elif pos>1:
        pos =1.

    if col1 is None and col2 is None:
        pcols = makeColors(2, cmap=cmap)
        col1=pcols[0]
        col2 = pcols[1]
    elif col1 is None or col2 is None:
        assert False, "either give me both col1 and 2, or neither"
    else:
        col1=np.array(col1)
        col2 = np.array(col2)
    return (pos*(col2-col1) + col1)[:3]



def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties

    FROM: https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def get_correlated_dataset(n, dependency, mu, scale):
    """ see above, confidence_ellipse (helper function for that)"""
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


def saveMultToPDF(path, figs):
    """ saved multiple figs (list of fig obejcts)
    to path as pdf. path should not have pdf in the name"""
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(f"{path}.pdf")
    for fig in figs:
        pdf.savefig(fig, dpi=100, bbox_inches='tight')
    pdf.close()

def shadedErrorBar(x, y, yerr=None, ylowupp = None, ax=None):
    """ Draw plot with error band and extra formatting to match seaborn style
    - pass eitehr yerr (symmetric) or ylowupp, a list of [yupper, ylower], or neither,
    if no plot error
    from https://stackoverflow.com/questions/56203420/how-to-use-custom-error-bar-in-seaborn-lineplot
    """
    a = yerr is not None
    b = ylowupp is not None
    if a or b:
        assert a!=b, "only guive me one of them"

    if a:
        lower = y - yerr
        upper = y + yerr
    elif b:
        lower, upper = ylowupp
    else:
        lower, upper = None, None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(9,5))

    ax.plot(x, y, label='')
    if a or b:
        ax.plot(x, lower, color='tab:blue', alpha=0.1)
        ax.plot(x, upper, color='tab:blue', alpha=0.1)
        ax.fill_between(x, lower, upper, alpha=0.2)
    # ax.set_xlabel('timepoint')
    # ax.set_ylabel('signal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.show()
    return ax


def plotScatterXreduced(X, dims_to_take, nplot = 20, ax=None, 
                       color="k", textcolor="r", alpha=0.05,
                       plot_text_over_examples=False, return_inds_text=False):
    """ 
    Scatter plot of X, picking out 2 dimensions. useful is X is output after
    dim reduction. 
    INPUT:
    - X, array, shape N x D, where N is n samp, D is dim.
    - dims_to_take, list, which 2 dims to take, e.g., [0, 1]
    """
    import random
    assert len(dims_to_take)==2

    Xfit = X[:,dims_to_take]

    # 1) Scatter plot all trials
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,15))
    else:
        fig = None
    ALPHA = Xfit.shape[0]/500
    ax.plot(Xfit[:,0], Xfit[:,1], "o", color=color, alpha=alpha)

    # === pick out random indices, highlight them in the plot, and plot them
    if plot_text_over_examples:
        indsrand = random.sample(range(Xfit.shape[0]),nplot)
        indsrand = sorted(indsrand)

        for i in indsrand:
            ax.text(Xfit[i,0], Xfit[i,1], i, color=textcolor, fontsize=10)
    else:
        indsrand = None

    if return_inds_text:
        return fig, ax, indsrand
    else:
        return fig, ax


def plotScatterOverlay(X, labels, dimsplot=[0,1], alpha=0.2, ver="overlay",
    downsample_auto=True):
    """ overlay multiple datasets on top of each other
    or separate.
    - X, array shape NxD.
    - labels, vector length N, with label for each sample. 
    Will color differnetly by label.
    - downsample_auto, then subsamples in case there are too many datapts
    """
    
    if downsample_auto:
        import random
        thresh = 20000
        nthis = X.shape[0]
        if nthis> thresh:
            indsthis = sorted(random.sample(range(X.shape[0]), thresh))
            print(f"Randomly subsampling to {thresh}")
            X = X[indsthis,:]
            labels = [labels[i] for i in indsthis]
    # Color the labels
    from pythonlib.tools.plottools import makeColors
    labellist = set(labels)
    pcols = makeColors(len(labellist), cmap="rainbow")

    # Plot
    if ver=="overlay":
        for i, l in enumerate(labellist):
            inds = [i for i, lab in enumerate(labels) if lab==l]
            Xthis = X[inds]
            if i==0:
                # initiate a plot
                fig, ax = plotScatterXreduced(Xthis, dimsplot, ax=None,
                                              color=pcols[i], textcolor=pcols[i], alpha=alpha)
            else:
                plotScatterXreduced(Xthis, dimsplot, ax=ax,
                                              color=pcols[i], textcolor=pcols[i], alpha=alpha)
        ax.legend(labellist)

    elif ver=="separate":
        fig, axes = plt.subplots(len(labellist), 1, figsize=(8, 8*len(labellist)))
        for i, (l, ax) in enumerate(zip(labellist, axes.flatten())):
            inds = [i for i, lab in enumerate(labels) if lab==l]
            Xthis = X[inds]
            # initiate a plot
            plotScatterXreduced(X, dimsplot, ax=ax,
                                          color="k", textcolor="k", alpha=alpha/5)
            plotScatterXreduced(Xthis, dimsplot, ax=ax,
                                          color=pcols[i], textcolor=pcols[i], alpha=alpha)
            ax.set_title(f"label: {l}")

    elif ver=="separate_no_background":
        fig, axes = plt.subplots(len(labellist), 1, figsize=(8, 8*len(labellist)))
        if len(labellist)==1:
            axes =  [axes]
        
        for i, (l, ax) in enumerate(zip(labellist, axes)):
            inds = [i for i, lab in enumerate(labels) if lab==l]
            Xthis = X[inds]
            # initiate a plot
#             plotScatter(X, None, dimsplot, ax=ax,
#                                           color="k", textcolor="k", alpha=alpha/5)
            plotScatterXreduced(Xthis, dimsplot, ax=ax,
                                          color=pcols[i], textcolor=pcols[i], alpha=alpha)
            ax.set_title(f"label: {l}")        
    else:
        print(ver)
        assert False
        
    return fig, ax

def plotScatter45(x, y, ax, plot_string_ind=False, dotted_lines="unity", 
    means=False, labels=None, alpha=0.8):
    """ scatter plot, but making sure is square, 
    xlim and ylim are identical, and plotting a unity line
    - plot_string_ind, THen plots 0, 1, 2..., on the pts
    - dotted_lines, {"unity", "plus", "none"}, to verlay
    - means, overlay dot for mean
    - labels, list, same len as x and y, to overlay on dots (text), takes 
    precedence over plot_string_ind
    """

    ax.plot(x, y, "x", alpha=alpha)

    # ax.set_axis("square")
    minimum = np.min((ax.get_xlim(),ax.get_ylim()))
    maximum = np.max((ax.get_xlim(),ax.get_ylim()))
    ran = maximum - minimum
    MIN = minimum - 0.1*ran
    MAX = maximum + 0.1*ran
    if dotted_lines=="unity":
        ax.plot([MIN, MAX], [MIN, MAX], '--k', alpha=0.5)
    elif dotted_lines=="plus":
        ax.plot([0, 0], [MIN, MAX], '--k', alpha=0.5)
        ax.plot([MIN, MAX], [0, 0], '--k', alpha=0.5)
    else:
        assert dotted_lines=="none", "what you want?"

    if means:
        xmean = np.mean(x)
        ymean = np.mean(y)
        ax.plot(xmean, ymean, "ok")

    # overlay strings?
    if labels is not None:
        assert len(labels)==len(x)
        for l, xx, yy in zip(labels, x,y):
            ax.text(xx, yy, l)
    else:

        if plot_string_ind:
            for i, (xx, yy) in enumerate(zip(x,y)):
                ax.text(xx, yy, i)



    ax.set_xlim(MIN, MAX)
    ax.set_ylim(MIN, MAX)
    ax.axis("square")
    # ax.set_aspect('equal', adjustable='datalim')

def hist_with_means(ax, vals, **kwargs):
    """ same, but overlays line for mean
    """
    if len(np.unique(vals))==1:
        print(vals)
        assert False, "need variety"
    from scipy.stats import sem
    tmp = ax.hist(vals, **kwargs)
    bins = tmp[1]
    vmean = np.mean(vals)
    vsem = sem(vals)
    ax.axvline(vmean, color="k")
    ax.axvline(vmean-vsem, color="r")
    ax.axvline(vmean+vsem, color="r")
    return bins


def histogramMult(vals_list, nbins, ax=None, separate_plots=False):
    """ overlays multiple historgrams, making sure to 
    use same bins. first finds bins by taking all values and getting 
    common bins.
    Overlays using steps, using standard mpl colors
    INPUT:
    - vals_list, list of vals, each a (N,) array
    RETURNS:
    - fig
    """

    # Find common bins
    allvals = np.concatenate([x[:] for x in vals_list])
    assert not any(np.isnan(allvals))
    assert len(allvals.shape)==1
    bins = getHistBinEdges(allvals, nbins)
    if separate_plots:
        nc = 3
        nr = int(np.ceil(len(vals_list)/3))
        fig, axes= plt.subplots(nr, nc, figsize=(3*nc, 2*nr))
        for x, ax in zip(vals_list, axes.flatten()):
            ax.hist(x, bins=bins, histtype="step", density=True)
            # print(x)
            # print(bins)
    else:
        if ax is None:
            nc = 1
            nr = 1
            fig, ax= plt.subplots(nr, nc, figsize=(3*nc, 2*nr))
        else:
            fig=None
        for x in vals_list:
            # weights = np.ones_like(x) / (len(x))
            hist_prob(ax, x, bins, "step")
            # ax.hist(x, bins=bins, histtype="step", weights=weights)
        
    return fig

def hist_prob(ax, x, bins, histtype="step"):
    """ liek hist, but plots in prob. not prob density.
    """
    weights = np.ones_like(x) / (len(x))
    ax.hist(x, bins=bins, histtype=histtype, weights=weights)


def getHistBinEdges(vals, nbins):
    """ return array of edges containing all vals,
    and having n bins (so values is nbins+1)
    """
    # get range for binning
    xmin = np.min(vals)
    xmax = np.max(vals)
    
    # append a bit to edges to get all data.
    width = xmax-xmin
    delt = 0.01*width/nbins
    xs = np.linspace(xmin-delt, xmax+delt, nbins+1)
    
    return xs

# def plotHistOfLabels(labels):
#     """ Plot historgram of labels,
#     INPUT:
#     - labels, vector of labels, could be string or num, 
#     """
#     import seaborn as sns
#     plt.figure(figsize=(15,5))
#     sns.histplot(data=SF, x="label", hue="animal_dset", stat="probability", multiple="dodge", element="bars", shrink=1.5)



def plotGridWrapper(data, plotfunc, cols=None, rows=None, SIZE=2.5, 
                   origin="lower_left", max_n_per_grid=None, 
                   col_labels = None, row_labels=None, tight=True,
                   aspect=0.8, ncols=6, titles=None, naked_axes=False, 
                   titles_on_y=False, return_axes =False, fig_axes=None,
                   xlabels = None):
    """ wrapper to plot each datapoint at a given
    col and row.
    INPUT:
    - data = list of datapts. must be compatible with 
    plotfunc
    - plotfunct, func, signature is plotfunc(data[ind], ax)
    - cols and rows, where 0,0 is bottom left of screen (assumes 
    0indexed (same length as data). if either is None, then plots data starting from 0,0 and
    going row by row
    --- ncols, only matters if have to automticalyl get cols and rows.
    - max_n_per_grid, then only plots max n per col/row combo. each time will shuffle
    so that is different. Leave none to plot all
    - col_labels, row_labels, list of things to label the cols and rows.
    - titles, same length as data. If present, will overwrite row_labels and col_labels.
    - aspect, w/h
    - clean_axes, then no x or y labels.
    - fig_axes, can pass in axes, but must be in exact shape expected given the other input params.
    (fig, axes)
    - xlabels, list of strings, same len as df, for x axis labels.
    RETURNS:
    - fig, 
    NOTE: will overlay plots if multiple pltos on same on.
    """

    if xlabels is not None:
        tight = False # or else cannot see

    if titles is not None:
        assert isinstance(titles, list)
    if cols is None or rows is None:
        # get so rows is 0, 1, 2, ... and cols is 0,0,0,...
        cols = []
        rows = []
        n = len(data)
        for i in range(n):
            cols.append(i%ncols)
            rows.append(int(np.floor(i/ncols)))
        cols = np.array(cols)
        rows = np.array(rows)

    # if lost row=0, then shift all values down. this could happen if removed datapts becuase of 
    # column pruning. subsequent code needs first row to be 0.
    rows = rows - min(rows)
    
    if min(rows)!=0 or min(cols)!=0:
        print(rows, cols)
        print(min(rows), min(cols))
        print(min(rows)!=0)
        print(min(cols)!=0)
        assert False, "if not, messes up plotting of titles"

    if xlabels is None:
        xlabels = [None for _ in range(len(data))]

    # Package all variables that should not be broken apart
    allvars = [[d, r, c, xl] for d, r, c, xl in zip(data, rows, cols, xlabels)]
    def unpack(allvars):
        data = [t[0] for t in allvars]
        rows = [t[1] for t in allvars]
        cols = [t[2] for t in allvars]
        xlabels = [t[3] for t in allvars]
        return data, rows, cols, xlabels

    if max_n_per_grid is not None:
        assert isinstance(max_n_per_grid, int)

        # then shuffle
        import random
        # tmp = [[d, r, c] for d, r, c in zip(data, rows, cols)]
        random.shuffle(allvars)
        data, rows, cols, xlabels = unpack(allvars)
        # data = [t[0] for t in tmp]
        # rows = [t[1] for t in tmp]
        # cols = [t[2] for t in tmp]

    nr = int(max(rows)+1)
    nc = int(max(cols)+1)
    
    if origin=="lower_left":
        rows = max(rows)-rows
    else:
        assert origin=="top_left", "not coded"

    if fig_axes is None:
        fig, axes = plt.subplots(nr, nc, sharex=True, sharey=True, 
                                 figsize=(nc*SIZE*aspect, nr*SIZE), squeeze=False)
    else:
        fig, axes = fig_axes[:]

    done = {}
    minrow = min(rows)
    mincol = min(cols)
    for col, row in zip(cols, rows):
        done[(row, col)] = 0

    for i, (dat, col, row, xl) in enumerate(zip(data, cols, rows, xlabels)):
        if max_n_per_grid is not None:
            if done[(row, col)]==max_n_per_grid:
                continue

        ax = axes[row][col]
        plotfunc(dat, ax)

        done[(row, col)] += 1

        if xl is not None:
            ax.set_xlabel(xl)

        if titles is not None:
            if isinstance(titles[i], str):
                if titles_on_y:
                    ax.set_ylabel(f"{titles[i]}")
                else:
                    ax.set_title(f"{titles[i]}")
            else:
                if titles_on_y:
                    ax.set_ylabel(f"{titles[i]:.2f}")
                else:
                    ax.set_title(f"{titles[i]:.2f}")
        # else:
        #     if col_labels:
        #         if row==0:
        #             ax.set_title(col_labels[col])
        #     if row_labels:
        #         if col==0:
        #             ax.set_ylabel(row_labels[row])
        # if naked_axes:
        #     ax.set_yticklabels([])
        #     ax.set_xticklabels([])
        #     ax.tick_params(axis='both', which='both',length=0)

    if tight:
        fig.subplots_adjust(wspace=0, hspace=0)

    # Set the labels
    for col in range(max(cols)+1):
        for row in range(max(rows)+1):

            ax = axes[row][col]
            if titles is None:
                # then labels and columns titles, not panel specific.
                if col_labels:
                    if row==0:
                        ax.set_title(col_labels[col])
                if row_labels:
                    if col==0:
                        ax.set_ylabel(row_labels[row])
            if naked_axes:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.tick_params(axis='both', which='both',length=0)


    # x and y lim should contain all data
    if return_axes:
        return fig, axes
    else:
        return fig


def get_ylim(vals, pertile=[1.5, 98.5]):
    """ helper, to get ylims for plotting, whch remove outliers,
    """
    # Then remove outliers.
    YLIM = np.percentile(vals[~np.isnan(vals)], pertile)
    ydelt = YLIM[1]-YLIM[0]

    # THis, so doesnt cut off lower data
    YLIM[0]-=0.1*ydelt
    YLIM[1]+=0.1*ydelt
    return YLIM
