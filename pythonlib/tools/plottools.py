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
        pdf.savefig(fig, dpi=100)
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
        for i, (l, ax) in enumerate(zip(labellist, axes.flatten())):
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