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


def annotate(s, color="k"):
    """ puts on top left corner"""
    plt.annotate(s, (0.05, 0.9), color=color, size=12, xycoords="axes fraction")


def makeColors(numcol, alpha=1, cmap="plasma"):
    """ gets evensly spaced colors. currntly uses plasma map"""
    import matplotlib.pylab as pl

    if cmap=="plasma":
        pcols = pl.cm.plasma(np.linspace(0,1, numcol), alpha=alpha)
    elif cmap=="jet":
        pcols = pl.cm.jet(np.linspace(0,1, numcol), alpha=alpha)
    # cool
    return pcols
