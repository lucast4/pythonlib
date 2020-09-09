""" things that take in strokes (list of np arrays) and plots """
import numpy as np
import matplotlib.pyplot as plt
from pythonlib.tools.stroketools import fakeTimesteps, strokesInterpolate

def plotDatStrokes(strokes, ax, plotver="strokes", fraction_of_stroke=[],
    add_stroke_number=True, markersize=6, pcol=None, alpha=0.55, 
    interpN=None):
    """given strokes (i.e. [stroke, stroke2, ...], with stroke2 N x 3)
    various ways of plotting
    fraction_of_stroke, from 0 to 1, indicates how much of the trial (i.e., in terms of time) 
    to plot (e.g., can plot timelapse running this multiple times..)
    - pcol, overwrites any gradation in color with one color
    - interpN = 20, then interpolates to fill each stroke.
    """
    
    if len(strokes)==0:
        print("[pythonlib/plotDatStrokes] EMPTY STROKES - not plotting")
        return 
        
    from pythonlib.tools.stroketools import fakeTimesteps, strokesInterpolate
    # this code to deal with single dot problems.
    for i, s in enumerate(strokes):
        if len(s.shape)==1:
            # then is just onetimestep, i.e., shape= (3,)
            assert s.shape[0]==3
            s = s.reshape(1,3)
            strokes[i]=s
        # s = s.reshape(-1, 


    # convert all times to the mean time for each stroek, so that is diff color
    import copy
    strokes2 = copy.deepcopy(strokes)

    # append fake times
    if strokes2[0].shape[1]<3:
        strokes2 = fakeTimesteps(strokes2, None, "in_order")
    if not interpN is None:
        strokes2 =strokesInterpolate(strokes2, interpN)

    CMAP = "jet"
    # CMAP = "plasma"
    # CMAP = "winter"
    ax.set_facecolor((0.9, 0.9, 0.9))
    if plotver=="strokes":
        # make color reflect stroke number
        for s in strokes2:
            if len(s)>0:
                s[:,2] = np.mean(s[:,2])
    elif plotver=="strokes_order":
        # same as storke,s, but ignore time, use order of arrays
        count = 0
        for s in strokes2:
            if len(s)>0:
                s[:,2] = count
                count+=1
    elif plotver=="onecolor":
        for s in strokes2:
            if len(s.shape)>1:
                s[:,2] = 0
        CMAP="coolwarm"
    elif plotver=="randcolor":
        pcol = np.random.rand(1,3) * 0.7
    elif isinstance(plotver, list) and len(plotver)==3:
        pcol = np.array(plotver).reshape(1,3)
    elif plotver=="raw":
        # keep as is, so that color reflects time.
        pass
    else:
        assert False, "dont know this"

    # print(strokes2)
    strokescat = np.array([ss for s in strokes2 for ss in s])
    # print(strokescat[0])
    # print(len(strokescat))
    # print(strokescat)
    if fraction_of_stroke:
        T = strokescat.shape[0]
        strokescat = strokescat[:int(np.ceil(fraction_of_stroke * T)),:]
        
        # get timepoints
        tvals = np.array([ss for s in strokes2 for ss in s])[:,2]
        timeon = tvals[0]
        timeoff = tvals[int(np.ceil(fraction_of_stroke * T))-1]
    # print(strokescat)
    # print(strokescat[:,0])
    if not pcol is None:
        ax.scatter(strokescat[:,0], strokescat[:,1], c=pcol, 
            marker="o", alpha=alpha, s=markersize)
    else:
        ax.scatter(strokescat[:,0], strokescat[:,1], c=strokescat[:,2], 
            marker="o", cmap=CMAP, alpha=alpha, s=markersize)

    if add_stroke_number:
        for i, s in enumerate(strokes2):
            ax.plot(s[0,0], s[0,1], 'o', color=[0.7, 0.7, 0.7], markersize=11)
            # ax.text(s[0,0]-10, s[0,1]-11, f"{i+1}", color='k', fontsize=12)
            ax.text(s[0,0], s[0,1], f"{i+1}", color='k', fontsize=12)
    if not isinstance(fraction_of_stroke, list):
        return (timeon, timeoff)


def plotDatStrokesTimecourse(strokes, ax, plotver="raw"):
    """given strokes (i.e. [stroke, stroke2, ...], with stroke2 N x 3)
    various ways of plotting, with time on the x axis"""
    YLIM = []
    if len(strokes)>0:
        if plotver=="raw":
            # then is standard. plot timecourse of strokes
            tdim = 2
            xyt = np.array([ss for s in strokes for ss in s])
            YLIM = (np.nanmin(xyt[:,(0,1)].reshape((-1,1)))-50, 
                    np.nanmax(xyt[:,(0,1)].reshape((-1,1)))+50)
            for s in strokes:
                ax.plot(s[:,tdim], s[:,0], ".b-", label='x')
                ax.plot(s[:,tdim], s[:,1], ".r-", label='y')
        if plotver=="speed":
            # then is standard. plot timecourse of strokes
            tdim =1
            xyt = np.array([ss for s in strokes for ss in s])
            YLIM = (np.nanmin(xyt[:,(0)].reshape((-1,1)))-50, 
                    np.nanmax(xyt[:,(0)].reshape((-1,1)))+50)
            for s in strokes:
                # print(s)
                ax.plot(s[:,tdim], s[:,0], ".b-", label='speed')
        # -- overlay extracted strokes
        for s in strokes:
            if len(s)>0:
                plt.plot([s[0,tdim], s[-1,tdim]], [YLIM[1], YLIM[1]], '-m')
        ax.legend()
        ax.set_xlim((-0.1, strokes[-1][-1,tdim]+0.1))

    return YLIM

