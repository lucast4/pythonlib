"""general purpose thing that works with stroke objects, whicha re generally lists of arrays (T x 2) (sometimes Tx3 is there is time)
and each element in a list being a stroke.
- takes in single strokes and outputs transformed strokse or some feature...
"""
import numpy as np
from pythonlib.drawmodel.features import *
from pythonlib.drawmodel.strokedists import distanceDTW, distanceBetweenStrokes
import matplotlib.pyplot as plt
from ..drawmodel.behtaskalignment import assignStrokenumFromTask

# =============== TIME SERIES TOOLS
def strokesInterpolate(strokes, N, uniform=False, Nver="numpts"):
    """interpoaltes each stroke, such that there are 
    N timesteps in each stroke. uses the actual time to interpoatle
    strokes must be list of T x 3 np ararys) 
    - uniform, then interpolate uniformly based on index, will give 
    fake timesteps from 0 --> 1
    OBSOLETE - use strokesInterpolate2 isbntead, since is more flexible 
    in deciding how to interpllate."""
    strokes_new = []
    for s in strokes:
        if uniform:
            t_old = np.linspace(0,1, s.shape[0])
            t_new = np.linspace(0,1, N)
        else:
            t_old = s[:,2]
            t_new = np.linspace(t_old[0], t_old[-1], num=N)
        s_new = np.concatenate([
                        np.interp(t_new, t_old, s[:,0]).reshape(-1,1), 
                        np.interp(t_new, t_old, s[:,1]).reshape(-1,1), 
                        t_new.reshape(-1,1)], 
                        axis=1)
        strokes_new.append(s_new)
    return strokes_new

def strokesInterpolate2(strokes, N, kind="linear", base="time", plot_outcome=False):
        """ 
        NEW - use this instaed of strokesInterpolate.
        N is multipurpose to determine how to interpolate. e.g,.
        N = ["npts", 100], same time range, but 100 pts
        N = ["updnsamp", 1.5] then up or down samples (here up by 1.5)
        N = ["fsnew", 1000, 125] then targets new fs 1000, assuming
        N = ["interval", interval], for spatial,
        initially 125.
        - base, 
        -- index, then replaces time with index before interpolating.
        -- time, then just uses time
        -- space, then uses cum dist.
        RETURNS:
        - returns a copy
        NOTE:
        - if creates an empyt strok, then fixes by replacing with endpoints 
        of original strokes.
        """

        from scipy.interpolate import interp1d
        strokes_interp = []

        for strok in strokes:
            
            if base=="index":
                strok = strok.copy()
                strok[:,2] = np.arange(len(strok))
            elif base=="space":
                strok = strok.copy()
                strok = convertTimeCoord([strok], ver="dist")[0]
            else:
                assert base=="time"

            if strok.shape[0]==1:
                # then dont interpolate, only one pt
                strokinterp = strok
            else:

                # get new timepoints
                t = strok[:,2]
                nold = len(t)
                if N[0]=="npts":
                    nnew = N[1]
                elif N[0]=="updnsamp":
                    nnew = int(np.round(N[1]*nold))
                    # tnew = np.linspace(t[0], t[-1], nnew)
                elif N[0]=="fsnew":
                    nnew = int(np.round(N[1]/N[2]*nold))
                    # tnew = np.linspace(t[0], t[-1], nnew)
                elif N[0]=="interval":
                    # goal is a constant interval btw pts.
                    interval = N[1]
                    total = t[-1] - t[0]
                    nnew = int(np.ceil(total/interval))
                    # print(interval)
                    # print(total)
                    # print(nnew)
                    # assert False
                else:
                    print(N)
                    assert False, "not coded"
                tnew = np.linspace(t[0], t[-1], nnew)
    #             print(t)
    #             print(tnew)

                strokinterp = np.empty((len(tnew), 3))
                strokinterp[:,2] = tnew
            
                # fill the x,y, columns
                for i in [0, 1]:
                    f = interp1d(t, strok[:,i], kind=kind)
                    strokinterp[:,i] = f(tnew)

                    if False:
                        plt.figure()
                        plt.plot(t, strok[:,0], '-ok');
                        plt.plot(tnew, f(tnew), '-or');

            strokes_interp.append(strokinterp)

        # If new length is 0, replace with previous endpoints.
        for i, s in enumerate(strokes_interp):
            if len(s)==0:
                strokes_interp[i] = strokes[i][[0, -1], :]

        if plot_outcome:
            fig, axes = plt.subplots(1,2, figsize=(12,6))
            ax = axes.flatten()[0]
            for s in strokes:
                ax.plot(s[:,0], s[:,1], "-o")
            ax.set_title('original')
            
            ax = axes.flatten()[1]
            for s in strokes_interp:
                ax.plot(s[:,0], s[:,1], "-o")
            ax.set_title('after interpolation')
    
        return strokes_interp

def smoothStrokes(strokes, sample_rate, window_time=0.05, window_type="hanning",
                 adapt_win_len="adapt"):
    """ returns copy of strokes, smoothed with window_time (seconds)
    - sample_rate in samp/sec (e.g., fd["params"]["sample_rate"])
    - adapt_win_len, what to do fro strokes that are shoerter than window.
    """
    from .timeseriestools import  smoothDat

    window_len = np.floor(window_time/(1/sample_rate))
    if window_len%2==0:
        window_len+=1
    window_len = int(window_len)

    # -- check that no strokes are shorter than window
    strokes_sm = []
    for s in strokes:
        did_adapt = False
        if len(s)<window_len:
            if adapt_win_len=="adapt":
                window_len = len(s)
                if window_len%2==0:
                    window_len-=1
                window_len = int(window_len)
#                 strokes_sm.append(np.array([
#                     smoothDat(s[:,0], window_len=wlen_tmp, window=window_type), 
#                     smoothDat(s[:,1], window_len=wlen_tmp, window=window_type), 
#                     s[:,2]]).T)
                did_adapt=True
                if False:
                    print("window_len too large for this stroke... adapting length")
            elif adapt_win_len=="error":
                assert False, "window too long..."
            elif adapt_win_len=="remove":
                # print("removing stroke since shorter than window")
                pass
        # Do smoothing
        strokes_sm.append(np.array([
            smoothDat(s[:,0], window_len=window_len, window=window_type), 
            smoothDat(s[:,1], window_len=window_len, window=window_type), 
            s[:,2]]).T)
        if False:
            # debugging
            if did_adapt:
                print('smoothed')
                print(strokes_sm[-1].shape)
                print('orig')
                print(s.shape)
    return strokes_sm

def strokesFilter(strokes, Wn, fs, N=9, plotresponse=False, 
    plotprepost=False, dims=[0,1], demean=False):
    """ filter each dimension of strokes (x,y).
    strokes is list of strok where a strok is N x 2(or 3, for t)
    array. assumes evenly sampled in time.
    - Wn is critical frequencies. in same units as fs
    [None <num>] does lowpass
    [<num> None] hp
    [<num> <num>] bandpass
    - returns copy
    """
    from scipy import signal
    assert dims==[0,1], "niot yet coded"
#     # normalize the frequency based rel to nyquist freq
#     nyq = 
    if Wn[0] is None:
        btype = "lowpass"
        Wn = Wn[1]
    elif Wn[1] is None:
        btype = "highpass"
        Wn = Wn[0]
    else:
        btype = "bandpass"
#     else:
#         print(Wn)
#         assert False, "not coded"
        
    sos = signal.butter(N, Wn, btype, analog=False, fs=fs, output='sos')
    # print(sos.shape)
    padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(),
                        (sos[:, 5] == 0).sum()))

    if plotresponse:
        w, h = signal.sosfreqz(sos, fs=fs)
        plt.semilogx(w, 20 * np.log10(abs(h)))
        plt.margins(0, 0.1)
        plt.title('Butterworth filter frequency response')

        plt.xlabel('Frequency [radians / second]')

        plt.ylabel('Amplitude [dB]')
        plt.grid(which='both', axis='both')
        
    # == apply filter
    strokesfilt = []
    for strok in strokes:

        if np.all(np.isnan(strok[:,0])):
            # dont bother trying to smooth
            strokf = np.copy(strok)
        elif btype=="lowpass" and len(strok)<=padlen:
            # instead of filtering, uses smoothingw with adaptive windowsize

            tmp = smoothStrokes([strok], fs, window_time=1/Wn, window_type="hanning",
                         adapt_win_len="adapt")
            # print('--')
            # print(strok.shape)
            # print(tmp[0].shape)
            # print(strok[:,2])
            # print(tmp[0][:,2])
            strokf = tmp[0]
            if False:
                print("not enough data to filter - using smoothing instead, and adaptive windowsize")
        else:
            strokf = np.copy(strok)
            if demean:
                # demean first, filter, then add mean.
                # NOTE: this doesnt amke a difference...
                strokfmean = np.mean(strokf, axis=0)
                strokf -= strokfmean
                strokf[:,dims] = signal.sosfiltfilt(sos, strokf[:,dims], axis = 0)
                strokf += strokfmean
            else:
                strokf[:,dims] = signal.sosfiltfilt(sos, strokf[:,dims], axis = 0)
        
        strokesfilt.append(strokf)
        
    # -- compare strokes pre and post
    if plotprepost:
        plt.figure(figsize=(10,10))
        ax = plt.subplot(211)
        plotDatStrokesTimecourse(strokes, ax=ax)
        ax = plt.subplot(212)
        plotDatStrokesTimecourse(strokesfilt, ax=ax)

    return strokesfilt
        

def strokesCurvature(strokes, fs, LP=5, fs_new = 30, absval = True, do_pre_filter=True, ploton=False):
    """ from Abend Bizzi 1982:
    Trajectory curvature = (X Y—X Y)/X 2 + Y 2 ) 3p , where X and
    Y are the time derivatives of the X-Y co-ordinates of the hand in the horizontal plane, and X and Y are
    the corresponding accelerations.
    Also see Miall Haggard 1995, for other measure of curvature of entire stroke. they argue
    that too noisey to the moment by moennt curvature.
    
    LP and fs_new are for computing velocity and in turn accel. Lower is more smooth. emprically
    is very noisy at edges, and noisy in middle too. See devo_strokestuff notebook for thoughts.
    
    """
    from pythonlib.tools.stroketools import strokesVelocity
    
    # 1) Get velocity and accel
    strokes_vel = strokesVelocity(strokes, fs, fs_new = fs_new, lowpass_freq=LP, do_pre_filter=do_pre_filter, ploton=ploton)[0]
    strokes_accel = strokesVelocity(strokes_vel, fs, fs_new=fs_new, lowpass_freq=LP, do_pre_filter=False, ploton=ploton)[0]
#     print(strokes_vel[0].shape)
#     print(strokes_accel[0].shape)
#     print(strokes[0].shape)
    
#     if ploton:
#         import matplotlib.pyplot as plt
#         plt.figure()
#         for S in strokes_vel
    def curv(v, a):
        """ v and a are N x 2, velocityu and accel"""
        return (v[:,0]*a[:,1] - v[:,1]*a[:,0])/(v[:,0]**2 + v[:,1]**2)**(3/2)
        
    strokes_curv = []
    for v,a in zip(strokes_vel, strokes_accel):
        
#         print(v.shape)
#         print(a.shape)
#         print(curv(v,a).shape)
        c = curv(v,a)
        c = np.concatenate((c.reshape(-1,1), v[:,2].reshape(-1,1)), axis=1) # give time axis.
        strokes_curv.append(c)
        
    if absval:
        for strok in strokes_curv:
            strok[:,0] = np.abs(strok[:,0])
    
    if ploton:
        fig, ax = plt.subplots(1,1)
        # plot curvature
        plotDatStrokesTimecourse(strokes_curv, ax=ax, plotver="speed", label="curv", overlay_stroke_periods=False)
        ax.set_title("curvature")
        YMAX = 1/50
        if absval:
            YMIN = 0
        else:
            YMIN = -YMAX

        plt.ylim([YMIN, YMAX])
        plt.ylabel("1/pix (1/radius)")
        
    return strokes_curv


def strokesVelocity(strokes, fs, ploton=False, lowpass_freq = 15,
    fs_new = 30, do_pre_filter=False, clean=False):
    """ gets velocity and speeds. 
    should first have filtered strokes to ~15-20hz. if not, then 
    activate flag to run filter here. 
    INPUTS:
    - fs, original sample rate (use this even if already filtered, as is always
    done if use the getTrialsStrokes code in uitls, which Dataset does use). e..g
    fs = filedata["params"]["sample_rate"]
    - lowpass_freq, applies this to smooth at end.
    - fs_new, how much to downsample before doing 5pt differnetiation. 
    assumes that have already filtered with lowpass below fs_new/2 [yes, filters
    at 15hz if use getTrialsStrokes. if have not, then set do_pre_filter=True,
    and will first filter data.
    empriically, 30hz for fs_new wiorks well. the ;opwer ot is thje smoother
    the velocy (e.g, 10-15 woudl be good). 
    - clean, then this is what I consider good for plotting, smooth bumps, etc. used
    in the wrapper code for plotting. 
    - NOTES:
    Processing steps:
    - downsamples (linear)
    - 5pt differentiation method
    - upsamples
    - smooths by filtering.
    
    - if a strok is too short to compute velocity, then returns puts
    nan.
    - note: speed is first taking norm at each timebin, adn then smoothing then norm,.

    RETURNS:
    strokesvel, which should be same size as strokes
    strokesspeed, which should be list of Nx2 arrays, where col2 is time.
    the timesteps for both of these outputs will match exactly the time inputs for
    strokes.

    """

    if clean:
        lowpass_freq = 5
        
    sample_rate = fs

    if ploton:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 25))

    
    if do_pre_filter:
        strokes = strokesFilter(strokes, Wn = [None, fs_new/2], 
        fs = sample_rate)

    # ----- 1) downsample, this leads to better estimate for velocity.
    # before downsasmple, save vec length for later upsample
    n_each_stroke = [s.shape[0] for s in strokes]
    strokes_down = strokesInterpolate2(strokes, N=["fsnew", fs_new, sample_rate])

    # containers for velocities and speeds
    strokes_vels = []
    strokes_speeds = []
    for j, (strok, n_orig) in enumerate(zip(strokes_down, n_each_stroke)):
        
        time = strok[:,2].reshape(-1,1)

        # minimum length or else will not return velocity
        if len(time)<6:
            # then don't bother getting belocity
            print("skipping differnetiation, stroke too short. giving NAN")
            strok_vels_t = np.empty((n_orig, 3))
            strok_vels_t[:,[0,1]] = np.nan
            strok_vels_t[:,2] = np.linspace(time[0], time[-1], n_orig).reshape(-1,)

            strok_speeds_t = np.empty((n_orig, 2))
            strok_speeds_t[:,0] = np.nan
            strok_speeds_t[:,1] = np.linspace(time[0], time[-1], n_orig).reshape(-1,)

            strokes_vels.append(strok_vels_t)
            strokes_speeds.append(strok_speeds_t)
        else:

            if ploton:
                ax = plt.subplot(5, 1, 1)
                plt.title('after downsample')
                for col in [0,1]:
                    plt.plot(strok[:,2], strok[:,col],  "-o", label=f"orig pos, dim{col}")
                    plt.legend()
            
            # ------------ 2) differntiate
            # what is sample peridocity? (actually compute exact value, since might
                #be slightly different from expected due to rounding erroes in interpoalte.
            a = np.diff(strok[:,2]).round(decimals=4)
            per = np.unique(a)
            if len(per)>1:
                if max(np.diff(per))<=0.002:
                    # then ok, just take mean
                    per = np.mean(np.diff(strok[:,2]))
                else:
                    print(per)
                    assert False, "why multiple periods?"

            strok_vels = np.empty((strok.shape[0],2))
            for i in [0,1]:
                strok_vels[:,i] = diff5pt(strok[:,i], h=per)

            # Conver to speed
            strok_speeds = np.linalg.norm(strok_vels[:,[0,1]], axis=1).reshape(-1,1)

            # ------------- 4) interpolate and upsample velocity back to original timesteps.
            # 1) velocities
            strok_vels_t = np.concatenate([strok_vels, time], axis=1)
            kind_upsample = "cubic" # doesnt matter too much, cubix vs. linear
            # seems like cubic better.
            if ploton:
                for col in [0,1]:
                    axv= plt.subplot(5, 1,2 )
                    plt.title("overlaying before and after upsample (and x and y)")
                    axv.plot(strok_vels_t[:,2], strok_vels_t[:,col],  "-o",label=f"vel, dim{col}")
            
            strok_vels_t = strokesInterpolate2([strok_vels_t], N=["npts", n_orig], kind=kind_upsample)[0]
            
            if ploton:
                for col in [0,1]:
                    axv.plot(strok_vels_t[:,2], strok_vels_t[:,col], "-o", label=f"vel, dim{col}")
                plt.legend()
                
            # 2) speed
            strok_speeds_t = np.concatenate([strok_speeds, strok_speeds, time], axis=1)
            if ploton:
                axs= plt.subplot(5, 1, 3)
                plt.title("overlaying before and after upsample")
                for col in [0]:
                    axs.plot(strok_speeds_t[:,2], strok_speeds_t[:,col], "-o", label=f"speed")
            
            strok_speeds_t = strokesInterpolate2([strok_speeds_t],  N=["npts", n_orig], kind=kind_upsample)[0]
            if ploton:
                for col in [0]:
                    axs.plot(strok_speeds_t[:,2], strok_speeds_t[:,col], "-o",label=f"speed")
                plt.legend()

            strokes_vels.append(strok_vels_t)
            strokes_speeds.append(strok_speeds_t[:,[0, 2]])

    # -- filter velocity to smooth
    strokes_vel = strokesFilter(strokes_vels, Wn = [None, lowpass_freq], fs = sample_rate)
    if ploton:
        ax = plt.subplot(5, 1, 4)
        plt.title("after filter (verlay x and y)")

        for S in strokes_vel:
            for col in [0,1]:
                ax.plot(S[:,2], S[:,col], "-o", label="vel, after filter")
        plt.legend()
            
    # -------- filter speed
    tmp = [np.concatenate([S[:,0, None], S[:,0, None], S[:,1, None]], axis=1) for S in strokes_speeds]
    # print(tmp)
    strokes_speeds = strokesFilter(tmp, Wn = [None, lowpass_freq], fs = sample_rate)
    # print(strokes_speeds)
    # assert False
    strokes_speeds = [np.concatenate([S[:,0, None], S[:,2, None]], axis=1) for S in strokes_speeds]
    if ploton:
        ax = plt.subplot(5, 1, 5)
        plt.title("after filter")
        for S in strokes_speeds:
            for col in [0]:
                ax.plot(S[:,1], S[:,col], "-o", label="speed, after filter")    
    
    return strokes_vel, strokes_speeds





def diff5pt(x, h=1):
    """ given timeseries x get devirative,
    using 5-point differentiation. see:
    https://en.wikipedia.org/wiki/Numerical_differentiation
    this gives error (compared to true derivative) or 
    order h^4, where h is timestep. 
    - to deal with edgepts (i.e.,. first and last 4 pts), pad
    edges by repeating the 1-interval distance.
    - expects input to be around 100hz (doesnt relaly matter).
    - x can be list or nparray
    - h is time interval. leave as 1 to be unitless. (e..g, if
    100hz, then h=10)
    """
    
    if isinstance(x, list):
        x = np.array(x)

    if False:
        # doesnt work well - large discontnuities at edges.
        x = np.concatenate([x[0, None], x[0, None], x, x[-1, None], x[-1, None]])
    else:
        for _ in range(2):
            x = np.concatenate([
                x[0, None]+(x[0]-x[1]), 
                x, 
                x[-1, None]+(x[-1]-x[-2])])

    return (-x[4:] + 8*x[3:-1] - 8*x[1:-3] + x[:-4])/(12*h)


def convertTimeCoord(strokes_in, ver="dist", fakegapdist=0.):
    """ methods to convert the time column to other things.
    Will return copy of strokes and leave input strokes unchanged.
    - ver:
    -- "dist", is distance traveled in pixels, starting from 
    onset of first stroke. will assume zero dist from stroke 
    offsets to next strok onsets, unless enter a gap dist for
    fakegapdist (in pix)
    
    """
    from copy import copy
    strokes = [np.copy(s) for s in strokes_in]
    
    if ver=="dist":
        cumdist=0.
        for i, strok in enumerate(strokes):
            c = np.cumsum(np.linalg.norm(np.diff(strok[:,[0,1]], axis=0), axis=1))
            c = np.r_[0., c]
            distthis = c[-1] # copy to add to cumdist
            c += cumdist
            cumdist += distthis + fakegapdist
            if strok.shape[1]>2:
                strok[:,2] = c
            else:
                strok = np.c_[strok, c]
                assert strok.shape[1]==3
            strokes[i] = strok

    else:
        print(ver)
        assert False, "not coded"
    return strokes

def strokesDiffsBtwPts(strokes):
    """ baseically taking diffs btw successive pts separately fro x and y
    , but using  diff5pt (i./.e, like np.diff)"""
    from pythonlib.tools.stroketools import diff5pt
    def _diff(strok):
        if len(strok)<2:
            # then cant' do this
            assert False, "need at least 2 pts"
        x = diff5pt(strok[:,0]).reshape(-1,1)
        y = diff5pt(strok[:,1]).reshape(-1,1)
        return np.concatenate([x, y], axis=1)
    return [_diff(strok) for strok in strokes]




################# OTHER FEATURES

def getStrokesFeatures(strokes):
    """output dict with features, one value for
    each stroke, eg. center of mass
    NOTE: This is generally obsolete, see 
    pythonlib.drawmodel.features instead"""

    outdict = {
    "centers_median":[np.median(s[:,:2], axis=0) for s in strokes]
    }
    return outdict


def splitTraj(traj, num=2):
    return _splitarray(traj, num)

def _splitarray(A, num=2, reduce_num_if_smallstroke=False):
    # split one numpy array into a list of two arrays
    # idx = int(np.ceil(A.shape[0]/2))
    if not reduce_num_if_smallstroke:
        assert A.shape[0]>=num, "not enough pts to split"
    else:
        if A.shape[0]<num:
            num = A.shape[0] 
    edges = np.linspace(0, len(A), num+1)
    edges = [int(np.ceil(e)) for e in edges]
    return [A[i1:i2,:] for i1, i2 in zip(edges[:-1], edges[1:])]

def splitStrokesOneTime(strokes, num=2, reduce_num_if_smallstroke=False):
    """like splitStrokes, but only ouptuts one list of np arrays,
    so will have 2x num arrays as input strokes. will maitnain input
    order (based on position. will ignore timepoints)"""

    strokes_split = []
    for A in strokes:
        strokes_split.extend(_splitarray(A, num=num, reduce_num_if_smallstroke=reduce_num_if_smallstroke))
        # appends: [[a,b], [b,a]], where a and b are np arrays
    return strokes_split


def splitStrokes(strokes, num=2):
    """ given list of np arays, splits each one into <num> 
    pieces. so if there are three arrays in strokes, then
    outputs a list of 2^3 new strokes (i.e., outputs
    a list of list of np arrays, one for each ordering) """

    assert num==2, "have not coded for other than 2."

    strokes_split = []
    for A in strokes:
        strokes_split.append([_splitarray(A), _splitarray(np.flipud(A))])
        # appends: [[a,b], [b,a]], where a and b are np arrays

    # --- get outer products
    from itertools import product
    X = product(*strokes_split)
    strokes_all =[]
    for x in X:
        strokes_all.append([xxx for xx in x for xxx in xx]) # to flatten
    return strokes_all
    # return list(product(*strokes_split))

    # e.g.,:
    # B = [[1,2],[3,4],[5,6]]
    # list(product(*B)) --> [(1, 3, 5), (1, 3, 6), (1, 4, 5), (1, 4, 6), (2, 3, 5), (2, 3, 6), (2, 4, 5), (2, 4, 6)]

    if False:
        # DEBUGGING PLOTS
        strokes_split = splitStrokes(strokes, 2)

        fig = plt.figure(figsize=(10,10))
        ax = plt.subplot(3,3,1)
        plotDatStrokes(strokes, ax=ax, plotver="strokes_order")

        for i, S in enumerate(strokes_split):
            ax = plt.subplot(3,3,i+2)
            # plotDatStrokes(strokes, ax)
            plotDatStrokes(S, ax, plotver="strokes_order")



def fakeTimesteps(strokes, point=np.array([0.,0.]), ver="in_order"):
    """strokes is a list with each stroke an nparray 
    - each stroke can be (T x 3) or T x 2. doesnt mater.
    for each array, replaces the 3rd column with new timesteps, such 
    that goes outward from point closer to origin
    - determines what timesteps to use independelty for each stroke.
    - to use origin as point, do origin=(getTrialsFix(filedata, 1)["fixpos_pixels"])
    NOTE: could pull this out of getTrialsTaskAsStrokes, but would 
    need to be able to pass in origin sopecific for eafch trial.
    - point is just any point, will make the poitn closer to this as the onset.
    NOTE: mutates strokes.
    """
    assert isinstance(strokes, (tuple, list)), "se above"
    if isinstance(strokes, tuple):
        strokes = list(strokes)

    start_ind = 0
    for i, s in enumerate(strokes):
    # s = strokes[0]

        if ver=="from_point":
            # distances from each end of the stroke to the origin
            dstart = np.linalg.norm((s[0, [0,1]] - point))
            dend = np.linalg.norm((s[-1, [0,1]] - point))

            if dstart<dend:
                # then start is closer to origin than is end
                s_pos = s[:,[0,1]]
            else:
                s_pos = np.flipud(s[:,[0,1]])
            
            # append a new timesteps
            s = np.concatenate((s_pos, np.arange(start_ind, start_ind+s_pos.shape[0]).reshape(-1,1)), axis=1)
        elif ver=="in_order":
            # just uses whatever order the coords are in currently.
            s_pos = s[:,[0,1]]
            # print(s_pos.shape)
            # print(np.arange(start_ind, start_ind+s_pos.shape[0]).reshape(-1,1).shape)
            s = np.concatenate((s_pos, np.arange(start_ind, start_ind+s_pos.shape[0]).reshape(-1,1)), axis=1)
        elif ver=="from_end_of_previous_stroke":
            # for first stroke keeps as currently is"
            s_pos = s[:,[0,1]]
            if i>0:
                s_prev_end = strokes[i-1][-1,[0,1]]

                dstart = np.linalg.norm((s_pos[0, [0,1]] - s_prev_end))
                dend = np.linalg.norm((s_pos[-1, [0,1]] - s_prev_end))

                if dstart<dend:
                    # then start is closer to origin than is end
                    s_pos = s_pos[:,[0,1]]
                else:
                    s_pos = np.flipud(s_pos[:,[0,1]])
                
            # append a new timesteps
            s = np.concatenate((s_pos, np.arange(start_ind, start_ind+s_pos.shape[0]).reshape(-1,1)), axis=1)

        elif ver:
            print(ver)
            assert False, "have not coded this"
        
        start_ind=start_ind+s_pos.shape[0]
        strokes[i]=s    
    return strokes


# def computeDistTraveled(strokes, origin, include_lift_periods=True):
#     """ assume start at origin. assumes straight line movements.
#     by default includes times when not putting down ink.
#     IGNORES third column(time) - i.e., assuems that datpoints are in 
#     chron order."""
    
#     cumdist = 0
#     prev_point = origin
#     for S in strokes:
#         cumdist += np.linalg.norm(S[0,[0,1]] - prev_point)
#         cumdist += np.linalg.norm(S[-1,[0,1]] - S[0,[0,1]])
#         prev_point = S[-1, [0,1]]
#     return cumdist
                                                

def getOnOff(strokes, relativetimesec=False):
    """ strokes is list of strokes, each stroke is list of (x,y,t). [i.e. from single trial]
    outputs onsets, offsets, in absolute time """
#     print(strokes)
#     [print(st) for st in strokes if not st]
    # print(strokes)
    onsets = [np.array(st[0][2]) if len(st)>0 else np.nan for st in strokes]
    offsets = [np.array(st[-1][2]) if len(st)>0 else np.nan for st in strokes]
    if relativetimesec:
        # then subcrat onset of first stroke
        a = onsets[0]
        onsets = (onsets-a)/1000
        offsets = (offsets-a)/1000
    return (onsets, offsets)



########################### SPATIAL MANIPULATIONS
def _stripTimeDimension(strokes):
    """ returns strokes where each strok is
    only N x 2, not N x 3.
    Returns also T, which can be passed into 
    _appendTimeDimension to put it back.
    RETURNS:
    strokes, t
    """
    t = [s[:,2] for s in strokes]
    strokes = [s[:,[0,1]] for s in strokes]
    return strokes, t

def _appendTimeDimension(strokes, T):
    """ Re-append T back to strokes.
    see _stripTimeDimension
    """
    strokes = [np.c_[s, t] for s, t in zip(strokes, T)]
    return strokes


def translateStrokes(strokes, xy):
    """ translates strokes by x, y
    xy, np.array, shape (1,2), (2,1), or (2,)
    RETURNS:
    copy.
    """

    # assert xy.shape == (1,2)

    # strokes_copy = [S.copy for S in strokes]
    # strokes_copy = [S[:,[0,1]] + xy for S in strokes]

    from ..drawmodel.primitives import transform
    if strokes[0].shape[1]==3:
        strokes, time = _stripTimeDimension(strokes)
    else:
        time=None

    strokes = transform(strokes, x=xy[0], y=xy[1])

    if time is not None:
        strokes = _appendTimeDimension(strokes, time)
    return strokes


def rescaleStrokes(strokes, ver="stretch_to_1"):
    """ 
    Apply rescale, taking into account entire strokes.
    - ver,
    --- "stretch_to_1", rescale so that max in either x or y is 1. 
    finds max over all pts across all strok (in absolute val) and divides all 
    values by that. i..e make this as big as possible in a square
    [-1 1 -1 1]. This makes most sense if you have recentered already, so that (0,0) is 
    at center.
    """
    if ver=="stretch_to_1":
        pos = np.concatenate(strokes)
        maxval = np.max(np.abs(pos[:,[0,1]]))
        if strokes[0].shape[1]==3:
            strokes = [np.concatenate((s[:,[0,1]]/maxval, s[:,2].reshape(-1,1)), axis=1) for s in strokes]
        else:
            strokes = [s[:,[0,1]]/maxval for s in strokes]
    else:
        print(ver)
        assert False, "not codede"
    return strokes

    
def getCenter(strokes, method="extrema"):
    """ get center of strokes, with methods:
    --- "extrema", based on x and y max edges, then
    find center.
    --- "com", then center of mass
    RETURNS: 2-length list.
    """
    if method=="com":
        assert False, "not coded, see features calc."""
    elif method == "extrema":
        
        tmp = np.concatenate([s[:,0] for s in strokes])
        xminmax = (np.min(tmp), np.max(tmp))

        tmp = np.concatenate([s[:,1] for s in strokes])
        yminmax = (np.min(tmp), np.max(tmp))
        
        return [np.mean(xminmax), np.mean(yminmax)]


def standardizeStrokes(strokes, onlydemean=False, ver="xonly",
    return_tform_func=False):
    """ standardize in space (so centered at 0, and x range is from -1 to 1
    - if return_tform_func, then returns function that can apply to any 
    strokes, giving the same transformation: tform(strokes)
    ) """
    if ver=="xonly":
        center = np.mean(getCentersOfMass(strokes, method="use_mean"))
        xvals = [s[0] for S in strokes for s in S]
        xlims = np.percentile(xvals, [2.5, 97.5])
        x_scale = xlims[1]-xlims[0]
        if onlydemean:
            x_scale = 1.

        def tform(strokes):
            strokes = [np.concatenate(((S[:,[0,1]]-center)/x_scale, S[:,2].reshape(-1,1)), axis=1) for S in strokes]
            return strokes



    elif ver=="centerize":
        center = np.mean(np.concatenate(strokes), axis=0)[:2]
        def tform(strokes):
            if strokes[0].shape[1]==3:
                strokes = [s-np.r_[center,0] for s in strokes]
            else:
                strokes = [s-center for s in strokes]
            return strokes
    else:
        print(ver)
        assert False, "not coded"

    if return_tform_func:
        return tform(strokes), tform
    else:
        return tform(strokes)


    # print(x_scale)
    # print(xvals)
    # print(center)
    # print(xlims)
    # new_strokes = [np.concatenate(((S[:,[0,1]]-center)/x_scale, S[:,2].reshape(-1,1)), axis=1) for S in strokes]

    # print(new_strokes)
    # assert False


def alignStrokes(strokes, strokes_template, ver = "translate"):
    """ transforms strokes so that aligns with strokes_template, 
    based on method in ver.
    - ver:
    -- translate: strokes center will be aligned to center of strokes_template.
    will take center over all points across all arrays in strokes.
    """

    if ver == "translate":
        smean = np.mean(np.concatenate(strokes, axis=0), axis=0)[0:2]
        stmean = np.mean(np.concatenate(strokes_template, axis=0), axis=0)[0:2]
        delta = stmean - smean
        strokes_aligned = []
        for i, s in enumerate(strokes):
            s[:, [0,1]] = s[:, [0,1]] + delta
            strokes_aligned.append(s)
        # strokes = [ss+delta for ss in strokes]
    else:
        assert False, "not coded!!"

    return strokes_aligned

def convertFlatToStrokes(strokes, flatvec):
    """ given flatvec length N, and strokes, which
    when flattened gives vec of length N x 3, converts
    flatvec to list of vecs matching strokes"""
    
    assert len(np.concatenate(strokes, axis=0))==len(flatvec)
    tmp = []
    onsets = [len(s) for s in strokes] # indices
    onsets = np.r_[0, np.cumsum(onsets)]

    for on1, on2 in zip(onsets[:-1], onsets[1:]):
        tmp.append(np.array(flatvec[on1:on2]))
    return tmp



def getMinMaxVals(strokes):
    """ get min and max pts across all pts in storkes
    returns np array, [minx, maxx, miny, maxy]
    """
    xvals = np.concatenate([s[:,0] for s in strokes])
    yvals = np.concatenate([s[:,1] for s in strokes])
    
    return [np.min(xvals), np.max(xvals), np.min(yvals), np.max(yvals)]
    


def check_strokes_in_temporal_order(strokes):
    """ True/False. based on 3rd column"""
    assert strokes[0].shape[1]>=3, "need a time column"

    times = np.concatenate([s[:,2] for s in strokes],0)
    if not np.all(np.diff(times)>=0):
        print(times)
        print(np.diff(times)>=0)
        print(np.diff(times))
        assert np.all(np.diff(times)>=0)


def check_strokes_identical(strokes1, strokes2):
    """ returns True if strokes1 and 2 are identical.
    this includes having same sahpe, and same time stamps
    """
    return all([np.all(np.isclose(s1, s2)) for s1,s2 in zip(strokes1, strokes2)])
    

################### STROKE PERMUTATION TOOLS

def getStrokePermutationsWrapper(strokes, ver,  num_max=1000):
    """ wrapper, different moethods for permuting strokes.
    OUT:
    - list of things of same type as storkes
    """
    if ver=="circular_and_reversed":
        # Then get circular for both strokes, and 
        # after strokes is reversed. only includes reversed
        # if len(strokes)>2, since for length 2 
        # reversed doesnt add any new circular perms.

        strokes_list = getCircularPerm(strokes)
        if len(strokes)>2:
            strokes_list.extend(getCircularPerm(strokes[::-1]))
    elif ver=="all_orders":
        # Then all possible unique orderings. This can get large,
        # so use num_max. Does not modify the direction within each stroke.
        from math import factorial
        # print(f"expect to get (ignoreing num max): {factorial(len(strokes))}")
        strokes_list, _ = getAllStrokeOrders(strokes, num_max =num_max)
    elif ver=="all_orders_directions":
        # Then all possible unique orderings. This can get large,
        # so use num_max. Gets all direction within each stroke. num
        # max only applies BEFORE getting the diff orders.
        from math import factorial
        print(f"expect to get (ignoreing num max): {factorial(len(strokes)) * 2**len(strokes)}")
        strokes_list, _ = getAllStrokeOrders(strokes, num_max =num_max)
        strokes_bothdir = []
        # strokes_bothdir_orders = []
        for strokes in strokes_list:
            S = getBothDirections(strokes, fake_timesteps_ver = "in_order")
            strokes_bothdir.extend(S)
        strokes_list = strokes_bothdir
    else:
        print(ver)
        assert False, "not coded"
    return strokes_list

#### BELOW: IGNORE, INSTEAD USE WRAPPER.
def getCircularPerm(strokes):
    """ get circular permutation of storkes, returning
    a list containing all permutations.
    circular perm maintains the ordering of trajs in strokes,
    but changes which is the first traj. 
    e..g, [1,2,3] returns [[1 2 3], [2 3 1], [3 1 2]]
    """
    from more_itertools import circular_shifts
    return circular_shifts(strokes)


def getAllStrokeOrders(strokes, num_max=1000):
    """for strokes, list of np arrays, outputs a set
    of all possible stroke orders, all permuations.
    - num_max, None if want to get all.
    NOTE: each time run could be a different order
    NOTE: output will be in same memory location as inputs."""
    from pythonlib.tools.listtools import permuteRand
    from math import factorial
    
    nstrokes = len(strokes)
    if nstrokes==0:
        return []
    elif nstrokes==1:
        return [[strokes[0]]], [[0]] 
    else:
        if num_max:
            num_max = min([num_max, factorial(nstrokes)])
        else:
            num_max = factorial(nstrokes)
        stroke_orders_set = set(permuteRand(list(range(nstrokes)), N=num_max))
        strokes_allorders = [[strokes[i] for i in order] for order in stroke_orders_set] # convert from set to list.
    #     strokes_allorders = permuteRand(strokes, N=factorial(len(strokes)))
        
    #     print(strokes_allorders)
    #     return [[strokes[i] for i in order] for order in strokes_allorders]
        return strokes_allorders, stroke_orders_set 



def getBothDirections(strokes, fake_timesteps_ver = "in_order", fake_timesteps_point=None):
    """ given list of np arays, outputs list of list of arrays,
    where each inner list is same size as input list (and actually
    includes the list as one of them), but will all permutations of 
    orders of the storkes. so if input is list of 3 arrays, then
    output is list of 8 lists, each with the same 3 arrays but 
    in unique ordering.
    """
    from itertools import product

    strokes_split = []
    for A in strokes:
        strokes_split.append([A, np.flipud(A)])
        # appends: [a, a'], where a and a' are arrays

    # --- get outer products
    strokes_all = list(product(*strokes_split))

    # --- get fake timesteps?
    if not fake_timesteps_ver is None:
        strokes_all = [fakeTimesteps(strokes, point=fake_timesteps_point, ver=fake_timesteps_ver) for strokes in strokes_all]

    return strokes_all
    # return list(product(*strokes_split))

    # e.g.,:
    # B = [[1,2],[3,4],[5,6]]
    # list(product(*B)) --> [(1, 3, 5), (1, 3, 6), (1, 4, 5), (1, 4, 6), (2, 3, 5), (2, 3, 6), (2, 4, 5), (2, 4, 6)]

    if False:
        # DEBUGGING PLOTS
        strokes_split = splitStrokes(strokes, 2)

        fig = plt.figure(figsize=(10,10))
        ax = plt.subplot(3,3,1)
        plotDatStrokes(strokes, ax=ax, plotver="strokes_order")

        for i, S in enumerate(strokes_split):
            ax = plt.subplot(3,3,i+2)
            # plotDatStrokes(strokes, ax)
            plotDatStrokes(S, ax, plotver="strokes_order")


def concatStrokes(strokes, reorder=False, thresh=10, sanity_check=False):
    """ combines strokes (list of np array) into strokes of 
    length 1 (still a list of a single np array)
    - reorder, then makes sure that the first coord of n+1 is close to
    the last coord of n. to do so, first start with orientation of s1,
    then looks for the next stroke and so on. Fails if there are multiple
    possibiklities. (i.e., this only works if the strokes are a single
    connected path, no brnaching). must pass in thresh, for calling pts coonected.
    in pxiels
    - sanity_check, then checks that output matches input pts exactly. This is additional 
    to the default chewcks.
    NOTE:
    - Default sanity checks: if either fail to connect a single time, or findmultiple
    connections (e..g, branch), or your thresh was not good value, then will walways fail;.
    """
    if reorder==False:
        return [np.concatenate(strokes,axis=0)]
    else:

        def _dist(pt1, pt2):
            return np.linalg.norm(pt1-pt2)

        def _find_strok_to_append(strokes_reordered, strokes_orig):
            out = None
            for i, s in enumerate(strokes_orig):
                for pos in [0, -1]: # try both ends
                    if pos==0:
                        pt = strokes_reordered[0][0,:2]
                    else:
                        pt = strokes_reordered[-1][-1, :2]
                    if _dist(s[0,:2], pt)<thresh:
                        if pos==0:
                            flip = True
                        else:
                            flip = False
                        assert out is None
                        out = (i, pos, flip)
                    elif _dist(s[-1,:2], pt)<thresh:
                        if pos==-1:
                            flip = True
                        else:
                            flip = False
                        assert out is None
                        out = (i, pos, flip)
            if out is None:
                assert False, "didnt find"
            return out

        # OLD METHOD, DOESNT WORK WELL since doesnt work if new strok can variably go to front or back.
        # def _find_ind_matching_stroke(sthis, strokes, return_none_if_fail=True):
        #     """ finds which stroke in strokes is connected to 
        #     sthis (i.e. to the last coord in sthis)
        #     returns ind, flip, 
        #     where ind indexes into strokes, and flip is bool indicating whether to flip 
        #     it
        #     """
        #     pt = sthis[-1, :2]
        #     for i, s in enumerate(strokes):
        #         if _dist(s[0,:2], pt)<thresh:
        #             print(_dist(s[0,:2], pt))
        #             return i, False
        #         elif _dist(s[-1,:2], pt)<thresh:
        #             print(_dist(s[-1,:2], pt))
        #             return i, True
        #     if return_none_if_fail:
        #         return None, None
        #     else:
        #         assert False, "did not find any"

        # strokes_reordered = []
        # strokes_orig = [s.copy() for s in strokes]
        # strok_last = strokes_orig.pop(0)
        # strokes_reordered.append(strok_last) # start with first stroke.
        # ct = 0
        # while len(strokes_orig)>0 and ct<len(strokes)+1:
        #     # Try appending to the end
        #     ind, flip = _find_ind_matching_stroke(strok_last, strokes_orig)
        #     if ind is not None:
        #         if flip:
        #             strokes_reordered.append(strokes_orig.pop(ind)[::-1])
        #         else:
        #             strokes_reordered.append(strokes_orig.pop(ind))
        #     else:
        #         # Try appending to the front
        #         ind, flip = _find_ind_matching_stroke(strokes_reordered[0][::-1], strokes_orig)
        #         if ind is None:
        #             assert False, "strokes noit actually connected?"
        #         flip = not flip # since sthis was flipped to enter.
        #         if flip:
        #             strokes_reordered.insert(0, strokes_orig.pop(ind)[::-1])
        #         else:
        #             strokes_reordered.insert(0, strokes_orig.pop(ind))

        #     ct+=1


        if len(strokes)>2:
            assert False, "check this, havent checkdd on len >2"
        strokes_reordered = []
        strokes_orig = [s.copy() for s in strokes]
        strokes_reordered.append(strokes_orig.pop(0)) # start with first stroke.
        ct = 0
        while len(strokes_orig)>0 and ct<len(strokes)+1:
            # look for next stroke to append
            i, pos, flip = _find_strok_to_append(strokes_reordered, strokes_orig)
             # = out
            snew = strokes_orig.pop(i)
            if flip:
                snew = snew[::-1]
            if pos==-1:
                strokes_reordered.append(snew)
            elif pos==0:
                strokes_reordered.insert(0, snew)
            else:
                assert False

            ct+=1

        strokes_out = concatStrokes(strokes_reordered)
        if sanity_check:
            assert strokes_out[0].shape[0]==np.sum([s.shape[0] for s in strokes]), "lost pts?"
            from pythonlib.drawmodel.strokedists import distscalarStrokes
            assert np.isclose(distscalarStrokes(strokes, strokes_out, "position_hd"), 0)
        return strokes_out


def intersect_traj_by_circle(traj, circle_radius, max_dist_along_pts=0.25):
    """ Finds index along traj that is interesected by circle whose origin is at
    onset of the traj.
    INPUT:
    - traj, Nx2
    - circle_radius, number
    - max_dist_along_pts, see other code in this file
    OUT:
    - index into traj. if circle is too big, returns []
    NOTE:
    - if multiple intersections (e.g., traj turns back), then will return the latest position
    on traj that is a threshold crossing from inside to outside circle.
    - returns the index immediately before the crossing.
    """

    # out of all pts that intersect with circle of radius threshold, take the one that is furthest away
    radius = circle_radius
    # - find all pts that intersect this circle.
    dists = np.linalg.norm(traj-traj[0], axis=1) # pts --> distances from traj onset.
    ind_max = min([len(traj)-2, int(np.ceil(max_dist_along_pts*traj.shape[0]))]) # -2, since diff below.
    inds_intersect = np.where(np.diff(dists[:ind_max]-radius>0))[0]
    if len(inds_intersect)>=1:
        ind = inds_intersect.max() # get the last crossing
    else:
        ind = None
    return ind




def merge_pts(pts1, pts2, up_to_idx):
    """
    Given two identical shape pts, finds an in-between traj by
    taking weighted avg of pts1 and pts2, with weight depending on 
    location along trajectory.
    INPUT:
    - pts1, Nx2
    - pts2, Nx2,
    - up_to_idx, all indices before this will take weighted avg of pts1 and 2.
    See below.
    OUT:
    - pts_new
    --- will not modify inputs
    NOTE:
    - onset of traj will be weighted to be more like pts1
    - offset of traj (up to up_to_idx) will be weighted more like pts2
    - the weights will be linear from 1-->0
    - inds after up_to_idx will be takend from pts2
    """
    
    # pts must be same length
    assert len(pts1)==len(pts2)
    
    # weight
    weights = np.linspace(1, 0, up_to_idx)

    # make new pts
    pts_new = np.copy(pts2)
    pts_new[:up_to_idx,:] = pts1[:up_to_idx,:]*weights[:, None] + pts2[:up_to_idx, :]*(1-weights[:, None])
    
    return pts_new

def smooth_pts_within_circle_radius(pts, radius, ploton=False, max_dist_along_pts=0.5,
    must_work=False):
    """ smooths pts within circle radius from pts onset, by taking weighted average
    of two trajs, one is original pts, the other is with a straight line drawn from
    onset to where pts intersects with circle of radius (centered at pts onset), and then continuing.
    IN:
    - pts, N x 2
    - radius, of circle
    - max_dist_along_pts, max travel along pts within wihchi to consider a candidate interseciotn
    pt. This important for loops, otherwise will choose the one closer to the return pt, and leads to a
    mess. in units of fraction of the entire traj.
    - must_work, then throw error if fail to find an intersection (e..g, if radius is too big)
    OUT:
    - pts_new.
    --- Does not modify pts.
    """
    from pythonlib.tools.stroketools import merge_pts
    from pythonlib.tools.stroketools import intersect_traj_by_circle

    if len(pts)==0:
        return pts
    # find the cutoff index
    idx = intersect_traj_by_circle(pts, radius, max_dist_along_pts=max_dist_along_pts)

    if idx is None:
        if must_work:
            print(pts)
            print(radius)
            print(max_dist_along_pts)
            idx = intersect_traj_by_circle(pts, radius, max_dist_along_pts=1)
            print("ideal max_dist_along_pts:", idx, len(pts), idx/len(pts))
            assert False, "didnt find..."
        else:
            return np.copy(pts)

    # copy pts, but with straight line from start to the cutoff index.
    pt0 = pts[0]
    pt1 = pts[idx]
    a = np.linspace(pt0[0], pt1[0], idx) # x
    b = np.linspace(pt0[1], pt1[1], idx) # y
    pts_shortcut = np.copy(pts)
    pts_shortcut[:idx, :] = np.c_[a, b]

    # merge them
    pts_new = merge_pts(pts_shortcut, pts, idx)

    if ploton:
        plt.figure()
        plt.plot(pts[:,0], pts[:,1], "--", label="long")
        plt.plot(pts_shortcut[:,0], pts_shortcut[:,1], "--", label="shortcut")

        # for nthis in list_nodes:
        #     o = self.get_node_dict(nthis)["o"]
        #     plt.plot(o[0], o[1], "kd", label="old_node_pos")
        # plt.plot(onew[0], onew[1], "ks", label="new")
    
        plt.plot(pts_new[:,0], pts_new[:,1], '-xr', label="merged");

        plt.legend()

    return pts_new


def travel_on_loop(pts_loop, pt_start, angle):
    """ given a loop and starting pt, traverse along it by angle (rad)
    then return the pt end
    INPUT:
    - pts_loop, Nx2, makes a loop. start location arbitray
    - pt_start, 1x2, assumes you enter something reasonable.
    - angle, radians, how much to go CCW, from the starting pt.
    OUT:
    - pt_new, does not have to snap to loop, but is distance from center
    depending on average radius from center.
    """
    from math import pi
    from pythonlib.tools.vectools import get_vector_from_angle, get_angle
    
    maxes = pts_loop.max(0)
    mins = pts_loop.min(0)
    center = np.c_[mins, maxes].mean(axis=1)

    # starting vector
    vec_start = pt_start - center
    angle_start = get_angle(vec_start)
    
    # add angle
    angle_end = angle_start + angle
    angle_end = angle_end%(2*pi)
    
    # convert new angle to a vector
    radius = np.mean(np.linalg.norm(pts_loop - center, axis=1))
    u = get_vector_from_angle(angle_end) * radius
    pt_new = center + u
    if False:
        print("vector_start", "angle1", "angle end", "radius", "center", "u", "pt_start", "pt_new")
        print(vec_start, angle_start, angle_end, radius, center, u, pt_start, pt_new)
    return pt_new


def split_strokes_large_jumps(strokes, thresh=50):
    """ Split strokes wherever there are large jumps between adjacnet pts
    IN:
    - storkes, ..
    - thresh, scalar distance, any jump  larger than this will be split into strokes.
    50 is good for task pts (after 2nd half of 2021)
    NOTE:
    - currently only works if there is at most one jump per stroke.
    - confirms that the flattened pts will not change. just how they are in strokes
    """

    # splits = []
    splitsdict = {} # e.g, splitsdict[i]=j means stroke i has jump between ind j and j+1
    for i, s in enumerate(strokes):
        ct = 0
        for j, (pt1, pt2) in enumerate(zip(s[:-1, :2], s[1:, :2])):
            if np.linalg.norm(pt2-pt1)>thresh:
                # splits.append((i, j))
                splitsdict[i] = j
                ct+=1
        assert ct<=1, "Only split each stroke once. otherwise need to rerun above (iterate)"

    # Make the splits
    strokes_new =[]
    for i, s in enumerate(strokes):
        if i not in splitsdict.keys():
            strokes_new.append(s)
        else:
            loc = splitsdict[i]
            strokes_new.append(s[:loc+1, :])
            strokes_new.append(s[loc+1:, :])

    # Sanity check, flattened strokes are not changed
    strokes_flat_in = np.concatenate(strokes, axis=0)
    strokes_flat_out = np.concatenate(strokes_new, axis=0)
    assert all(np.isclose(strokes_flat_in, strokes_flat_out).flatten())

    if False:
        plt.figure()
        for s in P.Strokes:
            plt.plot(s[:,0], s[:,1], 'x')
        plt.figure()
        for s in P.StrokesInterp:
            plt.plot(s[:,0], s[:,1], 'x')   
        plt.figure()
        for s in strokes_new:
            plt.plot(s[:,0], s[:,1], 'x')       

    return strokes_new


def insert_strok_into_strokes_to_maximize_alignment(strokes_template, strokes_mod, traj, 
    do_insertion=False):
    """ Finds the optimal location to insert traj into strokes_mod, in order to maximize
    similarity between strokes_mod and strokes_template.
    INPUT:
    - strokes_template, list of traj arrays. This will stay unchaged.
    - strokes_mod, list of traj, this is what will have a new traj inserted into
    - traj, (n,2) array
    OUT:
    - slot, location to insert. i.e., strokes_mod.insert(slot, traj).
    - strokes_mod modified, if do_insertion==True
    """
    from pythonlib.drawmodel.strokedists import distscalarStrokes
    # try sticking it in all possible locations
    list_d = []
    possible_slots = range(len(strokes_mod)+1) 
    for slot in possible_slots:
        strokes_tmp = [np.copy(s) for s in strokes_mod]
        strokes_tmp.insert(slot, traj)
        # compute distance
        d = distscalarStrokes(strokes_template, strokes_tmp, ver="dtw_segments")
        list_d.append(d)
    slot = list_d.index(min(list_d))

    if do_insertion:
        strokes_mod.insert(slot, traj)
    return slot

def sliceStrokes(strokes, twind, retain_n_strokes=False):
    """ Get a time slice of strokes, returned in strokes format.
    PARAMS:
    - [IGNORE] list_twind, list of twinds, where keeps pts that are within any of the 
    windows in this list. A single twind:
    - strokes: lsat column must be time!!
    - twind, [tstart, tend], time window (inclusive), in whatever units
    strokes are in. Usuallyl seconds.
    - return_n_strokes, bool (False), if True, then output strokes is same length as input.
    So some trajs in strokes can be empty lists. if False, then will only keep trajs
    that have non-zero legnth.
    RETURNS:
    - strokes_out
    """


    def _slice(traj):
        dim_time = traj.shape[1]-1 # num columns
        # assume last col is time.
        inds = (traj[:, dim_time]>=twind[0]) & (traj[:, dim_time]<=twind[1])
        return traj[inds, :]

    strokes_out = [_slice(s) for s in strokes]
    if retain_n_strokes==False:
        strokes_out = [s for s in strokes_out if len(s)>0]
    return strokes_out


################# STROKE FEATURE TIMECOURSE
def timepoint_extract_features_continuous(strokes, twind, list_feature=["mean_xy"]):
    """ Extract feature for this stroke at this timepoint
    PARAMS:
    - twind, [t1, t2], where feature uses data within this windopw (inclusinve)
    - list_feature, list of string, name of feature
    RETURNS:
    - features, list of np array of this fatures, each feature shape can depend on feature
    -- returns list of None (saem len as list_feature) if this window has no data...
    """
    from pythonlib.tools.stroketools import sliceStrokes
        
    # Slice strokes to get this time window
    strokes_sliced = sliceStrokes(strokes, twind, False)
    
    if len(strokes_sliced)==0:
        # no data, return None
        return [None for _ in range(len(list_feature))]
    
    pts = np.concatenate(strokes_sliced) # combine across traj, if this time window spans them --> (N,3) array
    # take mean within this slice
    features = []
    for f in list_feature:
        if f=="mean_xy":
            # (x,y), shape (2,)
            val = np.mean(pts,0)[:2]
        else:
            print(feature)
            assert False, "code it"
        features.append(val)
    
    return features
        
