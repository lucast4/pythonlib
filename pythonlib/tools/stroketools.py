"""general purpose thing that works with stroke objects, whicha re generally lists of arrays (T x 2) (sometimes Tx3 is there is time)
and each element in a list being a stroke.
- takes in single strokes and outputs transformed strokse or some feature...
"""
import numpy as np
from pythonlib.drawmodel.features import *
from pythonlib.drawmodel.strokedists import distanceDTW
import matplotlib.pyplot as plt
from ..drawmodel.behtaskalignment import assignStrokenumFromTask
from pythonlib.tools.camtools import euclidAlign, corrAlign, get_lags, fps, fps2, plotTrialsTrajectories, normalizeGaps, plotHeat
import pandas as pd
from pythonlib.tools.exceptions import NotEnoughDataException

# =============== TIME SERIES TOOLS
def create_generate_strokes_fake_debug(nstrokes):
    strokes = [np.random.rand(5,3) for _ in range(nstrokes)]
    return strokes

def strokesInterpolate(strokes, N, uniform=False, Nver="numpts"):
    """interpoaltes each stroke, such that there are 
    N timesteps in each stroke. uses the actual time to interpoatle
    strokes must be list of T x 3 np ararys) 
    - uniform, then interpolate uniformly based on index, will give 
    fake timesteps from 0 --> 1
    OBSOLETE - use strokesInterpolate2 isbntead, since is more flexible 
    in deciding how to interpllate."""

    return strokesInterpolate2(strokes, ["npts", N])

    # strokes_new = []
    # for s in strokes:
    #     if uniform:
    #         t_old = np.linspace(0,1, s.shape[0])
    #         t_new = np.linspace(0,1, N)
    #     else:
    #         t_old = s[:,2]
    #         t_new = np.linspace(t_old[0], t_old[-1], num=N)
    #     s_new = np.concatenate([
    #                     np.interp(t_new, t_old, s[:,0]).reshape(-1,1),
    #                     np.interp(t_new, t_old, s[:,1]).reshape(-1,1),
    #                     t_new.reshape(-1,1)],
    #                     axis=1)
    #     strokes_new.append(s_new)
    # return strokes_new

def strokesInterpolate2(strokes, N, kind="linear", base="time", plot_outcome=False):
        """ 
        NEW - use this instaed of strokesInterpolate.
        N is multipurpose to determine how to interpolate. e.g,.
        N = ["npts", 100], same time range, but 100 pts. or ["npts", 100, 150] if want 100 and 150 for strokes 1 and 2.
        N = ["updnsamp", 1.5] then up or down samples (here up by 1.5)
        N = ["fsnew", 1000, 125] then targets new fs 1000, assuming
        N = ["interval", interval], for spatial,
        N = ["input_times", (ntimes, )array], to get at these specific times.
        initially 125.
        - base, 
        -- index, then replaces time with index before interpolating.
        -- time, then just uses time
        -- space, then uses cum dist.
        --> useful if you want to interpolate evenly in a given dimension...
        RETURNS:
        - returns a copy
        NOTE:
        - if creates an empyt strok, then fixes by replacing with endpoints 
        of original strokes.
        """

        from scipy.interpolate import interp1d
        strokes_interp = []
        for stroke_ind, strok in enumerate(strokes):
            strok = np.array(strok.copy())
            n_columns = strok.shape[1]

            if base=="index":
                strok[:,-1] = np.arange(len(strok))
            elif base=="space":
                t_orig_for_space = strok[:,-1]
                strok = convertTimeCoord([strok], ver="dist")[0]
            else:
                assert base=="time"

            if strok.shape[0]==1:
                # then dont interpolate, only one pt
                strokinterp = strok
            else:
                # get new timepoints
                t = strok[:,-1]
                nold = len(t)
                COMPUTE_TNEW = True
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
                elif N[0]=="input_times":
                    tnew = N[1+stroke_ind] # (m,) array
                    try:
                        assert max(tnew)<=max(t), "interpolation timepoints must be within the data"
                        assert min(tnew)>=min(t), "interpolation timepoints must be within the data"
                    except Exception as err:
                        print('pts to interp',np.min(t),np.max(t))
                        print('ref pts',np.min(tnew),np.max(tnew))
                        raise err
                    COMPUTE_TNEW=False
                else:
                    print(N)
                    assert False, "not coded"
                if COMPUTE_TNEW:
                    tnew = np.linspace(t[0], t[-1], nnew)

                strokinterp = np.empty((len(tnew), n_columns))
                strokinterp[:,-1] = tnew
            
                # fill the x,y,z columns
                col_range= range(n_columns - 1)
                for i in col_range:
                    f = interp1d(t, strok[:,i], kind=kind)
                    strokinterp[:,i] = f(tnew)

                    if False:
                        plt.figure()
                        plt.plot(t, strok[:,0], '-ok');
                        plt.plot(tnew, f(tnew), '-or');

            # Replace in units of time
            HACK = False # To replace with units approximately time.
            if HACK and base=="space":
                t_time = np.linspace(t_orig_for_space[0], t_orig_for_space[-1], nnew)
                print(strokinterp.shape)
                print(strokinterp[:5,:])
                print(t_time[:5])
                strokinterp[:,-1] = t_time
            strokes_interp.append(strokinterp)

        # If new length is 0, replace with previous endpoints.
        for i, s in enumerate(strokes_interp):
            if len(s)==0:
                strokes_interp[i] = strokes[i][[0, -1], :]

        if plot_outcome:
            fig, axes = plt.subplots(2,2, figsize=(8,8))
            ax = axes.flatten()[0]
            for s in strokes:
                ax.plot(s[:,0], s[:,1], "-o")
            ax.set_title('original')
            
            ax = axes.flatten()[1]
            for s in strokes_interp:
                ax.plot(s[:,0], s[:,1], "-o")
            ax.set_title('after interpolation')

            ax = axes.flatten()[2]
            for s in strokes:
                ax.plot(s[:,2], s[:,0], "-.", label="orig_x", alpha=0.5)
                ax.plot(s[:,2], s[:,1], "-.", label="orig_y", alpha=0.5)
            for s in strokes_interp:
                ax.plot(s[:,2], s[:,0], "-x", label="interp_x", alpha=0.5)
                ax.plot(s[:,2], s[:,1], "-x", label="interp_y", alpha=0.5)

            ax.legend()
    
        return strokes_interp

def smoothStrokes(strokes, sample_rate, window_time=0.05, window_type="hanning",
                 adapt_win_len="adapt", sanity_check_endpoint_not_different=True,
                 DEBUG=False, clean_endpoints=True):
    """ returns copy of strokes, smoothed with window_time (seconds)
    - sample_rate in samp/sec (e.g., fd["params"]["sample_rate"])
    - adapt_win_len, what to do fro strokes that are shoerter than window.
    PARAMS:
    - strokes, list of np arrays, always assumes that the last column is time.
    """
    from .timeseriestools import  smoothDat

    if strokes[0].shape[1] == 4 :
        #turn off for 3d points bc errors
        sanity_check_endpoint_not_different = False

    window_len = int(np.floor(window_time/(1/sample_rate)))
    if window_len%2==0:
        window_len+=1
    window_len = int(window_len)

    # -- check that no strokes are shorter than window
    strokes_sm = []
    for s in strokes:
        did_adapt = False
        if len(s)<window_len:
            if adapt_win_len=="adapt":
                if False:
                    # Old method, before 3/2025. it is too large, leads to output strokes that are too much changed.
                    window_len = len(s)
                else:
                    pad_dur = 0.01 # time to subtract from entire stroke dur.
                    pad_n_samp = int(pad_dur*sample_rate)
                    window_len = len(s)-pad_n_samp # Added, to have less negaitve effect on short strokes.

                if window_len<1:
                    window_len = 1

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

        ### Do smoothing
        if DEBUG:
            print(window_len, window_time, sample_rate)
            print(window_type)
            print(len(s))
        
        if s.shape[1] == 2:
            strokes_sm.append(np.array([
                smoothDat(s[:,0], window_len=window_len, window=window_type),
                s[:,1]]).T)
        elif s.shape[1] == 3:
            strokes_sm.append(np.array([
                smoothDat(s[:,0], window_len=window_len, window=window_type), 
                smoothDat(s[:,1], window_len=window_len, window=window_type), 
                s[:,2]]).T)
        elif s.shape[1] == 4:
            strokes_sm.append(np.array([
                smoothDat(s[:,0], window_len=window_len, window=window_type), 
                smoothDat(s[:,1], window_len=window_len, window=window_type),
                smoothDat(s[:,2], window_len=window_len, window=window_type),
                s[:,3]]).T)
        else:
            print(s.shape)
            assert False

        if False:
            # debugging
            if did_adapt:
                print('smoothed')
                print(strokes_sm[-1].shape)
                print('orig')
                print(s.shape)

    if clean_endpoints:
        # FIx edge artifacts the edges, by taking weighted average between orig and 
        # new smoothed strokes.
        # This solves problem where sometimes short strokes are smoothed to the point where
        # the endpojnts are changed too much, and then it fails below.
        strokes_final = []
        for s_raw, s_sm in zip(strokes, strokes_sm):

            # tmp = np.stack([s1, s2], axis=2) # (times, dims, 2)
            # np.mean(tmp, axis=2)
            # np.average(tmp, axis=2, weights=)

            assert s_raw.shape == s_sm.shape
            npts = s_raw.shape[0]
            w = np.zeros(npts)[:, None]
            if npts>=10:
                # For edges, take some of the onset pts.
                # This done by hand, seems reasonable. The logic is that one should trust the on and off
                # position for each stroke, bsaed on touchscreen.
                w[0] = 1.
                w[1] = 0.66
                w[2] = 0.33
                w[3] = 0.16
                w[4] = 0.08

                w[-5] = 0.08
                w[-4] = 0.16
                w[-3] = 0.33
                w[-2] = 0.66
                w[-1] = 1.

                if False:
                    plt.figure()
                    plt.plot(w)
            else:
                # Then modify fewer pts at the edges.
                assert len(w)>=6, "assuming so below... Why keep such a short stroke anyway?"
                w[0] = 0.66
                w[1] = 0.33
                w[2] = 0.16
                w[-3] = 0.16
                w[-2] = 0.33
                w[-1] = 0.66

            s_new = s_sm*(1-w) + s_raw*w
            
            # Assign the actual values of the times. if take mean, then may run into numerical errors when try to check if is equal...
            if False: # Not needed...
                from pythonlib.tools.nptools import isnear
                assert isnear(s_sm[:, -1], s_raw[:, -1]), "why are times not the same?"
                s_new[:, -1] = s_sm[:, -1]

            strokes_final.append(s_new)

        if False:
            fig, axes = plt.subplots(1,3, sharex=True, sharey=True)
            
            from pythonlib.drawmodel.strokePlots import plotDatStrokesTimecourse
            ax = axes.flatten()[0]
            plotDatStrokesTimecourse(strokes, ax=ax)

            ax = axes.flatten()[1]
            plotDatStrokesTimecourse(strokes_sm, ax=ax)
            ax.set_title("strokesFilter() --> Filtered")

            ax = axes.flatten()[2]
            plotDatStrokesTimecourse(strokes_final, ax=ax)
            ax.set_title("combined")

            fig.savefig("/tmp/tmp1.png")

            fig, axes = plt.subplots(1,3, sharex=True, sharey=True)
            ax = axes.flatten()[0]
            plotDatStrokesWrapper(strokes, ax)

            ax = axes.flatten()[1]
            plotDatStrokesWrapper(strokes_sm, ax)

            ax = axes.flatten()[2]
            plotDatStrokesWrapper(strokes_final, ax)

            fig.savefig("/tmp/tmp2.png")

            import random
            if random.random() < 0.02:
                assert False
        strokes_sm = strokes_final

    # if plotprepost_xy:
    #     # Overlay strokes on (x,y) plot
    #     fig, ax = plt.subplots()
    #     ax.plot(strok[:,0], strok[:,1], '-xk', alpha=0.8, label="input")
    #     ax.plot(strokf[:,0], strokf[:,1], '-or', alpha=0.2, label="filtered")
    #     # Compare to just smoothing
    #     stroksm = smoothStrokes([strok], fs, window_time=1/Wn, window_type="hanning",
    #                  adapt_win_len="adapt")[0]
    #     ax.plot(stroksm[:,0], stroksm[:,1], '-og', alpha=0.2, label="smoothed")
    #     plt.legend()
    #     ax.set_title(f"Stroke {_i} (Wn={Wn})")
    #
    # # Save
    # strokesfilt.append(strokf)
    #
    # # -- compare strokes pre and post (timecourses)
    # if plotprepost:
    #     from pythonlib.drawmodel.strokePlots import plotDatStrokesTimecourse
    #     fig, axes = plt.subplots(2,1, sharex=True, sharey=True)
    #     ax = axes.flatten()[0]
    #     plotDatStrokesTimecourse(strokes, ax=ax)
    #     ax = axes.flatten()[1]
    #     plotDatStrokesTimecourse(strokesfilt, ax=ax)
    #     ax.set_title("strokesFilter() --> Filtered")

    if sanity_check_endpoint_not_different:
        # Sanity check
        _, _, diag = strokes_bounding_box_dimensions(strokes)

        if diag > 20: # Otherwise dont fail for things like dots, ebcuase of small diagonal.
            for s, sf in zip(strokes, strokes_sm):
                for idx_pt in [0, -1]:
                    
                    dist_old_to_new = np.linalg.norm(s[idx_pt, :2] - sf[idx_pt, :2])
                    # print(s)
                    # print('meow')
                    duration = s[-1,-1] - s[0,-1]

                    # Shorter duration strokes are more likely to have larger diff from filtering, so
                    # give them a bit more leweway
                    if duration<0.075:
                        # max_frac = 0.37
                        # max_frac = 0.26
                        max_frac = 0.2
                    elif duration<0.2:
                        # max_frac = 0.25
                        # max_frac = 0.18
                        max_frac = 0.15
                    elif duration<0.34:
                        # max_frac = 0.2
                        # max_frac = 0.14
                        max_frac = 0.1
                    else:
                        # max_frac = 0.18
                        # max_frac = 0.11
                        max_frac = 0.08

                    # print("-----")
                    # print(d)
                    # print(diag)
                    # print("-----")
                    if dist_old_to_new/diag > max_frac:
                        from pythonlib.drawmodel.strokePlots import plotDatStrokesWrapper
                        print(s)
                        print(sf)
                        print(dist_old_to_new)
                        print(dist_old_to_new/diag)
                        print(max_frac)

                        fig, axes = plt.subplots(1,2, sharex=True, sharey=True)
                        
                        from pythonlib.drawmodel.strokePlots import plotDatStrokesTimecourse
                        ax = axes.flatten()[0]
                        plotDatStrokesTimecourse(strokes, ax=ax)
                        ax = axes.flatten()[1]
                        plotDatStrokesTimecourse(strokes_sm, ax=ax)
                        ax.set_title("strokesFilter() --> Filtered")
                        fig.savefig("/tmp/tmp1.png")

                        fig, axes = plt.subplots(1,2, sharex=True, sharey=True)
                        ax = axes.flatten()[0]
                        plotDatStrokesWrapper(strokes, ax)

                        ax = axes.flatten()[1]
                        plotDatStrokesWrapper(strokes_sm, ax)

                        # Find velocity
                        fig.savefig("/tmp/tmp2.png")

                        assert False, "why smoothing made such big change to poisitons?"

    return strokes_sm

def strokesFilter(strokes, Wn, fs, N=9, plotresponse=False, 
    plotprepost=False, dims=(0,1), demean=False,
                  plotprepost_xy=False, DEBUG=False):
    """
    1/4/23 -
    filter each dimension of strokes (x,y).
    strokes is list of strok where a strok is N x 2(or 3, for t)
    array. assumes evenly sampled in time.
    - Wn is critical frequencies. in same units as fs
    [None <num>] does lowpass
    [<num> None] hp
    [<num> <num>] bandpass
    - returns copy
    NOTE: Tested taht this works well, better than smoothing, as long as the fs
    is not too low. (for lowpass). If too low, then, obviously, becomes close to a single
    dot.
    """
    #meowmewmoew
    # plotprepost_xy = True

    from scipy import signal
    assert dims==(0,1,2) or dims==(0,1), "not yet coded"

    if Wn[0] is None:
        btype = "lowpass"
        Wn = Wn[1]
        if not DEBUG:
            assert Wn>=10, "for drawing task, going lower if prob mistake. esp if this strokes is velocity..."
    elif Wn[1] is None:
        btype = "highpass"
        Wn = Wn[0]
    else:
        btype = "bandpass"

    # Filtering params
    sos = signal.butter(N, Wn, btype, analog=False, fs=fs, output='sos')
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
        
    # Apply filter
    strokesfilt = []
    for _i, strok in enumerate(strokes):
        
        assert strok.shape[1] == len(dims)+1, "will fail below."

        if np.all(np.isnan(strok[:,0])):
            # dont bother trying to smooth
            strokf = np.copy(strok)
        elif btype=="lowpass" and len(strok)<=padlen:
            # instead of filtering, uses smoothingw with adaptive windowsize
            strokf = smoothStrokes([strok], fs, window_time=1/Wn, window_type="hanning",
                         adapt_win_len="adapt")[0]
            # print('--')
            # print(strok.shape)
            # print(tmp[0].shape)
            # print(strok[:,2])
            # print(tmp[0][:,2])
            # fig, ax = plt.subplots()
            # ax.plot(strok[:,0], strok[:,1], '-xk')
            # ax.plot(strokf[:,0], strokf[:,1], '-or')
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

        if plotprepost_xy:
            # Overlay strokes on (x,y) plot
            fig, ax = plt.subplots()
            ax.plot(strok[:,0], strok[:,1], '-xk', alpha=0.8, label="input")
            ax.plot(strokf[:,0], strokf[:,1], '-or', alpha=0.2, label="filtered")
            # Compare to just smoothing
            stroksm = smoothStrokes([strok], fs, window_time=1/Wn, window_type="hanning",
                         adapt_win_len="adapt")[0]
            ax.plot(stroksm[:,0], stroksm[:,1], '-og', alpha=0.2, label="smoothed")
            plt.legend()
            ax.set_title(f"Stroke {_i} (Wn={Wn})")

        # Save
        strokesfilt.append(strokf)
        
    # -- compare strokes pre and post (timecourses)
    if plotprepost:
        from pythonlib.drawmodel.strokePlots import plotDatStrokesTimecourse
        fig, axes = plt.subplots(2,1, sharex=True, sharey=True)
        ax = axes.flatten()[0]
        plotDatStrokesTimecourse(strokes, ax=ax)
        ax = axes.flatten()[1]
        plotDatStrokesTimecourse(strokesfilt, ax=ax)
        ax.set_title("strokesFilter() --> Filtered")

    return strokesfilt
        

def strokesCurvature(strokes, fs, LP=5, fs_new = 30, absval = True, do_pre_filter=True, ploton=False,
                     plot_final_simple=False):
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

    if True:
        # assert LP>10, "this leads to problesm see strokeVel"
        # 1) Get velocity and accel
        strokes_vel = strokesVelocity(strokes, fs, fs_new = fs_new, lowpass_freq=LP, do_pre_filter=do_pre_filter, ploton=ploton)[0]
        strokes_accel = strokesVelocity(strokes_vel, fs, fs_new=fs_new, lowpass_freq=LP, do_pre_filter=False, ploton=ploton)[0]
    #     print(strokes_vel[0].shape)
    #     print(strokes_accel[0].shape)
    #     print(strokes[0].shape)
    else:
        strokes_vel = strokesVelocity(strokes, fs, clean=True)[0]
        strokes_accel = strokesVelocity(strokes_vel, fs, clean=True)[0]

    
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

    if plot_final_simple:
        # Plot the final result. Just plot one exmaple.
        strok = strokes[0]
        strokcurv = strokes_curv[0]
        fig, axes = plt.subplots(2,1)

        ax = axes.flatten()[0]
        ax.plot(strokcurv[:, 1], strokcurv[:, 0],"-x")
        ax.set_ylim([0, 1])
        ax.set_ylabel("curvature")
        ax.set_xlabel("time")

        ax = axes.flatten()[1]
        ax.scatter(strok[:, 0], strok[:, 1], c=strokcurv[:, 0], alpha=1)        

    return strokes_curv


def strokes_bin_timesegments_wrapper(strokes, binsize = 0.01):
    """ bin strokes into short segments by time, and for each time bin
    extract mean x, y,
    PARAMS:
    - strokes, list of (N,3) arrays, where 3rd column is time in sec
    - binsize, in sec. aligns (starts at ) 0. NOTE: strokes could be 
    positon or velcoity.
    RETURNS:
    - list of dicts, each a time bin.
    """

    strokes_array = np.concatenate(strokes, axis=0) # concat

    # get time bins
    tmin = 0
    tmax = strokes_array[-1,2] + binsize
    binedges = np.arange(0, tmax, binsize)-0.0005 # to avoid numerical imprecision at edges

    # collect each time bin, in incraesing order
    out = []
    for ed1, ed2 in zip(binedges[:-1], binedges[1:]):
        
        # get indices within this
        inds = (strokes_array[:,2]>=ed1) & (strokes_array[:,2]<ed2)
        npts = sum(inds)
        if npts>0:
            # then keep
            vel = np.mean(strokes_array[inds, :], axis=0)
    #         list_angles_binned = bin_angle_by_direction(list_angles, num_angle_bins=8)
    #         self.Dat["velmean_normbin"] = bin_values(list_magn, nbins=4)
            t1_actual = np.min(strokes_array[inds, 2])
            t2_actual = np.max(strokes_array[inds, 2])
            
            out.append({
                "npts":npts,
                "x":vel[0],
                "y":vel[1],
                "t1_bin_inclusive":ed1,
                "t2_bin_exclusive":ed2,
                "t1_actual":t1_actual,
                "t2_actual":t2_actual
            })

    return out

def strokes_bin_velocity_wrapper(strokes, fs, binsize=0.01, return_as_dataframe=False):
    """ bin strokes into short segments by time, and for each
    extract mean velocity, in cartesian and polar coords
    """
    from pythonlib.tools.vectools import get_angle, bin_angle_by_direction, cart_to_polar

    # from pythonlib.tools
    strokes_vel, strokes_speed = strokesVelocity(strokes, fs, clean=True)
    out = strokes_bin_timesegments_wrapper(strokes_vel, binsize)

    for o in out:
        a, norm = cart_to_polar(o["x"], o["y"])
        o["angle"]=a
        o["length"]=norm

    if return_as_dataframe:
        import pandas as pd
        return pd.DataFrame(out)
    else:
        return out
    
def sample_rate_equalize_across_strokes(strokes):
    """
    Make all strok in strokes have the same sample rate, ie.. 
    If any strok is off, then solves by interpoalting to global fs,
    which is determined using the good strokes in strok.

    RETURNS:
    - strokes, copies for anything that must interpolate. references for others.
    """
    from pythonlib.tools.exceptions import NotEnoughDataException

    # First, get a single consistent sample rate
    fs = sample_rate_from_strokes(strokes, allow_failures=True)
    print("Found this sample rate automatically: ", fs)

    # Now interpolate all the strokes to this, if they are not fs
    list_strok_clean = []
    n_bad = 0
    n_good = 0
    for strok in strokes:
        try:
            # Check if has clean sample rate. if it does, then it will surely have the right fs, due to the above step.
            sample_rate_from_strok(strok, suppress_print=True)
            list_strok_clean.append(strok)
            n_good+=1
        except NotEnoughDataException as err:
            plt.close("all")
            t0 = strok[0, 2]
            t1 = strok[-1, 2]
            period = 1/fs
            times_new = np.arange(t0+0.0005, t1-0.0005, period)
            strok_new = strokesInterpolate2([strok], ["input_times", times_new], plot_outcome=False)[0]
            list_strok_clean.append(strok_new)
            n_bad+=1

    if n_bad/n_good > 0.01:
        print(n_bad, n_good)
        assert False

    return list_strok_clean

def sample_rate_from_strok(strok, suppress_print=False):
    """
    Get fs from using the time differences between samples.
    Does sanity check for low variance.
    RETURNS:
    - Raises NotEnoughDataException if gap durations are not consistent across
    gaps. 
    """
    gaps =np.diff(strok[:, 2])
    gap_mean = np.mean(gaps)
    gap_std = np.std(gaps)
    if gap_std>=(0.01 * gap_mean):
        if not suppress_print:
            print(gap_std, "greater than: ", (0.01 * gap_mean))
            print(gap_mean)
            print(np.min(gaps), np.max(gaps))
            print(gaps)

            fig, axes = plt.subplots(1,3, figsize=(9,3))
            ax = axes.flatten()[0]
            ax.plot(gaps, "ok")

            ax = axes.flatten()[1]
            ax.plot(strok[:, 2], strok[:, 0], "x", label="x")
            ax.plot(strok[:, 2], strok[:, 0], "x", label="y")
            ax.legend()

            ax = axes.flatten()[2]
            ax.scatter(strok[:, 0], strok[:, 1], c=strok[:, 2])

            fig.savefig("/tmp/debug.pdf")
            print("too much variation in gap durations across samples!")
        raise NotEnoughDataException
    fs = 1/gap_mean
    return fs

def sample_rate_from_strokes(strokes, allow_failures=False):
    """
    Get fs from using the time differences between samples. 
    Does separately fro each strok, then takes the median, doing sanity check for low variance.
    """
    if allow_failures:
        # Then skip any strokes that fail (ie have periods that are not clean)
        list_fs = []
        for strok in strokes:
            try:
                list_fs.append(sample_rate_from_strok(strok, suppress_print=True))
            except NotEnoughDataException as err:
                pass                 
    else:
        list_fs = [sample_rate_from_strok(strok) for strok in strokes]
    min_fs = np.min(list_fs)
    med_fs = np.median(list_fs)
    max_fs = np.max(list_fs)
    assert max_fs<1.02*med_fs
    assert min_fs>0.98*med_fs
    return med_fs

    # strok = np.concatenate(strokes[:2], axis=0)
    # gaps =np.diff(strok[:, 2])
    # gap_mean = np.mean(gaps)
    # gap_std = np.std(gaps)
    # if gap_std>=(0.01 * gap_mean):
    #     print(gap_std)
    #     print(gap_mean)
    #     print(gaps)
    #     fig, ax = plt.subplots()
    #     ax.plot(gaps, "ok")
    #     assert False, "too much variation in gap durations across samples!"
    # fs = 1/gap_mean
    # return fs

# def strokesVelocity(strokes, fs, ploton=False, lowpass_freq = 15,
#     fs_new = 30, do_pre_filter=False, clean=False):
#NOTE: 2/8/23 - changed lowpass_freq to 5, this is what is used for plots generally, which
# call this function..
def strokesVelocity(strokes, fs, ploton=False, lowpass_freq = None,
    fs_new = None, do_pre_filter=True, clean=True,
    DEBUG = False, SKIP_POST_FILTERING_LOWPASS = False,
                    ADAPTIVE_FS_NEW = True):
    """
    UPDATE 1/3/23 - Lots of testing.
    Tested vairation in parasm here. And cleaned up code in strokeFilter().
    Result:
    - Most important is that (SKIP_POST_FILTERING_LOWPASS, lowpass_freq) should not have
    lowpass_freq too low (<10hz), as this leads to weird behavior for short storkes, and is too
    much smoothing (and edge effects). Note that for longer strokes this actually looks good.
    Above 10hz is doing better, but 12hz is about as low as makes sense, if lower than edge is
    filtered out. Note that is ok to NOT do tjhis (i.e., SKIP_POST_FILTERING_LOWPASS=False), but
    this leads to noisy velocity traces.
    - Decided that fs_new=25 and do_pre_filter=True is useful, as this helps reduced jitteriness
    in the input x/y position data.
    - Tested for Diego and Pancho, PIG and char, and Luca (in colony)
    - Overall am satisfied that its working well, and generally across all stroke types.
    Limitations:
    - still too squiggly sometimes, but leave as is and can update in downstream code.
    Did:
    - Turned clean=True to default. This is becuase now clean actually means still quite swiggly, whears
    previously clean was super smoothed, but not good.
    To test:
    - Run DS.extract_strokes_as_velocity_debug(), which makes plots showing each step of this computaiton.
    - Looked closely at the velocioty traces (clear peaks) and speed (clear num peaks that matches
    what expect based on the xy image).

    OLDER DOC:
    gets velocity and speeds.
    should first have filtered strokes to ~15-20hz. if not, then 
    activate flag to run filter here. 
    INPUTS:
    - fs, original sample rate (use this even if already filtered, as is always
    done if use the getTrialsStrokes code in uitls, which Dataset does use). e..g
    fs = filedata["params"]["sample_rate"]
    - lowpass_freq, applies this to smooth at end (i.e., on vel traces, NOT on xy traces).
    Therefore make this high. Keep it None, since it doesnt do much. Or
    make it 15+ (ideally 20).
    - fs_new, how much to downsample before doing 5pt differnetiation. 
    assumes that have already filtered with lowpass below fs_new/2 [yes, filters
    at 15hz if use getTrialsStrokes. if have not, then set do_pre_filter=True,
    and will first filter data.
    empriically, 30hz for fs_new wiorks well. the ;opwer ot is thje smoother
    the velocy (e.g, 10-15 woudl be good). 
    - clean, then this is what I consider good for plotting, smooth bumps, etc. used
    in the wrapper code for plotting. Dont go lower than 15hz, can leads to very weid thigns.
    - SKIP_POST_FILTERING_LOWPASS, bool, keep True to skip  lowpass_freq, sicne its not
    doing much.
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
    - strokesvel, which should be same size as strokes
    - strokesspeed, which should be list of Nx2 arrays, where col2 is time.
    the timesteps for both of these outputs will match exactly the time inputs for
    strokes.
    
    UPDATE 10/15/23 - to cmpute speed after ALL steps for computing velocity (interpoalte, filter) otherwise
    you get negative values for speed...
    """ 

    # if fs_new is None or fs_new ==30:
    #     print("here", fs_new)
    #     assert False
    import matplotlib.pyplot as plt

    if fs is None:
        # get it autoamtically.
        fs = sample_rate_from_strokes(strokes)

    # Prep variables.
    strokes = [x.copy() for x in strokes]
    sample_rate = fs
    del fs

    if DEBUG:
        ploton=True

    # Defaults
    if fs_new is None:
        fs_new = 25
    if lowpass_freq is None:
        lowpass_freq = 12
    if False:
        if lowpass_freq<15:
            print(lowpass_freq)
            assert False, "too low leads to weird thigns, as this filter applies on VELOCITY, not on XY"

    # Overwrite with good set of params. (See docs).
    if clean:
    #     # used to be 5, but
    #     # assert False, "dont use <15hz,, leads to weird things"
        # Why these params? See above.
        # Turn on filtering of input xy
        do_pre_filter = True
        fs_new = 25
        # Turn on final filtering of vels
        SKIP_POST_FILTERING_LOWPASS = False
        lowpass_freq = 12
    #     lowpass_freq = 15

        # Filtering is adaptive
        ADAPTIVE_FS_NEW = True

    if ploton:
        fig, axes = plt.subplots(6,1, figsize=(10, 25))

    if do_pre_filter:
        # Filter, lowpass < half of fs_new.
        if ploton:
            ax = axes.flatten()[0]
            ax.set_title(f'stroke (xy) before and after fitlter ({fs_new/2}hz) (fsorig={sample_rate})')
            for i_s, strok in enumerate(strokes):
                for i in [0,1]:
                    ax.plot(strok[:,2], strok[:,i], label=f"s_{i_s}-dim{i}")

        if sample_rate>fs_new*2:
            strokes = strokesFilter(strokes, Wn = [None, fs_new/2],
                fs = sample_rate)
        if ploton:
            for i_s, strok in enumerate(strokes):
                for i in [0,1]:
                    ax.plot(strok[:,2], strok[:,i],  label=f"s_{i_s}-dim{i}")
            ax.legend()

    # ----- 1) downsample, this leads to better estimate for velocity.
    # before downsasmple, save vec length for later upsample
    # OBSOLETE - do filtering of each strok one by one instead.
    # n_each_stroke = [s.shape[0] for s in strokes]
    # strokes_down = strokesInterpolate2(strokes, N=["fsnew", fs_new, sample_rate])

    # containers for velocities and speeds
    strokes_vels = []
    strokes_speeds = []

    # for j, (strok, n_orig) in enumerate(zip(strokes_down, n_each_stroke)):
    for strok_orig in strokes:

        ## Interpolate.
        n_orig = strok_orig.shape[0]
        n_new_predicted = int(np.floor(n_orig * (fs_new/sample_rate)))
        if n_new_predicted<6 and ADAPTIVE_FS_NEW:
            # Then adaptively change fs to stil be able to compute

            fs_new_this = int(np.ceil((6 * sample_rate)/n_orig))
            n_new_predicted = int(np.floor(n_orig * (fs_new_this/sample_rate)))
            if DEBUG:
                print("ADAPTIVE (fsnew, npts)", fs_new_this, n_new_predicted)
        else:
            fs_new_this = fs_new
        strok = strokesInterpolate2([strok_orig], N=["fsnew", fs_new_this, sample_rate])[0]    
        time = strok[:,2].reshape(-1,1)

        if len(time)<6:
            if ADAPTIVE_FS_NEW:
                # THen too short. shouldnt get here if adaptve
                print("orig fs: ", sample_rate)
                print("new fs: ", fs_new_this)
                print("len strok_orig old: ", len(strok_orig))
                print("len strok new: ", len(strok))
                assert False

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
                ax = axes.flatten()[1]
                ax.set_title('stroke (xy) after downsample (interp)')
                for col in [0,1]:
                    ax.plot(strok[:,2], strok[:,col],  "-o", alpha=0.5, label=f"orig pos, dim{col}")
                    ax.plot(strok_orig[:,2], strok_orig[:,col],  "-x", alpha=0.5, label=f"orig pos, dim{col}")

            # ------------ 2) differntiate
            # what is sample peridocity? (actually compute exact value, since might
            # be slightly different from expected due to rounding erroes in interpoalte.
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
            if False:
                # Do below instead, after interpolating vel. reason: to avoid having negative speeds.
                strok_speeds = np.linalg.norm(strok_vels[:,[0,1]], axis=1).reshape(-1,1)

            # ------------- 4) interpolate and upsample velocity back to original timesteps.
            # 1) velocities
            strok_vels_t = np.concatenate([strok_vels, time], axis=1)
            kind_upsample = "cubic" # doesnt matter too much, cubix vs. linear
            # seems like cubic better.
            if ploton:
                ax = axes.flatten()[2]
                for col in [0,1]:
                    ax.set_title("vel, before and after upsample (and x and y)")
                    ax.plot(strok_vels_t[:,2], strok_vels_t[:,col],  "-o",alpha=0.5, label=f"vel, dim{col}")
            strok_vels_t = strokesInterpolate2([strok_vels_t], N=["npts", n_orig], kind=kind_upsample)[0]
            if ploton:
                for col in [0,1]:
                    ax.plot(strok_vels_t[:,2], strok_vels_t[:,col],  "-o",alpha=0.5, label=f"vel, dim{col}")

            if not SKIP_POST_FILTERING_LOWPASS:
                # FIlter again.
                # Skip, since tHis doesnt do much! Tested.
                # assert False, "this doesnt help.. comment out if you really want to do it."
                import matplotlib.pyplot as plt
                if sample_rate/2>lowpass_freq:
                    if DEBUG:
                        strok_vels_t = strokesFilter([strok_vels_t], Wn = [None, lowpass_freq], fs = sample_rate,
                                                     plotprepost_xy=ploton, plotprepost=ploton)[0]
                    else:
                        strok_vels_t = strokesFilter([strok_vels_t], Wn = [None, lowpass_freq], fs = sample_rate)[0]

                if ploton:
                    ax = axes.flatten()[2]
                    for col in [0,1]:
                        ax.plot(strok_vels_t[:,2], strok_vels_t[:,col], "-x", alpha=0.5, label="vel, after filter")
                    plt.legend()

            # Collect.
            strokes_vels.append(strok_vels_t)

            ################## SPEED
            # 2) speed
            strok_speeds = np.linalg.norm(strok_vels_t[:,[0,1]], axis=1).reshape(-1,1) # (N,1)
            strok_speeds_t = np.concatenate([strok_speeds, strok_speeds, strok_vels_t[:,2][:,None]], axis=1)

            if False:
                # vels have already been interpoalted...
                strok_speeds_t = strokesInterpolate2([strok_speeds_t],  N=["npts", n_orig], kind=kind_upsample)[0]

            assert np.all(strok_speeds_t[:,0]>=0), "sanity check, this used to happen sometimes if filter"
            assert not np.any(np.isnan(strok_speeds_t[:,0])), "sanity check, NAN, this used to happen sometimes if filter"

            if ploton:
                ax = axes.flatten()[4]
                ax.set_title("speed")
                for col in [0]:
                    ax.plot(strok_speeds_t[:,2], strok_speeds_t[:,col], "-og",label=f"speed")

            strokes_speeds.append(strok_speeds_t[:,[0, 2]])
    #
    # if DEBUG:
    #     # Debugging, show what filtering wopuld look like
    #     strokesFilter(strokes_vels, Wn = [None, lowpass_freq], fs = sample_rate,
    #           plotprepost_xy=ploton, plotprepost=ploton)
    #     assert False, "just plotting what would look like if you filtered... i.e,.,m filtering is not necesary."

    if False:
        # dont filter, you've already filter vels avbove.
        # -------- filter speed
        tmp = [np.concatenate([S[:,0, None], S[:,0, None], S[:,1, None]], axis=1) for S in strokes_speeds]
        strokes_speeds = strokesFilter(tmp, Wn = [None, lowpass_freq], fs = sample_rate)
        strokes_speeds = [np.concatenate([S[:,0, None], S[:,2, None]], axis=1) for S in strokes_speeds]

    ## SANITY CHECKS
    # for s in strokes_speeds:
    #     if not np.all((s[:,0]>=0) | (np.isnan(s[:,0]))):
    #         print(s)
    #         print(s[:,0]<0)
    #         print(np.isnan(s[:,0]))
    #         print(np.all(np.isnan(s[:,0])))
    #         print((s[:,0]>=0) | (np.isnan(s[:,0])))
    #         assert False

    if ploton:
        ax = axes.flatten()[5]
        ax.set_title("FINAL")
        for S in strokes_speeds:
            ax.plot(S[:,1], S[:,0], "-o", alpha=0.5, label="speed")
        for i, S in enumerate(strokes_vels):
            for col in [0, 1]:
                ax.plot(S[:,2], S[:,col], "-o", alpha=0.5, label=f"vel {col}")
        for ax in axes.flatten():
            ax.legend()

    return strokes_vels, strokes_speeds


def feature_velocity_vector_angle_norm(strok):
    """
    A single vector, which is the average vector over all timesteps
    RETURNS:
    - velmean, (2,) which is mean x and y vel during stroke.
    - angle, radians
    - norm, length of vector.
    """
    from pythonlib.tools.vectools import cart_to_polar

    # First, convert to velocity
    strok_vel = strokesVelocity([strok], fs=None)[0][0]

    # tvals = strok[:,2]
    # tvals = tvals-tvals[0] # get relative to onset
    # inds = (tvals>=twind[0]) & (tvals<=twind[1]) # bool mask

    # s = strok[inds, :2]
    velmean = np.mean(strok_vel[:, :2],axis=0) # (x,y)

    # Also get polar
    angle, norm = cart_to_polar(velmean[0], velmean[1])

    return velmean, angle, norm

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
    - h is time interval. this tells the code what the scale of the 
    output is (i.e, the units). leave as 1 if the input is unitless. 
    (e..g, if 100hz, then the intervals are 10ms, can input h=10)
    """
    
    if isinstance(x, list):
        x = np.array(x)

    if False:
        # doesnt work well - large discontnuities at edges.
        x = np.concatenate([x[0, None], x[0, None], x, x[-1, None], x[-1, None]])
    else:
        # Pad edges.
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
    -- "dist", is distance traveled in pixels at each pt,
    starting from onset of first stroke. will assume zero dist from stroke
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



def fakeTimesteps(strokes, point=None, ver="in_order"):
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
    if point is None:
        point = np.array([0.,0.])

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
    RETURNS:
    - copy of strokes, modified.
    """

    strokes = [s.copy() for s in strokes]

    if ver=="stretch_to_1":
        # First, make all pts positive
        strokes = strokes_make_all_pts_positive(strokes)
        pos = np.concatenate(strokes)
        maxval = np.max(np.abs(pos[:,[0,1]]))
        if strokes[0].shape[1]==3:
            strokes = [np.concatenate((s[:,[0,1]]/maxval, s[:,2].reshape(-1,1)), axis=1) for s in strokes]
        else:
            strokes = [s[:,[0,1]]/maxval for s in strokes]
    elif ver=="stretch_to_1_diag":
        # Keeps aspect ratio, stregth so that longest diagonal is 1
        w, h, d = strokes_bounding_box_dimensions(strokes)
        strokes = strokes_make_all_pts_positive(strokes)
        for s in strokes:
            s[:, [0,1]] = s[:, [0,1]]/d
    else:
        print(ver)
        assert False, "not codede"
    return strokes

def getCentersOfMass(strokes):
    """
    """
    from pythonlib.drawmodel.features import getCentersOfMass
    return getCentersOfMass(strokes, method="use_median")

def get_centers_strokes_list(strokes, method="bounding_box"):
    """
    Retrun list of (x,y) centers at same length of strokes
    RETURNS:
    - centers, list of (x,y) tuples
    """ 

    if method=="bounding_box":
        def _get_center(strok):
            xmin, xmax, ymin, ymax = strokes_bounding_box([strok])
            center = (xmin+(xmax-xmin)/2, ymin+(ymax-ymin)/2)
            return center
    else:
        assert False, "coode it"

    centers = [_get_center(strok) for strok in strokes]
    return centers

def getCenter(strokes, method="extrema"):
    """ get center of strokes (all strok at once), with methods:
    --- "extrema", based on x and y max edges, then
    find center.
    --- "com", then center of mass
    RETURNS: 2-length list.
    """
    if method=="com":
        assert False, "not coded, see features calc."""
    elif method == "extrema":
        # for each of x and y, take the mean of the extreme
        # values.
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

def strokes_bounding_box_dimensions(strokes):
    """ Return (w, h, d)
    """
    [minx, maxx, miny, maxy] = strokes_bounding_box(strokes)
    w = maxx-minx
    h = maxy-miny
    d = (w**2 + h**2)**0.5
    #
    # if d==0:
    #     print(strokes)
    #     assert False

    return w, h, d

def strokes_bounding_box(strokes):
    """ returns [minx, maxx, miny, maxy] that
    bounds strokes (all strokes concatted)"""
    return getMinMaxVals(strokes)

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

def concatStrokesTimeGap(strokes, MINTIME, DEBUG=False):
    """
    PARAMS:
    - MINTIME, in sec
    """

    # If adjacent strokes off and on close in time, then assume this is one stroke
    # and is touhscreeen error.
    def try_concatting_close_strokes(strokes_this):
        """ concats adj strokes if finds
        If dins contat, does it then immed returns. so max concats 1 gap.
        """
        from pythonlib.tools.stroketools import concatStrokesSpecific
        list_concat = []
        for i in range(len(strokes_this)-1):
            j = i+1
            s1 = strokes_this[i]
            s2 = strokes_this[j]

            toff = s1[-1, 2]
            ton = s2[0, 2]

            if ton - toff < MINTIME:

                if DEBUG:
                    print(i, j, toff, ton, MINTIME)

                # then concat them
                strokes_this = concatStrokesSpecific(strokes_this, [i, j])
                return strokes_this, True

        # failed to concat
        return strokes_this, False

    # Keep trying to concat until no concats
    did_concat = True
    while did_concat:
        strokes, did_concat = try_concatting_close_strokes(strokes)

    return strokes


def concatStrokesSpecific(strokes, inds_concat):
    """ Concat concescutive strok in strokes, returning strokes with
    fewer strok
    PARAMS:
    - inds_concat, list of adjacent indices to concat into single strok
    e.g, [1,2] for a len 4 strokes returns len 3, with first aned last unchagned,
    but middle on concatted.
    """
    # make sure adjacent indices only
    inds_concat = sorted(inds_concat)

    i_prev = inds_concat[0]
    for i in inds_concat[1:]:
        assert i == i_prev+1
        i_prev = i
    for i in inds_concat:
        assert i>=0
        assert i<len(strokes)

    strokes_new = strokes[:inds_concat[0]] + [np.concatenate([strokes[i] for i in inds_concat], axis=0)] + strokes[inds_concat[-1]+1:]

    return strokes_new

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

def split_strok_into_two_by_time(strok, off_time_1, on_time_2, PRINT=False):
    """
    Split a trajectory into two by inserting a gap.
    PARAMS:
    - off_time_1, the time (in sec) to cut off first stroke (ie start of gap)
    - on_time_2, the time onset of 2nd stroke (ie end of gap).
    RETURNS:
    - strok1, strok2
    """
    
    # sanity checks
    assert isinstance(strok, np.ndarray)
    idx_time = strok.shape[1]-1
    times = strok[:, idx_time]

    assert off_time_1 > times[0]
    assert on_time_2 < times[-1]
    assert on_time_2 > off_time_1

    ind_off = np.argmin(np.abs(times - off_time_1))
    ind_on = np.argmin(np.abs(times - on_time_2))
    
    strok1 = strok[:ind_off+1, :].copy()
    strok2 = strok[ind_on:, :].copy()

    if PRINT:
        print(strok.shape, ind_off, ind_on)
        print(strok[0, idx_time], strok[-1, idx_time])
        print(strok1[0, idx_time], strok1[-1, idx_time])
        print(strok2[0, idx_time], strok2[-1, idx_time])

    return strok1, strok2

def split_strokes2_to_align_to_strokes1(strokes1, strokes2, DEBUG=False):
    """
    You believe that strokes1 is subset fo strokes2 (in correct order),
    except that one strok in strokes2 is actually split into two in
    strokes1. e..g,
    strokes1 = [3, 4, 5a, 5b, 6]
    strokes2 = [1, 2, 3, 4, 5, 6, 7]. 

    And the onsets (in sec) for the strokes are aligned.

    Then this finds how to split storkes2 to mach strokes1.
    i.e, returns:
    strokes2 = [1, 2, 3, 4, 5a, 5b, 6, 7]. 
    """

    assert len(strokes1) <= len(strokes2), "this is assumed, in step that pads strokes 1"

    ### Get the arrays of onset times
    idx_time_1 = strokes1[0].shape[1]-1
    idx_time_2 = strokes2[0].shape[1]-1
    onset_times_dataset = np.array([s[0,idx_time_1] for s in strokes1])
    onset_times_cam = np.array([s[0,idx_time_2] for s in strokes2])

    ### Find where strok1 matches strok2
    # - hacky, to make sure that edge strokes for strokes 2 are not incorrectly matched to strokes in storkes1
    tmp = np.insert(onset_times_dataset, 0, onset_times_dataset[0]-0.2)
    tmp = np.append(tmp, tmp[-1]+0.2)
    matches = [np.argmin(np.abs(tmp - t)) for t in onset_times_cam]
    if matches[0]<matches[1]:
        matches = [m-1 for m in matches]

    if DEBUG:
        print("for each cam stroke, which dataset stroke does it match: ", matches)
        print(np.   diff(matches))

    ### Get the cam stroke that skips an index
    if False:
        if len(np.argwhere(np.diff(matches)==2))==1:
            # Then something like matches = [0, 0, 1, 2, 4].
            idx_cam_stroke_split = int(np.argwhere(np.diff(matches)==2)[0]) # the strok to split into two
        elif len(np.argwhere(np.diff(matches)==2))==0:
            # Still possible. This means the last strok in strokes2 should be split.
            idx_cam_stroke_split = len(strokes2)-1
        else:
            print(matches)
            assert False, "probably strokes2 and strokes1 are not related"
    else:
        assert len(np.argwhere(np.diff(matches)==2))==1
        idx_cam_stroke_split = int(np.argwhere(np.diff(matches)==2)[0]) # the strok to split into two
    
    # Which dataset strokes to consider
    idx_dataset_first_stroke = matches[idx_cam_stroke_split]
    idx_dataset_second_stroke = idx_dataset_first_stroke+1

    if DEBUG:
        print("Splitting cam stroke: ", idx_cam_stroke_split, "to match dataset strokes : ", idx_dataset_first_stroke, idx_dataset_second_stroke)
        # strokes1[idx_dataset_first_stroke][0, 2], strokes1[idx_dataset_second_stroke][0, 2]
        # strokes2[idx_cam_stroke_split][0, 3], strokes2[idx_cam_stroke_split+1][0, 3]

    # Check that the onset of the next cam stroke is after the offset of the 2nd dataset stroke
    if len(strokes2)>idx_cam_stroke_split+1:
        assert strokes2[idx_cam_stroke_split+1][0, 3] > strokes1[idx_dataset_second_stroke][-1, 2]
    off_time_first_stroke = strokes1[idx_dataset_first_stroke][-1, 2]
    on_time_second_stroke = strokes1[idx_dataset_second_stroke][0, 2]

    # Split the cam stroke into two
    strok1, strok2 = split_strok_into_two_by_time(strokes2[idx_cam_stroke_split], off_time_first_stroke, on_time_second_stroke)
    if DEBUG:
        print(strok1.shape, strok2.shape)

    # Replace the cam stroke with these two
    strokes2_split = strokes2[:idx_cam_stroke_split] + [strok1, strok2] + strokes2[idx_cam_stroke_split+1:] 

    return strokes2_split

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

def angle_of_stroke_segment(strokes, twind=[0, 0.2]):
    """
    REturn angles, one for each strok in strokes, for segments
    sliced using twind.
    :param strokes:
    :param twind: [t1, t2], in sec, relative to strok onset
    :return: angles, lsit of angles same len as strokes
    """
    from pythonlib.tools.vectools import get_angle

    if False: # no need -- should allow doing this with task strokes too (psychometric)
        tdiff = strokes[0][1,2] - strokes[0][0,2]
        if tdiff>=1:
            print("did you enter task strokes?")
            print(strokes)
            assert False

    strokes_sliced = sliceStrokes(strokes, twind, time_is_relative_each_onset=True)

    # Angle from on to off of stroke sequemnt
    angles = []
    for strok in strokes_sliced:
        v = strok[-1,:2] - strok[0, :2]
        angles.append(get_angle(v))

    return angles


def slice_strok_by_frac_bounds(strok, frac_low, frac_high):
    """
    Returns strok slices by fraction of the num pts. 
    PARAMS:
    - frac_low, frac_high, will return segement of stroke within bounds (inclusive).
    """

    npts = strok.shape[0]
    ind1 = int(frac_low*npts)
    ind2 = int(frac_high*npts)
    assert ind1<ind2

    return strok[ind1:ind2+1, :]

def sliceStrokes(strokes, twind, retain_n_strokes=False,
                 time_is_relative_each_onset=False, assert_no_lost_strokes=False):
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

    strokes = [s.copy() for s in strokes]
    n_in = len(strokes)

    if time_is_relative_each_onset:
        for s in strokes:
            s[:,2] = s[:,2] - s[0,2]

    def _slice(traj):
        assert traj.shape[1]>=2, "you dont have a time column..."
        # dim_time = traj.shape[1]-1 # num columns
        dim_time = 2
        # assume last col is time.
        inds = (traj[:, dim_time]>=twind[0]) & (traj[:, dim_time]<=twind[1])
        if len(inds)==0:
            print(traj)
            print(twind)
            assert False, "twind is wrong time base. maybe you confused whether this is task (0,1,2..) or beh (seconds) strokes?"
        return traj[inds, :]

    strokes_out = [_slice(s) for s in strokes]
    if retain_n_strokes==False:
        strokes_out = [s for s in strokes_out if len(s)>0]

    if assert_no_lost_strokes:
        assert len(strokes_out)==n_in
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
            print(f)
            assert False, "code it"
        features.append(val)
    
    return features
        

def strokes_average(strokes, Ninterp=70, center_at_onset=False, centerize=False,
                    ver="mean", rescale_strokes_ver = None):
    """ Get average of trajs in strokes, after linearly interpolating them
    to all be the same
    PARAMS:
    - center_at_onset, bool, if True, then first subtracts the onset of each stroke
    RETURNS:
    - strokes, list of np array
    = strokes_stacked, np array, shape (n strokes, Ninterp, 3)
    """

    # Do this first, since that's what i have been doing in DS._cluster_compute_mean_stroke
    if rescale_strokes_ver is not None:
        strokes = rescaleStrokes(strokes, rescale_strokes_ver)

    if centerize:
        strokes = strokes_centerize(strokes)

    if center_at_onset:
        strokes = strokes_alignonset(strokes)
        # def _center(strok):
        #     return strok - strok[0,:]
        # strokes = [_center(s) for s in strokes]

    # interpolate each strokes (using actual time)
    # stack the arrays and then take average
    stroklist_interp = strokesInterpolate(strokes, Ninterp)    
    strokes_stacked = np.stack(stroklist_interp)
    if ver=="mean":
        strok_mean = np.mean(strokes_stacked, axis=0)
    elif ver=="median":
        strok_mean = np.median(strokes_stacked, axis=0)
    else:
        assert False
    return strok_mean, strokes_stacked

def strokes_centerize_combined(strokes, method="bounding_box"):
    """ Return list ofs trokes (copy) which are all trasnalted such that the
    center of the entire strokes is (0,0). i.e. related position of stroks to each
    other unchanged"""

    if method=="center_of_mass":
        assert False, "code it"
        strok_total = concatStrokes(strokes)
        # strokes = [_centerize(strok) for strok in strokes]
    elif method=="bounding_box":
        strok_total = concatStrokes(strokes)[0]
        cen = get_centers_strokes_list([strok_total], method="bounding_box")[0]
        strokes = [s.copy() for s in strokes]
        for strok in strokes:
            strok[:,0] -= cen[0]
            strok[:,1] -= cen[1]
    else:
        assert False
        
    return strokes

def strokes_centerize(strokes, method="center_of_mass"):
    """ Return list ofs trokes (copy) which are EACH
    cetnred in xy coord, using the mean (i.e, center of mass)"""

    if method=="center_of_mass":
        def _centerize(strok):
            # c = np.mean(strok[:,:2], axis=0)
            # s = strok.copy()
            # s[:,:2] = s[:,:2] - c
            s = strok.copy()
            s[:,:2] = s[:,:2] - np.mean(s[:,:2], axis=0)
            return s
        strokes = [_centerize(strok) for strok in strokes]
    elif method=="bounding_box":
        strokes = [s.copy() for s in strokes]
        centers = get_centers_strokes_list(strokes, method="bounding_box")
        for cen, strok in zip(centers, strokes):
            strok[:,0] -= cen[0]
            strok[:,1] -= cen[1]
    else:
        assert False
        
    return strokes

def strokes_alignonset(strokes):
    """ Return list ofs trokes (copy) which are translated 
    so that (0,0) as the onset"""

    def F(strok):
        return strok - strok[0,:]
    strokes = [F(strok) for strok in strokes]
    return strokes

def strokes_make_all_pts_positive(strokes):
    """ Does this by translation.
    Does this by finding global xmin and ymin,
    then subtracting (xmin, ymin) from all strokes
    RETURNS:
        - translated copy of strokes
    """
    # FInd minimum
    pts = np.concatenate(strokes)
    xmin = np.min(pts[:,0])
    ymin = np.min(pts[:,1])
    ptmin = (xmin, ymin)

    strokes = [s.copy() for s in strokes]
    for s in strokes:
        s[:,[0,1]] = s[:,[0,1]] - ptmin

    return strokes

def strokes_to_hash_unique(strokes, nhash = 6, centerize=False, align_to_onset=False):
    """Helper to convert strokes to a unique hash number, which takes into acocunt onseta nd offest
    pts for eacch stroke. Is generalyl very good at distinguishing strokes, e.g., same semantic label, but
    diff psychometric rotration, etc.
    :param strokes: _description_
    :param nhash: n digits in hash. Usually I have 6 for fixed and 10 for random tasks
    :param include_taskstrings: _description_, defaults to True
    :param include_taskcat_only: _description_, defaults to False
    :return: _description_
    """

    if centerize:
        strokes = strokes_centerize(strokes)

    strokes = [s[:,:2] for s in strokes]

    # Collect each x,y coordinate, and flatten it into vals.
    vals = []
    # for S in self.Strokes:
    #     vals.extend(S[0])
    #     vals.extend(S[-1])
    for S in strokes:
        for SS in S:
            vals.extend(SS)
            # vals.extend(SS[1])
    # print(np.diff(vals))
    # tmp = np.sum(np.diff(vals))
    # vals = np.asarray(vals)

    vals = np.asarray(vals)
    # vals = vals+MIN # so that is positive. taking abs along not good enough, since suffers if task is symmetric.

    # Take product of sum of first and second halves.
    # NOTE: checked that np.product is bad - it blows up.
    # do this splitting thing so that takes into account sequence.
    tmp1 = np.sum(vals[0::4])
    tmp2 = np.sum(vals[1::4])
    tmp3 = np.sum(vals[2::4])
    tmp4 = np.sum(vals[3::4])

    # rescale to 1
    # otherwise some really large, some small.
    # divie by 10 to make sure they are quite large.
    tmp1 = tmp1/np.max([np.floor(tmp1)/10, 1])
    tmp2 = tmp2/np.max([np.floor(tmp2)/10, 1])
    tmp3 = tmp3/np.max([np.floor(tmp3)/10, 1])
    tmp4 = tmp4/np.max([np.floor(tmp4)/10, 1])


    # tmp1 = 1+tmp1-np.floor(tmp1)
    # tmp2 = 1+tmp2-np.floor(tmp2)
    # tmp3 = 1+tmp3-np.floor(tmp3)
    # print(tmp1, tmp2, tmp3, tmp4)
    # assert False

    # tmp1 = np.sum(vals)
    # tmp = np.sum(vals)

    # Take only digits after decimal pt.
    if True:
        tmp = tmp1*tmp2*tmp3*tmp4
        # print(tmp)
        tmp = tmp-np.floor(tmp)
        tmp = str(tmp)
        # print(tmp)
        # assert False
    else:
        # This doesnt work well, all tmps end up lookgin the same. 
        tmp = np.log(np.abs(tmp1)) + np.log(np.abs(tmp2)) + np.log(np.abs(tmp3))
        print(tmp)
        tmp = str(tmp)
        ind = tmp.find(".")
        tmp = tmp[:ind] + tmp[ind+1:]
    _hash = tmp[2:nhash+2]

    return _hash

def merge_interpolate_concat_strokes_halves(dfbasis, PLOT = False):
    """
    Create new strokes by taking first half of one stroke and 2nd half of another. Does this across all pairs of 
    rows in dfbasis.
    E.g., useful for created shuffled, null-hypothesis strokes

    PARAMS:
    - dfbasis, a dataframe with a column "strok". Will get pairwise merges across all pairs of values (rows) 
    of strok. Usualyl get this from DS...
    RETURNS:
    - dfbasis_merged, each row is a pair from dfbasis.
    """
    from pythonlib.tools.stroketools import strokes_bounding_box_dimensions

    # Confirm that all strokes are same length (assumes so below)
    tmp = list(set([len(strok) for strok in dfbasis["strok"]]))
    assert len(tmp)==1, "I thought basis strokes are all same length"
    npts = tmp[0]
    print("N pts in strokes: ", npts)
    idx_join = int(np.floor(npts/2))

    # Make sigmoid
    def sigmoid(x):
        # Note: slope of 5 was chosen by eye. is reasonable, the transition window is about 1/4 of total window.
        return 1 / (1 + np.exp(-5*x))
    x = np.linspace(-2, 2, npts)
    y = 1-sigmoid(x)[:, None] # (npts, 1)

    if PLOT:
        fig, ax = plt.subplots()
        ax.plot(x, y)

    ### Go thru al pairs of strokes
    res =[]
    for i1 in range(len(dfbasis)):
        for i2 in range(len(dfbasis)):
            if i1!=i2: # Do both ways, as the below is not symmetric
                print("Running: ", i1, i2)
                strok1 = dfbasis.iloc[i1]["strok"].copy()[:, :2]
                strok2 = dfbasis.iloc[i2]["strok"].copy()[:, :2]

                # transalte stroke 2, so that the onset location of 2nd half matches offset location of first half.
                on_second_half = strok2[idx_join, :]
                off_first_half = strok1[idx_join, :]

                shift_second_half = off_first_half - on_second_half
                strok2 = strok2 + shift_second_half
                strok_merged = (y * strok1) + ((1-y)*strok2)

                # Rescale so it matches size of the starting prims
                d1 = strokes_bounding_box_dimensions([strok1])[2]
                d2 = strokes_bounding_box_dimensions([strok2])[2]
                d3 = strokes_bounding_box_dimensions([strok_merged])[2]
                dmean = np.mean([d1, d2])
                strok_merged *= dmean/d3
                d3_final = strokes_bounding_box_dimensions([strok_merged])[2]
                # print(d1, d2, d3, d3_final)

                # recenter to onset.
                strok_merged -= strok_merged[0, :]

                # Put back time axis
                t = (dfbasis.iloc[i1]["strok"].copy()[:, 2] + dfbasis.iloc[i2]["strok"].copy()[:, 2])/2
                strok_merged = np.concatenate([strok_merged, t[:,None]], axis=1)

                if PLOT:
                    DS.plot_multiple_strok([strok1, strok2, strok_merged], overlay=False)

                ### Collect
                res.append({
                    "i1":i1,
                    "i2":i2,
                    "strok":strok_merged
                })

    # This holds each new merged basis set
    dfbasis_merged = pd.DataFrame(res)

    return dfbasis_merged

def has_self_intersection(traj):
    """
    Determine whether trajectory intersects itself (doesnt count if it's just one endpoint touching
    the traj).

    traj: numpy array of shape (n, 2)
    Returns True if the trajectory intersects itself, False otherwise

        # Example
        trajectory = np.array([
            [0, 0],
            [1, 1],
            [2, 0],
            [1, -1],
            [0, 0]  # back to start — self-intersection
        ])

        print(has_self_intersection(trajectory))  # Output: True
    """
    from shapely.geometry import LineString

    line = LineString(traj)
    return not line.is_simple  # is_simple is False if it self-intersects

def add_noise_jitter_to_stroke(strok, nland = 7, dist_int = 10.,
                               sigma = 20., plot=False):
    """
    Add noise (Generate new samples of a stroke), by converting to spline, jittering control pts,
    then reconverting to trajectory.
    """
    from pythonlib.drawmodel.splines import add_noise_jitter_to_stroke
    return add_noise_jitter_to_stroke(strok, nland, dist_int, sigma, plot)

   