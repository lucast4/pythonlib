
"""
Stuff for cam (x,y,z,t) points, especialyl for alignmenet with touchscreen data.
Written by D. Hanuska, originalyl in stroketools.py, and LT moved over here.

"""
import numpy as np
from pythonlib.drawmodel.features import *
from pythonlib.drawmodel.strokedists import distanceDTW
import matplotlib.pyplot as plt
from ..drawmodel.behtaskalignment import assignStrokenumFromTask

# Some additional functiosn for checking alignment of strokes. There may be similiar functions baove, but these ones I wrote for this specific puropose
def euclidAlign(cam_pts, touch_pts, ploton=False, UB = 0.15):
    plot_bound_size = 10

    fig, ax = plt.subplots(1,2,figsize=(30,10))
    large_len = len(cam_pts)
    small_len = len(touch_pts)
    cam_pts_xy = cam_pts[:,[0,1]]
    touch_pts_no_time = touch_pts[:,[0,1]]

    min_dist = float('inf')
    best_index = -1

    for i in range(large_len - small_len + 1):
        window = cam_pts_xy[i:i + small_len]
        distances = np.linalg.norm(window - touch_pts_no_time, axis=1)
        total_dist = np.sum(distances)

        if total_dist < min_dist:
            min_dist = total_dist
            best_index = i

    lag = [touch_pts[0,2],cam_pts[best_index,3]]
    lag_adj = lag[0] - lag[1]

    touch_lag_adj = touch_pts[:,2] - lag_adj

    if plot_bound_size <= best_index < len(cam_pts) - plot_bound_size:
        plot_bounds = (best_index-plot_bound_size,best_index+plot_bound_size)
    elif best_index < plot_bound_size and best_index < len(cam_pts) - plot_bound_size:
        plot_bounds = (0, best_index+plot_bound_size)
    elif best_index >= plot_bound_size and best_index > len(cam_pts) - plot_bound_size:
        plot_bounds = (best_index-plot_bound_size,len(cam_pts)-1)
    else:
        plot_bounds = (0,len(cam_pts)-1)

    best_ts = cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,3]

    if ploton:
        ax[0].plot(best_ts,cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,0], '.-', label = 'cam pts')
        # ax[0].plot(cam_pts[:,3],cam_pts[:,0], label='all cam pts')
        ax[0].plot(touch_lag_adj, touch_pts[:,0], '.-', label='touch lag adj')
        ax[0].plot(touch_pts[:,2], touch_pts[:,0], '.-', color='grey', label='raw touch')
        # ax.set_title('Trial:', trial)

        ax[1].plot(best_ts,cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,1], '.-', label='cam pts')
        # ax[1].plot(cam_pts[:,3], cam_pts[:,1], label='all cam pts')
        ax[1].plot(touch_lag_adj,touch_pts[:,1], '.-', label='touch lag adj')
        ax[1].plot(touch_pts[:,2],touch_pts[:,1], '.-', color='grey', label='raw touch')
        plt.legend()

    return lag,fig

def corrAlign(cam_pts, touch_pts, ploton=True, UB = 0.15, method='corr'):
    """Aligns touch pts to cam pts based on correlation
    Params:
        cam_pts (array)
        touch_pts (array)
        ploton (bool): Make plots?
        UB (numeric): Upper bound for how big lag can be (if let be too big, can mess up)
        method (str): 'corr' for use correlation, 'sim' for use dot product sim
    Returns:
        lag : [t0_touch, t0_stroke]. Can then align data using these nums
        fig : 4 panel fgirue for looking at lag visually for inspection. x axis is time, y axis is x or y coordinate repsp.
    """
    plt.style.use('dark_background')
    plot_bound_size = 0

    fig, ax = plt.subplots(2,2,figsize=(40,20), sharex=True)
    large_len = len(cam_pts)
    small_len = len(touch_pts)
    
    touch_pts_norm = touch_pts[:,[0,1]] - np.mean(touch_pts[:,[0,1]], axis=0)
    # touch_pts_norm = np.divide(touch_pts[:,[0,1]],np.max(touch_pts[:,[0,1]],axis=0))
    
    max_sim = 0
    best_index = -1
    sim_course = []
    false_alarms = []
    found_good_sim = False

    if method == 'sim':
        for i in range(large_len-small_len+1):
            window = cam_pts[i:i+small_len]
            window_norm = window[:,[0,1]] - np.mean(window[:,[0,1]], axis=0)
            # window_norm = np.divide(window[:,[0,1]],np.max(window[:,[0,1]],axis=0))
            sim = np.einsum('ij,ij->', window_norm, touch_pts_norm)

            this_lag = touch_pts[0,2] - cam_pts[i,2]
            if sim > max_sim and np.abs(this_lag) < UB:
                max_sim = sim
                best_index = i
                found_good_sim = True
            elif sim > max_sim and max_sim != 0:
                false_alarms.append(cam_pts[i,2])
            sim_course.append((cam_pts[i,2],sim))

    elif method == 'corr':
        # Normalize the segment
        touch_pts_ = touch_pts[:,[0,1]]
        ts_mean = np.mean(touch_pts_,axis=0)
        ts_std = np.std(touch_pts_,axis=0)
        ts_norm = (touch_pts_ - ts_mean)/(ts_std+1e-8)

        for i in range(large_len-small_len+1):
            window = cam_pts[i:i+small_len,[0,1]]
            window_mean = np.mean(window,axis=0)
            window_std = np.std(window,axis=0)

            # Normalize the window
            if np.any(window_std == 0):  # Avoid division by zero
                corr = np.zeros(2)
            else:
                window_norm = (window-window_mean)/(window_std+1e-8)
                corr = np.sum(ts_norm*window_norm,axis=0)/small_len
            total_corr = np.mean(corr)
            this_lag = touch_pts[0,2] - cam_pts[i,2]
            if total_corr > max_sim and np.abs(this_lag) < UB:
                max_sim = total_corr
                best_index = i
                found_good_sim = True
            elif total_corr > max_sim and max_sim != 0:
                false_alarms.append(cam_pts[i,2])
            
            sim_course.append((cam_pts[i,2],total_corr)) 

    else:
        assert False, 'Pick a valid method or learn to type fool.'

    #Only save if good peak found
    sim_course = np.array(sim_course)
    if found_good_sim:
        lag = [touch_pts[0,2],cam_pts[best_index,2]]
        lag_adj = lag[0] - lag[1]
    else:
        return None,None, 'no good match found'
    
    left_peak = False
    right_peak = False
    for i,s in enumerate(sim_course):
        if s[0] == cam_pts[best_index,2]:
            if i > 5:
                left_peak = np.all(sim_course[i-5:i,1] < max_sim)
            else:
                left_peak = np.all(sim_course[:i,1] < max_sim)
            if len(sim_course) > i+5:
                right_peak = np.all(sim_course[i+1:i+6,1] < max_sim)
            else:
                right_peak = np.all(sim_course[i+1:,1] < max_sim)
    if not (left_peak and right_peak):
        return None,None,'no good peak in sim course'

    touch_lag_adj = touch_pts[:,2] - lag_adj

    if plot_bound_size <= best_index < len(cam_pts) - plot_bound_size:
        plot_bounds = (best_index-plot_bound_size,best_index+plot_bound_size)
    elif best_index < plot_bound_size and best_index < len(cam_pts) - plot_bound_size:
        plot_bounds = (0, best_index+plot_bound_size)
    elif best_index >= plot_bound_size and best_index > len(cam_pts) - plot_bound_size:
        plot_bounds = (best_index-plot_bound_size,len(cam_pts)-1)
    else:
        plot_bounds = (0,len(cam_pts)-1)

    best_ts = cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,2]

    if ploton:
        ax[0,0].plot(best_ts,cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,0], '.-', label = 'cam pts y')
        # ax[0].plot(cam_pts[:,3],cam_pts[:,0], label='all cam pts')
        ax[0,0].plot(touch_lag_adj, touch_pts[:,0], '.-', label='touch lag adj')
        ax[0,0].plot(touch_pts[:,2], touch_pts[:,0], '.-', color='grey', label='raw touch', alpha=0.5)
        ax[0,0].legend()
        # ax.set_title('Trial:', trial)

        ax[0,1].plot(best_ts,cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,1], '.-', label='cam pts x')
        # ax[1].plot(cam_pts[:,3], cam_pts[:,1], label='all cam pts')
        ax[0,1].plot(touch_lag_adj,touch_pts[:,1], '.-', label='touch lag adj')
        ax[0,1].plot(touch_pts[:,2],touch_pts[:,1], '.-', color='grey', label='raw touch',alpha=0.5)
        ax[0,1].legend()

        ax[1,0].plot(best_ts,cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,0], '.-', color = 'indianred')
        # ax[0].plot(cam_pts[:,3],cam_pts[:,0], label='all cam pts')
        # ax[1,0].plot(touch_lag_adj, touch_pts[:,0], '.-', label='touch lag adj')
        ax[1,0].plot(touch_pts[:,2], touch_pts[:,0], '.-', color='indianred', label='raw touch x', alpha=0.5)
        # ax.set_title('Trial:', trial)

        ax[1,0].plot(best_ts,cam_pts[plot_bounds[0]:plot_bounds[1]+small_len,1], '.-', color = 'lightgreen')
        # ax[1].plot(cam_pts[:,3], cam_pts[:,1], label='all cam pts')
        # ax[1,0].plot(touch_lag_adj,touch_pts[:,1], '.-', label='touch lag adj')
        ax[1,0].plot(touch_pts[:,2],touch_pts[:,1], '.-', color='lightgreen', label='raw touch y',alpha=0.5)
        
        ax[1,0].plot(cam_pts[:,2], cam_pts[:,0], label='x coord')
        ax[1,0].plot(cam_pts[:,2], cam_pts[:,1], label = 'y coord')

        for p in false_alarms:
            ax[1,0].axvline(p, color='w', zorder=0, alpha = 0.1)
        ax[1,0].legend()

        print(sim_course.shape)
        ax[1,1].plot(*zip(*sim_course))
        plt.axvline(cam_pts[best_index,2], color ='w', linestyle='--')

    return lag,fig,'success'

def get_lags(dfs_func, sdir, coefs, ploton=True):
    """Function to get different types to calc lag vetween the ts and teh cam data. The euclid lag calc
    minimizes the euclidean distacne between ts stroke and cam data. This method is not as good as corr method. 
    Corr maximizes correlation between touch creen stroke and cam data. Nonetheless, funcotin will output results fo rboth methods.
    Will also plot, asusmning you to plot (will be neded for manual curation of corraltions, i.e. cut out trials that have flat lines 
    as in thes ecase the corrlation is not as good).

    Args:
        dfs_func (dict): dfs for function. Should have extracted dat from ht.process_data_single_trials with trial num keys 
            (assuming vid indexing, 0 indexing)
        sdir (str, dir-like): Name of dir to save alignment plots
        coefs (str): Coeff name used for coordinates

    Returns:
        2 dicts, with corr and euc lags indexed by trial (trial nums from input df)
    """
    plt.style.use('dark_background')
    euc_lags = {}
    corr_lags = {}
    import os
    import shutil
    from pythonlib.tools.stroketools import strokesInterpolate2,smoothStrokes
    corr_dir = f'{sdir}/corr_figs'
    if os.path.exists(corr_dir):
        shutil.rmtree(corr_dir)
    os.makedirs(corr_dir, exist_ok=True)
    dfs_func = dfs_func[coefs]

    for trial, dat in dfs_func.items():
        if len(dat) == 0:
            continue
        corr_lags[trial] = []
        euc_lags[trial] = []
        cam_pts = dat['pts_time_cam_all']
        strokes_touch = dat['strokes_touch']
        pnut_strokes_touch = dat['pnut_strokes']

        touch_fs = 1/np.mean(np.diff(strokes_touch[0][:,2]))
        cam_fs = 1/np.mean(np.diff(cam_pts[:,3]))

        
        t_stroke_start = pnut_strokes_touch[0][0,2]
        t_stroke_end = pnut_strokes_touch[-1][-1,2]

        cush = 0.5

        # restrict data to be within desired times
        all_cam = cam_pts[(cam_pts[:,3] >= t_stroke_start-cush) & (cam_pts[:,3] <= t_stroke_end+cush)]
        touch_strokes_rest = []
        for strok in strokes_touch:
            if strok[0,2] >= t_stroke_start and strok[-1,2] <= t_stroke_end:
                touch_strokes_rest.append(strok)
        strokes_touch = touch_strokes_rest
        assert len(strokes_touch) == len(pnut_strokes_touch), f'{len(strokes_touch)}, {len(pnut_strokes_touch)}'

        
        if len(all_cam) == 0:
            print('Skipping trial:', trial)
            continue

        cam_interp = strokesInterpolate2([all_cam],kind='linear',N=["fsnew",1000,cam_fs])
        cam_interp_smth = smoothStrokes(cam_interp, 1000, window_type='median')[0]
        cam_interp_smth = cam_interp_smth[:,[0,1,3]]

        touch_interp = strokesInterpolate2(strokes_touch,kind='linear',N=["fsnew",1000,touch_fs])
        touch_interp_noz = []
        for stroke in touch_interp:
            touch_interp_noz.append(stroke[:,[0,1,2]])
        # if len(touch_interp_noz) > 1:
        #     touch_interp_noz = touch_interp_noz[1:-1]
        for i,touch_stroke in enumerate(touch_interp_noz):
            touch_stroke_filt = touch_stroke
            if len(touch_stroke_filt) == 0:
                continue
            corr_lag, corr_fig, outcome = corrAlign(cam_interp_smth,touch_stroke_filt, UB=0.25, method='corr', ploton=True)
            corr_lags[trial].append(corr_lag)
            if ploton:
                if corr_fig is not None:
                    corr_fig.savefig(f'{corr_dir}/trial{trial}-{i}_corr.png')
                else:
                    print(f'Fail {trial}-{i}', outcome)
                plt.close('all')
    return corr_lags
def finalize_alignment_data(lags, good_inds):
    """ Function to generate plots on alignment/lag data

    Args:
        lags (dict): trial keyed dict with list of lags (one for each stroke) for corr method {trial:[lags],trial:[lags]}
        good_inds (list): trials and strokes with good corr plots. Should be a list fo strings formatted like:
            ['trial-stroke', ... ,'trial-stroke'] e.g. ['10-0','12-1','20-0']
            * Uses the same labelling scheme as the get_lags function in stroketools thay is called below
    """
    plt.style.use('dark_background')
    all_corr_lags = []
    corr_lag_nums = []
    for lag_trial in lags['corr_lags'].values():
        for lag in lag_trial:
            if lag is None:
                continue
            else:
                all_corr_lags.append(lag[0]-lag[1])
    for index in good_inds:
        trial = int(index.split('-')[0])
        stroke = int(index.split('-')[1])
        this_lag = lags['corr_lags'][trial][stroke]
        if this_lag is None:
            continue
        lag_num = this_lag[0]-this_lag[1]
        corr_lag_nums.append(lag_num)
        print(index,lag_num)
    euc_lag_nums = []
    # for index in good_inds:
    #     trial = int(index.split('-')[0])
    #     stroke = int(index.split('-')[1])
    #     this_lag = euc_lags[trial][stroke]
    #     if this_lag is None:
    #         continue
    #     lag_num = this_lag[0]-this_lag[1]
    #     euc_lag_nums.append(lag_num)
    #     # print(index,lag_num)
    bins = 30
    fig,ax = plt.subplots(4,1,figsize=(15,30))
    ax[0].hist(all_corr_lags, bins=bins, color='indianred', alpha=0.5, label='corr lag')
    ax[0].set_xlabel('Lag times (pos means touch_t0 > cam_t0)')
    ax[0].set_title('All lags')
    # plt.hist(all_euc_lags, bins=bins, color='lightgreen', alpha=0.5, label='euc lag')
    ax[1].hist(corr_lag_nums, bins=bins, color='indianred', alpha=0.5, label='corr lag')
    ax[1].set_title('Good inds lags')
    # plt.hist(euc_lag_nums, bins=bins, color='lightgreen', alpha=0.5, label='euc lag')

    corr_mean = round(np.mean(corr_lag_nums),4)
    # euc_mean = round(np.mean(euc_lag_nums),4)
    # plt.boxplot([corr_lag_nums,euc_lag_nums], label=[f'corr lag {corr_mean}', f'euc lag {euc_mean}'])
    ax[2].boxplot(corr_lag_nums)
    ax[2].set_title(f'Good inds lags boxplot, mean: {corr_mean}')
    ax[3].plot(all_corr_lags,'.-')
    ax[3].set_title('Lags over trials')
    return fig,corr_mean

## Gap tools
## General tools for gaps, may overlap with stroke tools but with different intentionbs
def fps(x, fs):
    '''Five point stentil function for discrete derivative, scales to m/s auto'''
    v = [(-x[i+2] + 8*x[i+1] - 8*x[i-1] + x[i-2])/12 for i in range(len(x)) if 2<=i<len(x)-2]
    return np.array(v) * fs

def fps2(x, fs):
    '''Same as above but for second derivative scales to m/s**2 auto'''
    a = [(-x[i+2] + 16*x[i+1] - 30*x[i] + 16*x[i-1] - x[i-2])/12 for i in range(len(x)) if 2<=i<len(x)-2]
    return np.array(a) * fs**2

def plotTrialsTrajectories(dat, trial_ml2, data_use='trans'):
    """Plot some relevant trajectories"""
    from pythonlib.tools.stroketools import strokesInterpolate2, smoothStrokes

    plt.style.use('dark_background')

    assert len(dat) > 0, "No data here"
    if data_use == 'trans':
        cam_pts = dat['trans_pts_time_cam_all']
        strokes_touch = dat["strokes_touch"]
    elif data_use == 'raw':
        cam_pts = dat['pts_time_cam_all']
        strokes_touch = dat["strokes_touch"]
    else:
        assert False, "Not sure what data you want to use"

    cushion = 0.1
    t_onfix_off = strokes_touch[0][-1,2]
    t_offfix_on = strokes_touch[-1][0,2]
    on_offs = {}
    on_offs['on_fix'] = [None,t_onfix_off]
    for i,strok in enumerate(strokes_touch[1:-1]):
        on_offs[f'stroke_{i}'] = []
        on_offs[f'stroke_{i}'].append(strok[0,2])
        on_offs[f'stroke_{i}'].append(strok[-1,2])
    on_offs['off_fix'] = [t_offfix_on,None]

    # filter data to be within desired times
    pts_cam = cam_pts[(cam_pts[:,3] >= t_onfix_off-cushion) & (cam_pts[:,3] <= t_offfix_on+cushion)]
    cam_fs = 1/np.mean(np.diff(pts_cam[:,3]))
    assert 49.5 <= cam_fs <= 50.5, f'cam fs of {cam_fs}hz is weird'

    #Interpolate to 100 points
    kind='linear'
    pts_cam_int = strokesInterpolate2([pts_cam],kind=kind,N=["fsnew",1000,cam_fs])[0]

    #Get z data and v data (raw and interp)
    raw_z = pts_cam[:,2]
    int_zt = np.column_stack((pts_cam_int[:,2],pts_cam_int[:,3]))
    raw_vt = np.column_stack((fps(raw_z,cam_fs),pts_cam[2:-2,3]))
    int_vt = np.column_stack((fps(pts_cam_int[:,2],1000),pts_cam_int[2:-2,3]))
    #Smooth interp data
    if_zt = smoothStrokes([int_zt], 1000, window_type='median')[0]
    if_vt = smoothStrokes([int_vt], 1000, window_type='flat')[0]

    fig = plt.figure(figsize=(20,10))
    #Plot data
    plt.plot(if_vt[:,1], if_vt[:,0], label='v_filt')
    plt.plot(if_zt[:,1], if_zt[:,0]*10, label='z_filt')
    plt.plot(pts_cam[:,3],pts_cam[:,2], '.-',color='orange',label='raw z')
    #Plot ts strokes
    ymin,ymax = plt.ylim()
    for stroke,onoff in on_offs.items():
        if stroke == 'on_fix':
            plt.fill_between([plt.xlim()[0],onoff[1]], plt.ylim()[0], plt.ylim()[1], fc='lightgreen',alpha=0.2, zorder=0)
        elif stroke == 'off_fix':
            plt.fill_between([onoff[0],plt.xlim()[1]], plt.ylim()[0], plt.ylim()[1], fc='indianred',alpha=0.2, zorder=0)
        else:
            plt.fill_between(onoff, plt.ylim()[0], plt.ylim()[1], fc='lightgrey',alpha=0.2, zorder=0)
        plt.autoscale(False)
    plt.ylim(ymin,ymax)
    # plt.xlim(xmin,xmax)
    plt.legend()
    plt.title(f'Beh {trial_ml2} : Vid {trial_ml2-1}')
    return fig

def normalizeGaps(gaps):
    """Normalize all gaps ts in list to occur in t=[0,1]. Will use minmax normal

    Args:
        gaps (array): Array of gaps (x,y,z,t)
        coord_ind (int): Index of relevant coord (default is 2/z)
    """
    gaps_norm = []
    for gap in gaps:
        ts = gap[:,-1]
        t_min = np.min(ts)
        t_max = np.max(ts)
        norm_ts = (ts - t_min)/(t_max-t_min)
        gaps_norm.append(np.column_stack((gap[:,:-1],norm_ts)))
    return gaps_norm
# def sort_gaps_by_disp_ratio(gaps,stroke_dists)
    """
    Will sort gaps based on ration between distance travelled and direct distance between strokes
    """
    

def plotGapHeat(gaps,color_ind=2,preprocess_method='norm', sort_method='disp_ratio'):
    """Plot heat maps of gaps, one gap per row could be normal. Will bad zeros on bottom if needed

    Args:
        gaps (array): Array of gaps (x,y,z,t) 
        coord_ind (int): Index of relevant coord (default is 2/z)
        preprocess_method (str):
            'pad': Pads end with zero rows
            'norm': Will normalize t to [0,1]
    """
    if sort_method == 'dur':
        gaps = sorted(gaps,key=len)
    if sort_method == 'disp_ratio':
        print('Not done yet')
    if preprocess_method == 'pad':
        max_gap_len = np.max([len(gap) for gap in gaps])
        pad_gaps = []
        for gap in gaps:
            pad_size = max_gap_len - len(gap)
            pad_gap = np.pad(gap,((0,pad_size),(0,0)))
            assert len(pad_gap) == max_gap_len, f'{len(pad_gap)},{max_gap_len}'
            pad_gaps.append(pad_gap)

        gaps = np.array(pad_gaps)
    elif preprocess_method == 'norm':
        gaps = normalizeGaps(gaps)
    color_values = gaps[:,:,color_ind]
    
    plt.figure(figsize=(10, 500))
    plt.imshow(color_values, aspect='auto', cmap='viridis', interpolation='nearest')

    # Label axes
    plt.xlabel("Time (t)")
    plt.ylabel("Gap Index")
    plt.colorbar()
    
    plt.show()

    