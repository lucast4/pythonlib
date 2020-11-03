""" functions for computing distances between objects, generally time-series
represetning strokes, behavior, etc. """

import numpy as np


def distStrokTimeptsMatched(strok_beh, strok_mod, fs, ploton=False, 
                           min_strok_dur=0.175, return_separate_scores=False, 
                           vec_over_spatial_ratio=1, lowpass_freq=5):
    """ if strok1 and 2 have matched timepoitns (at least 
    same num timepoint) then can compare pt by pt to 
    compute distance. 
    - distance function in both spatial (xy pos) and velocity 
    (x and y) domains. final score is the sum of these. use 
    vec_over_spatial_ratio to match theri scales (since spatial is
    usually lower magnitudes)
    - min_strok_dur, if strok shorter than this, then throws error
    - return_separate_scores, then returns tuple separating spatial and 
    vel scores, (spatial, vel)
    - vec_over_spatial_ratio, multiples sptial score by this, useful if want
    them match, or to more strongly weigh one over other.
    - NOTE: vec_over_spatial_ratio, if scalar, then multiplies spatial dist.
    if tuple of scalars, then should be length 2 and will multiply (spatial, vec)
    - returns cost normalzied by num timesteps. """
    from pythonlib.tools.stroketools import strokesVelocity
    
    if strok_beh[-1, 2]-strok_beh[0,2]<min_strok_dur:
        assert False, "stroke duration too short"

    # skip certain distances if instructed
    skip_spatial=False
    skip_vel=False
    if isinstance(vec_over_spatial_ratio, (list, tuple)):
        if vec_over_spatial_ratio[0]==0:
            skip_spatial=True
        if vec_over_spatial_ratio[1]==0:
            skip_vel=True

    # 1) spatial distance - use mean squared error - can do this since points are matched
    if skip_spatial:
        dist_spatial = 0.
    else:
        dist_spatial = np.linalg.norm(strok_beh[:,[0,1]] - strok_mod[:,[0,1]])

    # 2) velocity distance - also use mean squared error of velocity timecourses
    if skip_vel:
        dist_vel = 0.
    else:
        strok_beh_vel = strokesVelocity([strok_beh], fs=fs, lowpass_freq=lowpass_freq)[0][0]
        strok_mod_vel = strokesVelocity([strok_mod], fs=fs, lowpass_freq=lowpass_freq)[0][0]
        dist_vel = np.linalg.norm(strok_beh_vel[:,[0,1]] - strok_mod_vel[:,[0,1]])
    
    # normalize by timesteps
    dist_spatial/=strok_beh.shape[0]
    dist_vel/=strok_beh.shape[0]    

    # reweight the scores if desired
    if isinstance(vec_over_spatial_ratio, (list, tuple)):
        assert len(vec_over_spatial_ratio)==2
        dist_spatial*=vec_over_spatial_ratio[0]
        dist_vel*=vec_over_spatial_ratio[1]
    else:
        dist_spatial*=vec_over_spatial_ratio
    
    if ploton:
        import matplotlib.pyplot as plt
        from ..drawmodel.strokePlots import plotDatStrokes, plotDatStrokesTimecourse
        fig, axes = plt.subplots(3,1, figsize=(10,15))
        plotDatStrokes([strok_beh], axes[0])
        plotDatStrokes([strok_mod], axes[0])
        plotDatStrokesTimecourse([strok_beh_vel], axes[1])
        plt.ylabel("beh vel")
        plotDatStrokesTimecourse([strok_mod_vel], axes[2])
        plt.ylabel("mod vel")
    
        print(dist_spatial, dist_vel)
    
    if return_separate_scores:
        return (dist_spatial, dist_vel)
    else:
        cost = dist_spatial + dist_vel
        return cost




def modHausdorffDistance(itemA, itemB, dims=[0,1], ver1="mean", ver2="max"):
    """
    Modified Hausdorff Distance.

    M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
     International Conference on Pattern Recognition, pp. 566-568.

    :param itemA: [(n,2) array] coordinates of "inked" pixels
    :param itemB: [(m,2) array] coordinates of "inked" pixels
    :return dist: [float] distance

    dims = [0,1] means will use itemA[:,[0,1]] and so on.
    From Reuben Feynman and Brenden Lake
    """
    from scipy.spatial.distance import cdist 
    if dims:
        D = cdist(itemA[:,dims], itemB[:,dims])
    else:
        D = cdist(itemA, itemB)
    mindist_A = D.min(axis=1)
    mindist_B = D.min(axis=0)
    if ver1=="mean":
        mean_A = np.mean(mindist_A)
        mean_B = np.mean(mindist_B)
    elif ver1=="median":
        mean_A = np.median(mindist_A)
        mean_B = np.median(mindist_B)
    elif ver1=="max":
        mean_A = np.max(mindist_A)
        mean_B = np.max(mindist_B)

    if ver2=="mean":
        dist = np.mean((mean_A,mean_B))
    elif ver2=="max":
        dist = np.max((mean_A,mean_B))
    return dist

