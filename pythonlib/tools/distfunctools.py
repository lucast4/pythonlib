""" functions for computing distances between objects, generally time-series
represetning strokes, behavior, etc. """

import numpy as np

def distStrok(strok1, strok2, ver="euclidian", align_to_onset=False, rescale_ver=None,
             debug=False, auto_interpolate_if_needed=False, n_interp = 50):
    """ general purpose, distance between two strok
    - strok1, strok2, np arrays, N/M x 2, where N and 
    M could be different. This is wrapper for all other things.
    - auto_interpolate_if_needed, then if strok1 and strok2 are different lengths, and if
    ver requires same length, then will interpolate both to length n_interp
    """
    from pythonlib.tools.stroketools import strokesInterpolate2

    if align_to_onset:
        assert False, "not coded"
    if rescale_ver:
        assert False, "not coded"

    def _interp(strok1, strok2):
        # npts_space = 50
        # npts_diff = 25
        # interpolate based on spatial coordinate.
        strok1 = strokesInterpolate2([strok1], 
            N=["npts", n_interp], base="space")[0]
        strok2 = strokesInterpolate2([strok2], 
            N=["npts", n_interp], base="space")[0]
        return strok1, strok2

        
    if ver=="hausdorff":
        from pythonlib.tools.distfunctools import modHausdorffDistance
        return modHausdorffDistance(strok1, strok2) 
    elif ver=="hausdorff_means":
        # hausdorff, using means, to allow for more smooth distances
            #     # This helps to avoid jumps in the scores, i.e., if use "hausdorff" then 
    #     # slices (columns) will not be smooth gaussian-like things. This should 
    #     # be clear if I tink about it.

        from pythonlib.tools.distfunctools import modHausdorffDistance
        return modHausdorffDistance(strok1, strok2, ver1="mean", ver2="mean") 
    elif ver=="euclidian":
        # pt-by-pt euclidian distance, lengths must be matched.
        from pythonlib.tools.distfunctools import distStrokTimeptsMatched
        if auto_interpolate_if_needed:
            strok1, strok2 = _interp(strok1, strok2)
        return distStrokTimeptsMatched(strok1, strok2, min_strok_dur=None,
                                       vec_over_spatial_ratio=(1,0))
    elif ver=="euclidian_diffs":
        # pt by pt, comparing diffs between pts, using euclidian. is like velocity, but 
        # not taking into account time. 
        # By default, 
        from pythonlib.tools.distfunctools import distStrokTimeptsMatched
        from pythonlib.tools.stroketools import diff5pt, strokesVelocity, strokesDiffsBtwPts
        if auto_interpolate_if_needed:
            strok1, strok2 = _interp(strok1, strok2)
        
#         strok1diff = np.concatenate([diff5pt(strok1[:,0])[-1,None], diff5pt(strok1[:,1])], axis=1)
#         strok2diff = np.concatenate([diff5pt(strok2[:,0]), diff5pt(strok2[:,1])], axis=1)
    
        # downsample stroks by interpolation
        # want about equivalent to fs = [NO, ignore]

        a = strokesDiffsBtwPts([strok1])[0]
        b = strokesDiffsBtwPts([strok2])[0]
        
        # chop of last and first few pts...
        nremove = 3
        a = a[nremove:-nremove-2]
        b = b[nremove:-nremove-2]
        
        d = distStrokTimeptsMatched(a, b, min_strok_dur=None, vec_over_spatial_ratio=(1,0))
#         d = distStrokTimeptsMatched(
#             strokesVelocity([strok1], 125)[0][0],
#             strokesVelocity([strok2], 125)[0][0], 
#             min_strok_dur=None, vec_over_spatial_ratio=(1,0))

        if debug:
            plt.figure()
            plt.plot(a[:,0], label="1x")
            plt.plot(a[:,1], label="1y")        
            plt.plot(b[:,0], label="2x")
            plt.plot(b[:,1], label="2y")        
            plt.legend()
            plt.title(d)
    #         plt.plot(np.diff(strok1), label="x")
        return d
    else:
        print(ver)
        assert False


def distStrokTimeptsMatched(strok_beh, strok_mod, fs=None, ploton=False, 
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
    vel scores, (spatial, vel). make it None to ignore.
    - vec_over_spatial_ratio, multiples sptial score by this, useful if want
    them match, or to more strongly weigh one over other.
    - NOTE: vec_over_spatial_ratio, if scalar, then multiplies spatial dist.
    if tuple of scalars, then should be length 2 and will multiply (spatial, vec)
    - returns cost normalzied by num timesteps. """
    from pythonlib.tools.stroketools import strokesVelocity
    
    if min_strok_dur is not None:
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
        if fs is None:
            assert False, "to get vel distance, you must pass in fs"
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




def modHausdorffDistance(itemA, itemB, dims=[0,1], ver1="mean", ver2="max", D=None,
    return_marginals=False):
    """
    Modified Hausdorff Distance.

    M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
     International Conference on Pattern Recognition, pp. 566-568.

    :param itemA: [(n,2) array] coordinates of "inked" pixels
    :param itemB: [(m,2) array] coordinates of "inked" pixels
    :return dist: [float] distance

    dims = [0,1] means will use itemA[:,[0,1]] and so on.
    From Reuben Feynman and Brenden Lake
    ----------
    - D, distance matrix, optional. If pass this, then itemA, itemB, dims will not be used.
    - return_marginals, then returns marginals of distance matrix after taking min.

    """

    if D is None:
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
    elif ver1=="min":
        mean_A = np.min(mindist_A)
        mean_B = np.min(mindist_B)
        assert mean_A==mean_B, "min should be like this..."
    else:
        assert False

    if ver2=="mean":
        dist = np.mean((mean_A,mean_B))
    elif ver2=="max":
        dist = np.max((mean_A,mean_B))
    else:
        assert False

    if return_marginals:
        return dist, mindist_A, mindist_B
    else:
        return dist



