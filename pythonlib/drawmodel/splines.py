""" based on Reuben Feinman's pyBPL splines library
converting between strokes and splines
"""
import torch

def strokes2splines(strokes, npts_ver="unif_space", nland=7, dist_int=1.):
    """ 
    Convert from strokes to splines, with the splines "rendered" back into
    the spatial coords.
    INPUTS
    - strokes, list of np.arrays (N,2), where each is a stroke. could 
    also be (N,3) where 3rd col is time, this will be removed in splines.
    - npts_ver, how to decide how many pts in splines to return
    --- orig, keep origianl in input strokes
    --- unif_space, use npts from unif space
    - nland, number of control pts
    RETURNS
    - splines, list of np arrays, but diff num rows compared to inputs.
    NOTE: throws out time dimension
    """
    from pybpl.splines import fit_bspline_to_traj, get_stk_from_bspline
    from pybpl.data import unif_space
    
    # remove time dimenision
    strokes = [s[:,:2] for s in strokes]
        
    # keep track just in case
    # if npts_ver=="orig":
    npts = [len(s) for s in strokes]
    
    # strokes to tenorys
    strokes = [torch.tensor(s) for s in strokes]
    
    # make uniform in space
    strokes = [unif_space(s, dist_int=dist_int) for s in strokes]
    # if npts_ver=="unif_space":
    #     npts = [len(s) for s in strokes]
    
    # get splines
    if True:
        splines = [fit_bspline_to_traj(s, nland=nland) for s in strokes]
    else:
        assert False, "not coded"
        # Option 2
        # if you don't know the right number of control points, use
        # this function to determine the ncpts dynamically
        # (available at gns.omniglot.minimal_splines)
        # NOTE: make sure you have called unif_space()
        def fit_minimal_spline(stroke, thresh=0.7, max_nland=100, normalize=True):
            ntraj = stroke.size(0)
            max_nland = min(max_nland, ntraj+1)

            # determine num control points
            for nland in range(1, max_nland):
                spline, residuals = fit_bspline_to_traj(stroke, nland, include_resid=True)
                loss = residuals.sum()
                if normalize:
                    loss = loss / ntraj
                if loss < thresh:
                    return spline

            return spline


    # upsample pts to match original
    splines = [get_stk_from_bspline(spl, neval=n)
               for spl, n in zip(splines, npts)]
    
    # return to np arrays
    splines = [s.numpy() for s in splines]
    
    return splines


def add_noise_jitter_to_stroke(strok, nland = 7, dist_int = 10.,
                               sigma = 20., plot=False):
    """
    Add noise (Generate new samples of a stroke), by converting to spline, jittering control pts,
    then reconverting to trajectory.
    """
    # Generate new samples of a stroke, by converting to spline, jittering control pts.
    from pybpl.splines import fit_bspline_to_traj, get_stk_from_bspline
    from pybpl.data import unif_space
    import torch
    import numpy as np

    strok_this = strok.copy()

    # remove time dimenision
    if strok.shape[1]>2:
        t = strok_this[:, 2]
        strok_this = strok_this[:, :2].copy()

    # keep track just in case
    npts = strok_this.shape[0]

    # strokes to tenorys
    strok_this = torch.tensor(strok_this)

    # make uniform in space
    strok_this = unif_space(strok_this, dist_int=dist_int)

    # Convert to spline
    strok_spline = fit_bspline_to_traj(strok_this, nland=nland)

    ### Add noise to spline
    # Mean vector
    mean = [0, 0]
    # Covariance matrix
    cov = sigma * np.array([[1, 0], [0, 1]])
    # Sample from the multivariate normal distribution
    samples = np.random.multivariate_normal(mean, cov, size=nland)

    # print(cov)
    # print(samples)
    
    # print(samples.shape)  # (1000, 2)
    # print(type(strok_spline))
    strok_spline = strok_spline + torch.tensor(samples, dtype=strok_spline.dtype) # torch.float32
    # print(type(strok_spline))

    ### Convert from spline to stroke (and upsample pts to match original)
    strok_from_spline = get_stk_from_bspline(strok_spline, neval=npts)

    # return to np arrays
    strok_from_spline = strok_from_spline.numpy()

    if plot:
        from pythonlib.dataset.dataset_strokes import DatStrokes
        ds = DatStrokes()
        fig, axes = ds.plot_multiple_strok([strok, strok_spline.numpy(), strok_from_spline], overlay=False)
        
        for i, tit in enumerate(["original", "spline", "final(noise added)"]):
            ax = axes.flatten()[i]
            ax.set_title(tit)

    # Put time axis back in
    strok_from_spline = np.concatenate([strok_from_spline, t[:, None]], axis=1)

    return strok_from_spline