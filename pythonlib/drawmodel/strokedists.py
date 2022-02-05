""" things that take in strokes and compute distances"""
import numpy as np
import matplotlib.pyplot as plt

def distscalarStrokes(strokes1, strokes2, ver, params=None, 
    norm_by_numstrokes=True, splitnum1 = 5, splitnum2 = 2, do_spatial_interpolate=False,
    do_spatial_interpolate_interval = 10, return_strokes_only=False):
    """ general purpose wrapper for scoring similarity of two strokes,
    - ver, str, is method
    - params, is flexible dict depends on ver
    - do_spatial_interpolate, then first interpolates so all data have uniform spacing btw points, with
    space defined by do_spatial_interpolate_interval (pixels)
    - return_strokes_only, then returns (strokes1, strokes2) after preprocessing, e.g, interpolation,
    without doing distance.
    RETURNS:
    - scalra score,
    - [if return_strokes_only] strokes1, strokes2
    TODO:
    - Frechet distnace
    - linear sum assignment.
    NOTE:
    - does _not_ modify input.
    """

    if do_spatial_interpolate:
        # interpolate all
        from pythonlib.tools.stroketools import strokesInterpolate2
        N = ["interval", do_spatial_interpolate_interval]
        base = "space"

        strokes1 = strokesInterpolate2(strokes1, N=N, base=base, plot_outcome=False)
        strokes2 = strokesInterpolate2(strokes2, N=N, base=base, plot_outcome=False)

    if return_strokes_only:
        return strokes1, strokes2

    # ==== OLD CODE - NAMES ARE NOT SYSTEMATIC
    if ver=="mindist":
        # minimum pairwise distance between strok in strokes
        dmat = distmatStrokes(strokes1, strokes2)
        print(dmat)
        assert False, "have not confirm this is correct"
    elif ver=="mindist_offdiag":
        # useful if strokes1 and strokes2 are identical and want
        # to get pairwise dist for nonidentical s in strokes.
        dmat = distmatStrokes(strokes1, strokes1)
#         f, ax = plt.subplots()
#         plotDatStrokes(strokes1, ax)
        # get off diagonal
        idx = ~np.eye(dmat.shape[0],dtype=bool)
        return np.min(dmat[idx])

    # ==== NEW CODE - NAMES ARE MORE SYSTEMATIC
    elif ver=="position_hd":
        # only based on positions (ingore time and stroke num), using hausdorff
        return distancePos(strokes1, strokes2, "hd")
    elif ver=="position_hd_soft":
        # same, but using mean, mean for hd params.
        return distancePos(strokes1, strokes2, "hd_soft")
    elif ver=="dtw_timepoints":
        return distanceDTW(strokes1, strokes2, ver="timepoints", 
            asymmetric=False, norm_by_numstrokes=norm_by_numstrokes)[0]
    elif ver=="dtw_segments":
        return distanceDTW(strokes1, strokes2, ver="segments", 
            asymmetric=False, norm_by_numstrokes=norm_by_numstrokes)[0]
    elif ver=="dtw_split_segments":
        return distanceDTW(strokes1, strokes2, ver="split_segments", 
            asymmetric=False, norm_by_numstrokes=norm_by_numstrokes,
            splitnum1=splitnum1, splitnum2=splitnum2)[0]
    elif ver=="alignment_dtwsegments":
        return strokesAlignmentScore(strokes1, strokes2, ver="dtw_split_segments")
    else:
        print(ver)
        assert False, "not codede"

        

def distmatStrokes(strokes1, strokes2, ver="mindist"):
    """ pariwise distnace btween all strok in strokes1 and 2.
    returns distmat, which is size N x M, where N and M are lenght of
    strokes1 and 2.
    - ver is what distance metric to use.
    """
    if ver=="mindist":
        def d(s1, s2):
            from pythonlib.tools.distfunctools import modHausdorffDistance as hd
            # s1 and s2 are np arrays. 
            # returns scalar.
#             from scipy.spatial.distance import cdist
#             distmat = pdist(s1, s2)
            return hd(s1, s2, ver1="min")
    else:
        print(ver)
        assert False, "not coded"
            
    distmat = np.empty((len(strokes1), len(strokes2)))
    for i, s1 in enumerate(strokes1):
        for j, s2 in enumerate(strokes2):
            distmat[i, j] = d(s1, s2)    
    return distmat
        


def distMatrixStrok(idxs1, idxs2, stroklist=None, distancever="hausdorff_means", 
                   convert_to_similarity=True, normalize_rows=False, ploton=False, 
                   normalize_cols_range01=False, distStrok_kwargs={}, 
                   rescale_strokes_ver=None, doprint=False, 
                   similarity_method="divide_by_max", cap_dist=None):
    """ 
    [use this over distmatStrokes]
    Given list of stroks, gets distance/similarity matrix, between all pariwise strokes.
    (Note, by definition, strokes is a list of np arrays)
    - idxs1, 2 are either lists of indices into stroklist, or are lists of strokes
    (if stroklist is None).
    - distancever, which metric to use between pairs of strok
    - normalize_rows, then each datapoint (row) normalized so sum across cols is 1.
    - convert_to_similarity, then returns similairty matrix instead of dist,
    where sim is defined as S = 1-D/np.max(D).
    (done in final step)
    - ploton, then plots in heatmap
    - rescale_strokes_ver, then methods to rescale stroke before computing distance.
    - cap_dist, either None or scalar. caps all distances to max this value. this useful
    if there shold not be difference between strokes that are far apart and strokes that are
    very far apart - they are both "far". reaosnalbe value is distance between strokes adjacent 
    if on a grid. (e..g, 150)
    NOTE: if stroklist is None, then idxs1 and 2 must be lists of stroks
    RETURNS: 
    - D, returns distance matrix,
    size N,M, where N is len(idxs1)...
    idxs index into stroklist
    """
    
    from pythonlib.tools.distfunctools import modHausdorffDistance, distStrok

    
    # if distancever=="hausdorff":
    #     def distfunc(strok1, strok2):
    #         return modHausdorffDistance(strok1, strok2) 
    # elif distancever=="hausdorff_means":
    #     # This helps to avoid jumps in the scores, i.e., if use "hausdorff" then 
    #     # slices (columns) will not be smooth gaussian-like things. This should 
    #     # be clear if I tink about it.
    #     def distfunc(strok1, strok2):
    #         return modHausdorffDistance(strok1, strok2, ver1="mean", ver2="mean") 
        
    if stroklist is None:
        stroklist1 = idxs1
        stroklist2 = idxs2
    else:
        stroklist1 = [stroklist[i] for i in idxs1]
        stroklist2 = [stroklist[i] for i in idxs2]

    # rescale?
    if rescale_strokes_ver is not None:
        from pythonlib.tools.stroketools import rescaleStrokes
        stroklist1 = [rescaleStrokes([s], ver=rescale_strokes_ver)[0] for s in stroklist1]
        stroklist2 = [rescaleStrokes([s], ver=rescale_strokes_ver)[0] for s in stroklist2]

    n1 = len(stroklist1)
    n2 = len(stroklist2)

    D = np.empty((n1, n2))

    for i_dat, strokdat in enumerate(stroklist1):
        if doprint:
            if i_dat%250==0:
                print(i_dat)
        for i_bas, strokbas in enumerate(stroklist2):
            # print(strokdat)
            # print(strokbas)
            d = distStrok(strokdat, strokbas, ver=distancever, **distStrok_kwargs)
            D[i_dat, i_bas] = d

    # print(D)
    if cap_dist is not None:
        D[D>cap_dist] = cap_dist
    # print(D)
    # print(n1, n2)
    # assert False
        
    if normalize_rows:
        dnorm = np.sum(D, axis=1, keepdims=True)
        D = D/dnorm

    if normalize_cols_range01:
        # then each column convert to range 0,1 (min, max distance)
        dmin = np.min(D, axis=0, keepdims=True)
        D = D-dmin
        dmax = np.max(D, axis=0, keepdims=True)
        D = D/dmax
        
        
    if convert_to_similarity:
        if similarity_method=="divide_by_max":
            D = 1-D/np.max(D)
        elif similarity_method=="divide_by_median":
            tmp = D/np.median(D)
            D = 1-tmp/np.max(tmp)
        elif similarity_method=="divide_by_maxcap":
            assert cap_dist is not None
            D = 1-D/cap_dist
        else:
            assert False

        
    if ploton:
        plt.figure()
        plt.imshow(D, cmap="gray_r", vmin=0., vmax=1.)
        plt.colorbar()
        plt.xlabel("stroklist2")
        plt.ylabel("stroklist1")

    return D



def strokesAlignmentScore(strokes_constant, strokes_perm, ver, ratio_ver="divide_by_mean", 
    Nperm=50):
    """ Methods to compute alginment bweteen strokes, 
    Intuitively, how similar are overall sequences. 
    Here explicitly do care about timing (order or withinstroke
    order).
    INPUTS:
    - ver, str indicating what method
    - ratio_vs_permutations, returns a ratio, where smaller is better, which is 
    (score)/(mean score over permtuations). i.e., if sequence is really alignemd, then
    permutations should damage score. Keeps strokes_constant constant, while getting all permutations of
    strokes_perm.
    - ratio_ver, how to summarize the actual dist vs. shuffle? 
    --- divide_by_mean,
    --- prctile, within distriubtion of perm, where does actual fall? 0 to 1.0
    NOTE:
    - in general output is distance, so larger is worse.
    - in general assumes that strokes_constant is beh, strokes_perm is model. Mostly doesnt matter, only for
    assmetric scores.
    """
    from pythonlib.drawmodel.strokedists import distscalarStrokes
    from pythonlib.tools.stroketools import getStrokePermutationsWrapper
    from math import factorial
    print("REPLACE PARTS OPF THIS CODE with scoreAgainstAllPermutations")
    assert "alignment" not in ver, "this is infinite loop, cant call self."

    def func(x, y):
        return distscalarStrokes(x, y, ver=ver)
        
    # GEt ratio_vs_permutations:

    if factorial(len(strokes_perm))<Nperm:
        # then also get directions
        strokes2_list = getStrokePermutationsWrapper(strokes_perm, ver="all_orders_directions", num_max=Nperm)
    else:
        strokes2_list = getStrokePermutationsWrapper(strokes_perm, ver="all_orders", num_max=Nperm)
    # strokes2_list = getStrokePermutationsWrapper(strokes_perm, ver="all_orders_directions", num_max=Nperm)
    distances_perm = [func(strokes_constant, s2) for s2 in strokes2_list]
    if ratio_ver=="divide_by_mean":
        dist = func(strokes_constant, strokes_perm)/np.mean(distances_perm)
    elif ratio_ver=="prctile":
        a = func(strokes_constant, strokes_perm)
        b = np.array(distances_perm)
        dist = np.sum(a>b)/len(distances_perm)
    else:
        print(ratio_ver)
        assert False
    
    return dist



########################## SPECIFIC DISTANCE FUNCTIONS
def distancePos(strokes1, strokes2, ver="hd"):
    """ distance between strokes1 and 2 only based on positions,
    so no temporal information
    """
    from ..tools.distfunctools import modHausdorffDistance
    if ver=="hd":
        pos1 = np.concatenate(strokes1, axis=0)[:,:2]
        pos2 = np.concatenate(strokes2, axis=0)[:,:2]
        d = modHausdorffDistance(pos1, pos2, dims = [0,1])
    elif ver=="hd_soft":
        pos1 = np.concatenate(strokes1, axis=0)[:,:2]
        pos2 = np.concatenate(strokes2, axis=0)[:,:2]
        d = modHausdorffDistance(pos1, pos2, dims = [0,1], ver1="mean", ver2="mean")       
    else:
        print(ver)
        assert False, "not coded"

    return d

def distanceStroksMustBePaired(strokes_beh, strokes_model, ver, 
    norm_by_numstrokes=False):
    """ Wrapper, for ways of scoring when strokes_beh and strokes_model are
    same length. If they are different, this will return nan.
    - ver, method for strok-strok dist, this passes into distStrok
    --- {set of ver in distStrok}
    --- list [ver1, ver2, ...] then will do each ver, then normalize each ver (dividing by max)
    then average across vers. THIS will return something like a ratio, [0,1] 
    [OBSOLETE]
    RETURNS: 
    - dist, scalar.
    NOTE: this is like distanceDTW, but here forces to be paired.
    NOTE: if len not same, returns nan

    """
    from pythonlib.tools.distfunctools import distStrok

    if len(strokes_beh)!=len(strokes_model):
        return np.nan

    if isinstance(ver, str):
        dist = 0
        for s1, s2 in zip(strokes_beh, strokes_model):
            dist+=distStrok(s1, s2, ver=ver, auto_interpolate_if_needed=True)
    else:
        assert False

    return dist



def distanceDTW(strokes_beh, strokes_model, ver="timepoints", 
    asymmetric=True, norm_by_numstrokes=True, splitnum1 = 3, splitnum2 = 2):
    """Get dsitnace between strokes, taking into account temporal information.
    INPUTS:
    - strokes_beh, strokes_model, list of np ararys. if asymmetric==True, then it
    matters whish is beh and model. the way it matters depends on the ver, see
    within. otherwise doesnt matter.
    - ver, string for what method to use
    --- timepoints, first flattens the lists
    into single arrays each. then does dtw betweeen Nx2 and Mx2 arrays.
    uses euclidian distance in space as the distance function. 
    --- segments, matches up segements across strokes (based on order) but ignores
    timing withing each stroke
    --- split_segments, to inforporate timing within strokes, splits up strokes, before
    matching across strokes_beh and strokes_model.
    - asymmetric, relevant for DTW, if true, then it will use up all of strokes_beh, but not
    constrianed to use up all of strokes_model. useful if they are not nececsarily smae lenght.
    logic is that must take into account all of what mopnkey did, but allow for possibility that
    monkey did not complete the entire drawing.
    - splitnum1 and 2, only relevant if ver is split_segemnts. this dictates how many segmetns
    to split up into, for beh(1) and model(2). 5 and 2 empriically seems to work well. 
    RETURNS: 
    - (distscalar, best alignemnt.)
    NOTES:
    - Allows to not use up all of strokes_model, but must use up all of strokes_beh (if 
    assymetric)
    - A given point in strokes_beh is allowed to map onto multipe 
    points in strokes_model
    - NOTE: this should make distanceBetweenStrokes() obsolete
    - norm_by_numstrokes, divide by num strokes (beh only if assyum,.
    min of task and beh if syummeteric) [note, this fixed, taking min
    is better than max, since if take max this can be cheated]
    """
    from pythonlib.tools.timeseriestools import DTW
    from pythonlib.tools.stroketools import splitStrokesOneTime

    if ver=="timepoints":
        # consider each timepoint vs. timepoint. 
        # concatenate all strokes together.
        # ignore timepoints, just go in chron order
        A = np.concatenate(strokes_beh, axis=0)[:,:2]
        B = np.concatenate(strokes_model, axis=0)[:,:2]
        distfun = lambda x,y: np.linalg.norm(x-y)
        output = DTW(A, B, distfun, asymmetric=asymmetric)
        lengths = (len(A), len(B))
    elif ver=="segments":
        # distances is between pairs of np arrays, so this
        # ignores the timesteps within each arrays.
        from pythonlib.tools.vectools import modHausdorffDistance
        distfun = lambda x,y: modHausdorffDistance(x,y, dims=[0,1])
        # print(len(strokes_beh), len(strokes_model))
        output = DTW(strokes_beh,strokes_model,distfun, asymmetric=asymmetric)
        lengths = (len(strokes_beh), len(strokes_model))
    elif ver=="split_segments":
        # distances is between pairs of np arrays, so this
        # ignores the timesteps within each arrays.
        from pythonlib.tools.vectools import modHausdorffDistance
        # NUM1 = 5 
        # NUM2 = 2 
        distfun = lambda x,y: modHausdorffDistance(x,y, dims=[0,1])
        # reduce split num based on length of strokes
        tmp = min([len(s) for s in strokes_beh])
        splitnum1 = min([tmp, splitnum1])
        tmp = min([len(s) for s in strokes_model])
        splitnum2 = min([tmp, splitnum2])

        A = splitStrokesOneTime(strokes_beh, num=splitnum1)
        B = splitStrokesOneTime(strokes_model, num=splitnum2)
        # print([len(a) for a in A])
        # print([len(a) for a in B])
        output = DTW(A,B,distfun, asymmetric=asymmetric)
        lengths = (len(A), len(B))
    else:
        print(ver)
        assert False, "not coded"

    if norm_by_numstrokes:
        output = list(output)
        if asymmetric:
            output[0] = output[0]/lengths[0]
        else:
            output[0] = output[0]/min(lengths)
    return output

    # if False:
    #     # === DEBUGGING PLOTS - takes one dataspojtna nd compares to a
    #     # bunch of random permutations, and plots distances on those plots.
    #     # NOTE: best to make a dset object, and getting both directions.
    #     from pythonlib.tools.stroketools import distanceDTW
    #     s1 = strokes_all[0]
    #     s2 = strokes_all[1]
    #     distanceDTW(s1, s2, ver="segments")[0]



    #     # =============== plot overview of distances
    #     if False:
    #         stroke1 = strokes_all[1]
    #         strokeothers = random.sample([strokes_all[i] for i in range(len(strokes_all))], 8)
    #     else:
    #         t = 10
    #         stroke1=dset.trials[t]["behavior"]["strokes"]
    #         strokes_model = [d["strokes"] for d in dset.trials[t]["model_parses"]]
    #         strokeothers = random.sample([strokes_model[i] for i in range(len(strokes_model))], 8)
    #     VER = "segments"

    #     plt.figure(figsize=(10,10))
    #     ax = plt.subplot(3,3,1)
    #     plotDatStrokes(stroke1, ax, plotver="raw")
    #     plt.xlim([-400, 400])
    #     plt.ylim([-400, 400])

    #     distances = []
    #     for i, S in enumerate(strokeothers):
    #         # get distance
    #     #     d1 = distanceDTW(stroke1, S, ver="segments")[0]
    #     #     d2 = distanceDTW(stroke1, S, ver="timepoints")[0]
    #         d3, tmp = distanceDTW(stroke1, S, ver="split_segments")
    #         print(tmp)
    #     #     d4 = distanceBetweenStrokes(stroke1, S)
    #         distances.append(d3)
    #         ax = plt.subplot(3,3,i+2)
    #         plt.title(f"seg{d1:.0f}, tp{d2/1000:.0f}\n split{d3:.0f}, old{d4:.0f}")
    #         plotDatStrokes(S, ax=ax, plotver="raw")
    #         plt.xlim([-400, 400])
    #         plt.ylim([-400, 400])
    #     plt.figure()
    #     plt.plot(np.sort(distances), 'o-k');
    #     plt.ylim(bottom=0)


######################### BATCH FUNCTIONS
def scoreAgainstBatch(strokes_beh, strokes_task_list, 
    distfunc = "COMBO-euclidian-euclidian_diffs", sort=True, 
    plots=False, confidence_ver=None):
    """ hold strokes_beh constant, get distnaces against all strokes in strokes_list(task)
    Useful plotting functions to visualize scores over all task permutations./
    INPUTS:
    - strokes_beh, strokes_task, in strokes format (list of list). Generlaly expect these to
    be for the same task
    - distfunc, to score stroke_beh againste each permtuation.
    - sort, sorts in ascending order of distance before returning.
    - plots, useful stuff.
    - confidence_ver, then computes confidence, based on method here (str).
    --- "diff_first_vs_second", the difference between rank0 and rank1 (abs) the larger the better.
    RETURNS:
    - None (if error in computing, such as if diff length strokes, but score fun
    requires same length.)
    - beh_task_distances, strokes_task_perms
    - (if confidence_ver) then beh_task_distances, strokes_task_perms, confidence_ver
    """
    # Get all permutations (and orders) for task
    from pythonlib.tools.stroketools import getStrokePermutationsWrapper
    # from pythonlib.drawmodel.features import computeDistTraveled
    # from pythonlib.drawmodel.strokedists import distscalarStrokes

    

    # Convert string distfuncs to their functions:
    if isinstance(distfunc, str):
        if distfunc == "COMBO-euclidian-euclidian_diffs":
            def dfunc1(strokes1, strokes2):
                return distanceStroksMustBePaired(strokes1, strokes2, ver='euclidian') 
            def dfunc2(strokes1, strokes2):
                return distanceStroksMustBePaired(strokes1, strokes2, ver='euclidian_diffs')
            distfunc = [dfunc1, dfunc2]
        else:
            print(distfunc)
            assert False, "not coded"

    # (3) Get all beh-task distances
    if isinstance(distfunc, list):
        # Then is multiple, 
        outs = []
        for dfn in distfunc:
            tmp = [dfn(strokes_beh, S) for S in strokes_task_list]
            if np.isnan(tmp[0]):
                # this usually mean num strokes doint match and requird to do so.
                return [np.nan for _ in range(len(strokes_task_list))], strokes_task_list
            # noramlzie to max
            tmp = np.array(tmp)
            tmp = tmp/np.max(tmp)
            outs.append(tmp)

        # average them
        beh_task_distances = np.mean(np.r_[outs], axis=0)
        assert len(beh_task_distances)==len(strokes_task_list)
        # print(outs)
        # print(np.c_[outs].shape)
        # assert False
    else:
        beh_task_distances = [distfunc(strokes_beh, S) for S in strokes_task_list]
        if np.isnan(beh_task_distances[0]):
            return None

    # Sort 
    if sort:
        tmp = [(a, b) for a, b in zip(strokes_task_list, beh_task_distances)]
        tmp = sorted(tmp, key=lambda x:x[1])
        strokes_task_list = [t[0] for t in tmp]
        beh_task_distances = [t[1] for t in tmp]

    if plots:
        from pythonlib.drawmodel.strokePlots import plotDatStrokes

        # (1) Plot behavior
        fig, axes = plt.subplots(1, 1, figsize=(1*3, 1*3))
        plotDatStrokes(ax=axes, strokes=strokes_beh, clean_ordered=True)
        axes.set_title(f"behavior")

        # (2) plot all strokes_task, ordered
        n = len(strokes_task_list)
        if n>80:
            print("SKIPPING PLOTTING<M TOO many:", n)
        else:
            nc = 6
            nr = int(np.ceil(n/nc))
            fig, axes = plt.subplots(nr, nc, figsize=(nc*3, nr*3))
            for ax, dist, strokest in zip(axes.flatten(), beh_task_distances, strokes_task_list):
                plotDatStrokes(ax=ax, strokes=strokest, clean_ordered=True)
                ax.set_title(f"dist: {dist:.2f}")

        # plot histograms
        fig, axes = plt.subplots(1, 2, figsize=(10,4))
        ax=axes.flatten()[0]
        ax.hist(beh_task_distances, 20)
        ax.set_xlabel("distances")

        ax=axes.flatten()[1]
        ax.plot(range(len(beh_task_distances)), beh_task_distances, "-ok")
        ax.axhline(0)
        ax.set_xlabel("rank")
        ax.set_ylabel("distnace")

    if confidence_ver == "diff_first_vs_second":
        assert sort==True, "need to sort first"
        confidence = np.abs(beh_task_distances[0] - beh_task_distances[1])
    else:
        assert confidence_ver is None

    # print("HERE")
    # print(len(beh_task_distances), len(strokes_task_list))

    if confidence_ver is not None:
        return beh_task_distances, strokes_task_list, confidence
    else:
        return beh_task_distances, strokes_task_list

def scoreAgainstAllPermutations(strokes_beh, strokes_task, permver="all_orders_directions",
    distfunc = "COMBO-euclidian-euclidian_diffs", **kwargs):
    """ hold strokes_beh constant, get distnaces against all permtuations of storkes_task.
    Useful plotting functions to visualize scores over all task permutations./
    INPUTS:
    - strokes_beh, strokes_task, in strokes format (list of list). Generlaly expect these to
    be for the same task
    - permver, str, how to permutate strokes_task.
    - distfunc, to score stroke_beh againste each permtuation.
    - return_strokes_permuted, then returns the list of permutated strokes (list of lists)
    - sort, sorts in ascending order of distance before returning.
    - plots, useful stuff.
    RETURNS:
    - beh_task_distances, strokes_task_perms

    """
    # Get all permutations (and orders) for task
    from pythonlib.tools.stroketools import getStrokePermutationsWrapper
    # from pythonlib.drawmodel.features import computeDistTraveled
    # from pythonlib.drawmodel.strokedists import distscalarStrokes

    # (1) get all permutations of the task strokes
    assert permver=="all_orders_directions", "have not coded else"
    assert len(strokes_task)<6, "havent figured out how to deal with long tasks..." 
    strokes_task_perms = getStrokePermutationsWrapper(strokes_task, ver=permver)
    # print("got this many perms:len(strokes_task_perms))

    return scoreAgainstBatch(strokes_beh, strokes_task_perms, 
        distfunc = distfunc, **kwargs)



######################### OLD
def distanceBetweenStrokes(strokes_beh, strokes_model, include_timesteps=False,
                          long_output=False):
    """ gets distance bewteen two sequences of strokes. takes into account
    order of strokes, but not of timepoint within the storkes.
    effectively maps on each beh stroke to a corresponding model stroke with 
    constraint that not allowed to go backwards for beh strokes.

    strokes_beh and strokes_mod are both lists of np arrays, each of which is T x 3 (if include time) or T x 2.
    
    My notes on this:
    let's say you have two ordered lists of objects, here let's say objects are strokes. 
    I'll call these lists: (a,b,c,...) and (1,2,3,...). so "a", and "1" represent different strokes. 
    So you can make a distance matrix (using modified Haussdorf distance). 
    let's say this matrix has (a,b,c...) in dim 0 (indexing rows) - behaviora strokes
    and (1,2,3).. on dim 1 (indexing columns).
    I want to find the path, starting from the top-left entry, going down and to the right, that, 
    if you sum over the values along the path, minimizes this sum distance. 
    You are not allowed to go left or up. You don't necessarily have to end up in the bottom-right entry. 
    So a1, b1, c2, d4, e4, ... is a valid sequence. But a1, b2, c1, ... is not.

    I'm currently doing this in a greedy manner, stepping from top left towards bottom right, 
    and this is OK, but can miss some cases where you should "stay" in a column and keep going down, 
    so that you can minimize a lower row.
    """
    assert False, "[distanceBetweenStrokes OBSOLETE] use distscalarStrokes"
    # 2) get the minimum distance between strokes_beh and strokes_model
    from pythonlib.pythonlib.tools.vectools import modHausdorffDistance as hd

    if include_timesteps:
        # also include time..
        dims = [0,1,2]
    else:
        dims = [0,1]
        
    distances= []
    strokes_assigned = []
    
    # 1) first strokes are always compared to each other
    distances.append(hd(strokes_beh[0], strokes_model[0], dims))
    strokes_assigned.append(0)

    # 2) the next strokes to the end
    if len(strokes_beh)==1:
        # only one comparison needed - DONE
        pass
    elif len(strokes_model)==1:
        # then every beh stroke is compared to this same stroke
        for s in strokes_beh[1:]:
            distances.append(hd(s, strokes_model[0], dims))
            strokes_assigned.append(0)
    else:
        # go thru all beh strokes
        model_strokes_being_considered=[0,1]
        model_stroke_list = list(range(len(strokes_model)))
        for s in strokes_beh[1:]:
            distances_to_compare = [hd(s, strokes_model[i], dims) for i in model_strokes_being_considered]

            # take the minimum distance
            distances.append(min(distances_to_compare))
            strokes_assigned.append(model_strokes_being_considered[np.argmin(distances_to_compare)])

            # update the strokes being considered
            if np.argmin(distances_to_compare)==0:
                # then don't update anything
                pass
            else:
                # remove the first index from consideration. add a new index
                model_strokes_being_considered = [model_strokes_being_considered[1], model_strokes_being_considered[1]+1]
            # make sure the last stroke being considered is not past the length of the model strokes
            model_strokes_being_considered = [i for i in model_strokes_being_considered if i in model_stroke_list]
    
    # --------
    if long_output:
        return distances, strokes_assigned
    else:
        return np.mean(distances)

if False:
    DOTIME = False
    LONG = True
    print(distanceBetweenStrokes(strokes_beh, strokes_model, include_timesteps=DOTIME, long_output=LONG))
    print(distanceBetweenStrokes(strokes_beh[::-1], strokes_model[::-1], include_timesteps=DOTIME, long_output=LONG))
    print(np.mean(distanceBetweenStrokes(strokes_beh, strokes_model, long_output=LONG)[0]))
    # TODO:
    # not symmetric
    # is greedy, should run back and forth.
    # fixed at 2, but should allow for not 2?

    plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all_task", strokes_to_plot="all", plotver="strokes")
    plotDictCanvasOverlay(stroke_dict, filedata, "strokes_all", plotver="strokes")
    plotTrialSimple(filedata, trials_list[trial])


