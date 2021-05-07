""" things that take in strokes and compute distances"""
import numpy as np
import matplotlib.pyplot as plt

def distscalarStrokes(strokes1, strokes2, ver, params=None, 
    norm_by_numstrokes=True):
    """ general purpose wrapper for scoring similarity of two strokes,
    - ver, str, is method
    - params, is flexible dict depends on ver
    RETURNS:
    - scalra score,
    TODO:
    - Frechet distnace
    - linear sum assignment.
    
    """

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
    elif ver=="dtw_timepoints":
        return distanceDTW(strokes1, strokes2, ver="timepoints", 
            asymmetric=False, norm_by_numstrokes=norm_by_numstrokes)[0]
    elif ver=="dtw_segments":
        return distanceDTW(strokes1, strokes2, ver="segments", 
            asymmetric=False, norm_by_numstrokes=norm_by_numstrokes)[0]
    elif ver=="dtw_split_segments":
        return distanceDTW(strokes1, strokes2, ver="split_segments", 
            asymmetric=False, norm_by_numstrokes=norm_by_numstrokes,
            splitnum1=2, splitnum2=2)[0]
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
                   rescale_strokes_ver=None):
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
        if i_dat%250==0:
            print(i_dat)
        for i_bas, strokbas in enumerate(stroklist2):
            # print(strokdat)
            # print(strokbas)
            d = distStrok(strokdat, strokbas, ver=distancever, **distStrok_kwargs)
            D[i_dat, i_bas] = d
        
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
        D = 1-D/np.max(D)
        
    if ploton:
        plt.figure()
        plt.imshow(D, cmap="plasma")
        plt.colorbar()
        plt.xlabel("stroklist2")
        plt.ylabel("stroklist1")

    return D

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
    else:
        print(ver)
        assert False, "not coded"

    return d

def distanceDTW(strokes_beh, strokes_model, ver="timepoints", 
    asymmetric=True, norm_by_numstrokes=True, splitnum1 = 5, splitnum2 = 2):
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
        A = splitStrokesOneTime(strokes_beh, num=splitnum1)
        B = splitStrokesOneTime(strokes_model, num=splitnum2)
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
