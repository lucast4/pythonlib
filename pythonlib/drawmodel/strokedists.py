""" things that take in strokes and compute distances"""
import numpy as np
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

def distscalarStrokes(strokes1, strokes2, ver, params=None):
    """ general purpose, returns scalra score,
    - ver is method
    - params is flexible dict depends on ver
    """
    
    if ver=="mindist":
        # minimum pairwise distance between strok in strokes
        dmat = distmatStrokes(strokes1, strokes2)
        print(dmat)
        assert False
    if ver=="mindist_offdiag":
        # useful if strokes1 and strokes2 are identical and want
        # to get pairwise dist for nonidentical s in strokes.
        dmat = distmatStrokes(strokes1, strokes1)
#         f, ax = plt.subplots()
#         plotDatStrokes(strokes1, ax)
        # get off diagonal
        idx = ~np.eye(dmat.shape[0],dtype=bool)
        return np.min(dmat[idx])

    else:
        print(ver)
        assert False, "not codede"
        
        
        
def distanceDTW(strokes_beh, strokes_model, ver="timepoints", 
    asymmetric=True, norm_by_numstrokes=True):
    """inputs are lists of strokes. this first flattens the lists
    into single arrays each. then does dtw betweeen Nx2 and Mx2 arrays.
    uses euclidian distance in space as the distance function. 
    Allows to not use up all of strokes_model, but must use up all of strokes_beh
    - NOTE: a given point in strokes_beh is allowed to map onto multipe 
    points in strokes_model
    - RETURNS: (distscalar, best alignemnt.)
    - NOTE: this should make distanceBetweenStrokes() obsolete
    - norm_by_numstrokes, divide by num strokes (beh only if assyum,.
    min of task and beh if syummeteric) [note, this fixed, taking min
    is better than max, since if take max this can be cheated]
    """
    from pythonlib.tools.timeseriestools import DTW
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
        NUM1 = 5 
        NUM2 = 2 
        distfun = lambda x,y: modHausdorffDistance(x,y, dims=[0,1])
        A = splitStrokesOneTime(strokes_beh, num=NUM1)
        B = splitStrokesOneTime(strokes_model, num=NUM2)
        output = DTW(A,B,distfun, asymmetric=asymmetric)
        lengths = (len(A), len(B))
    else:
        assert False, "not coded"

    if norm_by_numstrokes:
        output = list(output)
        if asymmetric:
            output[0] = output[0]/lengths[0]
        else:
            output[0] = output[0]/min(lengths)
    return output

    if False:
        # === DEBUGGING PLOTS - takes one dataspojtna nd compares to a
        # bunch of random permutations, and plots distances on those plots.
        # NOTE: best to make a dset object, and getting both directions.
        from pythonlib.tools.stroketools import distanceDTW
        s1 = strokes_all[0]
        s2 = strokes_all[1]
        distanceDTW(s1, s2, ver="segments")[0]



        # =============== plot overview of distances
        if False:
            stroke1 = strokes_all[1]
            strokeothers = random.sample([strokes_all[i] for i in range(len(strokes_all))], 8)
        else:
            t = 10
            stroke1=dset.trials[t]["behavior"]["strokes"]
            strokes_model = [d["strokes"] for d in dset.trials[t]["model_parses"]]
            strokeothers = random.sample([strokes_model[i] for i in range(len(strokes_model))], 8)
        VER = "segments"

        plt.figure(figsize=(10,10))
        ax = plt.subplot(3,3,1)
        plotDatStrokes(stroke1, ax, plotver="raw")
        plt.xlim([-400, 400])
        plt.ylim([-400, 400])

        distances = []
        for i, S in enumerate(strokeothers):
            # get distance
        #     d1 = distanceDTW(stroke1, S, ver="segments")[0]
        #     d2 = distanceDTW(stroke1, S, ver="timepoints")[0]
            d3, tmp = distanceDTW(stroke1, S, ver="split_segments")
            print(tmp)
        #     d4 = distanceBetweenStrokes(stroke1, S)
            distances.append(d3)
            ax = plt.subplot(3,3,i+2)
            plt.title(f"seg{d1:.0f}, tp{d2/1000:.0f}\n split{d3:.0f}, old{d4:.0f}")
            plotDatStrokes(S, ax=ax, plotver="raw")
            plt.xlim([-400, 400])
            plt.ylim([-400, 400])
        plt.figure()
        plt.plot(np.sort(distances), 'o-k');
        plt.ylim(bottom=0)



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
    print("[distanceBetweenStrokes OBSOLETE] use distscalarStrokes")
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
