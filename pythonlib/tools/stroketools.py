"""general purpose thing that works with stroke objects, whicha re generally lists of arrays (T x 2) (sometimes Tx3 is there is time)
and each element in a list being a stroke."""
import numpy as np


def strokesInterpolate(strokes, N):
    """interpoaltes each stroke, such that there are 
    N timesteps in each stroke. uses the actual time to interpoatle
    strokes must be list of T x 3 np ararys) """
    strokes_new = []
    for s in strokes:
        t_old = s[:,2]
        t_new = np.linspace(t_old[0], t_old[-1], num=N)
        s_new = np.concatenate([
                        np.interp(t_new, t_old, s[:,0]).reshape(-1,1), 
                        np.interp(t_new, t_old, s[:,1]).reshape(-1,1), 
                        t_new.reshape(-1,1)], 
                        axis=1)
        strokes_new.append(s_new)
    return strokes_new


def distanceDTW(strokes_beh, strokes_model, ver="timepoints", asymmetric=True):
    """inputs are lists of strokes. this first flattens the lists
    into single arrays each. then does dtw betweeen Nx2 and Mx2 arrays.
    uses euclidian distance in space as the distance function. 
    Allows to not use up all of strokes_model, but must use up all of strokes_beh
    - NOTE: a given point in strokes_beh is allowed to map onto multipe 
    points in strokes_model
    - RETURNS: (distscalar, best alignemnt.)
    - NOTE: this should make distanceBetweenStrokes() obsolete
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
    elif ver=="segments":
        # distances is between pairs of np arrays, so this
        # ignores the timesteps within each arrays.
        from pythonlib.tools.vectools import modHausdorffDistance
        distfun = lambda x,y: modHausdorffDistance(x,y, dims=[0,1])
        output = DTW(strokes_beh,strokes_model,distfun, asymmetric=asymmetric)
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
    else:
        assert False, "not coded"
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


def getStrokesFeatures(strokes):
    """output dict with features, one value for
    each stroke, eg. center of mass"""

    outdict = {
    "centers_median":[np.median(s[:,:2], axis=0) for s in strokes]
    }
    return outdict

def stroke2angle(strokes, stroke_to_use="all_strokes", force_use_two_points=False):
    """ get angles. outputs a list.
    - "first", then takes first stroke and gets vector
    from first to last point, and gets
    angle (in rad, (relative to 1,0)) for that
    - UPDATE: can now output angle for each stroke as 
    a list. note this type will be list, 
    updated to output will always be a list.
    - note, angles will all be relative to a universal (1,0)
    - will use nan wherever there is no movement for a stroke."""
    from .vectools import get_angle
    def _angle_for_one_stroke(s):
        """ s in np array T x 2(or 3)"""
        v = s[-1,[0,1]] - s[0,[0,1]]
        if np.linalg.norm(v)==0:
            return np.nan
        else:
            a = get_angle(v)
            if np.isnan(a):
                print(v)
                print(s)
                assert False
            return a

    if stroke_to_use=="first":
        # s = strokes[0]
        angles = [_angle_for_one_stroke(strokes[0])]
    elif stroke_to_use=="first_two_points":
        if not force_use_two_points:
            assert False, "have to modify to make sure that the two points are not equal. this leads to nan for the angle."
        s = strokes[0]
        s = s[[0,1],:]
        angles = [_angle_for_one_stroke(s)]
    elif stroke_to_use=="all_strokes":
        angles = [_angle_for_one_stroke(s) for s in strokes]
    else:
        assert False, "have not coded"
    return angles


def getAllStrokeOrders(strokes, num_max=None):
    """for strokes, list of np arrays, outputs a set
    of all possible stroke orders, all permuations.
    NOTE: each time run could be a different order
    NOTE: output will be in same memory location as inputs."""
    from pythonlib.tools.listtools import permuteRand
    from math import factorial
    
    nstrokes = len(strokes)
    if nstrokes==1:
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



def getCentersOfMass(strokes, method="use_median"):
    """ list, which is center for each stroke(i.e., np array) within strokes """
    if method=="use_median":
        return getStrokesFeatures(strokes)["centers_median"] # list of (x,y) arrays
    elif method=="use_mean":
        return [np.mean(s[:,:2], axis=0) for s in strokes]
    else:
        assert False, "not coded"

def _splitarray(A, num=2):
    # split one numpy array into a list of two arrays
    # idx = int(np.ceil(A.shape[0]/2))
    edges = np.linspace(0, len(A), num+1)
    edges = [int(np.ceil(e)) for e in edges]
    return [A[i1:i2,:] for i1, i2 in zip(edges[:-1], edges[1:])]

def splitStrokesOneTime(strokes, num=2):
    """like splitStrokes, but only ouptuts one list of np arrays,
    so will have 2x num arrays as input strokes. will maitnain input
    order (based on position. will ignore timepoints)"""

    strokes_split = []
    for A in strokes:
        strokes_split.extend(_splitarray(A, num=num))
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




def fakeTimesteps(strokes, point, ver):
    """strokes is a list with each stroke an nparray 
    - each stroke can be (T x 3) or T x 2. doesnt mater.
    for each array, replaces the 3rd column with new timesteps, such 
    that goes outward from point closer to origin
    - determines what timesteps to use independelty for each stroke.
    - to use origin as point, do origin=(getTrialsFix(filedata, 1)["fixpos_pixels"])
    NOTE: could pull this out of getTrialsTaskAsStrokes, but would 
    need to be able to pass in origin sopecific for eafch trial.
    - point is just any point, will make the poitn closer to this as the onset.
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

def computeDistTraveled(strokes, origin, include_lift_periods=True):
    """ assume start at origin. assumes straight line movements.
    by default includes times when not putting down ink.
    IGNORES third column(time) - i.e., assuems that datpoints are in 
    chron order."""
    
    cumdist = 0
    prev_point = origin
    for S in strokes:
        cumdist += np.linalg.norm(S[0,[0,1]] - prev_point)
        cumdist += np.linalg.norm(S[-1,[0,1]] - S[0,[0,1]])
        prev_point = S[-1, [0,1]]
    return cumdist
                                                

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


def standardizeStrokes(strokes):
    """ standardize in space (so centered at 0, and x range is from -1 to 1
    ) """
    center = np.mean(getCentersOfMass(strokes, method="use_mean"))
    xvals = [s[0] for S in strokes for s in S]
    xlims = np.percentile(xvals, [2.5, 97.5])
    x_scale = xlims[1]-xlims[0]

    # print(x_scale)
    # print(xvals)
    # print(center)
    # print(xlims)
    # new_strokes = [np.concatenate(((S[:,[0,1]]-center)/x_scale, S[:,2].reshape(-1,1)), axis=1) for S in strokes]

    # print(new_strokes)
    # assert False

    return [np.concatenate(((S[:,[0,1]]-center)/x_scale, S[:,2].reshape(-1,1)), axis=1) for S in strokes]

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

