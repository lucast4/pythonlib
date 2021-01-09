# collect all code that takes in strokes or multipel strokes
# and outputs features
# In general functions will take in a strokes obejct, which is a 
# list of np arrays, each a stroke (N x 2). in general will
# return a list of features, one for each stroke.
# NOTE Confirmed that have moved everything here from the following
# places where I also tend to put feature analyuses:
# drawmonkey.calc , ...



import numpy as np

##################################
# def strokeCenters(strokes, ver="median"):
#   """ get center of mass fo stroke. 
#   ignores timepoints"""
#   if ver=="median"
#   return [np.median(s[:,:2], axis=0) for s in strokes]


def getCentersOfMass(strokes, method="use_median"):
    """ list, which is center for each stroke(i.e., np array) within strokes """
    if method=="use_median":
        return [np.median(s[:,:2], axis=0) for s in strokes]
    elif method=="use_mean":
        return [np.mean(s[:,:2], axis=0) for s in strokes]
    else:
        assert False, "not coded"


def stroke2angle(strokes, stroke_to_use="all_strokes", 
    force_use_two_points=False):
    """ get angles. outputs a list.
    - "first", then takes first stroke and gets vector
    from first to last point, and gets
    angle (in rad, (relative to 1,0)) for that
    - UPDATE: can now output angle for each stroke as 
    a list. note this type will be list, 
    updated to output will always be a list.
    - note, angles will all be relative to a universal (1,0)
    - will use nan wherever there is no movement for a stroke."""
    from ..tools.vectools import get_angle
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


###################################
def strokeDurations(strokes):
    """ time duration for each strok"""
    return [s[-1,2]-s[0,2] for s in strokes]


def strokeDistances(strokes):
    """ cumulative distance traveled along stroke
    resturns one scalar for each strok in strokes.
    """
    def D(s):
        # s is one np.array, N by 3, N = timepoints.
        # adds up sum of distnace btween adjacent points
        d = [np.linalg.norm(s1-s2) for s1, s2 in zip(s[:-1,:2], s[1:,:2])]
        return sum(d)
    return [D(s) for s in strokes]

def strokeDisplacements(strokes):
    """displacement from strok onset to offset"""
    return [np.linalg.norm(s[-1,:2] - s[0,:2]) for s in strokes]

def strokeCircularity(strokes):
    """ 0 for straint line, 1 for full circle.
    is based on ratio of displacement to distance """
    displace = strokeDisplacements(strokes)
    distance = strokeDistances(strokes)
    return [1-p/t for p,t in zip(displace,distance)]

def strokeCurvature(strokes):
    """ see https://psycnet.apa.org/fulltext/1989-10716-001.pdf
    """
    print("see stroketools")
    assert False, "not done!!"

####### ONE VALUE SUMMARIZING STROKES
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


######## GET FEATURES, GIVEN STROKES
def strokeFeatures(strokeslist):
    """ 
    input: list(trials) of list(strokes) of np array(strok)
    retursns: list of list of dict, matching input dimension,
    where each dict corresponds to one strok.
        
    """
    
    # --- Collect features, one for each strok
    strokesfeatures = []
    for strokes in strokeslist:
        tmp = []
        for a, b in zip(
            strokeCircularity(strokes),
            strokeDistances(strokes)):

            tmp.append(
                {
                "circularity":a,
                "distance":b
                }
            )
        strokesfeatures.append(tmp)
        
    return strokesfeatures
