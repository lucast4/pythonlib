""" holds quick coded prior functions
(modified from drawmodel.taskmodel)
"""

import numpy as np

# def makePriorFunction(ver="uniform"):
#     """ returns a function that can pass into Model,
#     which does the job of computing prior (i.e., for each model parse 
#     give unnromalized score)
#     - convention: positive is better
#     """
#     NORM_VER = "softmax"

#     if isinstance(ver, str):
#         # pre-coded prior models
#         if ver=="uniform":
#             # just give it a constant
#             priorFunction = lambda x, trial:1
#         elif ver=="prox_to_origin":
#             # first find closest line, then touch closest poitn
#             # on that line, and so on.
#             def getDistFromOrig(point, orig):
#                 return np.linalg.norm(point-orig)

#             def priorFunction(p, trial):
#                 strokes = p["strokes"]
#                 centers = getCentersOfMass(strokes)
#                 orig = trial["task"]["fixpos"]
#                 distances = [getDistFromOrig(c, orig) for c in centers]
#                 s = np.sum(np.diff(distances)) # this is most positive when strokes are ordered from close to far
#                 return s
#         elif ver=="distance_travel":
#             from pythonlib.tools.stroketools import computeDistTraveled
#             def priorFunction(p, trial):
#                 strokes = p["strokes"]
#                 orig = trial["task"]["fixpos"]
#                 cumdist = computeDistTraveled(strokes, orig, include_lift_periods=True)
#                 s = -cumdist # better if distance traveled is shorter
#                 return s
#         elif ver=="angle_test":
#             # qwuickly putting together fake angles, e.g, empirical distrubtion
#             # based on single stroke tasks.

#             from math import pi
#             probs_empirical = {
#                 (0, pi/2):1,
#                 (pi/2, pi):0.25,
#                 (pi, 3*pi/2):0.25,
#                 (3*pi/2, 2*pi):1
#             }

#             def _getprob(dat):
#                 for angles, prob in probs_empirical.items():
#                     if dat>=angles[0] and dat<angles[1]:
#                         return prob
                    
#             def priorFunction(p, trial):
#                 from pythonlib.tools.stroketools import stroke2angle
#                 strokes = p["strokes"]
#                 angles = stroke2angle(strokes)
#                 probs = [_getprob(A) for A in angles]
#                 s = np.sum(probs) # for now take sum over all strokes
#                 return s      
#             NORM_VER = "divide" # since this is already in units of probabilti.
#         else:
#             assert False, "not coded"
#     else:
#         # Then this needs to be the function. I won't check but assume so.
#         priorFunction = ver

#     return priorFunction, NORM_VER


def makePriorFunction(ver="uniform"):
    """ 
    Simplified version, mainly for testing
    """

    # pre-coded prior models
    if ver=="distance_travel":
        from pythonlib.tools.stroketools import computeDistTraveled
        def priorFunction(p, trial):
            strokes = p["strokes"]
            orig = trial["task"]["fixpos"]
            cumdist = computeDistTraveled(strokes, orig, include_lift_periods=True)
            s = -cumdist # better if distance traveled is shorter
            return s
    else:
        print(ver)
        assert False, "not coded"

    return priorFunction



############## USING FEATURE EXTRACTOR
def prior_feature_extractor(hack_lines5=True, 
    parser_names = ["parser_graphmod", "parser_nographmod"], rule=None):
    """ combines FeatureExtractor and MotorCost to do:
    - features, scoring, normalizing to probs
    OUT:
    - Scorer object
    """
    from pythonlib.behmodel.scorer.scorer import Scorer
    from pythonlib.behmodel.scorer.utils import normscore
    from pythonlib.behmodel.feature_extractor.feature_extractor import FeatureExtractor
    from pythonlib.drawmodel.efficiencycost import Cost


    def priorscorer(D, indtrial):
        # 1) Extract features
        F = FeatureExtractor()
        list_featurevec = F.list_featurevec_from_mult_parser(D, indtrial, 
            parser_names=parser_names, hack_lines5=hack_lines5)

        # Convert a list of featurevectors to a list of scores
        a = lambda: np.random.rand()
        params = {}
        if rule=="bent":
            params["thetas"] = {
                "circ":10.,
                "dist":0.,
                "circ_max":10.,
                "nstrokes":0.,
            }
        elif rule=="straight":
            params["thetas"] = {
                "circ":-10.,
                "dist":0.,
                "circ_max":-10.,
                "nstrokes":0.,
            }
        else:
            assert False


        # 2) Dot product of features with thetas
        MC = Cost(params)
        list_scores = np.array([MC.score_features(feature_vec) for feature_vec in list_featurevec])
        
        return list_scores

    # Normalizer
    def norm(list_scores):
        beta = 1
        params = [beta, False]
        probs = normscore(list_scores, "softmax", params)
        return probs

    # Scorer
    Pr = Scorer()
    Pr.input_score_function(priorscorer)
    Pr.input_norm_function(norm)

    return Pr


