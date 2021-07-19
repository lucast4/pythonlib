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
def prior_feature_extractor_base():
    """ has norm, but no scoring fucntion
    """
    
    from pythonlib.behmodel.scorer.scorer import Scorer
    from pythonlib.behmodel.scorer.utils import normscore
    from pythonlib.behmodel.feature_extractor.feature_extractor import FeatureExtractor
    from pythonlib.drawmodel.efficiencycost import Cost

    # Normalizer
    def norm(list_scores):
        beta = 1
        params = [beta, False]
        probs = normscore(list_scores, "softmax", params)
        return probs

    # Scorer
    Pr = Scorer()
    Pr.input_norm_function(norm)
    return Pr


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


def prior_scorer_quick(ver, input_parsesflat=False):
    """ Good quick, better than above (here latest) since doesnt pass in Parsers, but 
    passes in list_of_p, so dont bneed to specify here how to parse.
    """

    from pythonlib.behmodel.scorer.scorer import Scorer
    from pythonlib.behmodel.scorer.utils import normscore
    from pythonlib.behmodel.feature_extractor.feature_extractor import FeatureExtractor
    from pythonlib.drawmodel.efficiencycost import Cost


    if ver=="lines5":

        # 1) Extract features
        # One feature extract for all models, since they are working on the same parses
        F = FeatureExtractor()
        
        # 2) Scorers, these will differ based on model.
        dict_MC = {}
        params1 = {}
        params1["thetas"] = {
            "circ":10.,
            "dist":0.,
            "circ_max":10.,
            "nstrokes":0.,
        }
        dict_MC["bent"] = Cost(params1)

        params1 = {}
        params1["thetas"] = {
            "circ":-10.,
            "dist":0.,
            "circ_max":-10.,
            "nstrokes":0.,
        }
        dict_MC["straight"] = Cost(params1)


        if input_parsesflat:
            def priorscorer(list_parses, trialcode, rule):
                MC = dict_MC[rule]
                # NOTE: computing feature vectors is slow part.
                mat_features = F.list_featurevec_from_flatparses_directly(list_parses, trialcode, 
                    hack_lines5=True)
                list_scores = np.array([MC.score_features(feature_vec) for feature_vec in mat_features])
                return list_scores
        else:
            def priorscorer(D, indtrial, rule):
                MC = dict_MC[rule]
                # NOTE: computing feature vectors is slow part.
                if True:
                    # input dataset
                    mat_features = F.list_featurevec_from_flatparses(D, indtrial, 
                        hack_lines5=True)
                else:
                    mat_features = [np.zeros(4) for _ in range(len(D.Dat.iloc[indtrial]["parses_behmod"]))]
                # for feature_vec in mat_features:
                #     print(feature_vec.shape)
                #     assert False
                if True:
                    list_scores = np.array([MC.score_features(feature_vec) for feature_vec in mat_features])
                else:
                    list_scores = np.zeros(mat_features.shape[0])
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
    else:
        assert False, "not coded"

    return Pr


def prior_scorer_quick_with_params(params0):
    """ Good quick, better than above (here latest) since doesnt pass in Parsers, but 
    passes in list_of_p, so dont bneed to specify here how to parse.
    """

    from pythonlib.behmodel.scorer.scorer import Scorer
    from pythonlib.behmodel.scorer.utils import normscore
    from pythonlib.behmodel.feature_extractor.feature_extractor import FeatureExtractor
    from pythonlib.drawmodel.efficiencycost import Cost


    # 1) Extract features
    # One feature extract for all models, since they are working on the same parses
    F = FeatureExtractor()
    
    # 2) Scorers, these will differ based on model.
    # params1["thetas"] = {
    #     "circ":10.,
    #     "dist":0.,
    #     "circ_max":10.,
    #     "nstrokes":0.,
    # }
    MC = Cost(params0)

    # def priorscorer(list_parses, trialcode, params):
    #     """ params = tuple of scalars
    #     """
    #     # NOTE: computing feature vectors is slow part.
    #     mat_features = F.list_featurevec_from_flatparses_directly(list_parses, trialcode, 
    #         hack_lines5=True)
    #     MC.updateThetaVec(params)
    #     print(mat_features.shape)
    #     print(MC.Params)
    #     assert False
    #     list_scores = np.array([MC.score_features(feature_vec) for feature_vec in mat_features])
    #     return list_scores

    # # Normalizer
    # def norm(list_scores, params):
    #     beta = params[0]
    #     # beta = 1
    #     probs = normscore(list_scores, "softmax", [beta, False])
    #     return probs

    def priorscorer(list_parses, trialcode):
        """ params = tuple of scalars
        """
        # NOTE: computing feature vectors is slow part.
        mat_features = F.list_featurevec_from_flatparses_directly(list_parses, trialcode, 
            hack_lines5=True)
        list_scores = np.array([MC.score_features(feature_vec) for feature_vec in mat_features])
        return list_scores

    # Normalizer
    def norm(list_scores, params):
        beta = params[0]
        # beta = 1
        log_probs = normscore(list_scores, "log_softmax", [beta, False])
        return log_probs

    # Scorer
    Pr = Scorer()
    # Pr.Params = {
    #     "norm":(1.),
    # }
    Pr.input_score_function(priorscorer)
    Pr.input_norm_function(norm)
    # save F and MC
    Pr.Objects = {
        "FeatureExtractor":F,
        "MotorCost":MC
    }
    Pr._do_score_with_params=False
    Pr._do_norm_with_params=True

    return Pr
