""" model that takes in beahviora nd outputs scalar representing efficiencyc ost.
Is related to "Planner" model in drawgood, but here making more general, e.g., can take
in strokes objects (list of np arrays) and generic "task" object as pairs, instead of
needing the complex "datseg" object I used in drawgood.
- Make this as simple as possioble. Model fitting, etc, will work using the taskmodel
object, which will have this efficiencycost as a sub object
- Note: I made this by coping over the Planner code, then pruning to based elements.
"""

# from preprocess import loadPreprocessedData, getFlatData
# import pickle
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

import numpy as np
# from itertools import permutations
from scipy.special import logsumexp
# import copy
# from math import factorial
# from operator import itemgetter

# from pythonlib.tools.listtools import permuteRand
# from pythonlib.tools.dicttools import printOverviewKeyValues, filterSummary
# from pythonlib.tools.snstools import addLabel

class Cost:
    def __init__(
        self, params=None):
        """ params is dict. flexible, and whatever keys are active will dictate what
        features are considered
        """
        if params is None:
            params = self.getDefaultParams()
        
        self.initialize(params)
        self.updateThetas(params)

    def getDefaultParams(self):
        """
        in all cases, params are tuple, where
        params[0] is the model params, and 
        params[1,2 ,...] are flexible params useful 
        for computing that feature.
        """

        import numpy as np
        a = lambda: np.random.rand()
        params = {
            "screenHV":[1024, 768]
        }
        params["thetas"] = {
            "strokedir":(a(), 0),
            "numstrokes":a(),
            "startloc":(a(), 0., 0.),
            "endloc":(a(),  0., 0.),
            "jumpdir":(a(), 0.)
        }
        return params


    def initialize(self, params):
        """ initialize model based on keys in params
        """

        self.Params = params

        # 2) what is scale of screen? for normalizing units
        # (goal is diagoal is ~2)
        self.Params["screenDiagPixHalf"] = \
        0.5*np.linalg.norm(self.Params["screenHV"])


    def updateThetas(self, params):
        """ takes params["thetas"], which is dict, with
        paramname: (params), and updates self.Params.
        Leaves untouched things in self.Params not related to 
        thetas.
        - usually want to keep params["thetas"] keys uinchanges, but
        subsequent code should work fine even if dont.
        """

        self.Params["thetas"] = params["thetas"]

        # 1) make sure all params are in tuple or list format.
        for k, p in self.Params["thetas"].items():
            if not isinstance(p, tuple) and not isinstance(p, list):
                # then is scalar.
                self.Params["thetas"][k] = tuple([p])

        # 3) save feature param vectors, easy to take dot products with features
        self.Params["thetavec"] = []
        self.Params["thetanames"] = []
        for k, v in self.Params["thetas"].items():
            self.Params["thetavec"].append(v[0])
            self.Params["thetanames"].append(k)

        self.Params["thetavec"] = np.array(self.Params["thetavec"])

    # def updateParams(self, params):
    #     """ given new params, preprocesses and puts ot
    #     self.Params. 
    #     - makes usre they are in format tuple (theta, **others...)
    #     - updates the theta vector extracted from params
    #     """

    #     self.Params = params

    #     # 1) make sure all params are in tuple or list format.
    #     for k, p in self.Params["thetas"].items():
    #         if not isinstance(p, tuple) and not isinstance(p, list):
    #             # then is scalar.
    #             self.Params["thetas"][k] = tuple([p])

    #     # 3) save feature param vectors, easy to take dot products with features
    #     self.Params["thetavec"] = []
    #     self.Params["thetanames"] = []
    #     for k, v in self.Params["thetas"].items():
    #         self.Params["thetavec"].append(v[0])
    #         self.Params["thetanames"].append(k)

    #     self.Params["thetavec"] = np.array(self.Params["thetavec"])

    def score(self, strokes, task, return_feature_vecs = False):
        """ given behavior (strokes) and task (task), 
        return a scalar score 
        - Note on conventions: 
        - normalization in features is goal to 
        put params in range (-1,1), or generally range 
        of 2 units (or order of that magnitude)
        - smaller values are "better" meaning less costly.
        (although this is arbitrary, since params could be neg.)
        """


        if False:
            # 1) Feature vectors for each stroke
            fstrokes = []
            for s in strokes:
                fstrokes.append(self.featuresSingleStrok(s))

            # 2) Feature vector considering entire task
            ftask = self.featuresWholeTask(strokes)

            # -- make sure no Nones, which indicates feature
            # extraction failed
            for f in fstrokes:
                assert f is not None
            for f in ftask:
                assert f is not None

            # == Combine scores into a feature vector across strokes
            # (average faet vec over strokes)
            featvec = np.r_[np.array(fstrokes).mean(0), ftask]
        else:
            featvec = self.features(strokes)
            for f in featvec:
                assert f is not None

        # score, dot using combined feat vec
        score = np.dot(featvec, self.Params["thetavec"])

        if return_feature_vecs:
            return featvec, score
        else:
            return score

    def features(self, strokes):
        from ..tools.vectools import get_angle, angle_diff
        from math import pi
        from .features import stroke2angle

        def _features(pname):
            pval = self.Params["thetas"][pname]
            if pname == "numstrokes":
                # number of strokes (0, 1, 2, ...)
                # not normalized
                nstrokes = len(strokes)
                return nstrokes
            elif pname == "startloc":
                """ position of first touch on first
                stroke.
                pval[1,2] = (x,y) location.
                returns euclidian distance from this 
                location
                """
                loc0 = np.array((pval[1], pval[2]))
                locbeh = strokes[0][0, [0,1]]
                d = np.linalg.norm(locbeh - loc0)

                # normalize ()
                return d/self.Params["screenDiagPixHalf"]
            elif pname == "endloc":
                """ position of last touch on last
                stroke.
                pval[1,2] = (x,y) location.
                returns euclidian distance from this 
                location
                """
                loc0 = np.array((pval[1], pval[2]))
                locbeh = strokes[-1][-1, [0,1]]
                d = np.linalg.norm(locbeh - loc0)

                # normalize ()
                return d/self.Params["screenDiagPixHalf"]
            elif pname == "jumpdir":
                """ overall direction of s-s transitions
                pval[1] is "ground truth" angle.
                see featuresSingleStroke to see how angle differences
                and normalization occur.
                """

                # get angle of transition between endpoint of one stroke
                # to onset of next stroke.
                jumpangles = []
                for s1, s2 in zip(strokes[:-1], strokes[1:]):
                    jump = s2[0,[0,1]] - s1[-1, [0,1]]
                    jumpangles.append(get_angle(jump))

                # for each jump angle, get its distance fromt he desired angle
                angle0 = pval[1]
                anglediffs = [angle_diff(angle0, anglej) for anglej in jumpangles]

                # normalize each, and take avearge
                return np.mean([a/pi for a in anglediffs])

            elif pname == "totaldist":
                from pythonlib.tools.stroketools import computeDistTraveled
                assert False, "not finished coding"
                orig = trial["task"]["fixpos"]
                cumdist = computeDistTraveled(strokes, orig, include_lift_periods=True)
                s = -cumdist # better if distance traveled is shorter
                return s

            elif pname == "strokedir":
                # angle of strok, in radians, distance from
                # pval[1]. angle 0 is to right, and CCW 
                # increases.
                
                vals = []
                angle0 = pval[1]
                anglelist = stroke2angle(strokes)
                for anglebeh in anglelist:
                    a = angle_diff(angle0, anglebeh)

                    # - normalize to [0,1]
                    vals.append(a/pi)
                return np.mean(vals)
            else:
                assert False, "not coded"

        fvec = []
        for pname in self.Params["thetanames"]:
            fvec.append(_features(pname))
        return fvec


    # def featuresWholeTask(self, strokes):
    #     """ feature vector for entire task.
    #     """
    #     from ..tools.vectools import get_angle, angle_diff
    #     from math import pi


    #     def _features(pname, pval):
    #         if pname == "numstrokes":
    #             # number of strokes (0, 1, 2, ...)
    #             # not normalized
    #             nstrokes = len(strokes)
    #             return nstrokes
    #         elif pname == "startloc":
    #             """ position of first touch on first
    #             stroke.
    #             pval[1,2] = (x,y) location.
    #             returns euclidian distance from this 
    #             location
    #             """
    #             loc0 = np.array((pval[1], pval[2]))
    #             locbeh = strokes[0][0, [0,1]]
    #             d = np.linalg.norm(locbeh - loc0)

    #             # normalize ()
    #             return d/self.Params["screenDiagPixHalf"]
    #         elif pname == "endloc":
    #             """ position of last touch on last
    #             stroke.
    #             pval[1,2] = (x,y) location.
    #             returns euclidian distance from this 
    #             location
    #             """
    #             loc0 = np.array((pval[1], pval[2]))
    #             locbeh = strokes[-1][-1, [0,1]]
    #             d = np.linalg.norm(locbeh - loc0)

    #             # normalize ()
    #             return d/self.Params["screenDiagPixHalf"]
    #         elif pname == "dir":
    #             """ overall direction of s-s transitions
    #             pval[1] is "ground truth" angle.
    #             see featuresSingleStroke to see how angle differences
    #             and normalization occur.
    #             """

    #             # get angle of transition between endpoint of one stroke
    #             # to onset of next stroke.
    #             jumpangles = []
    #             for s1, s2 in zip(strokes[:-1], strokes[1:]):
    #                 jump = s2[0,[0,1]] - s1[-1, [0,1]]
    #                 jumpangles.append(get_angle(jump))

    #             # for each jump angle, get its distance fromt he desired angle
    #             angle0 = pval[1]
    #             anglediffs = [angle_diff(angle0, anglej) for anglej in jumpangles]

    #             # normalize each, and take avearge
    #             return np.mean([a/pi for a in anglediffs])

    #         elif pname == "totaldist":
    #             from pythonlib.tools.stroketools import computeDistTraveled
    #             assert False, "not finished coding"
    #             orig = trial["task"]["fixpos"]
    #             cumdist = computeDistTraveled(strokes, orig, include_lift_periods=True)
    #             s = -cumdist # better if distance traveled is shorter
    #             return s

    #         else:
    #             assert False, "not coded"

    #     fvec = []
    #     for pname, pval in self.Params["task"].items():
    #         fvec.append(_features(pname, pval))
    #     return fvec

    # def featuresSingleStrok(self, strok):
    #     """ get feature vector for a single strok
    #     """
    #     from .features import stroke2angle
    #     from ..tools.vectools import angle_diff
    #     from math import pi

    #     def _features(pname, pval):
    #         if pname == "dir":
    #             # angle of strok, in radians, distance from
    #             # pval[1]. angle 0 is to right, and CCW 
    #             # increases.
                
    #             # get distance between angles
    #             angle0 = pval[1]
    #             anglebeh = stroke2angle([strok])[0]
    #             a = angle_diff(angle0, anglebeh)

    #             # - normalize to [0,1]
    #             return a/pi
    #         else:
    #             assert False, "not coded"

    #     fvec = []
    #     for pname, pval in self.Params["strok"].items():
    #         fvec.append(_features(pname, pval))
    #     return fvec

    def oldFeatureExtractors():
        assert False, "this shoudlnt be here. figure out which should go to .vectools, and which throw out"
        def _convert_to_rel_units(x,y):
            # converts from x,y to relative units from top-left corner
            # top left = (0,0)
            # bottom right = (1,1)
#             print("---")
#             print((x,y))
#             print((xmin, xmax))
#             print((ymin, ymax))
#             print("---")
            x = (x - xmin)/(xmax-xmin)
            y = (ymax - y)/(ymax-ymin)
            
            return (x,y)

        def get_lefttop_extreme(t):
            """get x and y extreme that is most top left
            """
            x = min(t["x_extremes"])
            y = max(t["y_extremes"])
            x,y = _convert_to_rel_units(x,y)
            return x,y

        def get_center_pos(t):
            x,y = _convert_to_rel_units(t["centerpos"][0], t["centerpos"][1])
            return x,y

        
        def get_vector_between_points(A,B):
            # vector from A --> B
            x = B[0] - A[0]
            y = B[1] - A[1]
            return (x,y)
            
        def get_vector_norm(x,y):
            # norm of vector (x,y), scaled so that 1 is maximum (corner to corner)
            dist = ((x**2 + y**2)**0.5)/(2**0.5) #
            return dist

        def get_dot_between_unit_vectors(u, v):
            # ignores length, just get dot of unit vectors.
            u = np.array(u)
            v = np.array(v)
            c = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v) 
            return c
            
        def get_projection_along_diagonal(x,y):
            # if x,y is vector with origin at top left, with positive values pointing towards bottom right, then
            # this gives projection along diagonal. 0 means is at top-left. 1 means is at bottom right.
            
            proj = (x*1+y*1)/(2**0.5) # get projection
            proj = proj/(2**0.5) # normalize
            return proj
        
        
        def score_start(t):
            """scores the starting token"""

            # get distance from top-left corner
            if True:
                x,y = get_lefttop_extreme(t)
            else:
                x,y = get_center_pos(t)
            # print("###")
            # print(x,y)
            # print(get_vector_norm(x,y))
            # print("###")

            cost=0
            if hasattr(self, "params_start"):
                cost+=3*self.params_start*get_vector_norm(x,y)

            return cost


        def score_trans(t1, t2):
            """t1-->t2. score this transition"""

            # === motor costs
            def motor_distance(t1, t2):
                # distance between centers
                c1 = get_center_pos(t1)
                c2 = get_center_pos(t2)
                x,y = get_vector_between_points(c1, c2)
                
                cost = get_vector_norm(x,y)
                return cost
            
            def motor_direction(t1,t2):
                # direction between strokes
                # weak bias to go left to right
                c1 = get_center_pos(t1)
                c2 = get_center_pos(t2)                
                x,y = get_vector_between_points(c1, c2)
                
                cost = -get_projection_along_diagonal(x,y)
                return cost
            
            def motor_inertia(t1, t2):
                # v is vector 
                global previous_vector
#                 print("previous vector {}".format(previous_vector))
                c1 = get_center_pos(t1)
                c2 = get_center_pos(t2)    
                x,y = get_vector_between_points(c1, c2)
#                 print("current vector {}".format((x,y)))

                if len(previous_vector)==0:
                    cost = 0
                else:
                    # - get angle between current vector and old vector
                    cost = -get_dot_between_unit_vectors(previous_vector, (x,y))
                
                previous_vector = (x,y)
                    
                return cost
                
             

            cost = 0
            if hasattr(self, "params_motor_dist"):
                cost+=3*self.params_motor_dist * motor_distance(t1, t2)    
            if hasattr(self, "params_motor_dir"):
                cost+=2*self.params_motor_dir * motor_direction(t1, t2)
            if hasattr(self, "params_motor_inertia"):
                cost+=self.params_motor_inertia * motor_inertia(t1, t2)
            
            return cost
    
        # cognitive costs
        def score_cognitive(t1, t2):
            
            def cog_primtype(t1,t2):
                # cost for switching primitive type
                def get_prim_type(t):
                    if "L" in t["codes_unique"]:
                        return "line"
                    elif "C" in t["codes_unique"]:
                        return "circle"
                if get_prim_type(t1) == get_prim_type(t2):
                    cost = 0
                else:
                    cost = 1
                return cost
            
            def cog_vertchunker(t1, t2):
                # cost for moving horizontally.
                c1 = get_center_pos(t1)
                c2 = get_center_pos(t2)
                x,_ = get_vector_between_points(c1, c2)
                
                cost = get_vector_norm(x,0)
                return cost
                
            def cog_horizchunker(t1, t2):
                # cost for moving vertically.
                c1 = get_center_pos(t1)
                c2 = get_center_pos(t2)
                _,y = get_vector_between_points(c1, c2)
                
                cost = get_vector_norm(0, y)
                return cost

            
            

            cost=0
            if hasattr(self, "params_cog_primtype"):
                cost+=self.params_cog_primtype * cog_primtype(t1, t2)
            if hasattr(self, "params_cog_vertchunker"):
                # print("ASDAS")
                # -- decide whether to add to param a value indicating this transition start from LL
                if hasattr(self, "params_cog_vertchunker_LL") and "LL" in t1["codes_unique"]:
                    # print("ADASd")
                    paramthis = self.params_cog_vertchunker + self.params_cog_vertchunker_LL
                else:
                    paramthis = self.params_cog_vertchunker
                cost+=2*paramthis*cog_vertchunker(t1, t2)
            if hasattr(self, "params_cog_horizchunker"):
                cost+=2*self.params_cog_horizchunker * cog_horizchunker(t1, t2)
            
            return cost
        
    def scoreSoftmax(self, value, valueall, debug=False):
        # returns softmax probability for cost
        # valueall would be the distribution of options (e.g, all permuations)
        assert False, "not checked"
        if debug:
            print("value")
            print(value)
            print("value all:")
            print(valueall[:10])

            print("these are log of softmax probabiltiies. should be identical")
            print(logsumexp(valueall))
            print(value - logsumexp(valueall))
            print(np.log(np.exp(value)/sum(np.exp(valueall))))
            print("this is probabilit:")
            print(np.exp(value)/sum(np.exp(valueall)))
        return value - logsumexp(valueall)
    

    ## ====== HELPER FUNCTIONS
    def printPlotSummary(self, strokes, task):
        """ overview for this data,
        - prints featurevecs
        - prints score
        - plots task
        """
        featvec, score = self.score(strokes, task,
            return_feature_vecs=True)

        print("Feature names:")
        print(self.Params["thetanames"])

        print("Feature vector: ")
        print(featvec)

        print("Feature thetas:")
        print(self.Params["thetavec"])        

        print("Score")
        print(score)



# ==== OBJECTIVE FUNCTION
# def getObjFun(datsegs, Nperm, paramslist, regularize=None):
#     # call fun to use memoized paths to generate loss
#     planner = Planner(paramslist=paramslist)
#     planner.memoizePaths(datsegs, N=Nperm)
    
#     def fun(params):
#         planner.updateParams(params)
#         score = planner.scoreMultTasks(datsegs, usememoizedPerms=True, minimalprint=True)
#         if regularize=="l2":
#             lam=(1/25) # for a sigma=5?
#             score = score-lam*np.sum(params**2)
#         elif regularize is not None:
#             print("PROBLEM - dont knwo this regularizer")
#             assert False

#         cost = -score # since want to maximize score and minimize cost.
#         return cost
    
#     return fun, planner


# from scipy.optimize import minimize as minim
# import numpy as np
# def minimize(fun, numparams):
#     params0=np.random.uniform(-8, 8, size=numparams)
#     # params = tuple([1])
#     # fun(params)
#     # res = minimize(fun, (1,), method="L-BFGS-B")
#     # res = minimize(fun, (0.5,0.5,0.5,0.5,0.5,0.5,0.5), method="L-BFGS-B")
#     # params0 = np.random.uniform(-10, 10, size=6)
#     res = minim(fun, params0, method="L-BFGS-B")
#     # res = minimize(fun, (1,), method="Nelder-Mead")
#     return res

