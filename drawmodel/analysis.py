from pythonlib.tools.stroketools import *
import matplotlib.pyplot as plt
import numpy as np

# import sys
# sys.path.insert(0, "/data1/code/python/ec/analysis/")
# from utils import loadCheckpoint

""" general purpose analyusis, including model-based analysis .
"""



# takes in a task, and computes parses, prior probabilities.
# if given the behavior, will also compute likelihood.

class Model(object):
    """ 
    - this does not save any data """
    def __init__(self, name, priorFunction, likeliFunction, use_presaved_parses=False, 
        parses=None):
        # - priorFunction, should be a function that take in a 
        # parse object (including of strokes or segs) and output (unnormalized) prior

        # - likeliFunction, should be a function that takes in a trial object (one dict). This object should be a dict that has keys for: "parses" (list of dicts), "behavior" (dict). and computes the likelihood of the trial given the parses. 
        # output should be list of distances, same length as parses.
        # convention is that smaller is better (distance)
        # - use_presaved_parses, e.g, for dreamcoder or other things where large space of parses...

        self.name = name
        self.priorFunction = priorFunction
        self.likeliFunction = likeliFunction
        self.use_presaved_parses = use_presaved_parses
        
        if self.use_presaved_parses:
            # then here need to preload the parses
            # assuming for now that this is dreamcoder saved parses
            # ECTRAIN="S12.10.test5"
            # from ec.analysis.utils import loadCheckpoint
            self.parses = parses
            # print("test")
            # DAT = loadCheckpoint(trainset=ECTRAIN, loadparse=True, suppressPrint=True, loadbehavior=False)
            # self.parses = DAT["parses"]

    def getParses(self, trial, max_num = 2000, split=None, both_direction=True):
        """ given a trial, gets all parses that have non-zero probability based
        on this model's prior 
        - trial is a dict {"behavior", "task"}, that is both the behavior 
        and ground truth task for this trial.
        - outputs parses, which is a list of dicts, each item in list is
        one potential parse.
        - max_num is in case there are too many permutations...
        - split = 2 means for each parse also get all permutations with each stroke split into 2.
        - both_direction, then gets both direction for each stroke. NOTE: this could explode in size
        number of parses.
        """
        GET_ALL_PARSES = True

        assert split in [None, 2], "not yet coded.."
        if both_direction:
            assert split is None, "if get both dir, then splitting is overkioll"

        self.parses_both_direction=both_direction

        if GET_ALL_PARSES:
            
            # -- take ground truth strokes and return all permutations
            strokes_task = trial["task"]["strokes"]

            # --- get all sequence permutations
            assert len(strokes_task)<5, "Note: note sure if should allow if large number of permutations... should lmit."
            strokes_allorders, stroke_orders_set = getAllStrokeOrders(strokes_task, num_max=max_num)

            # -- if split ... then for each strokes, get all splits
            if not split is None:
                strokes_allorders_splits = []
                strokes_allorders_splits_orders = []
                for strokes, orders in zip(strokes_allorders, stroke_orders_set):
                    S = splitStrokes(strokes, num=split)
                    strokes_allorders_splits.extend(S)
                    strokes_allorders_splits_orders.extend([orders for _ in range(len(S))])
                strokes_allorders = strokes_allorders_splits
                stroke_orders_set = strokes_allorders_splits_orders

            if both_direction:
                # then for each parse, get all the posible ways of flipping directions
                # for each stroek, while not changing order of strokes.
                strokes_bothdir = []
                strokes_bothdir_orders = []
                for strokes, orders in zip(strokes_allorders, stroke_orders_set):
                    S = getBothDirections(strokes, fake_timesteps_ver = "in_order")
                    strokes_bothdir.extend(S)
                    strokes_bothdir_orders.extend([orders for _ in range(len(S))])
                assert len(strokes_bothdir)==len(strokes_bothdir_orders), "definitely bug"
                strokes_allorders = strokes_bothdir
                stroke_orders_set = strokes_bothdir_orders

            # --- for each permutation (i.e., parse) save infoermation.
            parses = []
            for strokes, orders in zip(strokes_allorders, stroke_orders_set):
                parses.append({
                    "strokes":strokes, # is not using extra memory.
                    "order":orders,
                    })
            
            trial["model_parses"] = parses
        else:
            assert False, "not coded yet"

    def getParsesPresaved(self, trial, PARSES, Ninterp=15):
        """ PARSES is list of dicts. each dict has keys:
        'name' (string) and 'parses' (list of programs; i.e., 
        list of list of np arrays)
        - Ninterp, for converting programs to strokes. """
        from .program import program2strokes, parses2datflat, program2strokesWrapper
        from .segmentation import getSegmentation
        from ..tools.stroketools import strokesInterpolate

        t=trial
        stim = t["task"]["str"]

        P = [D for D in PARSES if D["name"]==stim]
        assert len(P)==1
        P = P[0] # this is dict with name and parses.

        print(f"{len(P['parse'])} parses found.")

        # 1) Convert parses to datflat
        datflat, datsegs = program2strokesWrapper(P["parse"], get_datsegs=True)

        # 2) extract all parses.
        parses = []
        for p, df, ds in zip(P["parse"], datflat, datsegs):
            # strokes = program2strokes(p) # appends fake timesteps
            # strokes = strokesInterpolate(strokes, N=Ninterp) # do interpolation
            
            parses.append({
                "strokes":df["trialstrokes"],
                "datseg":ds,
                "orders":None})

        t["model_parses"]=parses


    def getPriors(self, trial,  NORM_VER = "divide"):
        """ for this one trial, get prior scores for all parses
        will get both normalized (prob) and unnorm
        - NORM_VER # should change, currentrly just divisive norm.
        - NOTE: this first subtracts lowerst score, so the prob of worst will always be 0.
        """

        parses = trial["model_parses"]

        scores = []
        for p in parses:
            p["score"] = self.priorFunction(p, trial=trial)
            scores.append(p["score"])
        scores = np.array(scores)

        # get scores
        probs = self.normscore(scores, ver=NORM_VER)
        for p, r in zip(parses, probs):
            p["prob"] = r

        assert np.sum([p["prob"] for p in parses])-1<0.0001, "Did not normalize properly, probs dont sum to 1."



    # def normscore(self, score, scores_all, ver):
    #     """given one scalar (score) and list of scalars 
    #     across parses, and a method (ver), output anormalized
    #     score
    #     TODO: dont use divide. 
    #     (it doesnt work well if there exist different signs.)
    #     """

    #     if ver=="divide":
    #         # simple, just divide by sum of alls cores
    #         return (score)/np.sum(scores_all)
    #     elif ver=="softmax":
    #         # softmax. first normalize scores so that within similar range
    #         # dividing by sum of absolute values of scores.. this is hacky...

    #     else:
    #         assert False, "not coded!"
    def normscore(self, scores_all, ver):
        """given lsit of scalars (scores_all)  
        across parses, and a method (ver), output probabilsitys in 
        a list.
        TODO: dont use divide. 
        (it doesnt work well if there exist different signs.)
        """

        if ver=="divide":
            # simple, just divide by sum of alls cores
            # if any is negative, then subtracts it.
            # is very hacky!!!
            if np.any(scores_all>0) and np.any(scores_all<0):
                scores_all = scores_all - np.min(scores_all)
            s_sum = np.sum(scores_all)
            return scores_all/s_sum
            # return (score)/np.sum(scores_all)
        elif ver=="softmax":
            # softmax. first normalize scores so that within similar range
            # dividing by mean of absolute values of scores.. this is hacky...
            from scipy.special import softmax
            # scores_all = np.array([-5, 10, 25])
            # sumabs = np.sum(np.absolute(scores_all))

            # standardize scores. first subtract mean score. then divide by MAD
            scores_all = scores_all - np.mean(scores_all)
            meanabs = np.mean(np.absolute(scores_all))
            if meanabs>0:
                probs = softmax(scores_all/meanabs)
            else:
                probs = softmax(scores_all)
            # probs = softmax(scores_all)
            return probs
        else:
            assert False, "not coded!"

    def getLikelis(self, trial):
        """ given a trial object, which already has parses extracted,
        gets distance between each parse and the actual ground truth task
        - trial is a dict with "behavior" abnd "task" and "parses" entries
        - CONVENTION: more positive is better score.
        """
        # print(trial.keys())
        for a in ["behavior", "task", "model_parses"]:
            assert a in trial.keys(), f"need to get {a} first"

        likelis = self.likeliFunction(trial)
        assert len(likelis)==len(trial["model_parses"])
        for d, p in zip(likelis, trial["model_parses"]):
            p["likeli"]=d


    def getPosterior(self, trial, ver="weighted"):
        """ gioven trial object, computes posterior."""
        parses = trial["model_parses"]
        priors = np.array([p["prob"] for p in parses])
        likelis = np.array([p["likeli"] for p in parses])

        if ver=="top1":
            c = np.random.choice(np.flatnonzero(priors == priors.max())) # this randomly chooses, if there is tiebreaker.
            post = likelis[c]
        elif ver=="maxlikeli":
            # is positive control, take the maximum likeli parse
            c = np.random.choice(np.flatnonzero(likelis == likelis.max()))
            post = likelis[c]
        elif ver=="weighted":
            # baseline - weighted sum of likelihoods by prior probabilities
            post = np.average(likelis, weights=priors)
        elif ver=="likeli_weighted":
            # uses likelihoods as weights.. this is like apositive control..
            # 1) convert likelis to probabilities by softmax
            probs = self.normscore(likelis, ver="softmax")
            post = np.average(likelis, weights=probs)
        else:
            assert False, "not coded"


        trial["posterior"] = post


    
class Dataset(object):
    """ 
    - one entry for each task. saves all the parses, prior scores, likelis, etc. 
    """
    
    
    def __init__(self, behavior, tasks, split=None, first_stroke_only=False):
        """ 
        - behavior is list of dicts. must have a field called "strokes"
        - tasks is a Taskset object - should be list of dicts.
        - first_stroke_only applies only to the behavior
        """
        assert len(behavior)==len(tasks), "trials should match perfectly"

        self.trials = []
        
        for b, t, in zip(behavior, tasks):
            if not split is None:
                # split up the strokes into smaller pieces
                assert False, "not coded yet!"


            self.trials.append({
                "behavior":{"strokes":b if first_stroke_only==False else [b[0]]},
                "task":t})

        # ==== no model applied yet
        self.model = None
    


    def applyModel(self, model, parses_split=None, parses_bothdir=True, prior_ver = "softmax", 
        posterior_ver="weighted", standardize_strokes=False):
        """ model is a Model object. this locks in for this dataset a given model
        used to analysis.
        also applies the model to get parses for all trials.
        """
        
        assert self.model==None, "cannot - already applied a model."
        self.model = model

        # ====== GET PARSES
        for t in self.trials:
            if self.model.use_presaved_parses:
                self.model.getParsesPresaved(t, self.model.parses, Ninterp=15)
            else:
                self.model.getParses(t, split=parses_split, both_direction=parses_bothdir)

            if standardize_strokes:
                from pythonlib.tools.stroketools import standardizeStrokes
                # for each stroke object, center and subtract range (X)
                t["behavior"]["strokes"] = standardizeStrokes(t["behavior"]["strokes"])
                t["task"]["strokes"] = standardizeStrokes(t["task"]["strokes"])
                for p in t["model_parses"]:
                    p["strokes"] = standardizeStrokes(p["strokes"])
                # t["model_parses"]["strokes"] = standardizeStrokes(t["task"]["strokes"])


            self.model.getPriors(t, NORM_VER = prior_ver)
            self.model.getLikelis(t)
            self.model.getPosterior(t, ver=posterior_ver)



#     def # something that computes and stores data for each task.

    ####################### PLOTS
    def plotPosteriorHist(self):
        posteriors = [t["posterior"] for t in self.trials]
        plt.figure()
        plt.subplot(1,2,1)
        plt.hist(posteriors, bins=100)
        plt.subplot(1,2,2)
        plt.errorbar([0], [np.mean(posteriors)], [np.std(posteriors)])
        # plt.ylim(top=0)
        # plt.xlim()

    def plotExampleTrial(self, trialind, max_parses = 34, sort_by_likeli=False):
        """ useful plots of beahviora nd scores/priors/likelies, for a given trial
        trial inds are from 0, 1, 2, ... (i.e,. not to original trial indices in 
        filedata)"""
        print("NOTE: the x and y lims are hacky, should change")
        from .strokePlots import plotDatStrokes
        ncols = 6
        nparse = len(self.trials[trialind]["model_parses"])
        nparse = min(max_parses, nparse)
        nrows = int(np.ceil((nparse+2)/ncols))

        plt.figure(figsize=(ncols*3,nrows*3))

        # 1) ==== BEHAVIOR
        strokes_beh = self.trials[trialind]["behavior"]["strokes"]
        ax = plt.subplot(nrows, ncols, 1)
        # ax = plotTrialSimple(filedata, 1, ax=ax, plotver="empty", nakedplot=True, plot_task_stimulus=False, plot_drawing_behavior=False)
        ax.plot(1,1)
        plotDatStrokes(strokes_beh, ax=ax, plotver="raw", add_stroke_number=False)
        # plt.xlim([-400, 400])
        # plt.ylim([-600, 600])
        plt.title("behavior")

        # 2) ==== TASK
        strokes = self.trials[trialind]["task"]["strokes"]
        ax = plt.subplot(nrows, ncols, 2)
        plotDatStrokes(strokes, ax=ax, plotver="raw", add_stroke_number=False)
        ax.plot(1,1,'o')
        # plt.xlim([-400, 400])
        # plt.ylim([-600, 600])
        plt.title("task")
        parses = self.trials[trialind]["model_parses"]
        if sort_by_likeli:
            # print(sorted(parses, key=lambda x:x["likeli"])[0])
            parses = sorted(parses, key=lambda x:-x["likeli"])
        for i, P in enumerate(parses):
            if i<max_parses:
                ax = plt.subplot(nrows,ncols,i+3)
                S = P["strokes"]
                plotDatStrokes(S, ax=ax, plotver="raw", add_stroke_number=False)
                plt.title(f"sc{P['score']:.2f}, pr{P['prob']:.2f}, li{P['likeli']:.2f}")
                # plt.xlim([-400, 400])
                # plt.ylim([-400, 400])


    ################## PRINT THINGS
    def printTrialsWithPostInRange(self, post_range):
        """post_range is [min, max]. prints 
        trials in domain (0,1,...,) 
        incluscive"""

        # [len(dset.trials[i]["behavior"]["strokes"]) for i, t in enumerate(dset.trials) if t["posterior"]<-500]
        print([i for i, t in enumerate(self.trials) if t["posterior"]>=post_range[0] and t["posterior"]<=post_range[1]])




    


    
class Taskset(object):
    
    def __init__(self, tasks):
        """ tasks should be list of dicts. this will format tasks 
        so that will go into the Dataset object smothly.
        [ION PROGRESS] Not sure I actually need this. can just enter
        tasks directly into the Dataset object.
        """

        self.tasks = tasks # list of dicts.


        
################### HELPER FUNCTIONS
def makeLikeliFunction(ver="split_segments", norm_by_num_strokes=True, standardize_strokes=False):
    """ returns a function that can pass into Model,
    which does the job of computing likeli (ie., model human dist) 
    CONVENTION: more positive is better match.
    NOTE: by defai;t tje split_segments version does take into acocunt directionality.
    # 
    - norm_by_num_strokes this default, shoudl be, for cases where distance is
    summed over strokes. This is true for distanceDTW.
    - standardize_strokes, then will subtract mean and divide by x liomits range
    """
    if ver in ["split_segments", "timepoints"]:
        # these differ depending on direction fo strokes
        print("NOTE: should get parses in both directions [default for getParses()], since this distance function cares about the chron order.")

    def likeliFunction(t):
        dists_all = []
        for p in t["model_parses"]:
            strokes_parse = p["strokes"]
            if ver=="modHaussdorf":
                dist = -distanceBetweenStrokes(t["behavior"]["strokes"], strokes_parse)
            else:
                from pythonlib.tools.stroketools import distanceDTW
                dist = -distanceDTW(t["behavior"]["strokes"], strokes_parse, ver=ver)[0]
            if norm_by_num_strokes:
                dist = dist/len(t["behavior"]["strokes"])
            dists_all.append(dist)
        return dists_all
    return likeliFunction

def makePriorFunction(ver="uniform"):
    """ returns a function that can pass into Model,
    which does the job of computing prior (i.e., for each model parse 
    give unnromalized score)
    - convention: positive is better
    """
    NORM_VER = "softmax"
        
    if ver=="uniform":
        # just give it a constant
        priorFunction = lambda x, trial:1
    elif ver=="prox_to_origin":
        # first find closest line, then touch closest poitn
        # on that line 
        def getDistFromOrig(point, orig):
            return np.linalg.norm(point-orig)

        def priorFunction(p, trial):
            strokes = p["strokes"]
            centers = getCentersOfMass(strokes)
            orig = trial["task"]["fixpos"]
            distances = [getDistFromOrig(c, orig) for c in centers]
            s = np.sum(np.diff(distances)) # this is most positive when strokes are ordered from close to far
            return s
    elif ver=="distance_travel":
        from pythonlib.tools.stroketools import computeDistTraveled
        def priorFunction(p, trial):
            strokes = p["strokes"]
            orig = trial["task"]["fixpos"]
            cumdist = computeDistTraveled(strokes, orig, include_lift_periods=True)
            s = -cumdist # better if distance traveled is shorter
            return s
    elif ver=="angle_test":
        # qwuickly putting together fake angles, e.g, empirical distrubtion
        # based on single stroke tasks.

        from math import pi
        probs_empirical = {
            (0, pi/2):1,
            (pi/2, pi):0.25,
            (pi, 3*pi/2):0.25,
            (3*pi/2, 2*pi):1
        }

        def _getprob(dat):
            for angles, prob in probs_empirical.items():
                if dat>=angles[0] and dat<angles[1]:
                    return prob
                
        def priorFunction(p, trial):
            from pythonlib.tools.stroketools import stroke2angle
            strokes = p["strokes"]
            angles = stroke2angle(strokes)
            probs = [_getprob(A) for A in angles]
            s = np.sum(probs) # for now take sum over all strokes
            return s      
        NORM_VER = "divide" # since this is already in units of probabilti.
    else:
        assert False, "not coded"
    
    return priorFunction, NORM_VER







